# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
from torch.nn.functional import one_hot as onehot
from torch.nn import CrossEntropyLoss as CEL

from tqdm import tqdm
import numpy as np
import math

from proteusAI import proteusAI
from utils.train_utils.parameter_utils import HyperParameters, TrainingParameters
from utils.train_utils.mask_utils import MASK_injection
from utils.train_utils.io_utils import Output
from utils.train_utils.data_utils import DataHolder

# ----------------------------------------------------------------------------------------------------------------------

class TrainingRun():
	'''
	the main class for training. holds all of the configuration arguments, and organnizes them into hyper-parameters, 
	training parameters, MASK_injection, data, and output objects
	also orchestrates the setup of training, training itself, validation, testing, outputs, etc.

	Attributes:
		hyper_parameters (HyperParameters): store model hyper_parameters
		training_parameters (TrainingParameters): stores training parameters
		data (DataHolder): object to store data (contains object os Data type, inherets from Dataset), splits into train, 
							val, and test depending on arguments, also capable of loading the Data into DataLoader for 
							easy integration
		output (Output): holds information about where to write output to, also contains the logging object. comes with
						methods to print logs, plot training, and save model parameters
		train_losses (dict(Losses)):
		val_losses (Losses):
		test_losses (Losses):
		gpu (torch.device): for convenience in moving tensors to GPU
		cpu (torch.device): for loading Data before moving to GPU

	'''

	def __init__(self, args):

		self.hyper_parameters = HyperParameters(    args.d_model,
													args.freeze_structure_weights,
													args.freeze_sequence_weights,
													args.cp_struct_enc_2_seq_enc,
													args.learnable_wavelengths,
													args.wf_type, args.anisotropic_wf,
													args.min_wl, args.max_wl, args.base_wl, 
													args.d_hidden_we, args.hidden_layers_we, 
													args.use_aa,
													args.d_hidden_aa, args.hidden_layers_aa, 
													args.esm2_weights_path, args.learnable_esm,
													args.struct_encoder_layers, args.seq_encoder_layers, args.num_heads,
													args.learnable_spreads,
													args.min_spread, args.max_spread, args.base_spread, args.num_spread,
													args.min_rbf, args.max_rbf, args.beta,
													args.d_hidden_attn, args.hidden_layers_attn, 
													args.temperature, args.use_model
												)
		
		self.training_parameters = TrainingParameters(  args.epochs,
														args.accumulation_steps, 
														args.lr_type, # cyclic, attn, or plataeu
														args.warmup_steps, # for attn
														args.lr_initial_min, args.lr_initial_max, args.lr_final_min, args.lr_final_max, args.lr_cycle_length, # for cyclic
														args.lr_scale, args.lr_patience, args.lr_step, # for plataeu
														args.beta1, args.beta2, args.epsilon, # for adam optim
														args.dropout, args.attn_dropout, args.wf_dropout,
														args.label_smoothing,
														args.loss_type, args.grad_clip_norm, 
														args.use_amp, args.use_chain_mask,
														args.noise_coords_std,
														args.early_stopping_thresh,
														args.early_stopping_tolerance
													)
		
		self.MASK_injection = MASK_injection(	args.mean_mask_pct,
												args.std_mask_pct,
												args.min_mask_pct,
												args.max_mask_pct,
												args.mean_span,
												args.std_span,
												args.randAA_pct, args.trueAA_pct
											)
		
		self.data = DataHolder(	args.data_path, 
								args.num_train, args.num_val, args.num_test, 
								args.batch_tokens, args.max_batch_size, 
								args.min_seq_size, args.max_seq_size, 
								args.use_chain_mask, args.min_resolution
							)
		
		self.output = Output(args.out_path, args.loss_plot, args.seq_plot, args.weights_path, args.model_checkpoints)

		self.train_losses = Losses()
		self.val_losses = Losses() # validation with all tokens being MASK
		self.val_losses_context = Losses() if args.use_aa else None # validation with the same MASK injection as the training batches from current epoch
		self.test_losses = Losses()

		self.gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.cpu = torch.device("cpu")

		self.debug_grad = args.debug_grad

	def setup_training(self):
		'''
		sets up the training by setting up the model, optimizer, scheduler, loss 
		function, scaler (if using AMP), and losses

		Args:
			None

		Returns:
			None
		'''

		self.setup_model()
		self.setup_optim()
		self.setup_scheduler()
		self.setup_loss_function()
		self.scaler = GradScaler(self.gpu) if (self.gpu is not None) and (self.training_parameters.use_amp) else None

		self.output.log_hyperparameters(self.training_parameters, self.hyper_parameters, self.MASK_injection, self.data)

	def setup_model(self):
		'''
		instantiates proteusAI with given Hyper-Parameters, moves it to gpu, and 
		optionally loads model weights from pre-trained model

		Args:
			None
		
		Returns:
			None
		'''
		
		self.output.log.info("loading model...")
		
		self.model = proteusAI(	self.hyper_parameters.d_model, 

								self.hyper_parameters.learnable_wavelengths,
								self.hyper_parameters.wf_type, 
								self.hyper_parameters.anisotropic_wf, 
								self.hyper_parameters.min_wl,
								self.hyper_parameters.max_wl,
								self.hyper_parameters.base_wl,

								self.hyper_parameters.d_hidden_we,
								self.hyper_parameters.hidden_layers_we,

								self.hyper_parameters.use_aa,
								self.hyper_parameters.d_hidden_aa,
								self.hyper_parameters.hidden_layers_aa,

								self.hyper_parameters.esm2_weights_path,
								self.hyper_parameters.learnable_esm,

								self.hyper_parameters.struct_encoder_layers,
								self.hyper_parameters.seq_encoder_layers,
								self.hyper_parameters.num_heads, 

								self.hyper_parameters.learnable_spreads,
								self.hyper_parameters.min_spread,
								self.hyper_parameters.max_spread,
								self.hyper_parameters.base_spread,
								self.hyper_parameters.num_spread,
								self.hyper_parameters.min_rbf,
								self.hyper_parameters.max_rbf,
								self.hyper_parameters.beta,

								self.hyper_parameters.d_hidden_attn,
								self.hyper_parameters.hidden_layers_attn,

								self.training_parameters.dropout,
								self.training_parameters.attn_dropout, # attention has less aggressive dropout, as it is already heavily masked
								self.training_parameters.wf_dropout
							)

		self.model.to(self.gpu)

		if self.hyper_parameters.use_model is not None:
			pretrained_weights = torch.load(self.hyper_parameters.use_model, map_location=self.gpu, weights_only=True)
			self.model.load_state_dict(pretrained_weights, strict=False) # allow modifications of the model in between transfer learning

		# first train just on structure, then when it converges, train on sequence w/ frozen structure weights
		if self.hyper_parameters.freeze_structure_weights:
			self.model.freeze_structure_weights()
		if self.hyper_parameters.freeze_sequence_weights:
			self.model.freeze_sequence_weights()
		if self.hyper_parameters.cp_struct_enc_2_seq_enc:
			self.model.cp_structEnc_2_seqEnc()

		# get number of parameters for logging
		self.training_parameters.num_params = sum(p.numel() for p in self.model.parameters())

		# print gradients at each step if in debugging mode
		if self.debug_grad:
			def print_grad(name):
				def hook(grad):
					print(f"Gradient at {name}: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}, max={grad.abs().max().item():.6f}")
				return hook

			# Attach hook to all parameters
			for name, param in self.model.named_parameters():
				param.register_hook(print_grad(name))

	def setup_optim(self):
		'''
		sets up the optimizer, zeros out the gradient

		Args:
			None
		
		Returns:
			None
		'''

		self.output.log.info("loading optimizer...")
		if self.training_parameters.lr_type == "plateu":
			lr = self.training_parameters.lr_step 
		else:
			lr = 1.0 # the lambda lr scales it
		self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, 
									betas=(self.training_parameters.beta1, self.training_parameters.beta2), 
									eps=self.training_parameters.epsilon)
		self.optim.zero_grad()

	def setup_scheduler(self):
		'''
		sets up the loss scheduler, right now only using ReduceLROnPlateu, but planning on making this configurable

		Args:
			None
		
		Returns:
			None
		'''

		self.output.log.info("loading scheduler...")

		def cyclic_lr(epoch):

			stage = epoch / self.training_parameters.epochs
			current_min = self.training_parameters.lr_initial_min - stage*(self.training_parameters.lr_initial_min - self.training_parameters.lr_final_min)
			current_max = self.training_parameters.lr_initial_max - stage*(self.training_parameters.lr_initial_max - self.training_parameters.lr_final_max)

			midpoint = (current_max + current_min) / 2
			amplitude = (current_max - current_min) / 2
			
			lr = midpoint + amplitude*math.sin(2*math.pi*epoch/self.training_parameters.lr_cycle_length)
			
			return lr

		# compute the scale
		if self.training_parameters.lr_step == 0.0:
			scale = self.hyper_parameters.d_model**(-0.5)
		else:
			scale = self.training_parameters.warmup_steps**(0.5) * self.training_parameters.lr_step # scale needed so max lr is what was specified
		
		def attn_lr(step):
			'''lr scheduler from attn paper'''
			return scale * min((step+1)**(-0.5), (step+1)*(self.training_parameters.warmup_steps**(-1.5)))
		
		if self.training_parameters.lr_type == "plateau":
			self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=self.training_parameters.lr_scale, patience=self.training_parameters.lr_patience) 
		elif self.training_parameters.lr_type == "cyclic":
			self.scheduler = lr_scheduler.LambdaLR(self.optim, cyclic_lr)
		elif self.training_parameters.lr_type == "attn":
			self.scheduler = lr_scheduler.LambdaLR(self.optim, attn_lr)

	def setup_loss_function(self):
		'''
		initializes the loss function

		Args:
			None

		Returns: 
			None
		'''

		self.output.log.info("loading loss function...") 
		self.loss_function = CEL(ignore_index=-1, reduction=self.training_parameters.loss_type, label_smoothing=self.training_parameters.label_smoothing)

	def train(self):
		'''
		entry point for training the model. loads train and validation data, loops through epochs, plots training, 
		runs testing and saves the model

		Args:
			None
		
		Return:
			None
		'''

		# load the data
		self.output.log.info("loading training data...")
		self.data.load("train")
		self.output.log.info("loading validation data...")
		self.data.load("val")

		# log training info
		self.output.log.info(f"\n\ninitializing training. "\
							f"training on {len(self.data.train_data)} batches "\
							f"of batch size {self.data.batch_tokens} tokens "\
							f"for {self.training_parameters.epochs} epochs.\n" )
		
		# loop through epochs
		for epoch_idx in range(self.training_parameters.epochs):
			epoch = Epoch(epoch_idx, self)
			epoch.epoch_loop()

			if (epoch_idx+1) % self.output.model_checkpoints == 0: # model checkpointing
				self.output.save_model(self.model, f"e{epoch_idx}_s{round(self.val_losses.matches[-1],2)}")

			if self.training_converged(): # early stopping
				self.output.log.info(f"training converged after {epoch_idx} epochs")
				break

		self.output.plot_training(self.train_losses, self.val_losses, self.val_losses_context)
		self.output.save_model(self.model)
				
	def training_converged(self):

		# val losses are already in avg seq sim format per epoch
		if self.training_parameters.early_stopping_tolerance+1 > len(self.val_losses.matches):
			return False
		current_seq_sims = self.val_losses.matches[-(self.training_parameters.early_stopping_tolerance):]
		old_seq_sim = self.val_losses.matches[-(self.training_parameters.early_stopping_tolerance+1)]

		best_delta_seq_sim = -float("inf")
		for current_seq_sim in current_seq_sims:
			delta_seq_sim = current_seq_sim - old_seq_sim
			best_delta_seq_sim = max(best_delta_seq_sim, delta_seq_sim) 

		return best_delta_seq_sim < self.training_parameters.early_stopping_thresh

	def all_MASK_validation(self):
		'''run validation with no AA context'''
		self.output.log.info(f"running validation w/ no context...\n")
		self.validation()

	def batch_MASK_validation(self, epoch):
		'''run validation with the same context as the training batches from the current epoch'''
		self.output.log.info(f"running validation w/ context...\n")
		self.validation(epoch)

	def validation(self, epoch=None):		
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		# store intermediate losses here temporarily before storing the avg in the training run losses
		val_losses = Losses()
		
		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			val_pbar = tqdm(total=len(self.data.val_data), desc="epoch_validation_progress", unit="step")
			
			# loop through validation batches
			for label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask in self.data.val_data:
					
				# init batch
				batch = Batch(	label_batch, coords_batch, 
								chain_mask, chain_idxs, 
								key_padding_mask, 
								epoch=epoch, 
								use_amp=False, auto_regressive=False
							)

				# mask tokens depending on required context
				if epoch is not None:
					self.MASK_injection.MASK_tokens(batch)
				else:
					self.MASK_all(batch)

				# run the model
				batch.batch_forward(self.model, self.loss_function, self.gpu)

				# store losses
				val_losses.extend_losses(batch.outputs.output_losses)

				# update pbar
				val_pbar.update(1)

			# add the avg losses to the global loss and log
			if epoch is None:
				self.output.log_val_losses(val_losses, self.val_losses)
			else:
				self.output.log_val_losses(val_losses, self.val_losses_context)

	def test(self):

		# switch to evaluation mode
		self.model.eval()
		
		# load testing data
		self.output.log.info("loading testing data...")
		self.data.load("test")

		# init losses
		test_losses = Losses()
		test_ar_losses = Losses()

		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			test_pbar = tqdm(total=len(self.data.test_data), desc="test_progress", unit="step")

			# loop through testing batches
			for label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask in self.data.test_data:
					
				# init batch
				batch = Batch(  label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask, 
								use_amp=False, 
								auto_regressive=self.hyper_parameters.use_aa, # only do autoregressive if using aa modules 
								temp=self.hyper_parameters.temperature, 
							)

				# mask all tokens
				self.MASK_all(batch)

				# run the model
				batch.batch_forward(self.model, self.loss_function, self.gpu)

				# add the losses
				test_losses.extend_losses(batch.outputs.output_losses)
				test_ar_losses.extend_losses(batch.outputs.ar_output_losses)

				# update pbar
				test_pbar.update(1)
		
		# log the losses
		self.output.log_test_losses(test_losses, test_ar_losses)

	def MASK_all(self, batch):

		batch.predictions = onehot(torch.full((batch.size(0), batch.size(1)), 20, device=batch.predictions.device), 21).float()

class Epoch():	
	def __init__(self, epoch, training_run):

		self.training_run_parent = training_run

		self.epoch = epoch
		self.epochs = self.training_run_parent.training_parameters.epochs
		self.stage = epoch / self.epochs

		self.losses = Losses()
		
	def epoch_loop(self):
		'''
		a single training loop through one epoch. sets up epoch input perturbation values depending on the stage (calculated in Epoch.__init__)
		then loops through batches, logs the losses, and runs validation

		Args:
			None

		Returns:
			None
		'''

		# make sure in training mode
		self.training_run_parent.model.train()

		# setup the epoch
		self.training_run_parent.output.log_epoch(self, self.training_run_parent.scheduler.get_last_lr()[0])

		# attn scheduler updates every step, others every epoch
		scheduler = self.training_run_parent.scheduler if self.training_run_parent.training_parameters.lr_type=="attn" else None 
		
		# loop through batches
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")
		for b_idx, (label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask) in enumerate(self.training_run_parent.data.train_data):
					
			# instantiate this batch
			batch = Batch(	label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask, 
							b_idx=b_idx, epoch=self, 
							use_amp=self.training_run_parent.training_parameters.use_amp, 
						)

			if self.training_run_parent.hyper_parameters.use_aa:
				# inject MASK tokens for prediction
				self.training_run_parent.MASK_injection.MASK_tokens(batch)

			# add random noise to the coordinates
			batch.noise_coords(self.training_run_parent.training_parameters.noise_coords_std)

			# learn
			batch.batch_learn(scheduler=scheduler)

			# compile batch losses for logging
			self.losses.extend_losses(batch.outputs.output_losses)

			epoch_pbar.update(1)
		
		# print batch losses
		self.training_run_parent.output.log_epoch_losses(self, self.training_run_parent.train_losses)

		# run validation
		if self.training_run_parent.hyper_parameters.use_aa:
			self.training_run_parent.batch_MASK_validation(self) # only run validation w/ context if training w/ aa info
		self.training_run_parent.all_MASK_validation() 

		# lr scheduler update for cyclic/plateu
		if scheduler is None:
			self.training_run_parent.scheduler.step(self.training_run_parent.val_losses_context.losses[-1])

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()

class Batch():
	def __init__(self, labels, coords, chain_mask, chain_idxs, key_padding_mask, 
					b_idx=None, epoch=None,
					use_amp=True, auto_regressive=False, temp=0.1
				):

		self.labels = labels
		self.coords = coords 
		self.chain_mask = chain_mask
		self.chain_idxs = chain_idxs # for computing virtual Cb if using Ca only model, otherwise is computed directly from backbone coords
		self.use_amp = use_amp

		self.predictions = onehot(torch.where(labels==-1, 20, labels), 21).float()

		self.key_padding_mask = key_padding_mask
		self.onehot_mask = torch.zeros(self.key_padding_mask.shape, dtype=torch.bool)

		self.b_idx = b_idx
		self.epoch_parent = epoch

		self.use_amp = use_amp
		self.auto_regressive = auto_regressive
		self.temp = temp

		self.outputs = None

	def move_to(self, device):

		self.predictions = self.predictions.to(device)
		self.labels = self.labels.to(device)
		self.coords = self.coords.to(device)
		self.chain_mask = self.chain_mask.to(device)
		self.key_padding_mask = self.key_padding_mask.to(device)
		self.onehot_mask = self.onehot_mask.to(device)

	def noise_coords(self, std=0.02):

		'''data augmentation via gaussian noise injection into coords, default is 0.02 A standard deviation, centered around 0'''

		# define noise
		noise = torch.randn_like(self.coords) * std

		self.coords = self.coords + noise

	def batch_learn(self, scheduler=None):
		'''
		a single iteration over a batch.

		Args:
			None

		Returns:
			None
		'''

		# forward pass
		self.batch_forward(self.epoch_parent.training_run_parent.model, self.epoch_parent.training_run_parent.loss_function, self.epoch_parent.training_run_parent.gpu)

		# backward pass
		self.batch_backward(scheduler)

	def batch_forward(self, model, loss_function, device):
		'''
		performs the forward pass, gets the outputs and computes the losses of a batch. 
		option to use Automatic Mixed Precision (AMP)

		Args:
			batch (Batch): Batch objects, contains info about batch and useful methods
			use_amp (bool): whether to use AMP or not
			auto_regressive (bool): option for auto-regressive inference
			temp (float): temperature, only applicable if auto_regressive==True

		Returns:
			None
		  
		'''
		
		# move batch to gpu
		self.move_to(device)

		# mask one hot positions for loss. also only compute loss for the representative chain of the sequence cluster (unless config says not to, then chain mask is all zeros)
		self.labels = torch.where(self.onehot_mask | self.chain_mask | self.key_padding_mask, -1, self.labels)

		if self.use_amp: # optional AMP
			with autocast('cuda'):
				self.outputs = self.get_outputs(model)
		else:
			self.outputs = self.get_outputs(model)

		self.outputs.get_losses(loss_function)

	def batch_backward(self, scheduler=None):

		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.accumulation_steps
		scaler = self.epoch_parent.training_run_parent.scaler
		optim = self.epoch_parent.training_run_parent.optim

		learn_step = (self.b_idx + 1) % accumulation_steps == 0
		loss = self.outputs.output_losses.losses[-1] / accumulation_steps

		if scaler is not None:
			scaler.scale(loss).backward()

			if learn_step:

				# Apply gradient clipping using the scaler
				if self.epoch_parent.training_run_parent.training_parameters.grad_clip_norm:
					scaler.unscale_(optim)  # Unscale gradients before clipping
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.grad_clip_norm)

				scaler.step(optim)
				scaler.update()
				optim.zero_grad()
				if scheduler is not None:
					scheduler.step()
		else:
			loss.backward()
			if learn_step:
				if self.epoch_parent.training_run_parent.training_parameters.grad_clip_norm:
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.grad_clip_norm)
				optim.step()
				optim.zero_grad()
				if scheduler is not None:
					scheduler.step()

	def get_outputs(self, model):
		'''
		used to get output predictions for various scenarious, such as one-shot predictions from engineered inputs,
		one-shot predictions using previous outputs as inputs, or autoregressive inference
		
		Args:
			batch (Batch): Batch object, holds features, labels, and its own losses
			auto_regressive (bool): whether to do autoregressive inference
			temp (float): temperature if using autoregression

		Returns:
			None
		'''

		# compute autoregressive prediction if specified
		# cant compute cel with one hot vector, this is only for testing
		with torch.no_grad():
			if self.auto_regressive: 
				ar_output_prediction = model(self.coords, self.predictions, self.chain_idxs, key_padding_mask=self.key_padding_mask, 
											auto_regressive=self.auto_regressive, temp=self.temp)
			else:
				ar_output_prediction = None

		# compute one shot also
		output_prediction = model(self.coords, self.predictions, self.chain_idxs, key_padding_mask=self.key_padding_mask, auto_regressive=False)

		return ModelOutputs(output_prediction, ar_output_prediction, self.labels)

	def size(self, idx):
		return self.labels.size(idx)

class ModelOutputs():

	def __init__(self, output_predictions, ar_output_predictions, labels, loss_type="sum"):
		
		self.output_predictions = output_predictions
		self.ar_output_predictions = ar_output_predictions

		self.labels = labels
		self.valid_toks = (labels!=-1).sum().item() if loss_type == "sum" else 1 # 1 valid tok if computing mean since already normalized
		self.loss_type = loss_type

		self.output_losses = Losses()
		self.ar_output_losses = Losses()

	def compute_matches(self, prediction):
		'''greedy selection, just for tracking progress'''
		
		prediction_flat = prediction.contiguous().view(-1, prediction.size(-1)) # batch*N x 20
		labels_flat = self.labels.view(-1) # batch x N --> batch*N,

		seq_predictions = torch.argmax(prediction_flat, dim=-1) # batch*N x 20 --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		matches = ((seq_predictions == labels_flat) & (valid_mask)).sum().item() # 1, 

		if self.loss_type == "mean": # not true seq sim per tok, but easier to deal with
			matches /= self.valid_toks
		
		return matches # seq_sim

	def compute_losses(self, prediction, loss_function=None):

		predictions_flat = prediction.view(-1, prediction.size(2)).to(torch.float32)
		labels_flat = self.labels.view(-1).long()

		loss = loss_function(predictions_flat, labels_flat)

		# compute seq sim
		matches = self.compute_matches(prediction)

		return loss, matches

	def get_losses(self, loss_function):

		for prediction, prediction_loss in zip(	[self.output_predictions, self.ar_output_predictions], 
												[self.output_losses, self.ar_output_losses]
											):

			# for some reason gives tuple to cel and None to seq sim if do 'cel, seq_sim = self.compute...'
			nll_and_matches = self.compute_losses(prediction, loss_function) if prediction is not None else (None, None)
			nll, matches = nll_and_matches
			prediction_loss.add_losses(nll, matches, self.valid_toks)

class Losses():
	'''
	class to store losses
	'''
	def __init__(self): 

		self.losses = []
		self.matches = []
		self.valid_toks = 0

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		avg_loss, avg_seq_sim = 0, 0 # will make None later, but 0 for now since operated on like a float later
		avg_loss = sum(loss.item() for loss in self.losses if loss is not None) / self.valid_toks
		avg_seq_sim = 100*sum(match for match in self.matches if match is not None) / self.valid_toks
		
		return avg_loss, avg_seq_sim

	def add_losses(self, nll, matches, valid=1):
		self.losses.append(nll)
		self.matches.append(matches)
		self.valid_toks += valid

	def extend_losses(self, other):
		self.valid_toks += other.valid_toks
		self.losses.extend(other.losses)
		self.matches.extend(other.matches)

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.losses = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.losses]
