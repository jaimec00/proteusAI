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

		self.hyper_parameters = args.hyper_parameters
		self.training_parameters = args.training_parameters
		
		self.MASK_injection = MASK_injection(	args.training_parameters.regularization.mask_injection.mean_mask_pct,
												args.training_parameters.regularization.mask_injection.std_mask_pct,
												args.training_parameters.regularization.mask_injection.min_mask_pct,
												args.training_parameters.regularization.mask_injection.max_mask_pct,
												args.training_parameters.regularization.mask_injection.mean_span,
												args.training_parameters.regularization.mask_injection.std_span,
												args.training_parameters.regularization.mask_injection.randAA_pct, 
												args.training_parameters.regularization.mask_injection.trueAA_pct
											)
		
		self.data = DataHolder(	args.data.data_path, 
								args.data.num_train, args.data.num_val, args.data.num_test, 
								args.data.batch_tokens, args.data.max_batch_size, 
								args.data.min_seq_size, args.data.max_seq_size, 
								args.training_parameters.regularization.use_chain_mask, args.data.max_resolution
							)
		
		self.output = Output(args.output.out_path, args.output.loss_plot, args.output.seq_plot, args.output.weights_path, args.output.model_checkpoints)

		self.train_losses = Losses()
		self.val_losses = Losses() # validation with all tokens being MASK
		self.val_losses_context = Losses() if args.hyper_parameters.aa.use_aa else None # validation with the same MASK injection as the training batches from current epoch
		self.test_losses = Losses()

		self.gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.cpu = torch.device("cpu")

		self.debug = args.debug

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

		self.output.log_hyperparameters(self.training_parameters, self.hyper_parameters, self.data)

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

								self.hyper_parameters.wf.learnable_wavelengths,
								self.hyper_parameters.wf.wf_mag_type, 
								self.hyper_parameters.wf.anisotropic_wf, 
								self.hyper_parameters.wf.min_wl,
								self.hyper_parameters.wf.max_wl,
								self.hyper_parameters.wf.base_wl,
								self.hyper_parameters.wf.d_hidden_we,
								self.hyper_parameters.wf.hidden_layers_we,

								self.hyper_parameters.aa.use_aa,
								self.hyper_parameters.aa.d_hidden_aa,
								self.hyper_parameters.aa.hidden_layers_aa,
								self.hyper_parameters.aa.esm2_weights_path,
								self.hyper_parameters.aa.learnable_esm,

								self.hyper_parameters.struct_encoders.layers,
								self.hyper_parameters.seq_encoders.layers,
								self.hyper_parameters.decoders.layers,

								# just use the same configs for all enc/dec rn, uses struct encoder config
								self.hyper_parameters.struct_encoders.num_heads, 
								self.hyper_parameters.struct_encoders.learnable_spreads,
								self.hyper_parameters.struct_encoders.min_spread,
								self.hyper_parameters.struct_encoders.max_spread,
								self.hyper_parameters.struct_encoders.base_spread,
								self.hyper_parameters.struct_encoders.num_spread,
								self.hyper_parameters.struct_encoders.min_rbf,
								self.hyper_parameters.struct_encoders.max_rbf,
								self.hyper_parameters.struct_encoders.beta,
								self.hyper_parameters.struct_encoders.d_hidden_attn,
								self.hyper_parameters.struct_encoders.hidden_layers_attn,

								self.training_parameters.regularization.dropout,
								self.training_parameters.regularization.attn_dropout, # attention has less aggressive dropout, as it is already heavily masked
								self.training_parameters.regularization.wf_dropout
							)

		self.model.to(self.gpu)

		if self.training_parameters.weights.use_model:
			pretrained_weights = torch.load(self.training_parameters.weights.use_model, map_location=self.gpu, weights_only=True)
			self.model.load_state_dict(pretrained_weights, strict=False) # allow modifications of the model in between transfer learning

		# first train just on structure, then when it converges, train on sequence w/ frozen structure weights
		if self.training_parameters.weights.freeze_structure_weights:
			self.model.freeze_structure_weights()
		if self.training_parameters.weights.freeze_sequence_weights:
			self.model.freeze_sequence_weights()
		if self.training_parameters.weights.freeze_decoder_weights:
			self.model.freeze_decoder_weights()
		if self.training_parameters.weights.cp_struct_enc_2_seq_enc:
			self.model.cp_structEnc_2_seqEnc()

		# get number of parameters for logging
		self.training_parameters.num_params = sum(p.numel() for p in self.model.parameters())

		# print gradients at each step if in debugging mode
		if self.debug.debug_grad:
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
		self.optim = torch.optim.Adam(self.model.parameters(), lr=1.0, 
									betas=(self.training_parameters.adam.beta1, self.training_parameters.adam.beta2), 
									eps=float(self.training_parameters.adam.epsilon))
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

		# compute the scale
		if self.training_parameters.lr.lr_step == 0.0:
			scale = self.hyper_parameters.d_model**(-0.5)
		else:
			scale = self.training_parameters.lr.warmup_steps**(0.5) * self.training_parameters.lr.lr_step # scale needed so max lr is what was specified
		
		def attn_lr(step):
			'''lr scheduler from attn paper'''
			return scale * min((step+1)**(-0.5), (step+1)*(self.training_parameters.lr.warmup_steps**(-1.5)))
		
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
		self.loss_function = CEL(ignore_index=-1, reduction=self.training_parameters.loss.loss_type, label_smoothing=self.training_parameters.regularization.label_smoothing)

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

		losses = self.val_losses_context.matches if self.hyper_parameters.aa.use_aa else self.val_losses.matches

		# val losses are already in avg seq sim format per epoch
		if self.training_parameters.early_stopping.tolerance+1 > len(losses):
			return False
		current_seq_sims = losses[-(self.training_parameters.early_stopping.tolerance):]
		old_seq_sim = losses[-(self.training_parameters.early_stopping.tolerance+1)]

		best_delta_seq_sim = -float("inf")
		for current_seq_sim in current_seq_sims:
			delta_seq_sim = current_seq_sim - old_seq_sim
			best_delta_seq_sim = max(best_delta_seq_sim, delta_seq_sim) 

		return best_delta_seq_sim < self.training_parameters.early_stopping.thresh

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
								use_amp=False, mask_predict=False
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
		test_mp_losses = Losses()

		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			test_pbar = tqdm(total=len(self.data.test_data), desc="test_progress", unit="step")

			# loop through testing batches
			for label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask in self.data.test_data:
					
				# init batch
				batch = Batch(  label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask, 
								use_amp=False, 
								mask_predict=self.hyper_parameters.aa.use_aa, # only do mask_predict if using aa modules 
								temp=self.training_parameters.inference.temperature, num_iters=self.training_parameters.inference.num_iters
							)

				# mask all tokens
				self.MASK_all(batch)

				# run the model
				batch.batch_forward(self.model, self.loss_function, self.gpu)

				# add the losses
				test_losses.extend_losses(batch.outputs.output_losses)
				test_mp_losses.extend_losses(batch.outputs.mp_output_losses)

				# update pbar
				test_pbar.update(1)
		
		# log the losses
		self.output.log_test_losses(test_losses, test_mp_losses)

	def MASK_all(self, batch):

		batch.predictions = onehot(torch.full((batch.size(0), batch.size(1)), 20, device=batch.predictions.device), 21).float()

class Epoch():	
	def __init__(self, epoch, training_run):

		self.training_run_parent = training_run
		self.epoch = epoch
		self.epochs = self.training_run_parent.training_parameters.epochs
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

		# loop through batches
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")
		for b_idx, (label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask) in enumerate(self.training_run_parent.data.train_data):
					
			# instantiate this batch
			batch = Batch(	label_batch, coords_batch, chain_mask, chain_idxs, key_padding_mask, 
							b_idx=b_idx, epoch=self, 
							use_amp=self.training_run_parent.training_parameters.use_amp, 
						)

			if self.training_run_parent.hyper_parameters.aa.use_aa:
				# inject MASK tokens for prediction
				self.training_run_parent.MASK_injection.MASK_tokens(batch)

			# add random noise to the coordinates
			batch.noise_coords(self.training_run_parent.training_parameters.regularization.noise_coords_std)

			# learn
			batch.batch_learn(scheduler=self.training_run_parent.scheduler)

			# compile batch losses for logging
			self.losses.extend_losses(batch.outputs.output_losses)

			epoch_pbar.update(1)
		
		# print batch losses
		self.training_run_parent.output.log_epoch_losses(self, self.training_run_parent.train_losses)

		# run validation
		if self.training_run_parent.hyper_parameters.aa.use_aa:
			self.training_run_parent.batch_MASK_validation(self) # only run validation w/ context if training w/ aa info
		self.training_run_parent.all_MASK_validation() 

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()

class Batch():
	def __init__(self, labels, coords, chain_mask, chain_idxs, key_padding_mask, 
					b_idx=None, epoch=None,
					use_amp=True, mask_predict=False, temp=0.1, num_iters=10
				):

		self.labels = labels
		self.coords = coords 
		self.chain_mask = chain_mask
		self.chain_idxs = chain_idxs # for computing virtual Cb if using Ca only model, otherwise is computed directly from backbone coords
		self.use_amp = use_amp

		self.predictions = onehot(torch.where(labels==-1, 20, labels), num_classes=21).float()

		self.key_padding_mask = key_padding_mask
		self.onehot_mask = torch.zeros(self.key_padding_mask.shape, dtype=torch.bool)

		self.b_idx = b_idx
		self.epoch_parent = epoch

		self.use_amp = use_amp
		self.mask_predict = mask_predict
		self.temp = temp
		self.num_iters = num_iters

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
			mask_predict (bool): option for mask_predict inference
			temp (float): temperature, only applicable if mask_predict==True
			num_iters (int): number of iterations to run mask-predict for, if applicable

		Returns:
			None
		  
		'''
		
		# move batch to gpu
		self.move_to(device)

		# mask one hot positions for loss. also only compute loss for the representative chain of the sequence cluster (unless config says not to, then chain mask is all zeros)
		# NO MORE MASKING ONE HOT, THIS IS WHY I COULDNT INCLUDE SEQ INFO< BC I WAS DUMB AND DIDNT REALIZE LESS TOKENS CONTRIBUTE TO LOSS
		# the wf embedding does not encode info at the individual token level, rather, it distributes it to all other tokens, so can still include in loss
		self.labels = torch.where( self.chain_mask | self.key_padding_mask, -1, self.labels)

		if self.use_amp: # optional AMP
			with autocast('cuda'):
				self.outputs = self.get_outputs(model)
		else:
			self.outputs = self.get_outputs(model)

		self.outputs.get_losses(loss_function)

	def batch_backward(self, scheduler):

		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.loss.accumulation_steps
		scaler = self.epoch_parent.training_run_parent.scaler
		optim = self.epoch_parent.training_run_parent.optim

		learn_step = (self.b_idx + 1) % accumulation_steps == 0
		loss = self.outputs.output_losses.losses[-1] / accumulation_steps

		if scaler is not None:
			scaler.scale(loss).backward()

			if learn_step:

				# Apply gradient clipping using the scaler
				if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
					scaler.unscale_(optim)  # Unscale gradients before clipping
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)

				scaler.step(optim)
				scaler.update()
				optim.zero_grad()
				scheduler.step()
		else:
			loss.backward()
			if learn_step:
				if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)
				optim.step()
				optim.zero_grad()
				scheduler.step()

	def get_outputs(self, model):
		'''
		used to get output predictions for various scenarious, such as one-shot predictions from engineered inputs,
		one-shot predictions using previous outputs as inputs, or mask_predict inference
		
		Args:
			batch (Batch): Batch object, holds features, labels, and its own losses
			mask_predict (bool): whether to do mask_predict inference
			temp (float): temperature if using mask_predict

		Returns:
			None
		'''

		# compute mask_predict prediction if specified
		# cant compute cel with one hot vector, this is only for testing
		with torch.no_grad():
			if self.mask_predict: 
				mp_output_prediction = model(self.coords, self.predictions, self.chain_idxs, key_padding_mask=self.key_padding_mask, 
											mask_predict=self.mask_predict, temp=self.temp, num_iters=self.num_iters)
			else:
				mp_output_prediction = None

		# compute one shot also
		output_prediction = model(self.coords, self.predictions, self.chain_idxs, key_padding_mask=self.key_padding_mask, mask_predict=False)

		return ModelOutputs(output_prediction, mp_output_prediction, self.labels)

	def size(self, idx):
		return self.labels.size(idx)

class ModelOutputs():

	def __init__(self, output_predictions, mp_output_predictions, labels, loss_type="sum"):
		
		self.output_predictions = output_predictions
		self.mp_output_predictions = mp_output_predictions

		self.labels = labels
		self.valid_toks = (labels!=-1).sum().item() if loss_type == "sum" else 1 # 1 valid tok if computing mean since already normalized
		self.loss_type = loss_type

		self.output_losses = Losses()
		self.mp_output_losses = Losses()

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

		for prediction, prediction_loss in zip(	[self.output_predictions, self.mp_output_predictions], 
												[self.output_losses, self.mp_output_losses]
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
