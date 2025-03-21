# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm
import numpy as np
import math

from proteusAI import proteusAI
from utils.train_utils.io_utils import Output
from utils.train_utils.data_utils import DataHolder

# ----------------------------------------------------------------------------------------------------------------------

class TrainingRun():
	'''
	the main class for training. holds all of the configuration arguments, and organnizes them into hyper-parameters, 
	training parameters, data, and output objects
	also orchestrates the setup of training, training itself, validation, testing, outputs, etc.

	Attributes:
		hyper_parameters (Box): store model hyper_parameters
		training_parameters (Box): stores training parameters
		data (DataHolder): object to store data (contains object os Data type, inherets from Dataset), splits into train, 
							val, and test depending on arguments, also capable of loading the Data into DataLoader for 
							easy integration
		output (Output): holds information about where to write output to, also contains the logging object. comes with
						methods to print logs, plot training, and save model parameters
		losses (TrainingRunLosses):
		gpu (torch.device): for convenience in moving tensors to GPU
		cpu (torch.device): for loading Data before moving to GPU

	'''

	def __init__(self, args):

		self.hyper_parameters = args.hyper_parameters
		self.training_parameters = args.training_parameters
		
		self.data = DataHolder(	args.data.data_path, 
								args.data.num_train, args.data.num_val, args.data.num_test, 
								args.data.batch_tokens, args.data.max_batch_size, 
								args.data.min_seq_size, args.data.max_seq_size, 
								args.training_parameters.regularization.use_chain_mask, args.data.max_resolution
							)
		
		self.output = Output(args.output.out_path, args.output.loss_plot, args.output.seq_plot, args.output.weights_path, args.output.model_checkpoints)

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
		
		self.model = proteusAI(	# model params
								d_model=self.hyper_parameters.d_model, 
								num_aas=self.hyper_parameters.num_aa,

								# wf embedding params
								embedding_min_wl=self.hyper_parameters.embedding.min_wl,
								embedding_max_wl=self.hyper_parameters.embedding.max_wl,
								embedding_base_wl=self.hyper_parameters.embedding.base_wl,
								embedding_learnable_aa=self.hyper_parameters.embedding.learnable_aa,		

								# wf diffusion params

								# beta scheduler
								diffusion_beta_min=self.hyper_parameters.diffusion.scheduler.beta_min, 
								diffusion_beta_max=self.hyper_parameters.diffusion.scheduler.beta_max,
								diffusion_beta_schedule_type=self.hyper_parameters.diffusion.scheduler.beta_schedule_type, 
								diffusion_t_max=self.hyper_parameters.diffusion.scheduler.t_max,

								# timestep params
								diffusion_min_wl=self.hyper_parameters.diffusion.timestep.min_wl,
								diffusion_max_wl=self.hyper_parameters.diffusion.timestep.max_wl,
								diffusion_mlp_timestep=self.hyper_parameters.diffusion.timestep.use_mlp,
								diffusion_d_hidden_timestep=self.hyper_parameters.diffusion.timestep.d_hidden,
								diffusion_hidden_layers_timestep=self.hyper_parameters.diffusion.timestep.hidden_layers, 
								diffusion_norm_timestep=self.hyper_parameters.diffusion.timestep.use_norm,

								# wf preprocessing
								diffusion_mlp_pre=self.hyper_parameters.diffusion.pre_process.use_mlp, 
								diffusion_d_hidden_pre=self.hyper_parameters.diffusion.pre_process.d_hidden,
								diffusion_hidden_layers_pre=self.hyper_parameters.diffusion.pre_process.hidden_layers,
								diffusion_norm_pre=self.hyper_parameters.diffusion.pre_process.use_norm,

								# wf post_process
								diffusion_mlp_post=self.hyper_parameters.diffusion.post_process.use_mlp,
								diffusion_d_hidden_post=self.hyper_parameters.diffusion.post_process.d_hidden,
								diffusion_hidden_layers_post=self.hyper_parameters.diffusion.post_process.hidden_layers,
								diffusion_norm_post=self.hyper_parameters.diffusion.post_process.use_norm,

								# encoder layers
								diffusion_encoder_layers=self.hyper_parameters.diffusion.encoders.layers,
								diffusion_heads=self.hyper_parameters.diffusion.encoders.heads,
								diffusion_learnable_spreads=self.hyper_parameters.diffusion.encoders.learnable_spreads,
								diffusion_min_spread=self.hyper_parameters.diffusion.encoders.min_spread,
								diffusion_max_spread=self.hyper_parameters.diffusion.encoders.max_spread,
								diffusion_base_spreads=self.hyper_parameters.diffusion.encoders.base_spreads,
								diffusion_num_spread=self.hyper_parameters.diffusion.encoders.num_spread,
								diffusion_min_rbf=self.hyper_parameters.diffusion.encoders.min_rbf,
								diffusion_max_rbf=self.hyper_parameters.diffusion.encoders.max_rbf,
								diffusion_beta=self.hyper_parameters.diffusion.encoders.beta,
								diffusion_d_hidden_attn=self.hyper_parameters.diffusion.encoders.d_hidden_attn,
								diffusion_hidden_layers_attn=self.hyper_parameters.diffusion.encoders.hidden_layers_attn,

								# wf extraction params

								# wf preprocessing
								extraction_mlp_pre=self.hyper_parameters.extraction.pre_process.use_mlp, 
								extraction_d_hidden_pre=self.hyper_parameters.extraction.pre_process.d_hidden,
								extraction_hidden_layers_pre=self.hyper_parameters.extraction.pre_process.hidden_layers,
								extraction_norm_pre=self.hyper_parameters.extraction.pre_process.use_norm,

								# wf post_process
								extraction_mlp_post=self.hyper_parameters.extraction.post_process.use_mlp,
								extraction_d_hidden_post=self.hyper_parameters.extraction.post_process.d_hidden,
								extraction_hidden_layers_post=self.hyper_parameters.extraction.post_process.hidden_layers,
								extraction_norm_post=self.hyper_parameters.extraction.post_process.use_norm,

								# encoder layers
								extraction_encoder_layers=self.hyper_parameters.extraction.encoders.layers,
								extraction_heads=self.hyper_parameters.extraction.encoders.heads,
								extraction_learnable_spreads=self.hyper_parameters.extraction.encoders.learnable_spreads,
								extraction_min_spread=self.hyper_parameters.extraction.encoders.min_spread,
								extraction_max_spread=self.hyper_parameters.extraction.encoders.max_spread,
								extraction_base_spreads=self.hyper_parameters.extraction.encoders.base_spreads,
								extraction_num_spread=self.hyper_parameters.extraction.encoders.num_spread,
								extraction_min_rbf=self.hyper_parameters.extraction.encoders.min_rbf,
								extraction_max_rbf=self.hyper_parameters.extraction.encoders.max_rbf,
								extraction_beta=self.hyper_parameters.extraction.encoders.beta,
								extraction_d_hidden_attn=self.hyper_parameters.extraction.encoders.d_hidden_attn,
								extraction_hidden_layers_attn=self.hyper_parameters.extraction.encoders.hidden_layers_attn,

							)

		self.model.to(self.gpu)

		# which pretrained weights to use
		if self.training_parameters.weights.use_model:
			pretrained_weights = torch.load(self.training_parameters.weights.use_model, map_location=self.gpu, weights_only=True)
			self.model.load_state_dict(pretrained_weights, strict=False) # allow modifications of the model in between transfer learning
		if self.training_parameters.weights.use_embedding_weights:
			self.model.load_WFEmbedding_weights(self.training_parameters.weights.use_embedding_weights, self.device)
		if self.training_parameters.weights.use_diffusion_weights:
			self.model.load_WFDiffusion_weights(self.training_parameters.weights.use_diffusion_weights, self.device)
		if self.training_parameters.weights.use_extraction_weights:
			self.model.load_WFExtraction_weights(self.training_parameters.weights.use_extraction_weights, self.device)

		# what weights should be frozen depending on training type (see config file for explanation)
		if self.training_parameters.train_type == "extraction":
			self.model.freeze_WFDiffusion_weights()
		if self.training_parameters.train_type == "diffusion":
			self.model.freeze_WFEmbedding_weights()
			self.model.freeze_WFExtraction_weights()
		if self.training_parameters.train_type == "extraction_denoised":
			self.model.freeze_WFEmbedding_weights()
			self.model.freeze_WFDiffusion_weights()

		# get number of parameters for logging
		self.training_parameters.num_embedding_params = sum(p.numel() for p in self.model.wf_embedding.parameters())
		self.training_parameters.num_diffusion_params = sum(p.numel() for p in self.model.wf_diffusion.parameters())
		self.training_parameters.num_extraction_params = sum(p.numel() for p in self.model.wf_extraction.parameters())
		self.training_parameters.num_params = self.training_parameters.num_embedding_params + self.training_parameters.num_diffusion_params + self.training_parameters.num_extraction_params 

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

		if self.training_parameters.train_type in ["extraction", "extraction_denoised"]:
			loss_function = CrossEntropyLoss(ignore_index=-1, reduction="sum", label_smoothing=self.training_parameters.regularization.label_smoothing)
		elif self.training_parameters.train_type == "diffusion":
			loss_function = MSELoss(reduction="sum")
		else:
			raise ValueError("invalid training type specified. options are extraction, diffusion, or extraction_denoised")

		self.losses = TrainingRunLosses(loss_function)

	def train(self):
		'''
		entry point for training the model. loads train and validation data, loops through epochs, plots training, 
		runs testing and saves the model
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
			epoch = Epoch(self, epoch_idx)
			epoch.epoch_loop()

			self.model_checkpoint(epoch_idx)
			if self.training_converged(): # early stopping
				self.output.log.info(f"training converged after {epoch_idx} epochs")
				break

		self.output.plot_training(self.losses, self.training_parameters.train_type)
		self.output.save_model(self.model, train_type=self.training_parameters.train_type)
	
	def model_checkpoint(self, epoch_idx):
		if (epoch_idx+1) % self.output.model_checkpoints == 0: # model checkpointing
			self.output.save_model(self.model, appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_match().item(),2)}")

	def training_converged(self):

		losses = self.losses.val.matches

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

	def validation(self):		
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.clear_tmp_losses()

		# dummy epoch so can still access training run parent
		dummy_epoch = Epoch(self)
		
		# turn off gradient calculation
		with torch.no_grad():

			# logging
			self.output.log.info("running validation...")

			# progress bar
			val_pbar = tqdm(total=len(self.data.val_data), desc="epoch_validation_progress", unit="step")
			
			# loop through validation batches
			for coords, labels, chain_idxs, chain_mask, key_padding_mask in self.data.val_data:
					
				# init batch
				batch = Batch(coords, labels, chain_idxs, chain_mask, key_padding_mask, epoch=dummy_epoch)

				# run the model
				batch.batch_forward()

				# update pbar
				val_pbar.update(1)

			# add the avg losses to the global loss and log
			self.output.log_val_losses(self.losses)

	def test(self):

		# switch to evaluation mode
		self.model.eval()
		
		# load testing data
		self.output.log.info("loading testing data...")
		self.data.load("test")

		# init losses
		self.losses.clear_tmp_losses()

		# dummy epoch so can still access training run parent
		dummy_epoch = Epoch(self)

		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			test_pbar = tqdm(total=len(self.data.test_data), desc="test_progress", unit="step")

			# loop through testing batches
			for coords, labels, chain_idxs, chain_mask, key_padding_mask in self.data.test_data:
					
				# init batch
				batch = Batch(  coords, labels, chain_idxs, chain_mask, key_padding_mask, 
								temp=self.training_parameters.inference.temperature, 
								cycles=self.training_parameters.inference.cycles,
								inference=self.training_parameters.train_type in ["diffusion", "extraction_denoised"], # can only run inference once extraction AND diffusion have been trained
								epoch=dummy_epoch
							)

				# run the model
				batch.batch_forward()

				# update pbar
				test_pbar.update(1)
		
		# log the losses
		self.output.log_test_losses(self.losses)

class Epoch():	
	def __init__(self, training_run, epoch=None):

		self.training_run_parent = training_run
		self.epoch = epoch
		self.epochs = training_run.training_parameters.epochs
		
	def epoch_loop(self):
		'''
		a single training loop through one epoch. sets up epoch input perturbation values depending on the stage (calculated in Epoch.__init__)
		then loops through batches, logs the losses, and runs validation
		'''

		# make sure in training mode
		self.training_run_parent.model.train()

		# setup the epoch
		self.training_run_parent.output.log_epoch(self.epoch, self.training_run_parent.scheduler.get_last_lr()[0])

		# clear temp losses
		self.training_run_parent.losses.clear_tmp_losses()

		# init epoch pbar
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")

		# loop through batches
		for b_idx, (coords, labels, chain_idxs, chain_mask, key_padding_mask) in enumerate(self.training_run_parent.data.train_data):

			# instantiate this batch
			batch = Batch(coords, labels, chain_idxs, chain_mask, key_padding_mask, b_idx=b_idx, epoch=self)

			# learn
			batch.batch_learn()

			# update pbar
			epoch_pbar.update(1)
		
		# log epoch losses and save avg
		self.training_run_parent.output.log_epoch_losses(self.training_run_parent.losses)

		# run validation
		self.training_run_parent.validation() 

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()
			self.training_run_parent.data.val_data.rotate_data()

class Batch():
	def __init__(self, 	coords, labels, chain_idxs, chain_mask, key_padding_mask, 
						b_idx=None, epoch=None, inference=False, temp=0.1, cycles=10):

		self.coords = coords 
		self.labels = labels
		self.aas = torch.where(labels==-1, torch.randint(0, epoch.training_run_parent.hyper_parameters.num_aa, labels.shape), labels) # set to random aa so wf kernel works (so not all threads query a single thread for these). masking is already dealt with
		self.chain_idxs = chain_idxs
		self.chain_mask = chain_mask
		self.key_padding_mask = key_padding_mask

		self.b_idx = b_idx
		self.epoch_parent = epoch

		self.inference = inference
		self.temp = temp
		self.cycles = cycles

	def move_to(self, device):

		self.aas = self.aas.to(device)
		self.labels = self.labels.to(device)
		self.coords = self.coords.to(device)
		self.chain_mask = self.chain_mask.to(device)
		self.key_padding_mask = self.key_padding_mask.to(device)

	def batch_learn(self):
		'''
		a single iteration over a batch.
		'''

		# add random noise to the coordinates
		self.noise_coords()

		# forward pass
		self.batch_forward()

		# backward pass
		self.batch_backward()

	def batch_forward(self):
		'''
		performs the forward pass, gets the outputs and computes the losses of a batch. 
		'''
		
		# move batch to gpu
		self.move_to(self.epoch_parent.training_run_parent.gpu)

		# update labels for accurate loss computation (used by output objects)
		self.labels = self.labels.masked_fill(self.chain_mask | self.key_padding_mask, -1)

		# get model outputs
		self.outputs = self.get_outputs()

		# get losses (adds them to training run tmp losses)
		self.outputs.get_losses()

	def batch_backward(self):

		# utils
		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.loss.accumulation_steps
		optim = self.epoch_parent.training_run_parent.optim
		scheduler = self.epoch_parent.training_run_parent.scheduler

		# whether to take a step
		learn_step = (self.b_idx + 1) % accumulation_steps == 0

		# get last loss
		loss = self.epoch_parent.training_run_parent.losses.tmp.get_last_loss() # no scaling by accumulation steps, as already handled by grad clipping and scaling would introduce batch size biases

		# perform backward pass to accum grads
		loss.backward()

		if learn_step:
		
			# grad clip
			if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)

			# step
			optim.step()
			optim.zero_grad()
			scheduler.step()

	def noise_coords(self):

		'''data augmentation via gaussian noise injection into coords, default is 0.02 A standard deviation, centered around 0'''

		# define noise
		noise = torch.randn_like(self.coords) * self.epoch_parent.training_run_parent.training_parameters.regularization.noise_coords_std

		# add noise
		self.coords = self.coords + noise

	def get_outputs(self):
		'''
		used to get output predictions
		'''

		# get alpha carbon and beta carbon coords
		coords_alpha, coords_beta = self.epoch_parent.training_run_parent.model.get_CaCb_coords(self.coords, self.chain_idxs)

		# run model depending on training type
		match self.epoch_parent.training_run_parent.training_parameters.train_type:

			case "extraction":

				# get wf
				wf = self.epoch_parent.training_run_parent.model(coords_alpha, coords_beta=coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)

				# predict sequence
				seq_pred = self.epoch_parent.training_run_parent.model(coords_alpha, wf=wf, key_padding_mask=self.key_padding_mask, extraction=True)
			
				# convert to output object
				output = ExtractionOutput(self, seq_pred)

			case "diffusion":

				# get clean wavefunction
				wf = self.epoch_parent.training_run_parent.model(coords_alpha, coords_beta, aas=self.predictions, key_padding_mask=self.key_padding_mask, embedding=True)

				# get timesteps from uniform distribution
				timesteps = self.epoch_parent.training_run_parent.model.wf_diffusion.get_random_timesteps(wf.size(0), wf.device)

				# add noise
				noised_wf, noise = self.epoch_parent.training_run_parent.model.wf_diffusion.noise(wf, timesteps)

				# predict noise
				noise_pred = self.epoch_parent.training_run_parent.model(coords_alpha, wf=noised_wf, key_padding_mask=self.key_padding_mask, diffusion=True, t=timesteps)

				# convert to output object
				output = DiffusionOutput(self, noise_pred, noise)

			case "extraction_denoised": # need to preprocess wf outputs of diffusion models starting from different timesteps, and edit DataHolder to give these

				raise NotImplementedError

		# if self.inference: # run inference pipeline also, not implemented


		return ModelOutputs(output)

	def size(self, idx):
		return self.labels.size(idx)

class DiffusionOutput():

	def __init__(self, batch_parent, noise_pred, true_noise):
		self.batch_parent = batch_parent
		self.valid_toks = (batch_parent.labels!=-1).sum()
		self.noise_pred = noise_pred.masked_fill(batch_parent.labels==-1, 0) # set both to 0 for invalid positions so sum of squared errors is not affected
		self.true_noise = true_noise.masked_fill(batch_parent.labels==-1, 0)

	def compute_losses(self):

		# compute loss
		loss = self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(self.noise_pred, self.true_noise)

		# compute seq sim
		matches = 0 # Diffusion does not predict amino acids

		return loss, matches

class ExtractionOutput():

	def __init__(self, batch_parent, seq_pred):
		self.batch_parent = batch_parent 
		self.seq_pred = seq_pred
		self.valid_toks = (batch_parent.labels!=-1).sum()

	def compute_matches(self):
		'''greedy selection, just for tracking progress'''
		
		prediction_flat = self.seq_pred.view(-1, self.seq_pred.size(-1)) # batch*N x 20
		labels_flat = self.batch_parent.labels.view(-1) # batch x N --> batch*N,
		seq_predictions = torch.argmax(prediction_flat, dim=-1) # batch*N x 20 --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		matches = ((seq_predictions == labels_flat) & (valid_mask)).sum() # 1, 
		
		return matches 

	def compute_losses(self):

		# flatten labels and predictions
		seq_pred_flat = self.seq_pred.view(-1, self.seq_pred.size(2)).to(torch.float32)
		labels_flat = self.batch_parent.labels.view(-1).long()

		# compute loss
		loss = self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(seq_pred_flat, labels_flat)

		# compute seq sim
		matches = self.compute_matches()

		return loss, matches

# class InferenceOutput():
# 	def __init__(self, batch_parent, seq_pred):
# 		self.batch_parent = batch_parent 
# 		self.seq_pred = seq_pred
# 		self.valid_toks = (batch_parent.labels!=-1).sum()


class ModelOutputs():
	def __init__(self, output):
		self.output = output

	def get_losses(self):
		losses, matches = self.output.compute_losses()
		self.output.batch_parent.epoch_parent.training_run_parent.losses.tmp.add_losses(losses, matches, self.output.valid_toks)

class Losses():
	'''
	class to store losses
	'''
	def __init__(self, loss_function): 

		self.loss_function = loss_function
		self.losses = []
		self.matches = []
		self.valid_toks = 0

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		avg_loss, avg_seq_sim = 0, 0 
		avg_loss = sum(loss.item() for loss in self.losses if loss is not None) / self.valid_toks
		avg_seq_sim = 100*sum(match.item() for match in self.matches if match is not None) / self.valid_toks
		
		return avg_loss, avg_seq_sim

	def add_losses(self, loss, matches, valid=1):
		self.losses.append(loss)
		self.matches.append(matches)
		self.valid_toks += valid

	def extend_losses(self, other):
		self.valid_toks += other.valid_toks
		self.losses.extend(other.losses)
		self.matches.extend(other.matches)

	def clear_losses(self):
		self.losses = []
		self.matches = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.losses[-1]

	def get_last_match(self):
		return self.matches[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.losses = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.losses]
		self.matches = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches]

	def __len__(self):
		return len(self.losses)

class TrainingRunLosses():

	def __init__(self, loss_function):
		self.train = Losses(loss_function)
		self.val = Losses(loss_function)
		self.test = Losses(loss_function)
		self.tmp = Losses(loss_function)
		self.loss_function = loss_function

	def clear_tmp_losses(self):
		self.tmp.clear_losses()

	def to_numpy(self):
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

