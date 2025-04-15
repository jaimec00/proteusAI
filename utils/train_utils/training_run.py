# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from proteusAI import proteusAI
from utils.train_utils.io_utils import Output
from utils.train_utils.data_utils import DataHolder
from utils.train_utils.training_run_utils import Epoch, Batch
from utils.train_utils.losses import TrainingRunLosses

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training, particularly nans when training the VAE
torch.autograd.set_detect_anomaly(True, check_nan=True) # throws error when nan encountered

class TrainingRun():
	'''
	the main class for training. holds all of the configuration arguments, and organnizes them into hyper-parameters, 
	training parameters, data, and output objects
	also orchestrates the setup of training, training itself, validation, testing, outputs, etc.

	Attributes:
		hyper_parameters (Box): 	store model hyper_parameters
		training_parameters (Box): 	stores training parameters
		data (DataHolder): 			object to store data (contains object of Data type), splits into train, 
									val, and test depending on arguments, also capable of loading the Data
		output (Output): 			holds information about where to write output to, also contains the logging object. comes with
									methods to print logs, plot training, and save model parameters
		losses (TrainingRunLosses):	stores losses over the training run
		gpu (torch.device): 		for convenience in moving tensors to GPU
		cpu (torch.device): 		for loading Data before moving to GPU

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
		
		self.losses = TrainingRunLosses(	# general
											args.training_parameters.train_type, 
											# for extraction
											args.training_parameters.loss.cel.label_smoothing, 
											# for kl annealing
											args.training_parameters.loss.kl.beta, 
											args.training_parameters.loss.kl.kappa, 
											args.training_parameters.loss.kl.midpoint, 
											args.training_parameters.loss.kl.annealing, 
											# for diffusion nll
											args.training_parameters.loss.nll.gamma
										)

		self.output = Output(	args.output.out_path,
								cel_plot=args.output.cel_plot, seq_plot=args.output.seq_plot, 
								mse_plot=args.output.mse_plot, kldiv_plot=args.output.kldiv_plot, 
								vae_plot=args.output.vae_plot,
								test_plot=args.output.test_plot, 
								weights_path=args.output.weights_path, model_checkpoints=args.output.model_checkpoints
							)

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
		self.output.log_trainingrun(self.training_parameters, self.hyper_parameters, self.data)

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
		
		# so many fucking hyperparameters
		self.model = proteusAI(	# model params
								d_model=self.hyper_parameters.d_model, 
								d_latent=self.hyper_parameters.d_latent, # same as dmodel for preliminary tests 
								num_aas=self.hyper_parameters.num_aa,
								old=self.training_parameters.train_type == "old",

								# wf embedding params (everything is learnable, so no configs)

								# wf encoding params

								# wf preprocessing
								encoding_d_hidden_pre=self.hyper_parameters.encoding.pre_process.d_hidden,
								encoding_hidden_layers_pre=self.hyper_parameters.encoding.pre_process.hidden_layers,

								# wf post_process
								encoding_d_hidden_post=self.hyper_parameters.encoding.post_process.d_hidden,
								encoding_hidden_layers_post=self.hyper_parameters.encoding.post_process.hidden_layers,

								# encoder layers
								encoding_encoder_layers=self.hyper_parameters.encoding.encoders.layers,
								encoding_heads=self.hyper_parameters.encoding.encoders.heads,
								encoding_use_bias=self.hyper_parameters.encoding.encoders.use_bias,
								encoding_min_spread=self.hyper_parameters.encoding.encoders.min_spread,
								encoding_min_rbf=self.hyper_parameters.encoding.encoders.min_rbf,
								encoding_max_rbf=self.hyper_parameters.encoding.encoders.max_rbf,
								encoding_d_hidden_attn=self.hyper_parameters.encoding.encoders.d_hidden_attn,
								encoding_hidden_layers_attn=self.hyper_parameters.encoding.encoders.hidden_layers_attn,

								# wf diffusion params

								# beta scheduler
								diffusion_alpha_bar_min=self.hyper_parameters.diffusion.scheduler.alpha_bar_min,
								diffusion_beta_min=self.hyper_parameters.diffusion.scheduler.beta_min,
								diffusion_beta_max=self.hyper_parameters.diffusion.scheduler.beta_max,
								diffusion_noise_schedule_type=self.hyper_parameters.diffusion.scheduler.noise_schedule_type, 
								diffusion_t_max=self.hyper_parameters.diffusion.scheduler.t_max,

								# timestep params for FiLM
								diffusion_d_in_timestep=self.hyper_parameters.diffusion.timestep.d_in,
								diffusion_d_hidden_timestep=self.hyper_parameters.diffusion.timestep.d_hidden,
								diffusion_hidden_layers_timestep=self.hyper_parameters.diffusion.timestep.hidden_layers, 

								# wf post_process
								diffusion_d_hidden_post=self.hyper_parameters.diffusion.post_process.d_hidden,
								diffusion_hidden_layers_post=self.hyper_parameters.diffusion.post_process.hidden_layers,

								# encoder layers
								diffusion_encoder_layers=self.hyper_parameters.diffusion.encoders.layers,
								diffusion_heads=self.hyper_parameters.diffusion.encoders.heads,
								diffusion_use_bias=self.hyper_parameters.diffusion.encoders.use_bias,
								diffusion_min_spread=self.hyper_parameters.diffusion.encoders.min_spread,
								diffusion_min_rbf=self.hyper_parameters.diffusion.encoders.min_rbf,
								diffusion_max_rbf=self.hyper_parameters.diffusion.encoders.max_rbf,
								diffusion_d_hidden_attn=self.hyper_parameters.diffusion.encoders.d_hidden_attn,
								diffusion_hidden_layers_attn=self.hyper_parameters.diffusion.encoders.hidden_layers_attn,

								# wf decoding params

								# wf preprocessing
								decoding_d_hidden_pre=self.hyper_parameters.decoding.pre_process.d_hidden,
								decoding_hidden_layers_pre=self.hyper_parameters.decoding.pre_process.hidden_layers,

								# wf post_process
								decoding_d_hidden_post=self.hyper_parameters.decoding.post_process.d_hidden,
								decoding_hidden_layers_post=self.hyper_parameters.decoding.post_process.hidden_layers,

								# encoder layers
								decoding_encoder_layers=self.hyper_parameters.decoding.encoders.layers,
								decoding_heads=self.hyper_parameters.decoding.encoders.heads,
								decoding_use_bias=self.hyper_parameters.decoding.encoders.use_bias,
								decoding_min_spread=self.hyper_parameters.decoding.encoders.min_spread,
								decoding_min_rbf=self.hyper_parameters.decoding.encoders.min_rbf,
								decoding_max_rbf=self.hyper_parameters.decoding.encoders.max_rbf,
								decoding_d_hidden_attn=self.hyper_parameters.decoding.encoders.d_hidden_attn,
								decoding_hidden_layers_attn=self.hyper_parameters.decoding.encoders.hidden_layers_attn,

								# wf extraction params

								# wf preprocessing
								extraction_d_hidden_pre=self.hyper_parameters.extraction.pre_process.d_hidden,
								extraction_hidden_layers_pre=self.hyper_parameters.extraction.pre_process.hidden_layers,

								# wf post_process
								extraction_d_hidden_post=self.hyper_parameters.extraction.post_process.d_hidden,
								extraction_hidden_layers_post=self.hyper_parameters.extraction.post_process.hidden_layers,

								# encoder layers
								extraction_encoder_layers=self.hyper_parameters.extraction.encoders.layers,
								extraction_heads=self.hyper_parameters.extraction.encoders.heads,
								extraction_use_bias=self.hyper_parameters.extraction.encoders.use_bias,
								extraction_min_spread=self.hyper_parameters.extraction.encoders.min_spread,
								extraction_min_rbf=self.hyper_parameters.extraction.encoders.min_rbf,
								extraction_max_rbf=self.hyper_parameters.extraction.encoders.max_rbf,
								extraction_d_hidden_attn=self.hyper_parameters.extraction.encoders.d_hidden_attn,
								extraction_hidden_layers_attn=self.hyper_parameters.extraction.encoders.hidden_layers_attn,

								# dropout
								dropout=self.training_parameters.regularization.dropout,
								wf_dropout=self.training_parameters.regularization.wf_dropout,
							)

		self.model.to(self.gpu)

		# which pretrained weights to use
		if self.training_parameters.weights.use_model:
			pretrained_weights = torch.load(self.training_parameters.weights.use_model, map_location=self.gpu, weights_only=True)
			self.model.load_state_dict(pretrained_weights, strict=False) # allow modifications of the model in between transfer learning
		if self.training_parameters.weights.use_embedding_weights:
			self.model.load_WFEmbedding_weights(self.training_parameters.weights.use_embedding_weights, self.gpu)
		if self.training_parameters.weights.use_encoding_weights:
			self.model.load_WFEncoding_weights(self.training_parameters.weights.use_encoding_weights, self.gpu)
		if self.training_parameters.weights.use_diffusion_weights:
			self.model.load_WFDiffusion_weights(self.training_parameters.weights.use_diffusion_weights, self.gpu)
		if self.training_parameters.weights.use_decoding_weights:
			self.model.load_WFDecoding_weights(self.training_parameters.weights.use_decoding_weights, self.gpu)
		if self.training_parameters.weights.use_extraction_weights:
			self.model.load_WFExtraction_weights(self.training_parameters.weights.use_extraction_weights, self.gpu)
		
		# what weights should be frozen depending on training type 
		if self.training_parameters.train_type == "extraction": # train embedding and extraction only
			self.model.freeze_WFEncoding_weights()
			self.model.freeze_WFDiffusion_weights()
			self.model.freeze_WFDecoding_weights()

		elif self.training_parameters.train_type == "vae": # train encoder and decoder using trained (and frozen) embedding
			self.model.freeze_WFEmbedding_weights()
			self.model.freeze_WFDiffusion_weights()
			self.model.freeze_WFExtraction_weights()
		
		elif self.training_parameters.train_type == "extraction_finetune": # finetune extractor on decoder outputs, only extractor is not frozen
			self.model.freeze_WFEmbedding_weights()
			self.model.freeze_WFEncoding_weights()
			self.model.freeze_WFDiffusion_weights()
			self.model.freeze_WFDecoding_weights()

		elif self.training_parameters.train_type == "diffusion": # train diffusion on encoder outputs, diffusion is the only one not frozen
			self.model.freeze_WFEmbedding_weights()
			self.model.freeze_WFEncoding_weights()
			self.model.freeze_WFDecoding_weights()
			self.model.freeze_WFExtraction_weights()

		elif self.training_parameters.train_type == "old":
			self.model.freeze_WFEncoding_weights()
			self.model.freeze_WFDiffusion_weights()
			self.model.freeze_WFDecoding_weights()

		else:
			raise ValueError(f"invalid train_type selected, {self.training_parameters.train_type=}. options are ['extraction', 'vae', 'extraction_finetune', 'diffusion']") 

		# get number of parameters for logging
		self.training_parameters.num_embedding_params = sum(p.numel() for p in self.model.wf_embedding.parameters())
		self.training_parameters.num_encoding_params = sum(p.numel() for p in self.model.wf_encoding.parameters())
		self.training_parameters.num_diffusion_params = sum(p.numel() for p in self.model.wf_diffusion.parameters())
		self.training_parameters.num_decoding_params = sum(p.numel() for p in self.model.wf_decoding.parameters())
		self.training_parameters.num_extraction_params = sum(p.numel() for p in self.model.wf_extraction.parameters())
		self.training_parameters.num_params = self.training_parameters.num_embedding_params + self.training_parameters.num_encoding_params + self.training_parameters.num_diffusion_params + self.training_parameters.num_decoding_params +  self.training_parameters.num_extraction_params 

		# compile the model (more precisely the computational graph). doesnt seem to work, no time to debug anyways
		# self.model = torch.compile(self.model)

		# print gradients at each step if in debugging mode
		if self.debug.debug_grad:
			def print_grad(name):
				def hook(grad):
					print(f"Gradient at {name}: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}, max={grad.abs().max().item():.6f}")
				return hook

			# Attach hook to all parameters
			for name, param in self.model.named_parameters():
				if param.requires_grad:
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

		if self.training_parameters.lr.lr_type == "attn":

			# compute the scale
			if self.training_parameters.lr.lr_step == 0.0:
				scale = self.hyper_parameters.d_model**(-0.5)
			else:
				scale = self.training_parameters.lr.warmup_steps**(0.5) * self.training_parameters.lr.lr_step # scale needed so max lr is what was specified
			
			def attn(step):
				'''lr scheduler from attn paper'''
				return scale * min((step+1)**(-0.5), (step+1)*(self.training_parameters.lr.warmup_steps**(-1.5)))

			self.scheduler = lr_scheduler.LambdaLR(self.optim, attn)

		elif self.training_parameters.lr.lr_type == "static":
			def static(step):
				return self.training_parameters.lr.lr_step
			self.scheduler = lr_scheduler.LambdaLR(self.optim, static)
			
		else:
			raise ValueError(f"invalid lr_type: {self.training_parameters.lr.lr_type}. options are ['attn', 'static']")


	def model_checkpoint(self, epoch_idx):
		if (epoch_idx+1) % self.output.model_checkpoints == 0: # model checkpointing
			self.output.save_model(self.model, appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_loss(),2)}")

	def training_converged(self, epoch_idx):

		if self.training_parameters.train_type in ["extraction", "extraction_finetune", "old"]:
			criteria = self.losses.val.cel
		if self.training_parameters.train_type == "vae":
			criteria = self.losses.val.all_losses
		elif self.training_parameters.train_type == "diffusion":
			criteria = self.losses.val.squared_errors

		choose_best = min # choose best
		best = float("inf")
		converged = lambda best, thresh: best > thresh

		# val losses are already in avg seq sim format per epoch
		if self.training_parameters.early_stopping.tolerance+1 > len(criteria):
			return False

		current_n = criteria[-(self.training_parameters.early_stopping.tolerance):]
		old = criteria[-(self.training_parameters.early_stopping.tolerance+1)]

		for current in current_n:
			delta = current - old
			best = choose_best(best, delta) 

		has_converged = converged(best, self.training_parameters.early_stopping.thresh)

		if has_converged:
			self.output.log.info(f"training converged after {epoch_idx} epochs")

		return has_converged

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
			if self.training_converged(epoch_idx): break

		self.output.plot_training(self.losses, self.training_parameters.train_type)
		self.output.save_model(self.model, train_type=self.training_parameters.train_type)

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
			self.output.log_val_losses(self.losses, self.training_parameters.train_type)

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
								inference=self.training_parameters.train_type == "diffusion", # can only run inference once extraction AND diffusion have been trained
								epoch=dummy_epoch
							)

				# run the model
				batch.batch_forward()

				# update pbar
				test_pbar.update(1)
		
		# log the losses
		self.output.log_test_losses(self.losses, self.training_parameters.train_type)
