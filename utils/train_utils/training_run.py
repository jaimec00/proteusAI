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
import os

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

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
		data (DataHolder): 			object to store data (contains objects of Data type), splits into train, 
									val, and test depending on arguments, also capable of loading the Data
		output (Output): 			holds information about where to write output to, also contains the logging object. comes with
									methods to print logs, plot training, and save model parameters
		losses (TrainingRunLosses):	stores losses over the training run and holds the loss functions
		gpu (torch.device): 		for convenience in moving tensors to GPU
		cpu (torch.device): 		for loading Data before moving to GPU

	'''

	def __init__(self, args):

		world_size = torch.cuda.device_count()
		os.environ['MASTER_ADDR'] = '127.0.0.1'
		os.environ['MASTER_PORT'] = '29500'
		mp.spawn(self.start_training, args=(world_size, args), nprocs=world_size, join=True)

	def start_training(self, local_rank, world_size, args):
	
		dist.init_process_group(
			backend='nccl',
			init_method='env://',
			world_size=world_size,
			rank=local_rank
		)

		torch.cuda.set_device(local_rank)
		self.rank = int(local_rank)
		self.world_size = int(world_size)
		self.gpu = torch.device(f'cuda:{local_rank}')
		self.cpu = torch.device("cpu")
		self.debug = args.debug

		self.setup_training(args)
		self.train()
		self.test()

	def setup_training(self, args):
		'''
		sets up the training by setting up the model, optimizer, scheduler, loss 
		function, scaler (if using AMP), and losses

		Args:
			None

		Returns:
			None
		'''

		self.hyper_parameters = args.hyper_parameters
		self.training_parameters = args.training_parameters
		self.step = 0 # log the step number
		self.toks_processed = 0 # log the number of valid tokens used
		
		self.data = DataHolder(	(args.data.single_chain_data_path if args.training_parameters.single_chain else args.data.multi_chain_data_path), # single chain or multichain
								args.data.num_train, args.data.num_val, args.data.num_test, 
								args.data.batch_tokens, args.data.max_batch_size, 
								args.data.min_seq_size, args.data.max_seq_size, 
								args.training_parameters.regularization.use_chain_mask, args.data.max_resolution,
								args.training_parameters.ca_only_model, self.rank, self.world_size, self.training_parameters.rng
							)
		
		self.losses = TrainingRunLosses(	# general
											args.training_parameters.train_type, 
											# for extraction
											args.training_parameters.loss.cel.label_smoothing, 
											args.training_parameters.loss.distogram.label_smoothing, 
											args.training_parameters.loss.distogram.beta, 
											args.hyper_parameters.extraction.distogram.bins,
											args.hyper_parameters.extraction.distogram.min_dist,
											args.hyper_parameters.extraction.distogram.max_dist,
											# for kl annealing
											args.training_parameters.loss.kl.beta, 
											args.training_parameters.loss.kl.kappa, 
											args.training_parameters.loss.kl.midpoint, 
											args.training_parameters.loss.kl.annealing, 
											# for diffusion nll
											args.training_parameters.loss.nll.gamma
										)

		self.output = Output(args.output.out_path, model_checkpoints=args.output.model_checkpoints, rank=self.rank, world_size=self.world_size)

		self.setup_model()
		self.setup_optim()
		self.setup_scheduler()
		self.output.log_trainingrun(self.training_parameters, self.hyper_parameters, self.data)

	def setup_model(self):
		'''
		instantiates proteusAI with given Hyper-Parameters, moves it to gpu, 
		optionally loads model weights from pre-trained models, and freezes modules
		depending on train type

		Args:
			None
		
		Returns:
			None
		'''
		
		self.log("loading model...")
		
		self.model = proteusAI(	# model params
									d_model=self.hyper_parameters.d_model, d_latent=self.hyper_parameters.d_latent, d_wf=self.hyper_parameters.d_wf, num_aas=self.hyper_parameters.num_aa,
									old=self.training_parameters.train_type=="old", mlm=self.training_parameters.train_type=="mlm",

								# wf embedding params
									embedding_min_wl=self.hyper_parameters.embedding.min_wl, embedding_max_wl=self.hyper_parameters.embedding.max_wl, embedding_base_wl=self.hyper_parameters.embedding.base_wl, 
									embedding_learn_wl=self.hyper_parameters.embedding.learn_wl, embedding_learn_aa=self.hyper_parameters.embedding.learn_aa,

								# wf encoding params
									encoding_d_hidden_pre=self.hyper_parameters.encoding.pre_process.d_hidden, encoding_hidden_layers_pre=self.hyper_parameters.encoding.pre_process.hidden_layers, # wf pre_process
									encoding_d_hidden_post=self.hyper_parameters.encoding.post_process.d_hidden, encoding_hidden_layers_post=self.hyper_parameters.encoding.post_process.hidden_layers, # wf post_process 

									# encoder layers
									encoding_encoder_layers=self.hyper_parameters.encoding.encoders.layers, encoding_heads=self.hyper_parameters.encoding.encoders.heads,
									encoding_use_bias=self.hyper_parameters.encoding.encoders.use_bias, encoding_min_rbf=self.hyper_parameters.encoding.encoders.min_rbf,
									encoding_d_hidden_attn=self.hyper_parameters.encoding.encoders.d_hidden_attn, encoding_hidden_layers_attn=self.hyper_parameters.encoding.encoders.hidden_layers_attn,

								# wf diffusion params

									# beta scheduler
									diffusion_alpha_bar_min=self.hyper_parameters.diffusion.scheduler.alpha_bar_min, diffusion_noise_schedule_type=self.hyper_parameters.diffusion.scheduler.noise_schedule_type, diffusion_t_max=self.hyper_parameters.diffusion.scheduler.t_max,

									# timestep params for FiLM
									diffusion_d_in_timestep=self.hyper_parameters.diffusion.timestep.d_in, diffusion_d_hidden_timestep=self.hyper_parameters.diffusion.timestep.d_hidden, diffusion_hidden_layers_timestep=self.hyper_parameters.diffusion.timestep.hidden_layers, 

									# wf post_process
									diffusion_d_hidden_post=self.hyper_parameters.diffusion.post_process.d_hidden, diffusion_hidden_layers_post=self.hyper_parameters.diffusion.post_process.hidden_layers,

									# encoder layers
									diffusion_encoder_layers=self.hyper_parameters.diffusion.encoders.layers, diffusion_heads=self.hyper_parameters.diffusion.encoders.heads,
									diffusion_use_bias=self.hyper_parameters.diffusion.encoders.use_bias, diffusion_min_rbf=self.hyper_parameters.diffusion.encoders.min_rbf,
									diffusion_d_hidden_attn=self.hyper_parameters.diffusion.encoders.d_hidden_attn, diffusion_hidden_layers_attn=self.hyper_parameters.diffusion.encoders.hidden_layers_attn,

								# wf decoding params

									decoding_d_hidden_pre=self.hyper_parameters.decoding.pre_process.d_hidden, decoding_hidden_layers_pre=self.hyper_parameters.decoding.pre_process.hidden_layers, # wf preprocessing
									decoding_d_hidden_post=self.hyper_parameters.decoding.post_process.d_hidden, decoding_hidden_layers_post=self.hyper_parameters.decoding.post_process.hidden_layers, # wf post_process

									# encoder layers
									decoding_encoder_layers=self.hyper_parameters.decoding.encoders.layers, decoding_heads=self.hyper_parameters.decoding.encoders.heads,
									decoding_use_bias=self.hyper_parameters.decoding.encoders.use_bias, decoding_min_rbf=self.hyper_parameters.decoding.encoders.min_rbf,
									decoding_d_hidden_attn=self.hyper_parameters.decoding.encoders.d_hidden_attn, decoding_hidden_layers_attn=self.hyper_parameters.decoding.encoders.hidden_layers_attn,

								# wf extraction params

									# distogram params
									extraction_bins=self.hyper_parameters.extraction.distogram.bins, extraction_dk=self.hyper_parameters.extraction.distogram.dk,

									# wf preprocessing
									extraction_d_hidden_pre=self.hyper_parameters.extraction.pre_process.d_hidden, extraction_hidden_layers_pre=self.hyper_parameters.extraction.pre_process.hidden_layers,

									# wf post_process
									extraction_d_hidden_post=self.hyper_parameters.extraction.post_process.d_hidden, extraction_hidden_layers_post=self.hyper_parameters.extraction.post_process.hidden_layers,

									# encoder layers
									extraction_encoder_layers=self.hyper_parameters.extraction.encoders.layers, extraction_heads=self.hyper_parameters.extraction.encoders.heads,
									extraction_use_bias=self.hyper_parameters.extraction.encoders.use_bias, extraction_min_rbf=self.hyper_parameters.extraction.encoders.min_rbf,
									extraction_d_hidden_attn=self.hyper_parameters.extraction.encoders.d_hidden_attn, extraction_hidden_layers_attn=self.hyper_parameters.extraction.encoders.hidden_layers_attn,

								# dropout
								dropout=self.training_parameters.regularization.dropout,
								attn_dropout=self.training_parameters.regularization.attn_dropout
							)

		# parallelize the model
		self.model.to(self.gpu)
		self.model = DDP(self.model, device_ids=[self.rank])

		# which pretrained weights to use
		if self.training_parameters.weights.use_model:
			pretrained_weights = torch.load(self.training_parameters.weights.use_model, map_location=self.gpu, weights_only=True)
			self.model.module.load_state_dict(pretrained_weights, strict=False) # allow modifications of the model in between transfer learning
		if self.training_parameters.weights.use_embedding_weights:
			self.model.module.load_WFEmbedding_weights(self.training_parameters.weights.use_embedding_weights, self.gpu)
		if self.training_parameters.weights.use_encoding_weights:
			self.model.module.load_WFEncoding_weights(self.training_parameters.weights.use_encoding_weights, self.gpu)
		if self.training_parameters.weights.use_diffusion_weights:
			self.model.module.load_WFDiffusion_weights(self.training_parameters.weights.use_diffusion_weights, self.gpu)
		if self.training_parameters.weights.use_decoding_weights:
			self.model.module.load_WFDecoding_weights(self.training_parameters.weights.use_decoding_weights, self.gpu)
		if self.training_parameters.weights.use_extraction_weights:
			self.model.module.load_WFExtraction_weights(self.training_parameters.weights.use_extraction_weights, self.gpu)
		
		# what weights should be frozen depending on training type 
		if self.training_parameters.train_type in ["extraction", "old", "mlm"]: # train embedding and extraction only
			if self.training_parameters.weights.embedding.freeze_at_seq_sim == 0.0: # option to freeze embedding right away, only use if have pretrained embedding weights
				self.model.module.freeze_WFEmbedding_weights()
			if self.training_parameters.weights.geo_attn.init_bias_off:
				self.model.module.turn_off_bias() # turn off geo attn bias until certain seq sim
			if self.training_parameters.train_type == "extraction":
				self.model.module.freeze_WFEncoding_weights()
				self.model.module.freeze_WFDiffusion_weights()
				self.model.module.freeze_WFDecoding_weights()

		elif self.training_parameters.train_type == "vae": # train encoder and decoder using trained (and frozen) embedding
			self.model.module.freeze_WFEmbedding_weights()
			self.model.module.freeze_WFDiffusion_weights()
			self.model.module.freeze_WFExtraction_weights()
		
		elif self.training_parameters.train_type == "extraction_finetune": # finetune extractor on decoder outputs, only extractor is not frozen
			self.model.module.freeze_WFEmbedding_weights()
			self.model.module.freeze_WFEncoding_weights()
			self.model.module.freeze_WFDiffusion_weights()
			self.model.module.freeze_WFDecoding_weights()

		elif self.training_parameters.train_type == "diffusion": # train diffusion on encoder outputs, diffusion is the only one not frozen
			self.model.module.freeze_WFEmbedding_weights()
			self.model.module.freeze_WFEncoding_weights()
			self.model.module.freeze_WFDecoding_weights()
			self.model.module.freeze_WFExtraction_weights()

		else:
			raise ValueError(f"invalid train_type selected, {self.training_parameters.train_type=}. options are ['extraction', 'vae', 'extraction_finetune', 'diffusion', 'old', 'mlm']") 

		# get number of parameters for logging
		self.training_parameters.num_embedding_params = sum(p.numel() for p in self.model.module.wf_embedding.parameters())
		self.training_parameters.num_encoding_params = sum(p.numel() for p in self.model.module.wf_encoding.parameters()) if self.training_parameters.train_type not in ["old", "mlm"] else 0.0
		self.training_parameters.num_diffusion_params = sum(p.numel() for p in self.model.module.wf_diffusion.parameters()) if self.training_parameters.train_type not in ["old", "mlm"] else 0.0
		self.training_parameters.num_decoding_params = sum(p.numel() for p in self.model.module.wf_decoding.parameters()) if self.training_parameters.train_type not in ["old", "mlm"] else 0.0
		self.training_parameters.num_extraction_params = sum(p.numel() for p in self.model.module.wf_extraction.parameters())
		self.training_parameters.num_params = self.training_parameters.num_embedding_params + self.training_parameters.num_encoding_params + self.training_parameters.num_diffusion_params + self.training_parameters.num_decoding_params +  self.training_parameters.num_extraction_params 

		# print gradients at each step if in debugging mode
		if self.debug.debug_grad: # havent tested with DDP, but might interfere with DDP hooks for grad reduce, will check later, prob not until i need to debug grads lol
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

		self.log("loading optimizer...")
		self.optim = torch.optim.Adam(self.model.parameters(), lr=1.0,
									betas=(self.training_parameters.adam.beta1, self.training_parameters.adam.beta2), 
									eps=float(self.training_parameters.adam.epsilon))
		self.optim.zero_grad()
		if self.training_parameters.adam.use_adam:
			self.optim.load_state_dict(self.training_parameters.adam.use_adam)

	def setup_scheduler(self):
		'''
		sets up the loss scheduler, right now only using ReduceLROnPlateu, but planning on making this configurable

		Args:
			None
		
		Returns:
			None
		'''

		self.log("loading scheduler...")

		if self.training_parameters.lr.lr_type == "attn":

			# compute the scale
			if self.training_parameters.lr.lr_step == 0.0:
				scale = self.hyper_parameters.d_model**(-0.5)
			else:
				scale = self.training_parameters.lr.warmup_steps**(0.5) * self.training_parameters.lr.lr_step # scale needed so max lr is what was specified
			
			def attn(step):
				'''lr scheduler from attn paper'''
				step = step + self.training_parameters.lr.start_from_step # in case job gets cancelled and want to start from where left off
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
			if self.rank==0:
				self.output.save_model(self.model, adam=self.optim, appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_loss(),2)}")

	def training_converged(self, epoch_idx):

		if self.training_parameters.train_type in ["extraction", "extraction_finetune", "old", "mlm"]:
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

	def check_if_freeze_embedding(self):
		if self.rank == 0:
			if self.training_parameters.train_type in ["extraction", "old", "mlm"]:
				if self.model.module.wf_embedding.aa_magnitudes.requires_grad or self.model.module.wf_embedding.wavenumbers.requires_grad:
					if self.losses.val.get_last_match() >= self.training_parameters.weights.embedding.freeze_at_seq_sim:
						self.model.module.freeze_WFEmbedding_weights()

	def check_if_turn_bias_on(self):
		if self.rank == 0:
			if (self.training_parameters.train_type in ["extraction", "old", "mlm"]) and self.hyper_parameters.extraction.encoders.use_bias:
				if not any(encoder.attn.beta_weights.requires_grad for encoder in self.model.module.wf_extraction.encoders):
					if self.losses.val.get_last_match() >= self.training_parameters.weights.geo_attn.turn_bias_on_at_seq_sim:
						self.model.module.turn_on_bias() # turn the geo attention bias on now that embedding and the QKV weights are expressive enough 
	
	def train(self):
		'''
		entry point for training the model. loads train and validation data, loops through epochs, plots training, 
		runs testing and saves the model
		'''

		# load the data, note that all gpus are required to load all the data with the same random seeds, so they get unique data compared to other gpus each epoch
		self.log("loading training data...")
		self.data.load("train")
		self.log("loading validation data...")
		self.data.load("val")

		# log training info
		self.log(f"\n\ninitializing training. "\
					f"training on {len(self.data.train_data)} batches "\
					f"of batch size {self.data.batch_tokens} tokens "\
					f"for {self.training_parameters.epochs} epochs.\n" 
				)
		
		# loop through epochs
		for epoch_idx in range(self.training_parameters.epochs):

			epoch = Epoch(self, epoch_idx)
			epoch.epoch_loop()
			
			self.model_checkpoint(epoch_idx)
			if self.training_converged(epoch_idx): break
			
			self.check_if_freeze_embedding() # freeze embedding once validation seq sim gets high enough
			self.check_if_turn_bias_on() # turn geo attn bias on once QKV weights and embedding are well tuned

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
			self.log("running validation...")

			# progress bar
			if self.rank == 0:
				val_pbar = tqdm(total=len(self.data.val_data), desc="epoch_validation_progress", unit="step")
			
			# loop through validation batches
			for coords, labels, chain_idxs, chain_mask, key_padding_mask in self.data.val_data:
					
				# init batch
				batch = Batch(coords, labels, chain_idxs, chain_mask, key_padding_mask, epoch=dummy_epoch)

				# run the model
				batch.batch_forward()

				# update pbar
				if self.rank == 0:
					val_pbar.update(self.world_size)

			# add the avg losses to the global loss and log
			self.output.log_val_losses(self.losses, self.training_parameters.train_type)

	def test(self):

		# switch to evaluation mode
		self.model.eval()
		
		# load testing data
		self.log("loading testing data...")
		self.data.load("test")

		# init losses
		self.losses.clear_tmp_losses()

		# dummy epoch so can still access training run parent
		dummy_epoch = Epoch(self)

		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			if self.rank == 0:
				test_pbar = tqdm(total=len(self.data.test_data), desc="test_progress", unit="step")

			# loop through testing batches
			for coords, labels, chain_idxs, chain_mask, key_padding_mask in self.data.test_data:
					
				# init batch
				batch = Batch(  coords, labels, chain_idxs, chain_mask, key_padding_mask, 
								temp=self.training_parameters.inference.temperature, 
								cycles=self.training_parameters.inference.cycles,
								inference=self.training_parameters.train_type in ["diffusion", "mlm"], # can only run inference once extraction AND diffusion have been trained
								epoch=dummy_epoch
							)

				# run the model
				batch.batch_forward()

				# update pbar
				if self.rank == 0:
					test_pbar.update(self.world_size)
		
		# log the losses
		self.output.log_test_losses(self.losses, self.training_parameters.train_type)

	def log(self, message):
		if self.rank==0:
			self.output.log.info(message)
