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

# detect anomolies in training
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
		os.environ['MASTER_PORT'] = str(args.port)
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
								args.training_parameters.ca_only_model, args.training_parameters.regularization.homo_thresh, self.rank, self.world_size, self.training_parameters.rng
							)
		
		self.losses = TrainingRunLosses(args.training_parameters.regularization.label_smoothing)

		self.output = Output(args.output.out_path, model_checkpoints=args.output.model_checkpoints, rank=self.rank, world_size=self.world_size)

		self.checkpoint = torch.load(self.training_parameters.checkpoint.path, weights_only=True, map_location=self.gpu) if self.training_parameters.checkpoint.path else ""

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
		
		self.model = proteusAI(	K=self.hyper_parameters.topk, De=self.hyper_parameters.d_e, Dw=self.hyper_parameters.d_wf, Dv=self.hyper_parameters.d_v, Ds=self.hyper_parameters.d_e,
								min_wl=self.hyper_parameters.node_embedding.min_wl, max_wl=self.hyper_parameters.node_embedding.max_wl, base_wl=self.hyper_parameters.node_embedding.base_wl, anisotropic=self.hyper_parameters.node_embedding.anisotropic, learn_wl=self.hyper_parameters.node_embedding.learn_wl, learn_aa=self.hyper_parameters.node_embedding.learn_aa, 
								min_rbf=self.hyper_parameters.edge_embedding.min_rbf, max_rbf=self.hyper_parameters.edge_embedding.max_rbf, num_rbfs=self.hyper_parameters.edge_embedding.num_rbfs,
								enc_layers=self.hyper_parameters.encoder.layers, dec_layers=self.hyper_parameters.decoder.layers, dropout=self.training_parameters.regularization.dropout)
	
		# parallelize the model
		self.model.to(self.gpu)
		self.model = DDP(self.model, device_ids=[self.rank])

		if self.checkpoint:
			self.model.module.load_state_dict(self.checkpoint["model"])

		# get number of parameters for logging
		self.training_parameters.num_params = sum(p.numel() for p in self.model.module.parameters())

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
		self.optim = torch.optim.AdamW(	self.model.parameters(), lr=1.0,
										betas=(self.training_parameters.adam.beta1, self.training_parameters.adam.beta2), 
										eps=float(self.training_parameters.adam.epsilon), weight_decay=self.training_parameters.adam.weight_decay)
		self.optim.zero_grad()
		if self.checkpoint:
			self.optim.load_state_dict(self.checkpoint["adam"], weights_only=True, map_location=self.gpu)

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
				step = step # in case job gets cancelled and want to start from where left off
				return scale * min((step+1)**(-0.5), (step+1)*(self.training_parameters.lr.warmup_steps**(-1.5)))

			self.scheduler = lr_scheduler.LambdaLR(self.optim, attn)

		elif self.training_parameters.lr.lr_type == "static":
			def static(step):
				return self.training_parameters.lr.lr_step
			self.scheduler = lr_scheduler.LambdaLR(self.optim, static)

		else:
			raise ValueError(f"invalid lr_type: {self.training_parameters.lr.lr_type}. options are ['attn', 'static']")

		if self.checkpoint:
			self.scheduler.load_state_dict(self.checkpoint["scheduler"])

	def model_checkpoint(self, epoch_idx):
		if (epoch_idx+1) % self.output.model_checkpoints == 0: # model checkpointing
			if self.rank==0:
				self.output.save_checkpoint(self.model, adam=self.optim, scheduler=self.scheduler, appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_loss(),2)}")

	def training_converged(self, epoch_idx):

		criteria = self.losses.val.cel

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

		return has_converged

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
					f"training on approx. {len(self.data.train_data)} batches "\
					f"of batch size {self.data.batch_tokens} tokens "\
					f"for {self.training_parameters.epochs} epochs.\n" 
				)
		
		# loop through epochs
		for epoch_idx in range(self.training_parameters.epochs):

			# initialize epoch and loop through batches
			epoch = Epoch(self, epoch_idx)
			epoch.epoch_loop()
			
			self.model_checkpoint(epoch_idx)
			if self.training_converged(epoch_idx): break
			
		# announce trainnig is done
		self.log(f"training done after {epoch.epoch} epochs", fancy=True)

		# plot training losses
		self.output.plot_training(self.losses)

		# save the model
		self.output.save_checkpoint(self.model, self.optim, self.scheduler, appended_str="final")

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
			for data_batch in self.data.val_data:
					
				# init batch
				batch = Batch(data_batch, epoch=dummy_epoch)

				# run the model
				batch.batch_forward()

				# update pbar
				if self.rank == 0:
					val_pbar.update(self.world_size)

			# add the avg losses to the global loss and log
			self.output.log_val_losses(self.losses)

	def test(self):

		# switch to evaluation mode
		self.model.eval()

		self.log("starting testing", fancy=True)
		
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
			for data_batch in self.data.test_data:
					
				# init batch
				batch = Batch(  data_batch, 
								temp=self.training_parameters.inference.temperature, 
								inference=True, epoch=dummy_epoch
							)

				# run the model
				batch.batch_forward()

				# update pbar
				if self.rank == 0:
					test_pbar.update(self.world_size)
		
		# log the losses
		self.output.log_test_losses(self.losses)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"

		if self.rank==0:
			self.output.log.info(message)