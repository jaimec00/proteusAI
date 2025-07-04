# ----------------------------------------------------------------------------------------------------------------------

import torch
from tqdm import tqdm
from utils.train_utils.model_outputs import ModelOutputs
from data.constants import aa_2_lbl

# ----------------------------------------------------------------------------------------------------------------------

class Epoch():	
	def __init__(self, training_run, epoch=None):

		self.training_run_parent = training_run
		self.epoch = epoch
		self.epochs = training_run.training_parameters.epochs
		
	def epoch_loop(self):
		'''
		a single training loop through one epoch. loops through batches, logs the losses, and runs validation
		'''

		# make sure in training mode
		self.training_run_parent.model.module.train()

		# setup the epoch
		if self.training_run_parent.rank==0:
			self.training_run_parent.output.log_epoch(self.epoch, self.training_run_parent.step, self.training_run_parent.scheduler.get_last_lr()[0])

		# clear temp losses
		self.training_run_parent.losses.clear_tmp_losses()

		# init epoch pbar
		if self.training_run_parent.rank==0:
			epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")

		# loop through batches
		for b_idx, data_batch in enumerate(self.training_run_parent.data.train_data):

			# instantiate this batch
			batch = Batch(data_batch, b_idx=b_idx, epoch=self)

			# learn
			batch.batch_learn()

			# update pbar
			if self.training_run_parent.rank==0:
				epoch_pbar.update(self.training_run_parent.world_size)
		
		# log epoch losses and save avg
		self.training_run_parent.output.log_epoch_losses(self.training_run_parent.losses)

		# run validation
		self.training_run_parent.validation()

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			if self.training_run_parent.rank==0:
				self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()
			self.training_run_parent.data.val_data.rotate_data()

# ----------------------------------------------------------------------------------------------------------------------

class Batch():
	def __init__(self, data_batch, b_idx=None, epoch=None, inference=False, temp=1e-6):

		self.coords = data_batch.coords 
		self.labels = data_batch.labels
		self.aas = data_batch.labels if not inference else -torch.ones_like(data_batch.labels)# self.labels is edited for loss computation, keep this one for model input
		self.chain_idxs = data_batch.chain_idxs
		self.chain_mask = data_batch.chain_masks | data_batch.homo_masks
		self.key_padding_mask = data_batch.key_padding_masks


		self.b_idx = b_idx
		self.epoch_parent = epoch

		self.inference = inference
		self.temp = temp

		self.world_size = epoch.training_run_parent.world_size
		self.rank = epoch.training_run_parent.rank

	def move_to(self, device):

		self.labels = self.labels.to(device)
		self.aas = self.aas.to(device)
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
		self.labels = self.labels.masked_fill((~self.chain_mask) | self.key_padding_mask | (self.labels == aa_2_lbl("X")), -1)

		# get model outputs
		self.outputs = self.get_outputs()

		# get losses (adds them to training run tmp losses)
		self.outputs.get_losses()

	def batch_backward(self):

		# utils
		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.loss.accumulation_steps
		optim = self.epoch_parent.training_run_parent.optim
		scheduler = self.epoch_parent.training_run_parent.scheduler

		if self.epoch_parent.training_run_parent.training_parameters.loss.token_based_step:
			toks_proc = torch.tensor(self.epoch_parent.training_run_parent.toks_processed, device=self.epoch_parent.training_run_parent.gpu)
			torch.distributed.all_reduce(toks_proc, op=torch.distributed.ReduceOp.SUM)
			learn_step = (toks_proc.item() + 1) > accumulation_steps
		else:
			learn_step = (self.b_idx + 1) % accumulation_steps == 0

		# get last loss (ddp avgs the gradients, i want the sum, so mult by world size)
		loss = self.epoch_parent.training_run_parent.losses.tmp.get_last_loss() * self.epoch_parent.training_run_parent.world_size # no scaling by accumulation steps, as already handled by grad clipping and scaling would introduce batch size biases

		if self.epoch_parent.training_run_parent.debug.print_losses:
			if self.rank==0: # only printing for rank 0 for now
				self.epoch_parent.training_run_parent.output.log.info(f"loss: {loss}")

		# perform backward pass to accum grads
		loss.backward()

		if learn_step:
		
			if self.epoch_parent.training_run_parent.debug.print_grad_L2:
				if self.rank==0:
					L2 = sum(param.grad**2 for param in self.epoch_parent.training_run_parent.model.module.parameters() if param.grad is not None)
					self.epoch_parent.training_run_parent.output.log.info(f"grad L2: {L2}")

			# grad clip
			if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)

			# step
			optim.step()
			optim.zero_grad()
			scheduler.step()

			self.epoch_parent.training_run_parent.step += 1
			self.epoch_parent.training_run_parent.toks_processed = 0 # reset to 0 once target number of toks processed

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

		# run the model
		Z_mu, Z_logvar, noise, noise_pred, seq_logits = self.epoch_parent.training_run_parent.model.module(	self.coords, self.aas, self.chain_idxs, 
																											node_mask=self.key_padding_mask, 
																											inference=self.inference, temp=self.temp
																										)

		# convert to output object
		return ModelOutputs(self, Z_mu, Z_logvar, noise, noise_pred, seq_logits)

	def size(self, idx):
		return self.labels.size(idx)

# ----------------------------------------------------------------------------------------------------------------------
