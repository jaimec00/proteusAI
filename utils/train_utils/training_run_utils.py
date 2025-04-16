# ----------------------------------------------------------------------------------------------------------------------

import torch
from tqdm import tqdm
from utils.train_utils.model_outputs import ExtractionOutput, VAEOutput, DiffusionOutput, InferenceOutput, ModelOutputs

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
		self.train(self.training_run_parent.model)

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
		self.training_run_parent.output.log_epoch_losses(self.training_run_parent.losses, self.training_run_parent.training_parameters.train_type)

		# run validation
		self.training_run_parent.validation()

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()
			self.training_run_parent.data.val_data.rotate_data()

	def train(self, model):

		if self.training_run_parent.training_parameters.train_type == "extraction":
			model.wf_embedding.train()
			model.wf_encoding.eval()
			model.wf_diffusion.eval()
			model.wf_decoding.eval()
			model.wf_extraction.train()
		if self.training_run_parent.training_parameters.train_type == "vae":
			model.wf_embedding.eval()
			model.wf_encoding.train()
			model.wf_diffusion.eval()
			model.wf_decoding.train()
			model.wf_extraction.eval()
		if self.training_run_parent.training_parameters.train_type == "extraction_finetune":
			model.wf_embedding.eval()
			model.wf_encoding.eval()
			model.wf_diffusion.eval()
			model.wf_decoding.eval()
			model.wf_extraction.train()
		if self.training_run_parent.training_parameters.train_type == "diffusion":
			model.wf_embedding.eval()
			model.wf_encoding.eval()
			model.wf_diffusion.train()
			model.wf_decoding.eval()	
			model.wf_extraction.eval()
		if self.training_run_parent.training_parameters.train_type == "old":
			model.wf_embedding.train()
			model.wf_encoding.eval()
			model.wf_diffusion.eval()
			model.wf_decoding.eval()	
			model.wf_extraction.train()

# ----------------------------------------------------------------------------------------------------------------------

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

			if self.epoch_parent.training_run_parent.training_parameters.train_type=="vae":
				if self.epoch_parent.training_run_parent.training_parameters.loss.kl.annealing:
					# perform kl annealing, does sigmoidal increase
					self.epoch_parent.training_run_parent.losses.loss_function.kl_annealing_step += 1

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

		# get alpha carbon and beta carbon coords, done here rather than in init so can add noise to coords if specified
		self.coords_alpha, self.coords_beta = self.epoch_parent.training_run_parent.model.wf_embedding.get_CaCb_coords(self.coords, self.chain_idxs)

		# run model depending on training type
		match self.epoch_parent.training_run_parent.training_parameters.train_type:

			case "extraction":
				output = self.run_extraction_training()

			case "old": # old model with no diffusion is just extraction, but with no aa info
				output = self.run_extraction_training()

			case "vae":
				output = self.run_vae_training()

			case "extraction_finetune":
				output = self.run_extraction_finetune_training()

			case "diffusion":
				if self.inference:
					output = self.run_full_inference()
				else:
					output = self.run_diffusion_training()

		return ModelOutputs(output)

	def run_extraction_training(self):

		# get wf
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)

		# extract sequence 
		seq_pred = self.epoch_parent.training_run_parent.model(wf=wf, wf_no_aa=wf, key_padding_mask=self.key_padding_mask, extraction=True)

		# convert to output object
		return ExtractionOutput(self, seq_pred)

	def run_vae_training(self):
		
		# get wf
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)
		wf_no_aa = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True, no_aa=True)
		
		# predict mean and log var
		wf_encoded_mean, wf_encoded_log_var = self.epoch_parent.training_run_parent.model(wf=wf, key_padding_mask=self.key_padding_mask, encoding=True)

		# sample from the latent space given the mean and var
		wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.sample(wf_encoded_mean, wf_encoded_log_var)
		
		# decode from latent space to wf space, only computes mean to make reconstruction loss a simple squared error
		wf_decoded = self.epoch_parent.training_run_parent.model(latent=wf_encoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, decoding=True)

		# convert to output object
		return VAEOutput(self, wf_encoded_mean, wf_encoded_log_var, wf_decoded, wf)

	def run_extraction_finetune_training(self):
		
		# get wf
		wf = self.epoch_parent.training_run_parent.model(self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)
		wf_no_aa = self.epoch_parent.training_run_parent.model(self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True, no_aa=True)

		# make into latent representation
		wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.encode(wf, key_padding_mask=self.key_padding_mask)
		
		# decode from latent space to wf space, only computes mean to make reconstruction loss a simple squared error
		wf_decoded = self.epoch_parent.training_run_parent.model(latent=wf_encoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, decoding=True)

		# extract sequence
		seq_pred = self.epoch_parent.training_run_parent.model(wf=wf_decoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, extraction=True)

		# convert to output object
		return ExtractionOutput(self, seq_pred)

	def run_diffusion_training(self):
		
		# get clean wavefunction
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)
		wf_no_aa = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True, no_aa=True)

		# encode the wf in latent space
		wf_latent_mean, wf_latent_logvar = self.epoch_parent.training_run_parent.model(wf=wf, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, encoding=True)
		wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.sample(wf_latent_mean, wf_latent_logvar)

		# get timesteps from uniform distribution, as well as abars for reconstructing x0 for nll loss
		timesteps = self.epoch_parent.training_run_parent.model.wf_diffusion.get_random_timesteps(wf_encoded.size(0), wf_encoded.device)
		abars, _ = self.epoch_parent.training_run_parent.model.wf_diffusion.noise_scheduler(timesteps.unsqueeze(1).unsqueeze(2)) # for loss scaling

		# add noise
		noised_wf, noise = self.epoch_parent.training_run_parent.model.wf_diffusion.noise(wf_encoded, timesteps)

		# predict noise
		noise_pred = self.epoch_parent.training_run_parent.model(latent=noised_wf, t=timesteps, key_padding_mask=self.key_padding_mask, diffusion=True)

		# convert to output object
		return DiffusionOutput(self, noise_pred, noise, noised_wf, wf_latent_mean, wf_latent_logvar, abars)

	def run_full_inference(self):

		# inference uses random aas to test predictions from only structure, seq sim only computed for representative chain. model gets full structure info, but no sequence info
		self.aas = -torch.ones_like(self.aas) # no sequence info at all. diffusion starts by making all non fixed (eq. -1) positions to random AA before running
		t = torch.full((self.coords_alpha.size(0),), self.epoch_parent.training_run_parent.hyper_parameters.diffusion.scheduler.t_max)

		seq_pred = self.epoch_parent.training_run_parent.model(	self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask,
																inference=True, t=t,
																cycles=self.epoch_parent.training_run_parent.training_parameters.inference.cycles, 
																temp=self.epoch_parent.training_run_parent.training_parameters.inference.temperature
															)

		return InferenceOutput(self, seq_pred)

	def size(self, idx):
		return self.labels.size(idx)

# ----------------------------------------------------------------------------------------------------------------------
