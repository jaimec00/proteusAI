
import torch

# ----------------------------------------------------------------------------------------------------------------------
# outputs

class ExtractionOutput():
	'''need to edit so have 
		a kl div loss
		a reconstruction loss
		a cel loss
	'''
	def __init__(self, batch_parent, seq_pred):
		
		# batch parent
		self.batch_parent = batch_parent 

		# predictions
		self.seq_pred = seq_pred # seq pred from wf output

		# valid tokens for averaging
		self.valid_toks = (batch_parent.labels!=-1).sum()

		self.valid_samples = (batch_parent.labels!=-1).any(dim=1).sum()


	def compute_losses(self):
		return self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(self.seq_pred, self.batch_parent.labels)

class VAEOutput():
	'''need to edit so have 
		a kl div loss
		a reconstruction loss
		a cel loss
	'''
	def __init__(self, batch_parent, latent_mean_pred, latent_log_var_pred, wf_mean_pred, wf_mean_true):
		
		# batch parent
		self.batch_parent = batch_parent 

		# predictions
		# encoder outputs, loss is computed by comparing to gaussian, so dont need a true
		self.latent_mean_pred = latent_mean_pred # gaussian prior mean
		self.latent_log_var_pred = latent_log_var_pred # gaussian prior log var

		# decoder outputs
		self.wf_mean_pred = wf_mean_pred # wf prediction mean
		self.wf_mean_true = wf_mean_true # true

		# valid tokens for averaging
		self.valid_toks = (batch_parent.labels!=-1).sum()
		self.valid_samples = (batch_parent.labels!=-1).any(dim=1).sum()

	def compute_losses(self):
		return self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(	self.latent_mean_pred, self.latent_log_var_pred, 
																						self.wf_mean_pred, self.wf_mean_true, self.batch_parent.labels==-1, 
																					)

class DiffusionOutput():
	def __init__(self, batch_parent, noise_pred, true_noise, noised_latent, latent_mean, latent_logvar, abars):
		self.batch_parent = batch_parent
		self.noise_pred = noise_pred
		self.true_noise = true_noise
		self.latent_mean = latent_mean
		self.latent_logvar = latent_logvar
		self.noised_latent = noised_latent
		self.mask = batch_parent.labels.unsqueeze(2)==-1
		self.valid_toks = (~self.mask).sum()
		self.abars = abars
	def compute_losses(self):
		return self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(self.noise_pred, self.true_noise, self.noised_latent, self.latent_mean, self.latent_logvar, self.abars, self.mask)

class InferenceOutput():
	def __init__(self, batch_parent, seq_pred):
		self.batch_parent = batch_parent 
		self.seq_pred = seq_pred
		self.valid_toks = (batch_parent.labels!=-1).sum()
		self.valid_samples = (batch_parent.labels!=-1).any(dim=1).sum()

	def compute_matches(self):
		'''greedy selection, computed seq sim here for simplicity, will do it with other losses later '''
		
		prediction_flat = self.seq_pred.view(-1) # batch*N
		labels_flat = self.batch_parent.labels.view(-1) # batch x N --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		matches = ((prediction_flat == labels_flat) & (valid_mask)).sum() # 1, 
		
		return matches 

	def compute_losses(self):
		matches = self.compute_matches()

		return [matches]

class ModelOutputs():
	def __init__(self, output):
		self.output = output

	def get_losses(self):
		self.output.batch_parent.epoch_parent.training_run_parent.losses.tmp.add_losses(*self.output.compute_losses(), valid_toks=self.output.valid_toks, valid_samples=self.output.valid_samples)