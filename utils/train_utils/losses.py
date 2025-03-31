import torch
import torch.nn as nn
from torch.nn import KLDivLoss, CrossEntropyLoss, MSELoss
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# losses 

class TrainingRunLosses():

	def __init__(self, train_type, d_model, d_latent, num_aa, label_smoothing, cel_scaling_factor=1.0):
		loss_type = ExtractionLosses if train_type=="extraction" else DiffusionLosses
		self.train = loss_type()
		self.val = loss_type()
		self.test = loss_type()
		self.tmp = loss_type()
		self.loss_function = ExtractionLossFunction(d_model, d_latent, num_aa, label_smoothing, cel_scaling_factor) if train_type=="extraction" else DiffusionLossFunction(d_latent)

	def clear_tmp_losses(self):
		self.tmp.clear_losses()

	def to_numpy(self):
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

class ExtractionLosses():
	'''
	class to store losses
	'''
	def __init__(self): 

		# saved for logging
		self.kl_div = [] # kl div to match gaussian prior
		self.reconstruction = [] # reconstruction of the wf
		self.cel = [] # cel for aa prediction from wf
		self.matches = [] # number of matches from greedy selection of aa, just for logging

		# actual losses
		self.all_losses = [] # full scaled loss, contains kldiv, reconstruction, and cel

		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token per cha

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		valid_toks = self.valid_toks.item()
		avg_kl_div = sum(kl_div.item() for kl_div in self.kl_div if kl_div) / valid_toks
		avg_reconstruction = sum(reconstruction.item() for reconstruction in self.reconstruction if reconstruction) / valid_toks
		avg_cel = sum(cel.item() for cel in self.cel if cel) / valid_toks
		avg_loss = sum(loss.item() for loss in self.all_losses if loss) / valid_toks
		avg_seq_sim = 100*sum(match.item() for match in self.matches if match) / valid_toks
		
		return avg_kl_div, avg_reconstruction, avg_cel, avg_loss, avg_seq_sim

	def add_losses(self, kl_div, reconstruction, cel, full_loss, matches, valid=1):
		self.kl_div.append(kl_div)
		self.reconstruction.append(reconstruction)
		self.cel.append(cel)
		self.all_losses.append(full_loss)
		self.matches.append(matches)
		self.valid_toks += valid

	def extend_losses(self, other):
		self.kl_div.extend(other.kl_div)
		self.reconstruction.extend(other.reconstruction)
		self.cel.extend(other.cel)
		self.all_losses.extend(other.all_losses)
		self.matches.extend(other.matches)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.kl_div = []
		self.reconstruction = []
		self.cel = []
		self.all_losses = []
		self.matches = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.all_losses[-1]

	def get_last_match(self):
		return self.matches[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.kl_div = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.kl_div]
		self.reconstruction = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.reconstruction]
		self.cel = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.cel]
		self.all_losses = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.all_losses]
		self.matches = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches]

	def __len__(self):
		return len(self.all_losses)

class DiffusionLosses():
	def __init__(self):
		self.squared_errors = []
		self.valid_toks = 0

	def get_avg(self, is_inference=False):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		valid_toks = self.valid_toks.item()
		if is_inference: # store the seq sims in squared errors list, instead of dealing w seperate lists the whole run
			return 100*sum(match.item() for match in self.squared_errors if match) / valid_toks
		else:
			return sum(loss.item() for loss in self.squared_errors if loss) / valid_toks
		
	def add_losses(self, squared_error, valid=1):
		self.squared_errors.append(squared_error)
		self.valid_toks += valid

	def extend_losses(self, other):
		self.squared_errors.extend(other.squared_errors)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.squared_errors = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.squared_errors[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.squared_errors = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.squared_errors]

	def __len__(self):
		return len(self.squared_errors)

# ----------------------------------------------------------------------------------------------------------------------
# loss functions 

class ExtractionLossFunction(nn.Module):
	'''manually implementing the losses instead of torch built ins for more control, 
	since will want to also include histogram of seq sim and the like for each seq. functionality not included rn, but will add later
	all losses are scaled per token, so take the average across the feature dimensions
	'''
	def __init__(self, d_model, d_latent, num_aa, label_smoothing, cel_scaling_factor=1.0):
		super(ExtractionLossFunction, self).__init__()
		self.d_model = d_model
		self.d_latent = d_latent
		self.num_aa = num_aa
		self.cel_raw = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=label_smoothing)
		self.cel_scaling_factor = cel_scaling_factor


	def kl_div(self, prior_mean_pred, prior_log_var_pred, mask):
		kl_div = -0.5*torch.sum(1 + prior_log_var_pred - prior_mean_pred.pow(2) - torch.exp(prior_log_var_pred), dim=2) # Z x N
		
		return (kl_div*(~mask)).sum() / self.d_latent # will change this later, mask invalid and sum

	def reconstruction(self, reconstruct_mean_pred, reconstruct_mean_true, mask):
		return ((reconstruct_mean_true - reconstruct_mean_pred).pow(2) * (~mask).unsqueeze(2)).sum() / self.d_model

	def cel(self, seq_pred, seq_true):
		'''
		seq_pred: Z x N x num_aa (logits)
		seq_true: Z x N (labels, -1 not valid)
		'''

		cel = self.cel_raw(seq_pred.view(-1, self.num_aa), seq_true.view(-1)) # Z*N
		
		return cel

	def full_loss(self, kl_div, reconstruction, cel):# cel is typically larger than mse and kldiv, so scale it down so vae focuses on wf reconstruction more
		return kl_div + reconstruction + (cel * self.cel_scaling_factor)

	def compute_matches(self, seq_pred, seq_true):
		'''greedy selection, computed seq sim here for simplicity, will do it with other losses later '''
		
		prediction_flat = seq_pred.softmax(dim=2).argmax(dim=2).view(-1) # batch*N
		true_flat = seq_true.view(-1) # batch x N --> batch*N,
		valid_mask = true_flat != -1 # batch*N, 
		matches = ((prediction_flat == true_flat) & (valid_mask)).sum() # 1, 
		
		return matches 

	def forward(self, 	prior_mean_pred, prior_log_var_pred,
						reconstruct_mean_pred, reconstruct_mean_true, 
						seq_pred, seq_true
				):
		mask = seq_true == -1
		kl_div = self.kl_div(prior_mean_pred, prior_log_var_pred, mask)
		reconstruction = self.reconstruction(reconstruct_mean_pred, reconstruct_mean_true, mask)
		cel = self.cel(seq_pred, seq_true)
		full_loss = self.full_loss(kl_div, reconstruction, cel)
		matches = self.compute_matches(seq_pred, seq_true)

		return kl_div, reconstruction, cel, full_loss, matches # return all for logging, only full loss used for backprop

class DiffusionLossFunction(nn.Module):
	def __init__(self, d_latent):
		super(DiffusionLossFunction, self).__init__()
		self.d_latent = d_latent # to scale the losses on a per feature basis
	def forward(self, latent_pred, latent_true):
		'''simple sum of squared errors'''
		loss = ((latent_true - latent_pred).pow(2)).sum() / self.d_latent
		return [loss]

# ----------------------------------------------------------------------------------------------------------------------
