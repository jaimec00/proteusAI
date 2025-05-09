import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import math
# from utils.model_utils.base_modules.distogram_loss import DistogramLoss

# ----------------------------------------------------------------------------------------------------------------------
# losses 

class TrainingRunLosses():

	def __init__(self, train_type, label_smoothing=0.0, dist_label_smoothing=0.0, dist_beta=1.0, beta=1.0, bins=32, min_dist=2.0, max_dist=22.0, kappa=1.0, midpoint=4000, anneal=True, gamma=1.0):
		if train_type in ["extraction", "extraction_finetune", "old", "mlm"]:
			loss_type = ExtractionLosses 
			self.loss_function = ExtractionLossFunction(label_smoothing, dist_label_smoothing, dist_beta, bins, min_dist, max_dist)
		elif train_type=="vae":
			loss_type = VAELosses
			self.loss_function = VAELossFunction(beta, kappa, midpoint, anneal) 
		elif train_type == "diffusion":
			loss_type = DiffusionLosses
			self.loss_function = DiffusionLossFunction(gamma)

		self.train = loss_type()
		self.val = loss_type()
		self.test = loss_type()
		self.tmp = loss_type()

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
		self.cel = [] # cel for aa prediction from wf
		self.dist_cel = [] # cel for aa prediction from wf
		self.full_loss = [] # cel for aa prediction from wf
		self.matches = [] # number of matches from greedy selection of aa, just for logging

		# to scale losses for logging, does not affect backprop
		self.valid_toks = torch.tensor([0]) # valid tokens to compute avg per token per cha


	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		valid_toks = self.valid_toks.item()
		avg_cel = sum(cel.item() for cel in self.cel if cel) / valid_toks
		avg_dist_cel = sum(dist_cel.item() for dist_cel in self.dist_cel if dist_cel) / valid_toks
		avg_loss = sum(loss.item() for loss in self.full_loss if loss) / valid_toks
		avg_seq_sim = 100*sum(match.item() for match in self.matches if match) / valid_toks
		
		return avg_cel, avg_dist_cel, avg_loss, avg_seq_sim

	def add_losses(self, cel, dist_cel, full_loss, matches, valid_toks=1):
		self.cel.append(cel)
		self.dist_cel.append(dist_cel)
		self.full_loss.append(full_loss)
		self.matches.append(matches)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.cel.extend(other.cel)
		self.dist_cel.extend(other.dist_cel)
		self.full_loss.extend(other.full_loss)
		self.matches.extend(other.matches)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.cel = []
		self.dist_cel = []
		self.full_loss = []
		self.matches = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.full_loss[-1]

	def get_last_match(self):
		return self.matches[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.cel = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.cel]
		self.dist_cel = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.dist_cel]
		self.full_loss = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.full_loss]
		self.matches = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches]

	def __len__(self):
		return len(self.cel)

class VAELosses():
	'''
	class to store losses
	'''
	def __init__(self): 

		# saved for logging
		self.kl_div = [] # kl div to match gaussian prior
		self.reconstruction = [] # reconstruction of the wf

		self.kl_div_no_aa = []
		self.reconstruction_no_aa = []

		# actual losses
		self.all_losses = [] # full scaled loss, contains kldiv, reconstruction, and cel

		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token per cha

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		valid_toks = self.valid_toks.item()
		avg_kl_div = sum(kl_div.item() for kl_div in self.kl_div if kl_div) / valid_toks
		avg_reconstruction = sum(reconstruction.item() for reconstruction in self.reconstruction if reconstruction) / valid_toks
		avg_kl_div_no_aa = sum(kl_div.item() for kl_div in self.kl_div_no_aa if kl_div) / valid_toks
		avg_reconstruction_no_aa = sum(reconstruction.item() for reconstruction in self.reconstruction_no_aa if reconstruction) / valid_toks
		avg_loss = sum(loss.item() for loss in self.all_losses if loss) / valid_toks
		
		return avg_kl_div, avg_reconstruction, avg_kl_div_no_aa, avg_reconstruction_no_aa, avg_loss

	def add_losses(self, kl_div, reconstruction, kl_div_no_aa, reconstruction_no_aa, full_loss, valid_toks=1):
		self.kl_div.append(kl_div)
		self.reconstruction.append(reconstruction)
		self.kl_div_no_aa.append(kl_div_no_aa)
		self.reconstruction_no_aa.append(reconstruction_no_aa)
		self.all_losses.append(full_loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.kl_div.extend(other.kl_div)
		self.reconstruction.extend(other.reconstruction)
		self.kl_div_no_aa.extend(kl_div_no_aa)
		self.reconstruction_no_aa.extend(reconstruction_no_aa)
		self.all_losses.extend(other.all_losses)
		self.valid_toks += valid_toks

	def clear_losses(self):
		self.kl_div = []
		self.reconstruction = []
		self.kl_div_no_aa = []
		self.reconstruction_no_aa = []
		self.all_losses = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.all_losses[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.kl_div = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.kl_div]
		self.reconstruction = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.reconstruction]
		self.kl_div_no_aa = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.kl_div_no_aa]
		self.reconstruction_no_aa = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.reconstruction_no_aa]
		self.all_losses = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.all_losses]

	def __len__(self):
		return len(self.all_losses)

class DiffusionLosses():
	def __init__(self):
		self.squared_errors = []
		# self.nll = []
		self.total_loss = [] # squared_err + gamma*nll
		self.valid_toks = 0

	def get_avg(self, is_inference=False):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		valid_toks = self.valid_toks.item()
		if is_inference: # store the seq sims in squared errors list, instead of dealing w seperate lists the whole run
			return 100*sum(match.item() for match in self.squared_errors if match) / valid_toks
		else:
			squared_err = sum(loss.item() for loss in self.squared_errors if loss) / valid_toks
			# nll = sum(loss.item() for loss in self.nll if loss) / valid_toks
			total_loss = sum(loss.item() for loss in self.total_loss if loss) / valid_toks

			return squared_err, total_loss
		
	def add_losses(self, squared_error, total_loss, valid_toks=1):
		self.squared_errors.append(squared_error)
		# self.nll.append(nll)
		self.total_loss.append(total_loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.squared_errors.extend(other.squared_errors)
		# self.nll.extend(other.nll)
		self.total_loss.extend(other.total_loss)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.squared_errors = []
		# self.nll = []
		self.total_loss = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.total_loss[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.squared_errors = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.squared_errors]
		# self.nll = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.nll]
		self.total_loss = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.total_loss]

	def __len__(self):
		return len(self.total_loss)

# ----------------------------------------------------------------------------------------------------------------------
# loss functions 

class ExtractionLossFunction(nn.Module):

	def __init__(self, label_smoothing, dist_label_smoothing, beta, bins, min_dist, max_dist):
		super(ExtractionLossFunction, self).__init__()
		self.cel_raw = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=label_smoothing)
		# self.dist_cel = DistogramLoss(bins, min_dist, max_dist, dist_label_smoothing)
		self.beta = beta # weight

	def cel(self, seq_pred, seq_true):
		'''
		seq_pred: Z x N x num_aa (logits)
		seq_true: Z x N (labels, -1 not valid)
		'''

		cel = self.cel_raw(seq_pred.view(-1, seq_pred.size(-1)), seq_true.view(-1)) # Z*N
		
		return cel

	def compute_matches(self, seq_pred, seq_true):
		'''greedy selection, computed seq sim here for simplicity, will do it with other losses later '''
		
		prediction_flat = seq_pred.softmax(dim=2).argmax(dim=2).view(-1) # batch*N
		true_flat = seq_true.view(-1) # batch x N --> batch*N,
		valid_mask = true_flat != -1 # batch*N, 
		matches = ((prediction_flat == true_flat) & (valid_mask)).sum() # 1, 
		
		return matches 

	def forward(self, seq_pred, seq_true, dist_features=None, coords=None, mask=None):
		cel = self.cel(seq_pred, seq_true)
		# dist_cel = self.dist_cel(dist_features, coords, mask) if dist_features is not None else 0.0 # seperate mask than labels, since also predicting dists for non repre chains
		full_loss = cel #+ (self.beta*dist_cel) # divide by number of tokens, since cel is computed on NxN samples, basically sum of avg cel for each other token. this makes cel and dist cel be on same scale 
		matches = self.compute_matches(seq_pred, seq_true)

		return cel, 0.0, full_loss, matches # return all for logging, only full loss used for backprop

class VAELossFunction(nn.Module):

	def __init__(self, beta=1.0, kappa=1.0, midpoint=4000, anneal=True):
		super(VAELossFunction, self).__init__()
		self.beta = beta
		self.kappa = kappa
		self.midpoint = midpoint
		self.anneal = anneal
		self.kl_annealing_step = 0 # for kl annealing

	def kl_div(self, prior_mean_pred, prior_log_var_pred, mask):
		kl_div = -0.5*torch.sum(1 + prior_log_var_pred - prior_mean_pred.pow(2) - torch.exp(prior_log_var_pred), dim=2) # Z x N
		
		return (kl_div*(~mask)).sum() 

	def reconstruction(self, reconstruct_mean_pred, reconstruct_mean_true, mask):
		return ((reconstruct_mean_true - reconstruct_mean_pred).pow(2) * (~mask).unsqueeze(2)).sum()

	def full_loss(self, kl_div, reconstruction):# cel is typically larger than mse and kldiv, so scale it down so vae focuses on wf reconstruction more

		# beta starts small and gradualy increases	
		beta = self.beta if not self.anneal else self.beta/(1+math.exp(-self.kappa*(self.kl_annealing_step-self.midpoint)))
		return ( beta * kl_div) + reconstruction 

	def forward(self, 	prior_mean_pred, prior_log_var_pred,
						reconstruct_mean_pred, 
						prior_mean_pred_no_aa, prior_log_var_pred_no_aa,
						reconstruct_mean_pred_no_aa, 
						reconstruct_mean_true, mask
				):
		kl_div = self.kl_div(prior_mean_pred, prior_log_var_pred, mask)
		reconstruction = self.reconstruction(reconstruct_mean_pred, reconstruct_mean_true, mask)
		kl_div_no_aa = self.kl_div(prior_mean_pred_no_aa, prior_log_var_pred_no_aa, mask)
		reconstruction_no_aa = self.reconstruction(reconstruct_mean_pred_no_aa, reconstruct_mean_true, mask) 
		full_loss = self.full_loss(kl_div, reconstruction) + self.full_loss(kl_div_no_aa, reconstruction_no_aa)

		return kl_div, reconstruction, kl_div_no_aa, reconstruction_no_aa, full_loss # return all for logging, only full loss used for backprop

class DiffusionLossFunction(nn.Module):
	def __init__(self, gamma=1.0): # gamma scales the nll term
		super(DiffusionLossFunction, self).__init__()
		self.gamma = gamma

	def forward(self, noise_pred, noise_true, abar, mask):
		'''sum of squared errors plus an NLL term to evaluate the probability of the estimated x0 under encoders mean and var'''

		squared_err = ((noise_true - noise_pred).pow(2)*(~mask)).sum()
		loss = squared_err #+ self.gamma*nll # testing if nll implicitly improves squared err
		return squared_err, loss

# ----------------------------------------------------------------------------------------------------------------------
