import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import math
from data.constants import canonical_aas

# ----------------------------------------------------------------------------------------------------------------------
# losses 

class TrainingRunLosses():

	def __init__(self, label_smoothing=0.0):

		self.loss_function = LossFunction(label_smoothing)

		self.train = Losses()
		self.val = Losses()
		self.test = Losses()
		self.tmp = Losses()

	def clear_tmp_losses(self):
		self.tmp.clear_losses()

	def to_numpy(self):
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

class Losses():
	'''
	class to store losses
	'''
	def __init__(self): 

		# saved for logging
		self.cel = [] # cel for aa prediction from wf
		self.matches1 = [] # number of matches from greedy selection of aa, just for logging
		self.matches3 = [] 
		self.matches5 = [] 
		self.probs = []

		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token per cha

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		valid_toks = self.valid_toks
		avg_cel = sum(cel.item() for cel in self.cel if cel) / valid_toks
		avg_seq_sim = 100*sum(match.item() for match in self.matches1 if match) / valid_toks
		avg_seq_sim3 = 100*sum(match.item() for match in self.matches3 if match) / valid_toks
		avg_seq_sim5 = 100*sum(match.item() for match in self.matches5 if match) / valid_toks
		avg_probs = 100*sum(prob.item() for prob in self.probs if prob) / valid_toks
		
		return avg_cel, avg_seq_sim, avg_seq_sim3, avg_seq_sim5, avg_probs

	def add_losses(self, cel, matches1, matches3, matches5, probs, valid_toks=1):
		self.cel.append(cel)
		self.matches1.append(matches1)
		self.matches3.append(matches3)
		self.matches5.append(matches5)
		self.probs.append(probs)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.cel.extend(other.cel)
		self.matches1.extend(other.matches1)
		self.matches3.extend(other.matches3)
		self.matches5.extend(other.matches5)
		self.probs.extend(other.probs)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.cel = []
		self.matches1 = []
		self.matches3 = []
		self.matches5 = []
		self.probs = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.cel[-1]

	def get_last_match(self):
		return self.matches1[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.cel = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.cel]
		self.matches1 = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches1]
		self.matches3 = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches3]
		self.matches5 = [match.detach().to("cpu").numpy() if isinstance(match, torch.Tensor) else np.array([match]) for match in self.matches5]
		self.probs = [prob.detach().to("cpu").numpy() if isinstance(prob, torch.Tensor) else np.array([prob]) for prob in self.probs]

	def __len__(self):
		return len(self.cel)

# ----------------------------------------------------------------------------------------------------------------------
# loss functions 

class LossFunction(nn.Module):

	def __init__(self, label_smoothing):
		super(LossFunction, self).__init__()
		self.cel_raw = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=label_smoothing)

	def cel(self, seq_pred, seq_true):
		'''
		seq_pred: Z x N x num_aa (logits)
		seq_true: Z x N (labels, -1 not valid)
		'''

		cel = self.cel_raw(seq_pred.view(-1, seq_pred.size(-1)), seq_true.view(-1)) # Z*N
		# cel = self.cel_raw(seq_pred, seq_true)
		return cel

	def compute_matches(self, seq_pred, seq_true):
		'''
		greedy selection, computed seq sim here for simplicity, will do it with other losses later 
		also computes top3 and top5 accuracy
		'''
		
		# dont need softmax for topk analysis
		top1 = torch.argmax(seq_pred, dim=2).view(-1) # Z*N,
		top3 = torch.topk(seq_pred, 3, 2, largest=True, sorted=False).indices.view(-1, 3) # Z*N x 3
		top5 = torch.topk(seq_pred, 5, 2, largest=True, sorted=False).indices.view(-1, 5) # Z*N x 5

		true_flat = seq_true.view(-1) # Z x N --> Z*N,
		valid_mask = true_flat != -1 # batch*N, 

		matches1 = ((top1 == true_flat) & (valid_mask)).sum() # 1, 
		matches3 = ((top3 == true_flat[:, None]).any(dim=1) & (valid_mask)).sum() # 1, 
		matches5 = ((top5 == true_flat[:, None]).any(dim=1) & (valid_mask)).sum() # 1, 

		return matches1, matches3, matches5

	def compute_probs(self, seq_pred, seq_true):
		valid_mask = seq_true != -1
		probs = torch.softmax(seq_pred, dim=2)
		probs_sum = (valid_mask.unsqueeze(2)*torch.gather(probs, 2, (seq_true*valid_mask).unsqueeze(2))).sum()

		return probs_sum

	def forward(self, seq_pred, seq_true, inference=False):

		if inference: # seq pred is Z x N labels tensor of predictions, convert to one hot for simplicity
			no_seq = seq_pred == -1
			seq_pred = torch.nn.functional.one_hot(torch.where(no_seq, 0, seq_pred), num_classes=len(canonical_aas)).to(torch.float32) # Z x N x canonical_aas

		cel = self.cel(seq_pred, seq_true)
		matches1, matches3, matches5 = self.compute_matches(seq_pred, seq_true)
		probs = self.compute_probs(seq_pred, seq_true)

		return cel, matches1, matches3, matches5, probs # return all for logging, only full loss used for backprop
