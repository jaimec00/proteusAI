import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MLP(nn.Module):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, d_in=512, d_out=512, d_hidden=1024, hidden_layers=0, dropout=0.1, act="gelu"):
		super(MLP, self).__init__()

		self.in_proj = nn.Linear(d_in, d_hidden)
		self.hidden_proj = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for layer in range(hidden_layers)])
		self.out_proj = nn.Linear(d_hidden, d_out)

		self.in_dropout = nn.Dropout(dropout)
		self.hidden_dropout = nn.ModuleList([nn.Dropout(dropout) for layer in range(hidden_layers)])

		if act == "gelu":
			self.act = F.gelu 
		elif act == "silu":
			self.act = F.silu
		elif act == "relu":
			self.act = F.relu
		else:
			self.act = lambda x: x # no activation if none of the above 

		self.init_linears()

	def init_linears(self):

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		init_xavier(self.out_proj)  # Xavier for output layer

	def forward(self, x):
		x = self.in_dropout(F.gelu(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(self.act(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x

class CrossFeatureNorm(nn.Module):
	'''
	normalizes each feature independantly across the sequence. it is independant of batches (not batch norm)
	this is helpful because each feature for a given token (Ca atom) is the output of that token for the global 
	superposed wavefunction at a particular wavelength. thus, each feature in a given token is only relevant
	RELATIVE to the CORRESPONDING features of all other tokens in the sequence. 
	This essentially normalizes each wavefunction's (psi_k) output to have mean of 0 and std of 1. 
	Note that this normalizes the real part and the imaginary part independantly 
	the resulting features are then scaled by 1/sqrt(d_model), so that the variance of the whole wf is 1
	'''
	def __init__(self, d_model):
		super(CrossFeatureNorm, self).__init__()

	def forward(self, x, mask=None):

		batch, N, d_model = x.shape

		mask = mask if mask is not None else torch.ones(batch, N, device=x.device, dtype=torch.bool) # Z x N
		valid = mask.sum(dim=1, keepdim=True).unsqueeze(2).clamp(min=1) # Z x 1 x 1
		mean = (x*mask.unsqueeze(2)).sum(dim=1, keepdim=True) / valid # Z x 1 x D
		x = x - mean # Z x N x D
		std = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)/valid) # Z x 1 x D
		x = x/std # Z x N x D
		# x = x/(d_model**0.5) # Z x N x D

		return x

class StaticLayerNorm(nn.Module):
	'''just normalizes each token to have a mean of 0 and var of 1, no scaling and shifting'''
	def __init__(self, d_model):
		super(StaticLayerNorm, self).__init__()
		self.d_model = d_model
	def forward(self, x):
		centered = x - x.mean(dim=2, keepdim=True) 
		std = centered.std(dim=2, keepdim=True)
		std = std.masked_fill(std==0, 1)
		return centered / std

class FiLM(nn.Module):
	def __init__(self, d_model=512, d_hidden=1024, hidden_layers=0, dropout=0.1):
		super(FiLM, self).__init__()

		# single mlp that outputs gamma and beta, manually split in fwd
		self.gamma_beta = MLP(d_in=d_model, d_out=2*d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)

	def forward(self, e_t, x): # assumes e_t is Z x 1 x d_model
		gamma_beta = self.gamma_beta(e_t)
		gamma, beta = torch.split(gamma_beta, dim=-1, split_size_or_sections=gamma_beta.shape[-1] // 2)
		return gamma*x + beta

class adaLN(nn.Module):
	'''adaptive layer norm to perform affine transformation conditioned on timestep'''
	def __init__(self, d_in=512, d_model=512, d_hidden=1024, hidden_layers=0, dropout=0.0):
		super(adaLN, self).__init__()
		self.gamma_beta_alpha = MLP(d_in=d_in, d_out=3*d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout, act="silu")

	def forward(self, e_t):

		gamma_beta_alpha = self.gamma_beta_alpha(e_t)
		gamma, beta, alpha = torch.chunk(gamma_beta_alpha, chunks=3, dim=-1)
		return gamma, beta, alpha

# initializations for linear layers
def init_orthogonal(m):
	if isinstance(m, nn.Linear):
		init.orthogonal_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_kaiming(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, nonlinearity='relu')
		if m.bias is not None:
			init.zeros_(m.bias)
def init_xavier(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
