import torch
import torch.nn as nn
from utils.model_utils.geometric_attn.triton.geometric_attn import geometric_attn
from utils.model_utils.base_modules.base_modules import MLP

class Encoder(nn.Module):

	def __init__(self, 	d_model=512, d_hidden=2048, hidden_layers=0, 
						heads=8, min_spread=3, max_spread=15, base_spread=15, num_spread=8, 
						min_rbf=0.001, max_rbf=0.85, beta=2.0, learnable_spreads=True,
						dropout=0.0, attn_dropout=0.0
					):
		super(Encoder, self).__init__()

		# Self-attention layers
		self.attn = GeoAttention(	d_model=d_model, heads=heads, 
									min_spread=min_spread, max_spread=max_spread, base_spread=base_spread, num_spread=num_spread, 
									min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, 
									dropout=attn_dropout
								)
		self.attn_norm = nn.LayerNorm(d_model)
		self.attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.ffn_norm = nn.LayerNorm(d_model)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(self, x, coords, t=None, key_padding_mask=None): # use x bc applied to structure and seq

		x2 = self.attn(	x, x, x,
						coords=coords,
						key_padding_mask=key_padding_mask
					)

		x = self.attn_norm(x + self.attn_dropout(x2))

		# Feed-forward network for wavefunction
		x = self.ffn_norm(x + self.ffn_dropout(self.ffn(x)))

		return x

class GeoAttention(nn.Module):
	'''
	Geometric Attention (w/ Flash Attention 2 implementation)
	custom MHA module, in order to scale attention weights for each head based 
	on each head's spread in the RBF of PW distances 
	see the imported function (supports fwd and bwd) triton implementation
	note that if minrbf and maxrbf are set, this is essentially sparse attention, 
	so i recommend to do higher dropout on other layers and no dropout here
	this essentially noises the inputs to the geo attention module, rather than directly doing dropout on attention weights
	'''

	def __init__(self, 	d_model=512, 
						heads=8, 
						min_spread=1, max_spread=6, base_spread=20, num_spread=8, 
						min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, 
						dropout=0.1
					):
		super(GeoAttention, self).__init__()

		self.heads = heads
		self.d_model = d_model

		if self.d_model % self.heads != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.heads})")
		self.d_k = self.d_model // self.heads

		self.dropout = dropout

		self.learnable_spreads = learnable_spreads
		if self.learnable_spreads:
			# see the comments in wavefunction embedding, same method
			idxs = torch.arange(0, heads) 
			diag_idx = idxs.unsqueeze(1).expand(-1, heads) 
			col_idx = idxs.unsqueeze(0).expand(heads, -1)
			dist = (diag_idx - col_idx).abs() 
			dist_pct = dist / heads
			inv_dist_pct = 1 - (dist_pct**(1/(2*heads))) 
			log_inv = torch.log(inv_dist_pct) 

			# this is for when there are more spreads than heads, i.e. the learnable weights is not a square matrix. 
			# most weight goes to num_spreads//heads first spreads 
			# (i.e. first 4 idxs if num_spreads is 4 times bigger), etc.
			init_spread_weights = log_inv.unsqueeze(2).expand(-1, -1, num_spread//heads).reshape(heads, num_spread)
			self.spread_weights = nn.Parameter(init_spread_weights) # initialize the learnable weight matrix
		else:
			# make sure dont have more spreads than heads if it is not learnable
			num_spread = self.heads

		# define spreads and spread weights matrix so each head's spread is a weighted sum of the allowed spreads
		if base_spread == 1.0: # linear spacing
			self.register_buffer("spreads", min_spread + (max_spread - min_spread) * torch.linspace(0,1,num_spread))
		else: # log spacing
			self.register_buffer("spreads", min_spread + (max_spread - min_spread) * (torch.logspace(0,1,num_spread, base_spread) - 1) / (base_spread - 1))

		self.min_rbf = min_rbf
		self.max_rbf = max_rbf
		self.beta = beta

		# QKV projection weight and bias matrices

		# init xavier distribution
		xavier_scale = (6/(self.d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k

		self.q_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.k_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.v_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def get_spreads(self):
		if self.learnable_spreads:
			# get spread for each head, which is a learnable weighted sum of the allowed spreads
			spread_weights = torch.softmax(self.spread_weights, dim=1)
			spreads = torch.matmul(spread_weights, self.spreads.unsqueeze(1)).squeeze(1) # heads x nspread @ nspread x 1 -> heads
		else:
			spreads = self.spreads

		return spreads

	def forward(self, q, k, v, coords, key_padding_mask=None):
		'''
		performs scaled dot-product attention weighted by Gaussian RBFs
		'''

		# make sure shape is compatible
		assert q.shape == k.shape == v.shape
		assert q.dim() == 3
		batch, N, d_model = q.shape
		assert d_model == self.d_model

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k

		dropout = self.dropout if self.training else 0.0

		# perform attention
		out = geometric_attn(Q, K, V, coords, self.get_spreads(), mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf, beta=self.beta, dropout=dropout)  # batch x heads x N x d_k

		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x heads
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x heads --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model
