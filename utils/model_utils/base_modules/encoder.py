import torch
import torch.nn as nn
from utils.model_utils.geometric_attn.triton.geometric_attn import geometric_attn
from utils.model_utils.base_modules.base_modules import MLP, adaLN

class Encoder(nn.Module):

	def __init__(self, 	d_model=512, d_hidden=2048, hidden_layers=0, 
						heads=8, min_spread=3, 
						min_rbf=0.001, max_rbf=0.85,
						dropout=0.0,
						use_adaLN=False, d_in_t=512, d_hidden_t=2048, hidden_layers_t=512

					):
		super(Encoder, self).__init__()

		# Self-attention layers
		self.attn = GeoAttention(	d_model=d_model, heads=heads, 
									min_spread=min_spread, min_rbf=min_rbf, max_rbf=max_rbf,
								)

		if use_adaLN:
			self.attn_adaLN = adaLN(d_in=d_in_t, d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t)
			self.ffn_adaLN = adaLN(d_in=d_in_t,  d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t)
			
		self.attn_norm = nn.LayerNorm(d_model)
		self.ffn_norm = nn.LayerNorm(d_model)
		
		self.attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.ffn_dropout = nn.Dropout(dropout)

	# optional kwarg to apply adaLN w/ timestep embedding, defined in diffusion module, does not affect others
	def forward(self, x, coords, key_padding_mask=None, t=None):

		if t is not None: # ada ln
		
			gamma1, beta1, alpha1 = self.attn_adaLN(t)
			gamma2, beta2, alpha2 = self.ffn_adaLN(t)

			x = gamma1*self.attn_norm(x) + beta1

			x2 = self.attn(	x, x, x,
					coords=coords,
					key_padding_mask=key_padding_mask
				)

			x = x + self.attn_dropout(x2)*alpha1

			x = gamma2*self.ffn_norm(x) + beta2

			x = x + self.ffn_dropout(self.ffn(x))*alpha2

			return x

		else: # regular layer norm

			x2 = self.attn(	x, x, x,
					coords=coords,
					key_padding_mask=key_padding_mask
				)
			x = self.attn_norm(x + self.attn_dropout(x2))
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
						min_spread=2.0,
						min_rbf=0.01, max_rbf=0.99, 
					):
		super(GeoAttention, self).__init__()

		self.heads = heads
		self.d_model = d_model

		if self.d_model % self.heads != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.heads})")
		self.d_k = self.d_model // self.heads

		self.spread_weights = nn.Parameter(torch.zeros(heads))
		self.beta_weights = nn.Parameter(torch.zeros(heads))
		self.min_spread = min_spread
		self.min_rbf = min_rbf
		self.max_rbf = max_rbf

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
		return self.min_spread + torch.exp(self.spread_weights)

	def get_betas(self):
		return torch.exp(self.beta_weights)

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

		# perform attention
		out = geometric_attn(Q, K, V, coords, self.get_spreads(), self.get_betas(), mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf)  # batch x heads x N x d_k
			
		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x heads
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x heads --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model
