import torch
import torch.nn as nn
from utils.model_utils.attn.flash_attn import flash_attn
from utils.model_utils.attn.geometric_attn import geometric_attn
from utils.model_utils.base_modules.base_modules import MLP, adaLN, StaticLayerNorm

class Encoder(nn.Module):
	'''
	'''
	def __init__(self, 	d_model=1024, d_other=256, heads=8, min_rbf=0.001,
						d_hidden=2048, hidden_layers=0, dropout=0.10, attn_dropout=0.10, bias=True, # make bias config later
					):
		super(Encoder, self).__init__()

		# cross-attention layers
		if bias:
			self.attn = GeoAttention(d_model=d_model, heads=heads, min_rbf=min_rbf, dropout=attn_dropout)
		else:
			self.attn = Attention(d_model=d_model, d_other=d_other, heads=heads, dropout=attn_dropout)

		self.attn_norm = nn.LayerNorm(d_model)
	
		# Feed-forward network
		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.ffn_norm = nn.LayerNorm(d_model)

		# dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, coords, mask=None):

		# attn
		q = self.attn_norm(q + self.dropout(self.attn(q, k, v, coords, mask=mask)))

		# ffn
		q = self.ffn_norm(q + self.dropout(self.ffn(q)))

		return q


class DiTEncoder(nn.Module):
	'''
	encoder used in DiT w/ timestep conditioning via adaLN(Zero)
	'''
	def __init__(self, 	d_model=512, heads=8, 
						d_hidden=2048, hidden_layers=0, dropout=0.0,
						bias=False, min_rbf=0.000,
						d_in_t=512, d_hidden_t=2048, hidden_layers_t=512
					):
		super(DiTEncoder, self).__init__()

		# Self-attention layers
		if bias:
			self.attn = GeoAttention(d_model=d_model, heads=heads, min_rbf=min_rbf)
		else:
			self.attn = Attention(d_model=d_model, d_other=d_model, heads=heads)

		# adaptive layernorm
		self.static_norm = StaticLayerNorm(d_model)
		self.attn_adaLN = adaLN(d_in=d_in_t, d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t, dropout=dropout)
		self.ffn_adaLN = adaLN(d_in=d_in_t,  d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t, dropout=dropout)

		# feed forward network
		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)

		# dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, t, coords, key_padding_mask=None):
	
		# get the ada ln
		gamma1, beta1, alpha1 = self.attn_adaLN(t)
		gamma2, beta2, alpha2 = self.ffn_adaLN(t)

		# attn
		x2 = gamma1*self.static_norm(x) + beta1
		x2 = self.attn(x2, x2, x2, coords, mask=key_padding_mask)
		x = x + self.dropout(x2*alpha1)

		# ffn
		x2 = gamma2*self.static_norm(x) + beta2
		x = x + self.dropout(self.ffn(x2)*alpha2)

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
						min_rbf=0.001, max_rbf=1.00, 
						dropout=0.10 # not used
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

	def forward(self, q, k, v, coords, mask=None, flash=True, drop_p=0.1):
		'''
		performs scaled dot-product attention weighted by Gaussian RBFs
		'''

		# make sure shape is compatible
		assert q.shape == k.shape == v.shape
		assert q.dim() == 3
		batch, N, d_model = q.shape
		assert d_model == self.d_model
		drop_p = drop_p if self.training else 0.0

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k

		# perform attention
		if flash:
			out = geometric_attn(Q, K, V, coords, self.get_spreads(), self.get_betas(), mask=mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf)  # batch x heads x N x d_k
		else: # so i can do dropout on small single chain data set, had dropout in the triton kernel before, but not convinced it worked properly
			S = torch.matmul(Q, K.transpose(2,3)) / (Q.size(3)**0.5)
			D = torch.sqrt(((coords.unsqueeze(2) - coords.unsqueeze(1))**2).sum(dim=3))[:, None, :, :]
			R = torch.exp(-(D**2)/(2*(self.get_spreads().view(1,-1,1,1)**2)))
			R_norm = 2*((R - self.min_rbf) / (self.max_rbf - self.min_rbf)) - 1
			SR = torch.where(mask[:, None, None, :], float("-inf"), S + torch.abs(S)*self.get_betas().view(1,-1,1,1)*R_norm)
			SR = torch.where(torch.rand_like(SR) < drop_p, float("-inf"), SR / (1-drop_p) ) # dropout
			P = torch.softmax(SR, dim=3)
			out = torch.matmul(P, V)

		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x heads
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x heads --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

class Attention(nn.Module):
	'''
	works for self or cross attention
	'''

	def __init__(self, d_model=1024, d_other=256, heads=8, dropout=0.10):
		super(Attention, self).__init__()

		if d_model % heads != 0: raise ValueError(f"number of dimensions ({d_model}) must be divisible by number of attention heads ({heads})")
		d_k = d_model // heads # compute dk based on dmodel, dother just has more/less projections to match the numebr of heads

		# attention dropout
		self.dropout = dropout

		# QKV projection weight and bias matrices

		# init xavier distribution
		xavier_scale = (6/(d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_model, d_k) * (2*xavier_scale)) # H x d_model x d_k
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_other, d_k) * (2*xavier_scale)) # H x d_model x d_k
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(heads, d_other, d_k) * (2*xavier_scale)) # H x d_model x d_k

		self.q_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x d_k
		self.k_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x d_k
		self.v_bias = nn.Parameter(torch.zeros(heads, d_k)) # H x d_k

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, q, k, v, coords=None, mask=None):
		'''
		performs scaled dot-product attention 
		'''

		# make sure shape is compatible
		assert k.shape == v.shape
		assert q.dim() == k.dim() == 3, f"expected q and k to have the same number of dimensions, but got {q.dim()=} and {k.dim()=}"
		batch, N, d_model = q.shape

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # Z x 1 x N x d_model @ 1 x H x d_model x d_k --> Z x H x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # Z x 1 x N x d_other @ 1 x H x d_other x d_k --> Z x H x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # Z x 1 x N x d_other @ 1 x H x d_other x d_k --> Z x H x N x d_k

		# perform attention
		dropout = self.dropout if self.training else 0.0
		out = flash_attn(Q, K, V, mask=mask, dropout_p=dropout)

		# cat heads and project through final linear layer
		out = out.permute(0,2,3,1) # Z x N x d_k x heads
		out = out.reshape(batch, N, d_model) # batch x N x d_k x heads --> batch x N x d_model
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model
