# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein, based on 
				alpha carbon coordinates. uses wave function embedding and geometric attention
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils.wf_embedding import wf_embedding
from utils.model_utils.geometric_attn import geometric_attn

# ----------------------------------------------------------------------------------------------------------------------

class MLP(nn.Module):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, d_in=512, d_out=512, d_hidden=1024, hidden_layers=0, dropout=0.1):
		super(MLP, self).__init__()

		self.in_proj = nn.Linear(d_in, d_hidden)
		self.hidden_proj = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for layer in range(hidden_layers)])
		self.out_proj = nn.Linear(d_hidden, d_out)

		self.in_dropout = nn.Dropout(dropout)
		self.hidden_dropout = nn.ModuleList([nn.Dropout(dropout) for layer in range(hidden_layers)])

	def forward(self, x):
		x = self.in_dropout(F.gelu(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(F.gelu(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x

class AminoAcidEmbedding(nn.Module):
	'''
	simple mlp w/ layernorm
	'''
	def __init__(self, num_aas, d_model, d_hidden_aa, hidden_layers_aa, dropout):
		super(AminoAcidEmbedding, self).__init__()
		
		self.ffn = MLP(num_aas, d_model, d_hidden_aa, hidden_layers_aa, dropout)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, aa):
		return self.norm(self.ffn(aa))

class WavefunctionEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and each feature for each Ca corresponds 
	to the superposed wavefunction at a particular wavelength. each wavefunction output creates two features, a real part and 
	an imaginary part. Thus, d_model / 2 wave functions, each with a corresponding wavelength, are generated to create
	d_model features for each Ca 
	serve as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions. 
	also includes mlp after the embedding layer
	'''
	def __init__(self, d_model=512, min_wl=3.7, max_wl=20, base=20, d_hidden=1024, hidden_layers=0, dropout=0.1):
		super(WavefunctionEmbedding, self).__init__()

		# compute wavenumbers from wavelengths
		self.register_buffer("wavelengths", min_wl + (max_wl - min_wl) * (torch.logspace(0,1,d_model//2, base) - 1) / (base - 1))

		# compute the values to initialize the weights, so that post softmax, the diagonals have the most weight (close to 0.67 for d_model=512)
		# and the farther a column index is from the diagonal, the less weight it receives (farthest is about 5e-6, for d_model=512)
		idxs = torch.arange(0, d_model//2) # just the index of each wavenumber
		diag_idx = idxs.unsqueeze(1).expand(-1, d_model//2) # the index of the corresponding diagonal for each row (same for all columns)
		col_idx = idxs.unsqueeze(0).expand(d_model//2, -1)# the index of each column, same for each row
		dist = (diag_idx - col_idx).abs() # how far an index is from the diagonal of its corresponding row
		dist_pct = dist / (d_model//2) # as a percentage
		inv_dist_pct = 1 - (dist_pct**(1/d_model)) # invert by subtracting pct from 1, also take the d_model root to emphasize the diagonals more
		log_inv = torch.log(inv_dist_pct) # take the log to prepare it for softmax
		self.wavelength_weights = nn.Parameter(log_inv)# initialize the learnable weight matrix
		# note that will be doing a weighted sum of wavelengths, not wavenumbers, compute the wavenumbers after computing weighted sum 
		# of wavelengths, to ensure the wavelengths are always within min wavelength and max wavelength		

		self.mlp = MLP(d_model, d_model, d_hidden, hidden_layers, dropout)
		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, coords, key_padding_mask):

		# each feature index will be a weighted sum of the allowed wavelengths, it is initialized to be almost
		# an identity after softmax so that at the start, each index is basically just taking into account the
		# wavelength of its corresponding index
		wavelength_weights = torch.softmax(self.wavelength_weights, dim=1)
		wavelengths = torch.matmul(wavelength_weights, self.wavelengths.to().unsqueeze(1)).squeeze(1) # d_model//2 x d_model//2 @ d_model//2 x 1 -> d_model//2
		wavenumbers = 2 * torch.pi / wavelengths

		# convert to wf features if not already precomputed
		wf = wf_embedding(coords, wavenumbers, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		# pass through ffn
		wf2 = self.mlp(wf)

		# add and norm w dropout
		wf = wf + self.dropout(wf2)
		wf = self.norm(wf)

		return wf

class GeoAttention(nn.Module):
	'''
	Geometric Attention (w/ Flash Attention 2 implementation)
	custom MHA module, in order to scale attention weights for each head based 
	on each head's spread in the RBF of PW distances 
	see the imported function (supports fwd and bwd) triton implementation
	'''

	def __init__(self, d_model=512, nhead=8, min_spread=1, max_spread=6, base=20, min_rbf=0.1, max_rbf=0.9, dropout=0.1):
		super(GeoAttention, self).__init__()

		self.nhead = nhead
		self.d_model = d_model

		if self.d_model % self.nhead != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.nhead})")
		self.d_k = self.d_model // self.nhead

		self.min_rbf = min_rbf
		self.max_rbf = max_rbf
		self.dropout = dropout

		# define spreads and spread weights matrix so each head's spread is a weighted sum of the allowed spreads
		self.register_buffer("spreads", min_spread + (max_spread - min_spread) * (torch.logspace(0,1,nhead, base) - 1) / (base - 1))
		self.spread_weights = nn.Parameter(torch.randn([self.nhead, self.nhead])) 

		self.q_proj = nn.Parameter(torch.randn(self.nhead, self.d_model, self.d_k)) # nhead x d_model x d_k
		self.k_proj = nn.Parameter(torch.randn(self.nhead, self.d_model, self.d_k)) # nhead x d_model x d_k
		self.v_proj = nn.Parameter(torch.randn(self.nhead, self.d_model, self.d_k)) # nhead x d_model x d_k

		self.q_layernorm = nn.LayerNorm(self.d_k)
		self.k_layernorm = nn.LayerNorm(self.d_k)
		self.v_layernorm = nn.LayerNorm(self.d_k)

		self.out_proj = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, coords, key_padding_mask=None):
		'''
		performs scaled dot-product attention weighted by Gaussian RBFs
		'''

		# make sure shape is compatible
		assert (q.shape == k.shape) and (q.shape == v.shape)
		batch, N, d_model = q.shape
		assert d_model == self.d_model

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) # batch x nhead x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) # batch x nhead x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) # batch x nhead x N x d_k

		# apply layer norm after projection to ensure numerical stability
		Q = self.q_layernorm(Q)
		K = self.k_layernorm(K)
		V = self.v_layernorm(V)

		# define dropout
		dropout = self.dropout if self.training else 0.0
		
		# get spread for each head, which is a learnable weighted sum of the allowed spreads
		spread_weights = torch.softmax(self.spread_weights, dim=1) 
		spreads = torch.matmul(spread_weights, self.spreads.unsqueeze(1)).squeeze(1) # nhead x nhead @ nhead x 1 -> nhead
		
		# perform attention
		out = geometric_attn(Q, K, V, coords, spreads, mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf, dropout=dropout)  # batch x nhead x N x d_k

		out = out.permute(0,2,3,1) # batch x N x d_k x nhead
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x nhead --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

class GeoAttention_Unit(nn.Module):
	def __init__(self, d_model=512, nhead=8,  min_spread=1, max_spread=6, base=20, min_rbf=0.1, max_rbf=0.9, dropout=0.0):
		super(GeoAttention_Unit, self).__init__()
		'''
		utility class that combines attention and add+norm w/ dropout
		'''
		self.attn = GeoAttention(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, min_rbf=min_rbf, max_rbf=max_rbf, dropout=dropout)
		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, coords, key_padding_mask=None):
		
		# multi-head gaussian attention
		o = self.attn(	q, k, v,
						coords=coords,
						key_padding_mask=key_padding_mask,
					)

		# residual connection with dropout
		o = v + self.dropout(o)

		# norm
		o = self.norm(o)

		return o

class DualCoder(nn.Module):
	'''
	interleaved cross-attention module, structure queries sequence to update self, sequence queries structure to update itself
	'''

	def __init__(self, d_model=512, d_hidden=1024, hidden_layers=0, nhead=8, min_spread=1, max_spread=6, base=20, min_rbf=0.1, max_rbf=0.9, dropout=0.0):
		super(DualCoder, self).__init__()

		# Self-attention layers
		self.wf_self_attn = GeoAttention_Unit(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, min_rbf=min_rbf, max_rbf=max_rbf, dropout=dropout)
		self.aa_self_attn = GeoAttention_Unit(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, min_rbf=min_rbf, max_rbf=max_rbf, dropout=dropout)
		
		# cross attention layers
		self.wf_cross_attn = GeoAttention_Unit(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, min_rbf=min_rbf, max_rbf=max_rbf, dropout=dropout)
		self.aa_cross_attn = GeoAttention_Unit(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, min_rbf=min_rbf, max_rbf=max_rbf, dropout=dropout)

		# Feed-forward network
		self.wf_ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.aa_ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)

		self.wf_ffn_norm = nn.LayerNorm(d_model)
		self.wf_ffn_dropout = nn.Dropout(dropout)

		self.aa_ffn_norm = nn.LayerNorm(d_model)
		self.aa_ffn_dropout = nn.Dropout(dropout)

	def forward(self, wf, aas, coords, key_padding_mask=None):

		#### wf self attention ####
		wf2 = self.wf_self_attn(	wf, wf, wf,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)
		
		#### aa self attention ####
		aas2 = self.aa_self_attn(	aas, aas, aas,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)

		#### aa as K cross attention ####
		wf = self.wf_cross_attn(	aas2, aas2, wf2,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)

		#### wf as K cross attention ####
		aas = self.aa_cross_attn(	wf2, wf2, aas2,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)

		# Feed-forward network for wavefunction
		wf2 = self.wf_ffn(wf)
		wf = wf + self.wf_ffn_dropout(wf2)
		wf = self.wf_ffn_norm(wf)

		# Feed-forward network for AAs
		aas2 = self.aa_ffn(aas)
		aas = aas + self.aa_ffn_dropout(aas2)
		aas = self.aa_ffn_norm(aas)
		
		return wf, aas

class proteusAI(nn.Module):
	'''

	proteusAI.

	'''
	
	def __init__(self, 	d_model=512, # model dimension

						# wf embedding + wf mlp
						min_wl=3.7, max_wl=20, base_wl=20, 
						d_hidden_wl=1024, hidden_layers_wl=0, 

						# aa mlp
						d_hidden_aa=1024, hidden_layers_aa=0,

						# geometric attn + ffn
						dualcoder_layers=4,
						n_head=4,
						min_spread=3.7, max_spread=7, base_spreads=20, 
						min_rbf=0.05, max_rbf=0.99, 
						d_hidden_attn=1024, hidden_layers_attn=0,
						
						# dropout
						dropout=0.00,

						# include non-canonical AAs
						include_ncaa=False
					):

		super(proteusAI, self).__init__()

		# wavefunc
		self.wf_embedding = WavefunctionEmbedding(d_model, min_wl, max_wl, base_wl, d_hidden_wl, hidden_layers_wl, dropout)

		# amino acids
		self.aa_embedding = AminoAcidEmbedding(21, d_model, d_hidden_aa, hidden_layers_aa, dropout)

		# dual coders
		self.dual_coders = nn.ModuleList([DualCoder(d_model, d_hidden_attn, hidden_layers_attn, n_head, min_spread, max_spread, base_spreads, min_rbf, max_rbf, dropout) for _ in range(dualcoder_layers)])

		# map to aa probs
		self.out_proj = nn.Linear(d_model, 20)

	def dualcode(self, wf, aas, coords, key_padding_mask):

		for dual_coder in self.dual_coders:
			wf, aas = dual_coder(wf, aas, coords, key_padding_mask)

		# note that wf is the output, since the original wf is more 
		# representative of the unique protein than the initial aa probabilites
		seq_probs = self.out_proj(wf)

		return seq_probs

	def auto_regressive(self, wf, coords, aas, key_padding_mask=None, temp=0.1):

		# extract the already fixed positions
		aa_onehot = torch.where((aas == 1).any(dim=2, keepdim=True), aas, 0)

		for position in range(aas.size(1)):
			
			aa_embedded = self.aa_embedding(aas)

			# decode the wavefunction
			seq_probs = self.dualcode(wf, aa_embedded, coords, key_padding_mask) # batch x N x 512 --> batch x N X 20 
			
			# convert to probability distribution
			seq_probs = F.softmax(seq_probs, dim=2) # batch x N x 20

			# select aa at most confident position
			aas, aa_onehot = self.update_most_confident(seq_probs, aa_onehot, key_padding_mask, temp)

			# if all positions predicted, stop
			if torch.all(aa_onehot.any(dim=2) | key_padding_mask):
				break

		return aa_onehot

	def update_most_confident(self, seq_probs, aa_onehot, key_padding_mask=None, temp=0.1):
		
		# mask for predicted positions
		predicted_positions_mask = (aa_onehot==1).any(dim=2) | key_padding_mask # batch x N

		# get the most confident position in each batch by calculating entropy
		entropy = -torch.sum(seq_probs * torch.log(seq_probs + 1e-5), dim=-1) # batch x N

		# set predicted and non-valid positions to inf to exclude from argmin
		entropy = entropy.masked_fill(predicted_positions_mask, float("inf")) # batch x N

		# get most confident and the index
		most_confident = torch.argmin(entropy, dim=-1) # batch,
		most_confident_idx = most_confident.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_probs.size(-1)) # batch x 1 x 20

		# get the probability distributions of the most confident positions
		prob_distributions = torch.gather(seq_probs, dim=1, index=most_confident_idx).squeeze(1)  # batch x 20
		
		# Apply temperature scaling to most confident positions
		scaled_logits = torch.log(prob_distributions + 1e-5) / temp # batch x 20
		scaled_probs = F.softmax(scaled_logits, dim=-1)  # batch x 20
		
		# Sample an amino acid based on the scaled probabilities for each sequence
		sampled_aa = torch.multinomial(scaled_probs, num_samples=1).squeeze(-1)  # batch,

		# Create a one-hot encoding of the sampled amino acids to update the `seq_probs` tensor
		one_hot_samples = F.one_hot(sampled_aa, num_classes=seq_probs.size(-1)).float() # batch x 20

		# update filled_positions with most confident positions' sampled aa's 
		aa_onehot_temp = aa_onehot.scatter(1, most_confident_idx, one_hot_samples.unsqueeze(1))
		aa_onehot = torch.where((predicted_positions_mask.unsqueeze(-1).expand(-1, -1, aa_onehot.size(-1))), aa_onehot, aa_onehot_temp)
		seq_probs = torch.where((aa_onehot==1).any(dim=2, keepdim=True), aa_onehot, seq_probs)

		return seq_probs, aa_onehot

	def forward(self, coords, aas, key_padding_mask=None, auto_regressive=False, temp=0.1):
		"""
		Forward pass of the model with optional auto-regressive inference
		"""
		# coords: batch x N x 3 (or batch x N x d_model if self.as_coords is False)
		# aa_onehot: batch x N x 20

		# wave function embedding (replaces positional encoding)
		wf = self.wf_embedding(coords, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		# initial wf embedding is the same, so dont recompute each time in auto regressive
		if auto_regressive:
			return self.auto_regressive(wf, coords, aas, key_padding_mask, temp)

		# simple mlp to get aas to target feature space
		aas = self.aa_embedding(aas)

		# dual coder to communicate structure and sequence
		seq_probs = self.dualcode(wf, aas, coords, key_padding_mask) # batch x N x d_model (returns aa probability logits)

		return seq_probs

# ----------------------------------------------------------------------------------------------------------------------