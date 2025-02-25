# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein, based on 
				alpha carbon coordinates. uses wave function embedding and geometric attention, 
				and ESM2 pretrained weights for amino acid embedding
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.model_utils.esm2.get_esm_weights import get_esm_weights
from utils.model_utils.wf_embedding.cuda.wf_embedding import wf_embedding
from utils.model_utils.geometric_attn.triton.geometric_attn import geometric_attn

# ----------------------------------------------------------------------------------------------------------------------

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

		self.init_linears()

	def init_linears(self):

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		init_xavier(self.out_proj)  # Xavier for output layer

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
	def __init__(self, num_aas=21, d_model=512, esm2_weights_path="utils/model_utils/esm2/esm2_t33_650M_UR50D.pt", d_hidden_aa=1024, hidden_layers_aa=0, learnable_esm=False, dropout=0.0):
		super(AminoAcidEmbedding, self).__init__()

		self.use_esm = esm2_weights_path != ""

		if self.use_esm: # choose a esm2 model based on d_model and download
			if not esm2_weights_path.endswith(".pt"): # download the chosen model
				esm2_weights = get_esm_weights(esm2_weights_path)
			else: # load from precomputed file, note that this should be created w/ main func of utils/model_utils/esm2/get_esm_weights.py for proper mapping to proteusAI alphabet
				try:
					esm2_weights = torch.load(esm2_weights_path, weights_only=True)
				except FileNotFoundError as e:
					raise e(f"could not find ESM2 weights at {esm2_weights_path}")

			# initialize esm2 weights
			esm2_d_model = esm2_weights["esm2_linear_nobias.weight"].size(1)
			self.esm2_linear_nobias = nn.Linear(in_features=num_aas, out_features=esm2_d_model, bias=False)
			self.esm2_linear_nobias.weight.data = esm2_weights["esm2_linear_nobias.weight"].T

			self.esm2_layernorm = nn.LayerNorm(normalized_shape=esm2_d_model)
			self.esm2_layernorm.weight.data = esm2_weights["esm2_layernorm.weight"]
			self.esm2_layernorm.bias.data = esm2_weights["esm2_layernorm.bias"]

			if not learnable_esm:
				self.esm2_linear_nobias.weight.requires_grad = False
				self.esm2_layernorm.weight.requires_grad = False
				self.esm2_layernorm.bias.requires_grad = False
				
		else:
			esm2_d_model = num_aas

		self.linear = nn.Linear(esm2_d_model, d_model) # to map to d_model
		self.ffn = MLP(d_model, d_model, d_hidden_aa, hidden_layers_aa, dropout)
		self.dropout = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, aas, wf):

		if self.use_esm:
			aas = self.esm2_linear_nobias(aas)
			aas = self.esm2_layernorm(aas)

		aas = self.norm1(self.linear(aas) + wf)

		aas = self.norm2(aas + self.dropout(self.ffn(aas)))

		return aas

class WavefunctionEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and each feature for each Ca corresponds 
	to the superposed wavefunction at a particular wavelength. each wavefunction output creates two features, a real part and 
	an imaginary part. Thus, d_model / 2 wave functions, each with a corresponding wavelength, are generated to create
	d_model features for each Ca 
	serve as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions. 
	also includes mlp after the embedding layer
	actual function implemented in cuda, optimized for h100 (not really, only takes advantage of large shared mem, need a rewrite to use TMA, possibly WGMMA)
	'''
	def __init__(self, d_model=512, min_wl=3.7, max_wl=20, base=20, d_hidden=1024, hidden_layers=0, learnable_wavelengths=False, dropout=0.1):
		super(WavefunctionEmbedding, self).__init__()

		# compute wavelengths
		self.register_buffer("wavelengths", min_wl + (max_wl - min_wl) * (torch.logspace(0,1,d_model//2, base) - 1) / (base - 1))

		self.learnable_wavelengths = learnable_wavelengths
		if self.learnable_wavelengths:
			# compute the values to initialize the weights, so that post softmax, the diagonals have the most weight (close to 0.67 for d_model=512)
			# and the farther a column index is from the diagonal, the less weight it receives (farthest is about 5e-6, for d_model=512)
			# note that will be doing a weighted sum of wavelengths, not wavenumbers, compute the wavenumbers after computing weighted sum 
			# of wavelengths, to ensure the wavelengths are always within min wavelength and max wavelength		
			idxs = torch.arange(0, d_model//2) # just the index of each wavenumber
			diag_idx = idxs.unsqueeze(1).expand(-1, d_model//2) # the index of the corresponding diagonal for each row (same for all columns)
			col_idx = idxs.unsqueeze(0).expand(d_model//2, -1)# the index of each column, same for each row
			dist = (diag_idx - col_idx).abs() # how far an index is from the diagonal of its corresponding row
			dist_pct = dist / (d_model//2) # as a percentage
			inv_dist_pct = 1 - (dist_pct**(1/d_model)) # invert by subtracting pct from 1, also take the d_model root to emphasize the diagonals more
			log_inv = torch.log(inv_dist_pct) # take the log to prepare it for softmax
			self.wavelength_weights = nn.Parameter(log_inv)# initialize the learnable weight matrix
			
		self.ffn = MLP(d_model, d_model, d_hidden, hidden_layers, dropout)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def get_wavelengths(self):
		
		if self.learnable_wavelengths:
			wavelength_weights = torch.softmax(self.wavelength_weights, dim=1)
			wavelengths = torch.matmul(wavelength_weights, self.wavelengths.unsqueeze(1)).squeeze(1) # d_model//2 x d_model//2 @ d_model//2 x 1 -> d_model//2
		else:
			wavelengths = self.wavelengths

		return wavelengths		

	def forward(self, coords, key_padding_mask):

		wavenumbers = 2 * torch.pi / self.get_wavelengths()
		alpha = 1

		wf = wf_embedding(coords, wavenumbers, alpha, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		wf = self.norm1(wf)
		# wf = self.norm2(wf + self.dropout(self.ffn(wf)))

		return wf

class GeoAttention(nn.Module):
	'''
	Geometric Attention (w/ Flash Attention 2 implementation)
	custom MHA module, in order to scale attention weights for each head based 
	on each head's spread in the RBF of PW distances 
	see the imported function (supports fwd and bwd) triton implementation
	'''

	def __init__(self, d_model=512, nhead=8, min_spread=1, max_spread=6, base=20, num_spread=8, min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, dropout=0.1):
		super(GeoAttention, self).__init__()

		self.nhead = nhead
		self.d_model = d_model

		if self.d_model % self.nhead != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.nhead})")
		self.d_k = self.d_model // self.nhead

		self.dropout = dropout

		self.learnable_spreads = learnable_spreads
		if self.learnable_spreads:
			# see the comments in wavefunction embedding, same method
			idxs = torch.arange(0, nhead) 
			diag_idx = idxs.unsqueeze(1).expand(-1, nhead) 
			col_idx = idxs.unsqueeze(0).expand(nhead, -1)
			dist = (diag_idx - col_idx).abs() 
			dist_pct = dist / nhead
			inv_dist_pct = 1 - (dist_pct**(1/(2*nhead))) 
			log_inv = torch.log(inv_dist_pct) 

			# this is for when there are more spreads than heads, i.e. the learnable weights is not a square matrix. 
			# most weight goes to num_spreads//nhead first spreads 
			# (i.e. first 4 idxs if num_spreads is 4 times bigger), etc.
			init_spread_weights = log_inv.unsqueeze(2).expand(-1, -1, num_spread//nhead).reshape(nhead, num_spread)
			self.spread_weights = nn.Parameter(init_spread_weights) # initialize the learnable weight matrix
		else:
			# make sure dont have more spreads than heads if it is not learnable
			num_spread = self.nhead

		# define spreads and spread weights matrix so each head's spread is a weighted sum of the allowed spreads
		self.register_buffer("spreads", min_spread + (max_spread - min_spread) * (torch.logspace(0,1,num_spread, base) - 1) / (base - 1))

		self.min_rbf = min_rbf
		self.max_rbf = max_rbf
		self.beta = beta

		# QKV projection weight and bias matrices

		# init xavier distribution
		xavier_scale = (6/(self.d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.nhead, self.d_model, self.d_k) * (2*xavier_scale)) # nhead x d_model x d_k
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.nhead, self.d_model, self.d_k) * (2*xavier_scale)) # nhead x d_model x d_k
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.nhead, self.d_model, self.d_k) * (2*xavier_scale)) # nhead x d_model x d_k

		self.q_bias = nn.Parameter(torch.zeros(self.nhead, self.d_k)) # nhead x d_k
		self.k_bias = nn.Parameter(torch.zeros(self.nhead, self.d_k)) # nhead x d_k
		self.v_bias = nn.Parameter(torch.zeros(self.nhead, self.d_k)) # nhead x d_k

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def get_spreads(self):
		if self.learnable_spreads:
			# get spread for each head, which is a learnable weighted sum of the allowed spreads
			spread_weights = torch.softmax(self.spread_weights, dim=1)
			spreads = torch.matmul(spread_weights, self.spreads.unsqueeze(1)).squeeze(1) # nhead x nspread @ nspread x 1 -> nhead
		else:
			spreads = self.spreads

		return spreads

	def forward(self, q, k, v, coords, key_padding_mask=None):
		'''
		performs scaled dot-product attention weighted by Gaussian RBFs
		'''

		# make sure shape is compatible
		assert (q.shape == k.shape) and (q.shape == v.shape)
		batch, N, d_model = q.shape
		assert d_model == self.d_model

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k

		dropout = self.dropout if self.training else 0.0

		# perform attention
		out = geometric_attn(Q, K, V, coords, self.get_spreads(), mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf, beta=self.beta, dropout=dropout)  # batch x nhead x N x d_k

		out = out.permute(0,2,3,1) # batch x N x d_k x nhead
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x nhead --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

class Encoder(nn.Module):
	'''
	bidirectional encoder
	'''

	def __init__(self, d_model=512, d_hidden=1024, hidden_layers=0, nhead=8, min_spread=1, max_spread=6, base=20, num_spread=8, min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, dropout=0.0, attn_dropout=0.0):
		super(Encoder, self).__init__()

		# Self-attention layers
		self.attn = GeoAttention(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=False, dropout=attn_dropout)
		self.attn_norm = nn.LayerNorm(d_model)
		self.attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.ffn_norm = nn.LayerNorm(d_model)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(self, aas, coords, key_padding_mask=None):

		aas2 = self.attn(	aas, aas, aas,
							coords=coords,
							key_padding_mask=key_padding_mask
						)

		aas = self.attn_norm(aas + self.attn_dropout(aas2))

		# Feed-forward network for wavefunction
		aas2 = self.ffn(aas)
		aas = aas + self.ffn_dropout(aas2)
		aas = self.ffn_norm(aas)

		return aas

class proteusAI(nn.Module):
	'''

	proteusAI.

	'''
	
	def __init__(self, 	d_model=512, # model dimension

						# wf embedding + wf mlp
						learnable_wavelengths=False,
						min_wl=3.7, max_wl=20, base_wl=20, 
						d_hidden_wl=1024, hidden_layers_wl=0, 

						# aa mlp
						use_aa=True,
						d_hidden_aa=1024, hidden_layers_aa=0, 
						esm2_weights_path="esm2_t12_35M_UR50D", learnable_esm=False,

						# geometric attn + ffn
						encoder_layers=4,
						n_head=4,
						learnable_spreads=False,
						min_spread=3.7, max_spread=7, base_spreads=20, num_spread=4,
						min_rbf=0.01, max_rbf=0.99, beta=2.0,
						d_hidden_attn=1024, hidden_layers_attn=0,
						
						# dropout
						dropout=0.00,
						attn_dropout=0.00,
					):

		super(proteusAI, self).__init__()

		# wavefunc
		self.wf_embedding = WavefunctionEmbedding(d_model, min_wl, max_wl, base_wl, d_hidden_wl, hidden_layers_wl, learnable_wavelengths, dropout)

		# amino acids
		# init aa embedding even if not used, so that state dict loads properly when used without having to use strict=False kwarg
		self.use_aa = use_aa
		self.aa_embedding = AminoAcidEmbedding(21, d_model, esm2_weights_path, d_hidden_aa, hidden_layers_aa, learnable_esm, dropout)

		# encoders
		self.encoders = nn.ModuleList([
										Encoder(	d_model, d_hidden_attn, hidden_layers_attn, 
													n_head, min_spread, max_spread, base_spreads, 
													num_spread, min_rbf, max_rbf, beta, learnable_spreads, 
													dropout, attn_dropout
												) 
										for _ in range(encoder_layers)
									])

		# map to aa probs
		self.out_proj = nn.Linear(d_model, 20)
		init_xavier(self.out_proj)


	def forward(self, coords, aas, key_padding_mask=None, auto_regressive=False, temp=0.1):
		"""
		Forward pass of the model with optional auto-regressive inference
		"""
		# coords: batch x N x 3 or batch x N x 3 x 3 for backbone atoms (N, C, and Ca)
		# aas: batch x N x 21

		# still need to implement properly
		# coords_alpha, coords_beta = get_coords(coords, key_padding_mask, chain_mask)

		# wave function embedding (replaces positional encoding)
		wf = self.wf_embedding(coords, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		# initial wf embedding is the same, so dont recompute each time in auto regressive
		if auto_regressive:
			return self.auto_regressive(wf, coords, aas, key_padding_mask, temp)

		if self.use_aa:
			wf = self.aa_embedding(aas, wf)

		# bidirectional encoder
		seq_probs = self.encode(wf, coords, key_padding_mask) # batch x N x d_model (returns aa probability logits)

		return seq_probs

	def encode(self, aas, coords, key_padding_mask):

		for encoder in self.encoders:
			aas = encoder(aas, coords, key_padding_mask)

		seq_probs = self.out_proj(aas)

		return seq_probs

	# will prob move this outside the model, and just have the input be ca and cb coords directly
	# will import AA sizes so can update the cb magnitudes auto regressively (possibly include pKa based phase shifts too)
	# def get_coords(self, coords, key_padding_mask, chain_idxs): # chain dict
	# 	if coords.dim() == 3: # Ca ony model
	# 		coords_beta = get_Cb_from_Ca(coords, key_padding_mask, chain_idxs)
	# 		coords_alpha = coords
	# 	if coords.dim() == 4: # backbone model
	# 		coords_alpha, coords_beta = get_Cb_from_BB(coords, key_padding_mask, chain_idxs)

	# def get_Cb_from_Ca(self, coords, key_padding_mask, chain_idxs):
	# 	'''
	# 	note, that need to include chain mask and key_padding_mask to do this properly, in progress, same goes for get_Cb_from_BB

	# 	compute beta carbon coords (not absolute, just relative to Ca). not used yet, need to adapt wf_embedding to use this info
	# 	approximates N and C as being on the line connecting adjacent Ca, with ideal bond distances
	# 	uses the same experimentally determined constants as PMPNN to compute linear combination of b1, b2, and b3
	# 	terminal Ca do not have Cb (can't approximate N and C, since no adjacent Ca), so Cb coords (relative to Ca) are (0,0,0) for these
	# 	'''

	# 	logical_n = coords[:, 0:-2, :]
	# 	logical_ca = coords[:, 1:-1, :]
	# 	logical_c = coords[:, 2:, :]

	# 	b1 = ( logical_ca - logical_n)
	# 	b1 = 1.458 * b1 / torch.linalg.norm(b1, dim=2, keepdims=True)
	# 	b2 = (logical_c - logical_ca)
	# 	b2 = 1.525 * b2 / torch.linalg.norm(b2, dim=2, keepdims=True)
	# 	b3 = torch.linalg.cross(b1, b2)

	# 	cb = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

	# 	# the first and last ca are used as inplace cb, so make cb=ca for these
	# 	# no_cb = 
	# 	# cb = torch.cat([, cb, coords[:, -1, :][None, :]], dim=1) 

	# 	return cb

	# def get_Cb_from_BB(self, coords):

	# 	n = coords[:, :, 0, :]
	# 	ca = coords[:, :, 1, :]
	# 	c = coords[:, :, 2, :]
		
	# 	b1 = ca - n
	# 	b2 = c - ca
	# 	b3 = torch.linalg.cross(b1, b2)

	# 	cb = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

	# 	return ca, cb

	def auto_regressive(self, wf, coords, aas, key_padding_mask=None, temp=0.1):

		# extract the already fixed positions. unpredicted positions are the MASK token (idx 20), so extracting
		# all except mask index means unpredicted are all 0
		aa_onehot = aas[:, :, :20]

		for position in range(aas.size(1)):

			# update original aa tensor to include previously predicted positions
			aas = torch.where((aa_onehot==0).all(dim=2, keepdim=True), aas, torch.cat([aa_onehot, torch.zeros(aa_onehot.shape[:2] + (1,), device=aa_onehot.device)], dim=2))

			# embed to feature space
			aa_embedded = self.aa_embedding(aas, wf)

			# decode the wavefunction
			seq_probs = self.encode(aa_embedded, coords, key_padding_mask) # batch x N x 512 --> batch x N X 20 

			# convert to probability distribution
			seq_probs = F.softmax(seq_probs, dim=2) # batch x N x 20

			# select aa at most confident position
			aa_onehot = self.update_most_confident(seq_probs, aa_onehot, key_padding_mask, temp)

			# if all positions predicted, stop
			if torch.all( (~((aa_onehot==0).all(dim=2))) | key_padding_mask):
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

		return aa_onehot

# ----------------------------------------------------------------------------------------------------------------------
