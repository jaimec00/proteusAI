# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein, based on 
				alpha carbon coordinates
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils.featurization import protein_to_wavefunc
from utils.model_utils.gaussian_attn import attn

# ----------------------------------------------------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and each feature for each Ca corresponds 
	to the superposed wavefunction at a particular wavelength. each wavefunction output creates two features, a real part and 
	an imaginary part. Thus, d_model / 2 wave functions, each with a corresponding wavelength, are generated to create
	d_model features for each Ca 
	need to precompute these for training, but very manageable for inference to compute on the fly, especially with 
	optimized triton kernel
	serve as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions. 
	'''
	def __init__(self, d_model=512, min_wl=3.7, max_wl=50, min_base=2, max_base=100):
		super(SpatialEmbedding, self).__init__()

		self.d_model = d_model
		self.min_wl = min_wl
		self.max_wl = max_wl
		self.base = base

		# compute wavenumbers from wavelengths
		self.wavelengths = self.min_wl + (self.max_wl - self.min_wl) * (torch.logspace(0,1,self.d_model//2, self.base) - 1) / (self.base - 1)
		self.wavenumbers = 2 * torch.pi / wavelengths

	def forward(self, coords, key_padding_mask):

		# convert to wf features if not already precomputed
		wavefunc = protein_to_wavefunc(coords, self.wavenumbers, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		return wavefunc

class MHA(nn.Module):
	'''
	custom MHA module, in order to scale attention weights for each head based 
	on each head's spread in the RBF of PW distances 
	see the imported function (supports fwd and bwd) triton implementation
	'''

	def __init__(self, d_model=512, nhead=8, min_rbf=0.1, max_rbf=0.9, min_spread=1, max_spread=6, base=20, dropoout=0.1):
		super(MHA, self).__init__()

		self.nhead = nhead
		self.d_model = d_model

		if self.d_model % self.nhead != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.nhead})")
		self.d_k = self.d_model // self.nhead

		# weights matrix so each head's spread is a weighted sum of the custom wavelengths
		# self.spread_weights = nn.Parameter(torch.randn([d_model//2, self.nhead])) 

		self.min_rbf = min_rbf
		self.max_rbf = max_rbf
		self.dropout = dropout
		self.min_spread = min_spread
		self.max_spread = max_spread
		self.base = base

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

		spreads = self.min_spread + (self.max_spread - self.min_spread) * (torch.logspace(0,1,self.nhead, self.base, device=Q.device) - 1) / (self.base - 1)

		# perform attention
		dropout = self.dropout if self.training else 0.0
		out = attn(Q, K, V, coords, spreads.to(Q.device), mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf, dropout=dropout)  # batch x nhead x N x d_k

		out = out.permute(0,2,3,1) # batch x N x d_k x nhead
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x nhead --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

class Attention(nn.Module):
	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, min_rbf=0.1, max_rbf=0.9, min_spread=1, max_spread=6, base=20, dropout=0.0):
		super(Attention, self).__init__()

		self.mha = MHA(d_model, nhead, min_rbf=min_rbf, max_rbf=max_rbf, min_spread=min_spread, max_spread=max_spread, base=base, dropout=dropout)
		
		# Separate normalization layers
		self.norm = nn.LayerNorm(d_model)

		# Separate dropout layers
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, coords, key_padding_mask=None):
		
		# multi-head gaussian attention
		o = self.mha(	q, k, v,
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

	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, min_rbf=0.1, max_rbf=0.9, min_spread=1, max_spread=6, base=20, dropout=0.0):
		super(DualCoder, self).__init__()

		# Self-attention layers
		self.wf_self_attn = Attention(d_model, nhead, min_rbf=min_rbf, max_rbf=max_rbf, min_spread=min_spread, max_spread=max_spread, base=base, dropout=dropout)
		self.aa_self_attn = Attention(d_model, nhead, min_rbf=min_rbf, max_rbf=max_rbf, min_spread=min_spread, max_spread=max_spread, base=base, dropout=dropout)
		
		# cross attention layers
		self.wf_cross_attn = Attention(d_model, nhead, min_rbf=min_rbf, max_rbf=max_rbf, min_spread=min_spread, max_spread=max_spread, base=base, dropout=dropout)
		self.aa_cross_attn = Attention(d_model, nhead, min_rbf=min_rbf, max_rbf=max_rbf, min_spread=min_spread, max_spread=max_spread, base=base, dropout=dropout)

		# Feed-forward network
		self.feedforward_norm = nn.LayerNorm(d_model)
		self.feedforward_dropout = nn.Dropout(dropout)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

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
		wf = self.wf_cross_attn(	wf2, aas2, wf2,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)

		#### wf as K cross attention ####
		aas = self.aa_cross_attn(	aas2, wf2, aas2,
									coords=coords,
									key_padding_mask=key_padding_mask,
								)

		# Feed-forward network (with dropout)
		wf2 = self.linear2(self.feedforward_dropout(F.gelu(self.linear1(wf))))
		
		# add (with dropout) and norm
		wf = wf + self.feedforward_dropout(wf2)
		wf = self.feedforward_norm(wf)

		return wf

class proteusAI(nn.Module):
	'''

	proteusAI. 

	converts 3D coords to wavefunction features (wf), and one hot amino acid tensors into aa embeddings. performs masked 
	cross gaussian attention (lots of words) via the ContextModule, with wf as Q and V, and AA embeddings as K. Gaussian weighting is
	done based on pair-wise distance based RBFs. then go through all decoder layers. decoder layer and context module are essentially the same
	just the q, k, and v matrices vary. these are a stack of multi-scale gaussian attention layers, with each head operating at a 
	different scale, by varying the spread of the RBF for that head. after self-attention, do add + norm, FFN, and residual connection. 
	the subspace (d_k) each head operates at is close to the scale of the spread, since each feature in d_k (and d_model, where d_model=d_k*nhead) 
	corresponds to a distinct wavelength (and thus k in Green's function), with low index features corresponding to small wavelengths 
	and large index features to large wavelengths. the final decoder layer output goes through a linear layer that converts the 
	d_model feature space to a 20-d feature space (20 AAs). softmax is applied to get probabilities and 
	sample from the distribution autoregressively (during inference).

	'''
	
	def __init__(self, 	d_model, n_head, decoder_layers, hidden_linear_dim, dropout, 
						min_wl=1, max_wl=50, base=20, min_rbf=0.05, max_rbf=0.99, 
						min_spread=1, max_spread=30, 
						use_probs=False, include_ncaa=False
					):

		super(proteusAI, self).__init__()

		# configuration
		self.d_model = d_model
		self.n_head = n_head
		self.use_probs = use_probs

		# wavefunc
		self.spatial_embedding = SpatialEmbedding(d_model, min_wl, max_wl, base)

		# context
		num_aas = 20 if not include_ncaa else 21
		self.aa_embedding = nn.Linear(num_aas, d_model)

		# decoder
		self.decoders = nn.ModuleList([Decoder(d_model, n_head, hidden_linear_dim, dropout, min_rbf, max_rbf, min_spread, max_spread) for _ in range(decoder_layers)])
		self.encoders = nn.ModuleList([Encoder(d_model, n_head, hidden_linear_dim, dropout, min_rbf, max_rbf, min_spread, max_spread) for _ in range(decoder_layers)])

		# map to aa probs
		self.linear = nn.Linear(d_model, num_aas)


	def get_wavelengths(self, coords, key_padding_mask):

		max_wl = 8.0 #torch.full((coords.size(0), 1), 10.0, device=torch.device)
		base = 20 # hard code base for testing
		min_wl =  1 # for testing

		# # samples wavelengths near the average more frequently than those far away. max wl is twice the avg
		wavelengths = min_wl + (max_wl - min_wl) * ((base ** (torch.arange(0, self.d_model//2, device=coords.device, dtype=torch.float32)/(self.d_model//2))) - 1) / (base - 1)
		if wavelengths.dim() < 2:
			wavelengths = wavelengths.unsqueeze(0).expand(coords.size(0), -1)

		return wavelengths


	def auto_regressive(self, wavefunc, coords, aa_onehot, key_padding_mask=None, temp=0.1):

		# auto regressively sample AAs at most confident position
		for position in range(aa_onehot.size(1)):

			# add context, unless first iteration, in which case context was already added in forward method
			if position > 0:
				wavefunc = self.contextualize(wavefunc, coords, aa_onehot, key_padding_mask)

			# decode the wavefunction
			seq_probs = self.decode(wavefunc, coords, key_padding_mask) # batch x N x 512 --> batch x N X 20 
			
			# convert to probability distribution
			seq_probs = F.softmax(seq_probs, dim=-1) # batch x N x 20

			# select aa at most confident position
			aa_onehot = self.update_most_confident(seq_probs, aa_onehot, key_padding_mask, temp)

			# if all positions predicted, stop
			if torch.all(aa_onehot.any(dim=-1)):
				break	

		return aa_onehot

	def update_most_confident(self, seq_probs, aa_onehot, key_padding_mask=None, temp=0.1):
		
		# mask for predicted positions
		predicted_positions_mask = aa_onehot.any(dim=-1) | key_padding_mask # batch x N

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

	def forward(self, coords, aa_onehot, features=None, key_padding_mask=None, auto_regressive=False, temp=0.1, use_checkpoint=False):
		"""
		Forward pass of the model with optional auto-regressive inference
		"""
		# coords: batch x N x 3 (or batch x N x d_model if self.as_coords is False)
		# aa_onehot: batch x N x 20

		# get sample specific wavelengths via PCA and eigenvalue decomposition
		wavelengths = self.get_wavelengths(coords, key_padding_mask)

		# wave function encoding (replaces positional encoding)
		if features is None:
			wavefunc = self.spatial_embedding(coords, wavelengths, key_padding_mask) # batch x N x 3 --> batch x N x d_model
		else:
			wavefunc = features
			

		# note auto-regression breaks the computational chain, do not implement during training
		# if auto_regressive:
		# 	seq_probs = self.auto_regressive(wavefunc, coords, aa_onehot, key_padding_mask, temp=temp) # batch x N x 20 (returns one-hot tensor)
		# else: 
		wavefunc = self.decode(wavefunc, coords, wavelengths, key_padding_mask) # batch x N x 20 (returns aa probability logits)

		aas = self.aa_embedding(aa_onehot)
		seq_probs = self.encode(wavefunc, aas, coords, wavelengths, key_padding_mask)


		return seq_probs

# ----------------------------------------------------------------------------------------------------------------------