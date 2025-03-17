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
import torch.nn.functional as F

from utils.model_utils.esm2.get_esm_weights import get_esm_weights
from utils.model_utils.wf_embedding.isotropic.cuda.wf_embedding import wf_embedding as wf_embedding_iso
from utils.model_utils.wf_embedding.anisotropic.cuda.wf_embedding import wf_embedding as wf_embedding_aniso
from utils.model_utils.geometric_attn.triton.geometric_attn import geometric_attn
from utils.model_utils.base_modules import MLP, init_orthogonal, init_kaiming, init_xavier

from data.constants import aa_sizes # 20 d tensor containing the approximate size for each AA

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	'''
	proteusAI.
	'''
	
	def __init__(self, 	d_model=512, # model dimension

						learnable_wavelengths=False,
						wf_mag_type=1, anisotropic_wf=False,
						min_wl=3.7, max_wl=20, base_wl=20, 
						d_hidden_wl=1024, hidden_layers_wl=0, 

						struct_encoder_layers=4,
						n_head=4,
						learnable_spreads=False,
						min_spread=3.7, max_spread=7, base_spreads=20, num_spread=4,
						min_rbf=0.01, max_rbf=0.99, beta=2.0,
						d_hidden_attn=1024, hidden_layers_attn=0,
						
						dropout=0.00,
						attn_dropout=0.00,
						wf_dropout=0.00,

						aa_scale = 0.0, # what scale the AA sizes should be, 0.0 means use the original size, other nums >0 make the max size this number
						mask_id="zero" # whether to treat mask as glycine ("zero") or as the mean of all aas ("mean")
				):

		super(proteusAI, self).__init__()

		# inject the aa info into beta carbons
		self.register_buffer("aa_sizes", self.scale_aa_sizes(aa_sizes, aa_scale, mask_id))

		# structure modules
		self.anisotropic_wf = anisotropic_wf # false means use isotropic version, will prob change so that always uses anisotropic
		self.wf_embedding = WavefunctionEmbedding(	d_model, min_wl, max_wl, base_wl, 
													d_hidden_wl, hidden_layers_wl, 
													learnable_wavelengths, wf_mag_type, anisotropic_wf, 
													dropout, wf_dropout
												)

		self.structure_encoders = nn.ModuleList([ # structure encoders
													Encoder(	d_model, d_hidden_attn, hidden_layers_attn, 
																n_head, min_spread, max_spread, base_spreads, 
																num_spread, min_rbf, max_rbf, beta, learnable_spreads, 
																dropout, attn_dropout
															) 
													for _ in range(struct_encoder_layers)
												])

		# map to aa prob logits
		self.out_proj = nn.Linear(d_model, 20)
		init_xavier(self.out_proj)

	def forward(self, coords, aas, chain_idxs, key_padding_mask=None, mask_predict=False, temp=0.1, num_iters=10, remask=True):
		"""
		Forward pass of the model with optional auto-regressive inference
		
		coords: batch x N x 3 for Ca only model or batch x N x 3 x 3 for backbone atoms (N, C, and Ca)
		aas: batch x N x 21
		chain_idxs: list of lists of lists of start/stop idxs of each chain
		"""

		# get the virtual beta carbon coordinates, full backbone model is the same, just more accurate
		coords_alpha, coords_beta = self.get_coords(coords, chain_idxs)

		if mask_predict: # inference
			return self.mask_predict(coords_alpha, coords_beta, aas, key_padding_mask, temp, num_iters, remask)

		# encode the structure via wave function embedding and stack of geometric attention encoders
		wf = self.encode_structure(coords_alpha, coords_beta, aas, key_padding_mask)

		# map to probability logits
		logits = self.out_proj(wf)

		return logits

	def encode_structure(self, coords_alpha, coords_beta, aas, key_padding_mask=None):

		# scale the magnitude of the beta carbon proportionally to the size of the AA (uses avg size if MASK token)
		# testing if this can be done iteratively w/ mask predict inference strat. gets rid of the problem of the model overfitting to
		# sequence info when i have two seperate modalities, this marries them and is physically reasonable
		coords_beta = self.inject_aas(coords_beta, aas)

		# wave function embedding
		wf = self.wf_embedding(coords_alpha, coords_beta, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		# geometric attention encoders
		for encoder in self.structure_encoders:
			wf = encoder(wf, coords_alpha, key_padding_mask)

		return wf

	def mask_predict(self, coords_alpha, coords_beta, aas, key_padding_mask=None, temp=0.1, num_iters=10, remask=True):
		
		# new decoding strat of CMLM paper. constant number of iterations. 
		# predict full seq at each iter, and remask the least confident. 
		# each iter has a smaller percentage of masked tokens, until get a full prediction
		# allows for error correction as context increases. 
		# also have the option to do no remasking, just have the model use full output as input each iter

		# setup params
		batch, N, num_classes = aas.shape
		valid = (~key_padding_mask).sum(dim=1, keepdim=True)
		idxs = torch.arange(N, device=aas.device).unsqueeze(0).expand(batch, N)

		# loop until get prediction
		for i in range(num_iters):

			# run the model
			logits = self.out_proj(self.encode_structure(coords_alpha, coords_beta, aas, key_padding_mask)) # Z x N x C

			# get probs w/ temp scaling
			scaled_logits = logits / temp
			probs = torch.softmax(scaled_logits, dim=2)

			# sample from the probs distribution
			# multinomial doesnt do batched stuff, so do some reshaping
			predictions_idxs = torch.multinomial(probs.view(-1, num_classes-1), num_samples=1).view(batch, N) # Z x N

			# get model's confidence per position on unscaled probs
			entropy_probs = torch.softmax(logits, dim=2)
			entropy = -torch.sum(entropy_probs * torch.log(entropy_probs), dim=2) # Z x N

			# put the mask tokens as low entropy, so they are not included when counting the number of high entropy tokens
			entropy[key_padding_mask] = -float("inf")

			# sort by highest entropy, ie lowest confidence
			sorted_entropy_idxs = torch.argsort(entropy, descending=True, dim=1)

			# sort the predictions according to their entropies
			sorted_predictions = torch.gather(predictions_idxs, dim=1, index=sorted_entropy_idxs)

			# compute the number of tokens to mask for each sample
			pct_mask = (num_iters-(i+1)) / num_iters if remask else 0.0 # testing whether to do no remasking, just do multiple iterations of denoising
			num_mask = torch.floor(valid * pct_mask).int()

			# replace low confidence predictions w/ mask tokens
			sorted_predictions[idxs < num_mask] = num_classes-1

			# reverse the sort to get original prediction order
			reverse_sorted_entropy_idxs = torch.argsort(sorted_entropy_idxs, descending=False, dim=1)
			reverse_sorted_predictions = torch.gather(sorted_predictions, dim=1, index=reverse_sorted_entropy_idxs)

			# create the one hot tensor for the next iter
			aas = torch.nn.functional.one_hot(reverse_sorted_predictions.long(), num_classes=num_classes).float()

		# confidence scores are the entropy of the last iteration
		confidence_scores = entropy # will return this later

		return aas[:, :, :num_classes-1] # get rid of mask token dim, should be all zeros there

	# will import AA sizes so can update the cb magnitudes auto regressively (possibly include pKa based phase shifts too)
	def get_coords(self, coords, chain_idxs): 

		# in both cases, ca and cb coords are batch x N x 3
		if self.anisotropic_wf:
			if coords.dim() == 3: # Ca only model
				coords_alpha, coords_beta = self.get_Cb_from_Ca(coords, chain_idxs)
			elif coords.dim() == 4: # backbone model
				coords_alpha, coords_beta = self.get_Cb_from_BB(coords)
			else:
				raise ValueError(f"invalid input size for coordinates, expected (batch,N,3) for Ca only model or (batch,N,3,3) for backbone model, but got {coords.shape=}")
		
		else: # isotropic version is just the bland ca only model, will probably get rid of the option bc anisotropic is just better
			coords_alpha, coords_beta = coords, None

		return coords_alpha, coords_beta

	def get_Cb_from_Ca(self, coordsA, chain_idxs):
		'''
		compute beta carbon coords (not absolute, just relative to Ca). used for anisotropic wf embedding, in testing
		approximates N and C as being on the line connecting adjacent Ca, with ideal bond distances
		uses the same experimentally determined constants as PMPNN to compute linear combination of b1, b2, and b3

		chain_idxs is a list of lists of lists, like:
			[ 
				[ 
					[sample1_chain1_start, sample1_chain1_stop], 
					[sample1_chain2_start, sample1_chain2_stop] 
				], 
				[
					[sample2_chain1_start, sample2_chain1_stop], 
					[sample2_chain2_start, sample2_chain2_stop] 
				]
			]
		
		flattens the indexes to perform efficient batched computation of virtual Cb coordinates
		'''

		batch, N, space = coordsA.shape

		# default cb position is 0,0,0
		coordsB = torch.zeros_like(coordsA)

		# create flattened lists of idxs
		batch_idxs_flat, start_idxs_flat, end_idxs_flat = [], [], []
		for sample_idx, sample in enumerate(chain_idxs):
			for start_idx, stop_idx in sample:
				batch_idxs_flat.append(sample_idx)
				start_idxs_flat.append(start_idx)
				end_idxs_flat.append(stop_idx)

		# convert to flattened tensors and reshape
		batch_idxs_flat = torch.tensor(batch_idxs_flat).unsqueeze(0).unsqueeze(2) # 1 x num_chains x 1
		start_idxs_flat = torch.tensor(start_idxs_flat).unsqueeze(0).unsqueeze(2)  # 1 x num_chains x 1
		end_idxs_flat = torch.tensor(end_idxs_flat).unsqueeze(0).unsqueeze(2) # 1 x num_chains x 1

		# create Z x num_chains x N boolean tensor, where True is the positions corresponding to each chain within its batch
		seq_idxs = torch.arange(N).unsqueeze(0).unsqueeze(0) # 1 x 1 x N
		batch_idxs = torch.arange(batch).unsqueeze(1).unsqueeze(2) # Z x 1 x 1
		is_chain = (batch_idxs == batch_idxs_flat) & (seq_idxs >= start_idxs_flat) & (seq_idxs < end_idxs_flat) # Z x num_chains x N


		# convert the boolean tensors to Z x N
		# create boolean tensors defining if each position acts as a logical N, CA, or C, where N[i] is adjacent to CA[i] is adjacent to C[i]
		# the max works as an OR operation along num chains dim to get Z x N boolean tensor of where each logical backbone atom goes
		is_logical_N = (is_chain & ((seq_idxs+2) < end_idxs_flat)).max(dim=1).values # Z x N
		is_logical_CA = (is_chain & ((seq_idxs-1) >= start_idxs_flat) & ((seq_idxs+1) < end_idxs_flat)).max(dim=1).values # Z x N
		is_logical_C = (is_chain & ((seq_idxs-2) >= start_idxs_flat)).max(dim=1).values # Z x N

		# extract the coordinates for each
		logical_N = coordsA[is_logical_N, :] # num_positions_in_all_chains-(2*num_chains) x 3
		logical_CA = coordsA[is_logical_CA, :] # num_positions_in_all_chains-(2*num_chains) x 3
		logical_C = coordsA[is_logical_C, :] # num_positions_in_all_chains-(2*num_chains) x 3

		# compute the virtual beta carbons
		# shapes are all # num_positions_in_all_chains-(2*num_chains) x 3
		# uses ideal bond lengths
		b1 = logical_CA - logical_N
		b1 = 1.458 * b1 / torch.linalg.vector_norm(b1, dim=1, keepdim=True).clamp(min=1e-6)
		b2 = logical_C - logical_CA
		b2 = 1.525 * b2 / torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp(min=1e-6)
		b3 = torch.linalg.cross(b1, b2)

		# compute virtual cb w/ empirical constants
		virtual_CB = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

		# only the logical CA get a CB, so use the already computed boolean tensor to assign CB
		coordsB[is_logical_CA] = virtual_CB

		return coordsA, coordsB

	def get_Cb_from_BB(self, coords):

		'''
		don't need chain idxs here, since the input is a batch x N x 3(N,Ca,C) x 3 tensor, so can use the coords tensor directly
		no masking necessary, as cb ends up being 0,0,0 for masked vals, since masked coords are 0,0,0
		'''

		n = coords[:, :, 0, :]
		ca = coords[:, :, 1, :]
		c = coords[:, :, 2, :]
		
		b1 = ca - n
		b2 = c - ca
		b3 = torch.linalg.cross(b1, b2, dim=2)

		cb = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

		return ca, cb

	def scale_aa_sizes(self, aa_sizes, aa_scale, mask_id):
		
		# decide whether the mask token scales Cb to 0 or to the mean size of all aas
		if mask_id == "zero":
			mask_tok = torch.tensor([0], device=aa_sizes.device) # mask is glycine
		elif mask_id == "mean":
			mask_tok = aa_sizes.mean(dim=0, keepdim=True)
		elif isinstance(mask_id, float):
			mask_tok = torch.tensor([mask_id], device=aa_sizes.device)
		else:
			raise ValueError(f"invalid mask_id option: {mask_id}. options are 'zero' or 'mean', or float value")

		aa_sizes = torch.cat([aa_sizes, mask_tok], dim=0)

		if aa_scale != 0.0:
			aa_sizes = aa_scale * (aa_sizes / aa_sizes.max(dim=0, keepdim=True).values)

		return aa_sizes

	def inject_aas(self, coords_beta, aas):

		batch, N, num_classes = aas.shape

		# convert aas to token idxs
		token_idxs = torch.argmax(aas, dim=2) # Z x N

		# normalize beta carbon so can scale it
		coords_beta_norm = torch.linalg.vector_norm(coords_beta, dim=2, keepdim=True) # Z x N x 1
		coords_beta_norm[coords_beta_norm==0] = 1 # avoid div by zero
		coords_beta_unit = coords_beta / coords_beta_norm # create the unit vector

		# gather the corresponding distances and scale the beta carbon
		coords_beta_magnitudes = torch.gather(self.aa_sizes.unsqueeze(0).unsqueeze(0).expand(batch, N, -1), dim=2, index=token_idxs.unsqueeze(2))
		coords_beta_scaled = coords_beta_unit * coords_beta_magnitudes

		return coords_beta_scaled

class WavefunctionEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and superposing all wavefunction effects
	each feature for each Ca corresponds to the real/imaginary part of the superposed wavefunction at a particular 
	wavelength evaluated at that Cas position. 
	serve as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions. 
	also includes mlp after the embedding layer
	actual function implemented in cuda, optimized for h100 (not really, only takes advantage of large shared mem, need a rewrite to use TMA, possibly WGMMA)
	'''
	def __init__(self, 	d_model=512, 
						min_wl=3.7, max_wl=20, base=20, 
						d_hidden=1024, hidden_layers=0, 
						learnable_wavelengths=False, magnitude_type=1, anisotropic_wf=False, 
						dropout=0.0, wf_dropout=0.0
					):
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
		
		self.magnitude_type = magnitude_type
		self.anisotropic_wf = anisotropic_wf

		self.ffn = MLP(d_model, d_model, d_hidden, hidden_layers, dropout)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.wf_dropout = wf_dropout

	def get_wavelengths(self):
		
		if self.learnable_wavelengths:
			wavelength_weights = torch.softmax(self.wavelength_weights, dim=1)
			wavelengths = torch.matmul(wavelength_weights, self.wavelengths.unsqueeze(1)).squeeze(1) # d_model//2 x d_model//2 @ d_model//2 x 1 -> d_model//2
		else:
			wavelengths = self.wavelengths

		return wavelengths		

	def forward(self, coords_alpha, coords_beta=None, key_padding_mask=None):

		wavenumbers = 2 * torch.pi / self.get_wavelengths()
		wf_dropout = self.wf_dropout if self.training else 0.0

		if self.anisotropic_wf:
			wf = wf_embedding_aniso(coords_alpha, coords_beta, wavenumbers, self.magnitude_type, wf_dropout, key_padding_mask) # batch x N x 3 --> batch x N x d_model
		else:
			wf = wf_embedding_iso(coords_alpha, wavenumbers, self.magnitude_type, wf_dropout, key_padding_mask) # batch x N x 3 --> batch x N x d_model

		# norm, ffn, dropout, residual, norm
		wf = self.norm1(wf)
		wf = self.norm2(wf + self.dropout(self.ffn(wf)))

		return wf

		
class Encoder(nn.Module):
	'''
	bidirectional encoder
	'''

	def __init__(self, 	d_model=512, 
						d_hidden=1024, hidden_layers=0, 
						nhead=8, min_spread=1, max_spread=6, base=20, num_spread=8, 
						min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, 
						dropout=0.0, attn_dropout=0.0
					):
		super(Encoder, self).__init__()

		# Self-attention layers
		self.attn = GeoAttention(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, dropout=attn_dropout)
		self.attn_norm = nn.LayerNorm(d_model)
		self.attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.ffn_norm = nn.LayerNorm(d_model)
		self.ffn_dropout = nn.Dropout(dropout)

	def forward(self, x, coords, key_padding_mask=None): # use x bc applied to structure and seq

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
						nhead=8, 
						min_spread=1, max_spread=6, base=20, num_spread=8, 
						min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, 
						dropout=0.1
					):
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
		if base == 1.0: # linear spacing
			self.register_buffer("spreads", min_spread + (max_spread - min_spread) * torch.linspace(0,1,num_spread))
		else: # log spacing
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
		assert q.shape == k.shape == v.shape
		assert q.dim() == 3
		batch, N, d_model = q.shape
		assert d_model == self.d_model

		# project the tensors
		Q = torch.matmul(q.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k
		K = torch.matmul(k.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k
		V = torch.matmul(v.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # batch x nhead x N x d_k

		dropout = self.dropout if self.training else 0.0

		# perform attention
		out = geometric_attn(Q, K, V, coords, self.get_spreads(), mask=key_padding_mask, min_rbf=self.min_rbf, max_rbf=self.max_rbf, beta=self.beta, dropout=dropout)  # batch x nhead x N x d_k

		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x nhead
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x nhead --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

# ----------------------------------------------------------------------------------------------------------------------
