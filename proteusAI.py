# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a a protein, based on 
				alpha carbon coordinates
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils import protein_to_wavefunc

# ----------------------------------------------------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and each feature for each Ca corresponds 
	to the superposed wavefunction at a particular wavelength. each wavefunction output creates two features, a real part and 
	an imaginary part. Thus, d_model / 2 wave functions, each with a corresponding wavelength, are generated to create
	d_model features for each Ca 
	need to precompute these for training, but very manageable for inference to compute on the fly
	serve as a generalization of positional encoding for irregularly spaced tokens in arbitrary dimensions. probably need
	approximations for larger scale problems, but has potential
	'''
	def __init__(self, d_model=512):
		super(SpatialEmbedding, self).__init__()

		self.d_model = 512

	def forward(self, coords, key_padding_mask):

		# convert to wf features if not already precomputed
		wavefunc = protein_to_wavefunc(coords, key_padding_mask, self.d_model) # batch x N x 3 --> batch x N x d_model

		return wavefunc

class MHA(nn.Module):
	'''
	custom MHA module, in order to scale attention weights for each head based 
	on each head's spread in the RBF of PW distances 
	'''

	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(MHA, self).__init__()

		self.nhead = nhead
		self.d_model = d_model

		if self.d_model % self.nhead != 0: raise ValueError(f"number of dimensions ({self.d_model}) must be divisible by number of attention heads ({self.nhead})")
		self.d_k = self.d_model // self.nhead

		self.q_proj = nn.Parameter(torch.randn(1, self.nhead, self.d_model, self.d_k)) # 1 x nhead x d_model x d_k
		self.k_proj = nn.Parameter(torch.randn(1, self.nhead, self.d_model, self.d_k)) # 1 x nhead x d_model x d_k
		self.v_proj = nn.Parameter(torch.randn(1, self.nhead, self.d_model, self.d_k)) # 1 x nhead x d_model x d_k

		self.out_proj = nn.Linear(d_model, d_model)

	def get_attn(self, q, k, v):
		'''
		computes the attention matrix and the value matrix for each head
		'''
		
		# reshape so each head gets its own data
		q = q.unsqueeze(1).expand(-1, self.nhead, -1, -1) # batch x N x d_model --> batch x nhead x N x d_model
		k = k.unsqueeze(1).expand(-1, self.nhead, -1, -1) # batch x N x d_model --> batch x nhead x N x d_model
		v = v.unsqueeze(1).expand(-1, self.nhead, -1, -1) # batch x N x d_model --> batch x nhead x N x d_model

		# project q, k, and v to d_k for each head
		q = torch.matmul(q, self.q_proj) # batch x nhead x N x d_model @ 1, nhead, d_model, d_k --> batch x nhead x N x d_k
		k = torch.matmul(k, self.k_proj) # batch x nhead x N x d_model @ 1, nhead, d_model, d_k --> batch x nhead x N x d_k
		v = torch.matmul(v, self.v_proj) # batch x nhead x N x d_model @ 1, nhead, d_model, d_k --> batch x nhead x N x d_k

		# create the attention tensor
		attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # batch x nhead x N x d_k @ batch x nhead x d_k x N --> batch x nhead x N x N

		return attn, v

	def get_dists_and_spreads(self, coords, min_wl=3.7, max_wl=20, base=20, eps=2e-2):
		'''
		computes pairwise distances and the spread for each head
		'''
		# define the distance tensor and the spreads for each head
		dists = torch.sqrt(((coords.unsqueeze(1) - coords.unsqueeze(2))**2).sum(dim=-1)) # batch x N x 3 --> batch x N x N
		dists = dists.unsqueeze(1).expand(-1, self.nhead, -1, -1) # batch x N x N --> batch x nhead x N x N
		spreads = min_wl + (((torch.logspace(0, 1-eps, steps=self.nhead, base=base) - 1) / (base - 1)) * (max_wl - min_wl)) # nhead,
		spreads = spreads.to(dists.device)
		spreads = spreads.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # nhead, --> 1 x nhead x 1 x 1 

		return dists, spreads


	def get_dist_masks(self, dists, spreads):
		'''
		creates a clamp mask, used to clamp distances < head_spread to head_spread, and 
		dist_mask, which clamps large values to 3*head_spread, for numerical stability, since
		dist masked values will be masked later anyway
		'''
		# clamp small distances so not disproportionally important at larger spreads
		clamp_mask = dists < spreads # batch x nhead x N x N
		dists = torch.where( # batch x nhead x N x N
			clamp_mask,
			dists.clamp(min=spreads), # clamp distances so smallest possible is the spread, independant for each head
			dists
		)

		# clamp long distances to ensure numerical instability, will mask anyways later
		dist_mask = dists > (spreads*3) # batch x nhead x N x N
		dists = torch.where( # batch x nhead x N x N
			dist_mask,
			dists.clamp(max=spreads*3), # clamp distances to spread * 3 to avoid numerical instability,
			dists
		)

		return dists, dist_mask


	def rbf_scaling(self, attn, dists, spreads):
		'''
		scales the attention weights by the RBF computed based on pair-wise distances and head-specific spread
		'''
		# compute the rbf, and the inverse for negative logits
		rbf = torch.exp(-(dists**2 / (2*spreads**2))) # batch x nhead x N x N
		rbf = torch.where( 	# inverse rbf for negative logits, so close distances with negative logits are more pronounced 
							# than long distances, even if the attn score is close to zero since exponentiating negative value
			attn < 0,
			1/rbf,
			rbf
		)

		# apply the rbf scaling to the logits
		attn = attn * rbf # batch x nhead x N x N

		return attn
	
	def forward(self, q, k, v, coords=None, key_padding_mask=None, min_wl=3.7, max_wl=20, base=20, eps=2e-2, use_checkpoint=False):
		'''
		forward method, wrapper for _forward so can use gradient checkpointing in training
		'''
		if use_checkpoint:
			out = checkpoint(
				self._forward,
				q, k, v, coords, key_padding_mask,
				torch.tensor(min_wl), torch.tensor(max_wl),
				torch.tensor(base), torch.tensor(eps)
			)
		else:
			out = self._forward(q, k, v, coords, key_padding_mask, min_wl, max_wl, base, eps)

		return out

	def _forward(self, q, k, v, coords=None, key_padding_mask=None, min_wl=3.7, max_wl=20, base=20, eps=2e-2):
		'''
		performs scaled dot-product attention weighted by Gaussian RBF 
		'''

		# make sure shape is compatible
		assert (q.shape == k.shape) and (q.shape == v.shape)
		batch, N, d_model = q.shape
		assert d_model == self.d_model

		for a_name, a in zip("qkv", [q, k, v]):
			print(f"{a_name}: \n{a}\n{a.shape}")

		attn, v = self.get_attn(q, k, v)
		print(f"attn: \n{attn}\n{attn.shape}")

		dists, spreads = self.get_dists_and_spreads(coords, min_wl, max_wl, base, eps)
		print(f"dists: \n{dists}\n{dists.shape}")
		print(f"spreads: \n{spreads}\n{spreads.shape}")

		dists, dist_mask = self.get_dist_masks(dists, spreads)

		print(f"dists_mask: \n{dist_mask}\n{dist_mask.shape}")
		attn = self.rbf_scaling(attn, dists, spreads)

		# mask attention pairs and normalize
		attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | dist_mask
		attn = attn.masked_fill(attn_mask, -torch.inf) # batch x nhead x N x N
		attn = F.softmax(attn, dim=-1) # batch x nhead x N x N
		print(f"scaled_attn: \n{attn}\n{attn.shape}")

		# multiply attn tensor by v
		out = torch.matmul(attn, v) # batch x nhead x N x N @ batch x nhead x N x d_k --> batch x nhead x N x d_k

		print(f"qkv: \n{out}\n{out.shape}")

		# reshape to batch x N x d_model
		out = out.permute(0, 2, 1, 3) # batch x nhead x N x d_k --> batch x N x nhead x d_k
		print(f"qkv_permute: \n{out}\n{out.shape}")
		out = out.reshape(batch, N, self.d_model) # batch x N x d_k x nhead --> batch x N x d_model
		print(f"qkv_reshape: \n{out}\n{out.shape}")

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model
		print(f"qkv_proj: \n{out}\n{out.shape}")

		# return
		return out # batch x N x d_model

class Decoder(nn.Module):
	'''
	decoder module, contains self-attention, normalization, dropout, feed-forward network, and residual connections
	'''

	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(Decoder, self).__init__()

		# Self-attention layers
		self.self_attn = MHA(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)

		# Separate normalization layers
		self.norm = nn.LayerNorm(d_model)
		
		# Separate dropout layers
		self.dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.feedforward_norm = nn.LayerNorm(d_model)
		self.feedforward_dropout = nn.Dropout(dropout)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		# module list to store learnable parameters, to easily freeze and unfreeze them
		self.learnable = nn.ModuleList([self.self_attn, self.norm, self.feedforward_norm, self.linear1, self.linear2])
		
	def modify_weights(self, requires_grad=True):
		'''
		modifies decoder weights, so that stage one of training (no aa-context) complements
		stage 2, where freeze decoder weights so that only aa embeddings, qkv of context module
		and output embeddings are learnable. forces model to build off of learned spatial
		representation, rather than completely rewrite it
		'''
		# method to (un)freeze weights to have model learn aa representations
		for module in self.learnable:
			for param in module.parameters():
				param.requires_grad = requires_grad

	def forward(self, wavefunc, coords, key_padding_mask=None, use_checkpoint=False):

		# full self-attention
		wavefunc2 = self.self_attn(wavefunc, wavefunc, wavefunc,
						coords=coords,
						key_padding_mask=key_padding_mask,
						use_checkpoint=use_checkpoint
						)

		# residual connection with dropout
		wavefunc = wavefunc + self.dropout(wavefunc2)

		# norm
		wavefunc = self.norm(wavefunc)
		
		# Feed-forward network (with dropout)
		wavefunc2 = self.linear2(self.feedforward_dropout(F.gelu(self.linear1(wavefunc))))
		
		# add (with dropout) and norm
		wavefunc = wavefunc + self.feedforward_dropout(wavefunc2)
		wavefunc = self.feedforward_norm(wavefunc)

		return wavefunc


class ContextModule(Decoder):
	'''
	Performs masked self attention so that the environments query the amino acid identities,
	but masked so that only predicted positions with AA context are queried, but not 
	non-context positions. acts as a way to distribute the AA context before
	full self attention in decoder 

	same as decoder but forward method accounts for when some samples in a 
	batch have no context (all masked) but the rest only have some, avoids 
	numerical instability
	'''
	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(ContextModule, self).__init__(d_model, nhead, dim_feedforward, dropout)

								                    # vvvvvvvvvv repetive, as can do single mask, but just for clarity
	def forward(self, wavefunc, coords, aa_context, aa_context_mask=None, key_padding_mask=None, use_checkpoint=False):

		# masked attention; update Ca envs based on predicted positions env with AA context
		context_wavefunc2 = self.self_attn(wavefunc, aa_context, wavefunc,
							coords=coords,
							key_padding_mask=(aa_context_mask | key_padding_mask),
							use_checkpoint=use_checkpoint)

		# residual connection with dropout
		context_wavefunc = wavefunc + self.dropout(context_wavefunc2)
		# norm
		context_wavefunc = self.norm(context_wavefunc)

		# Feed-forward network (with dropout)
		context_wavefunc2 = self.linear2(self.feedforward_dropout(F.gelu(self.linear1(context_wavefunc))))
		
		# add (with dropout) and norm
		context_wavefunc = context_wavefunc + self.feedforward_dropout(context_wavefunc2)
		context_wavefunc = self.feedforward_norm(context_wavefunc)

		# only include context module for samples in the batch with already predicted positions
		has_context = torch.any(~(aa_context_mask | key_padding_mask), dim=-1).unsqueeze(-1).unsqueeze(-1) # batch x 1 x 1
		wavefunc = torch.where(has_context, context_wavefunc, wavefunc)

		return wavefunc

class proteusAI(nn.Module):
	'''

	proteusAI. 

	converts 3D coords to wavefunction features (wf), and one hot amino acid tensors into aa embeddings. performs masked 
	cross gaussian self-attention (lots of words) via the ContextModule, with wf as Q and V, and AA embeddings as K. Gaussian weighting is
	done based on pair-wise distance based RBFs. then go through all decoder layers. decoder layer and context module are essentially the same
	just the q, k, and v matrices vary. these are a stack of multi-head attention layers, with each head operating at a 
	different scale, by varying the spread of the RBF for that head. after self-attention, do add + norm, FFN, and residual connection. the subspace (d_k) each head operates at is close to the scale of
	d_k, since each feature in d_k (and d_model, where d_model=d_k*nhead) corresponds to a distinct wavelength (and thus k in Green's function),
	with low index features corresponding to small wavelengths and large index features to large wavelengths. the final
	decoder layer output goes through a linear layer converts the d_model feature space to a 21-d feature space (20 AAs + X (unknown AA)). softmax is applied to 
	get probabilities and sample from the distribution autoregressively (during inference).

	'''
	
	def __init__(self, N, d_model, n_head, decoder_layers, hidden_linear_dim, dropout, active_decoders=-1, use_probs=False):
		super(proteusAI, self).__init__()

		# configuration
		self.use_probs = use_probs
		self.active_decoders = active_decoders if active_decoders != -1 else decoder_layers

		# wavefunc
		self.spatial_embedding = SpatialEmbedding(d_model)

		# context
		self.aa_embedding = nn.Linear(20, d_model)
		self.context_module = ContextModule(d_model, n_head, hidden_linear_dim, dropout)

		# decoder
		self.decoders = nn.ModuleList([Decoder(d_model, n_head, hidden_linear_dim, dropout) for _ in range(decoder_layers)])

		# map to aa probs
		self.linear = nn.Linear(d_model, 20)

	def add_decoder(self, new_decoders=1):
		'''
		method to dynamicaly add decoders, probably wont use, but might be useful in future
		'''
		
		for new_decoder in new_decoders:
			if len(self.decoders) > self.active_decoders:

				# load new decoder with prev decoder weights
				last_decoder = self.decoders[self.active_decoders-1]
				new_decoder = self.decoders[self.active_decoders]
				new_decoder.load_state_dict(last_decoder.state_dict())

				self.active_decoders += 1

	def alter_decoder_weights(self, requires_grad=True):
		'''
		method to alter decoder weights, so can freeze spatial components and have
		aa context components be learnable only
		'''

		for decoder in self.decoders:
			decoder.modify_weights(requires_grad)

	def contextualize(self, wavefunc, coords, aa_onehot, key_padding_mask=None):

		# map the predicted aas to target feature space, non-predicted positions are all 0, so linear layer retains the zero vector for these
		aa_context = self.aa_embedding(aa_onehot) # batch x N x 20 --> batch x N x d_model

		# used for masked MHA to ignore keys with no context
		aa_context_mask = ~(aa_onehot.any(dim=-1))# batch x N

		# masked MHA to update all positions by ONLY the new context
		wavefunc = self.context_module(wavefunc, coords, aa_context, aa_context_mask, key_padding_mask)

		return wavefunc

	def decode(self, wavefunc, coords, key_padding_mask=None):

		for decoder_layer in self.decoders[:self.active_decoders]:
			
			# decode the updated environments
			wavefunc = decoder_layer(wavefunc, coords, key_padding_mask) # batch x N x d_model

		# map the decoded environment to aa probabilities
		seq_probs = self.linear(wavefunc) # batch x N x d_model --> batch x N x 20

		return seq_probs

	def auto_regressive(self, wavefunc, aa_onehot, key_padding_mask=None, distance_weights=None, temp=0.1):

		# auto regressively sample AAs at most confident position
		for position in range(aa_onehot.size(1)):

			# add context, unless first iteration, in which case context was already added in forward method
			if position > 0:
				wavefunc = self.contextualize(wavefunc, aa_onehot, key_padding_mask, distance_weights)

			# decode the wavefunction
			seq_probs = self.decode(wavefunc, key_padding_mask, distance_weights) # batch x N x 512 --> batch x N X 20 
			
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
		# prob_distributions = prob_distributions.masked_fill(predicted_positions_mask, 0.0) # mask out complete positions
		
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

		# wave function encoding (replaces positional encoding)
		if features is None:
			wavefunc = self.spatial_embedding(coords, key_padding_mask) # batch x N x 3 --> batch x N x d_model
		else:
			wavefunc = features

		# contextualize environment with AAs if any sequence has context
		has_context = (aa_onehot.any(dim=-1) & (~key_padding_mask)).any() # 1,
		if has_context:
			wavefunc = self.contextualize(wavefunc, aa_onehot, coords, key_padding_mask) # batch x N x d_model

		# note auto-regression breaks the computational chain, do not implement during training
		if auto_regressive:
			seq_probs = self.auto_regressive(wavefunc, aa_onehot, coords, key_padding_mask, temp=temp) # batch x N x 20 (returns one-hot tensor)
		else: 
			seq_probs = self.decode(wavefunc, coords, key_padding_mask) # batch x N x 20 (returns aa probability logits)

		return seq_probs

# ----------------------------------------------------------------------------------------------------------------------