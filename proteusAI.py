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

from utils import protein_to_wavefunc

# ----------------------------------------------------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
	'''
	converts the Ca coordinates (batch x N x 3) to the target feature space (batch x N x d_model) by modeling each Ca as
	a point source via the Green function solution to the Hemholtz equation, and each feature for each Ca corresponds 
	to the superposed wavefunction at a particular wavelength. each wavefunction output creates two features, a real part and 
	an imaginary part. Thus, d_model / 2 wave functions, each with a corresponding wavelength, are generated to create
	d_model features for each Ca 
	need to precompute these for training, but very manageable for inference to compute on the fly
	'''
	def __init__(self):
		super(SpatialEmbedding, self).__init__()

	def forward(self, x, key_padding_mask):

		x = protein_to_wavefunc(x, key_padding_mask) # batch x N x 3 --> batch x N x 512

		return x

class CrossFeatureNorm(nn.Module):
	'''
	normalizes each feature independantly across the sequence. it is independant of batches (not batch norm)
	this is helpful because each feature for a given token (Ca atom) is the output of that token for the global 
	superposed wavefunction at a particular k (wavelength). thus, each feature in a given token is only relevant
	RELATIVE to the CORRESPONDING features of all other tokens in the sequence. 
	This essentially normalizes each wavefunction's (psi_k) output. Note that this normalizes the real part and 
	the imaginary part independantly 
	'''
	def __init__(self, d_model, eps=1e-5):
		super(CrossFeatureNorm, self).__init__()

		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(1,1,d_model))
		self.beta = nn.Parameter(torch.zeros(1,1,d_model))

	def forward(self, x, key_padding_mask=None):

		if key_padding_mask is not None:
			# key_padding_mask is of shape (batch, N) - we invert it to use as a valid mask
			valid_mask = ~key_padding_mask.unsqueeze(-1)  # batch x N --> batch x N x 1
			
			# Mask invalid (padded) positions
			x_masked = x * valid_mask  # Zero out padded positions in x ; batch x N x d_model

			# Compute the mean and variance only for valid positions
			sum_valid = torch.sum(x_masked, dim=1) # batch x d_model
			num_valid = valid_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
			mean = sum_valid / num_valid  # shape (batch, d_model)

			# Subtract the mean for valid positions
			mean_expanded = mean.unsqueeze(1)  # shape (batch, 1, d_model)
			x_centered = (x - mean_expanded) * valid_mask  # Zero out padded positions again after centering

			# Compute variance only for valid positions
			variance = torch.sum(x_centered ** 2, dim=1) / num_valid  # shape (batch, d_model)
			variance_expanded = variance.unsqueeze(1)
			std = torch.sqrt(variance_expanded + self.eps)

			# Normalize the valid positions
			x_norm = (x_centered / std) * valid_mask 

		else:
			# compute mean and variance ; batch x N x d_model --> batch x 1 x d_model
			mean = x.mean(dim=1, keepdim=True)
			var = x.var(dim=1, keepdim=True, unbiased=False)

			# normalize each feature independently across the sequence ; batch x N x d_model
			x_norm = (x - mean) / torch.sqrt(var + self.eps)


		# apply learnable scaling (gamma) and shifting (beta) to each feature
		x = self.gamma * x_norm + self.beta

		return x

class Encoder(nn.Module):
	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(Encoder, self).__init__()

		# Self-attention layer
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		
		# Feed-forward network
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		
		# Normalization layers
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		
		# Dropout
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, src, src_key_padding_mask=None):

		# Multi-head self-attention
		src2, _ = self.self_attn(src, src, src,
							key_padding_mask=src_key_padding_mask,
							need_weights=False)
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		
		# Feed-forward network
		src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)

		return src
		
class Decoder(nn.Module):
	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(Decoder, self).__init__()

		# Self-attention layer
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.cross_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		
		# Feed-forward network
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		
		# Normalization layers
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		
		# Dropout
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, src, trgt, key_padding_mask=None):

		# first cross self attention to weight probabilities by structural relevance
		trgt2, _ = self.cross_self_attn(src, src, trgt,
								key_padding_mask=key_padding_mask,
								need_weights=False)
		trgt = trgt + self.dropout1(trgt2)
		trgt = self.norm1(trgt)
		
		# then self attention with structure informed probabiilties
		trgt2, _ = self.self_attn(trgt, trgt, trgt,
							key_padding_mask=key_padding_mask,
							need_weights=False)
		trgt = trgt + self.dropout1(trgt2)
		trgt = self.norm1(trgt)
		
		# Feed-forward network
		trgt2 = self.linear2(self.dropout(F.gelu(self.linear1(trgt))))
		trgt = trgt + self.dropout2(trgt2)
		trgt = self.norm2(trgt)

		return trgt

class proteusAI(nn.Module):
	def __init__(self, N, d_model, n_head, encoder_layers, decoder_layers, hidden_linear_dim, dropout, use_features):
		super(Model, self).__init__()

		self.use_features = use_features
		self.spatial_embedding = SpatialEmbedding()
		self.cross_feature_norm = CrossFeatureNorm(d_model)
		# self.layer_norm = nn.LayerNorm(d_model)

		self.encoders = nn.ModuleList([Encoder(d_model, n_head, hidden_linear_dim, dropout) for _ in range(encoder_layers)])

		self.output_embedding = nn.linear(20, d_model)
		
		self.decoders = nn.ModuleList([Decoder(d_model, n_head, hidden_linear_dim, dropout) for _ in range(decoder_layers)])

		self.linear = nn.Linear(d_model, 20)

	def forward(self, src, trgt, key_padding_mask, diffusion_cycles=0):
		"""
		Forward pass of the model with optional diffusion cycles.
		- diffusion_cycles=0 for one-shot prediction
		- diffusion_cycles > 0 for iterative diffusion-like inference
		"""
		# src: batch x N x 3 (or batch x N x d_model if self.use_features is True)
		# trgt: batch x N x 20

		if not self.use_features: # only compute spatial embedding if have not precomputed them to save memory
			src = self.spatial_embedding(src, key_padding_mask) # batch x N x 3 --> batch x N x 512

		src = self.cross_feature_norm(src, key_padding_mask)
		# src = self.layer_norm(src)

		for encoder_layer in self.encoders:
			src = encoder_layer(src, key_padding_mask) # batch x N x 512

		for diffusion_cycle in range(diffusion_cycles + 1):
			trgt = self.output_embedding(trgt) # batch x N x 20 --> batch x N x 512
			
			for decoder_layer in self.decoders:
				trgt = decoder_layer(src, trgt, key_padding_mask) # batch x N x 512
	
			trgt = self.linear(trgt) # batch x N x 512 --> batch x N x 20
			if diffusion_cycle < diffusion_cycles:
				trgt = F.softmax(trgt, dim=-1)  # let the loss function compute softmax on final output

		return trgt

# ----------------------------------------------------------------------------------------------------------------------
