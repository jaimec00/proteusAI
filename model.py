# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		model.py
description:	model definition
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

class PositionalEncoding(nn.Module):
	def __init__(self, N, d_model=512):
		super(PositionalEncoding, self).__init__()
		self.positional_encoding = torch.zeros(N, d_model) # N x d_model
		
		position = torch.arange(0, N, dtype=torch.float32).unsqueeze(1) # N x 1
		div_term = torch.pow(10000, (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)) # sin and cos terms for each div term, so has length d_model/2 ; d_model/2,
		
		self.positional_encoding[:, 0::2] = torch.sin(position / div_term) # N x d_model
		self.positional_encoding[:, 1::2] = torch.cos(position / div_term) # N x d_model
		
		# Register as buffer so it's part of the model but not trainable
        # self.register_buffer('positional_encoding', positional_encoding)

	def forward(self, x):
		pe = self.positional_encoding[None, :, :] # batch x N x d_model
		x = x + pe # batch x N x d_model
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
		src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)

		return src
		
# Define a small MLP for the transformation
class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class OutputEmbedding(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(OutputEmbedding, self).__init__()
		self.mlp = MLP(input_dim, hidden_dim, output_dim)

	def forward(self, x):
		x = self.mlp(x)
		return x

class Decoder(nn.Module):
	def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
		super(Decoder, self).__init__()

		# Self-attention layer
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.combined_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		
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


		# Multi-head self-attention
		trgt2, _ = self.self_attn(trgt, trgt, trgt,
								key_padding_mask=key_padding_mask,
								need_weights=False)
		trgt = trgt + self.dropout1(trgt2)
		trgt = self.norm1(trgt)
		
		# Multi-head self-attention
		trgt2, _ = self.combined_self_attn(src, src, trgt,
							key_padding_mask=key_padding_mask,
							need_weights=False)
		trgt = trgt + self.dropout1(trgt2)
		trgt = self.norm1(trgt)
		
		# Feed-forward network
		trgt2 = self.linear2(self.dropout(F.relu(self.linear1(trgt))))
		trgt = trgt + self.dropout2(trgt2)
		trgt = self.norm2(trgt)

		return trgt

class Model(nn.Module):
	def __init__(self, N, d_model, n_head, hidden_linear_dim, dropout, use_features):
		super(Model, self).__init__()

		self.use_features = use_features
		self.spatial_embedding = SpatialEmbedding()
		self.cross_feature_norm = CrossFeatureNorm(d_model)
		# self.layer_norm = nn.LayerNorm(d_model)

		self.encoder1 = Encoder(d_model, n_head, hidden_linear_dim, dropout)
		self.encoder2 = Encoder(d_model, n_head, hidden_linear_dim, dropout)
		self.encoder3 = Encoder(d_model, n_head, hidden_linear_dim, dropout)

		self.output_embedding = OutputEmbedding(20, hidden_linear_dim, d_model)
		self.positional_encoding = PositionalEncoding(N, d_model)
		
		self.decoder1 = Decoder(d_model, n_head, hidden_linear_dim, dropout)
		self.decoder2 = Decoder(d_model, n_head, hidden_linear_dim, dropout)
		self.decoder3 = Decoder(d_model, n_head, hidden_linear_dim, dropout)
		
		self.linear = nn.Linear(d_model, 20)

	def forward(self, src, trgt, key_padding_mask):

		# src: batch x N x 3
		# trgt: batch x N x 20

		if not self.use_features: # only compute spatial embedding if have not precomputed them to save memory
			src = self.spatial_embedding(src, key_padding_mask) # batch x N x 3 --> batch x N x 512

		src = self.cross_feature_norm(src, key_padding_mask)
		# src = self.layer_norm(src)

		src = self.encoder1(src, key_padding_mask) # batch x N x 512
		src = self.encoder2(src, key_padding_mask) # batch x N x 512
		src = self.encoder3(src, key_padding_mask) # batch x N x 512
	
		trgt = self.output_embedding(trgt) # batch x N x 20 --> batch x N x 512
		trgt = self.positional_encoding(trgt) # batch x N x 512
		
		trgt = self.decoder1(src, trgt, key_padding_mask) # batch x N x 512
		trgt = self.decoder2(src, trgt, key_padding_mask) # batch x N x 512
		trgt = self.decoder3(src, trgt, key_padding_mask) # batch x N x 512
		
		trgt = self.linear(trgt) # batch x N x 512 --> batch x N x 20
		# trgt = F.softmax(trgt, dim=-1) # no softmax for cross entropy loss, handled automatically

		return trgt

# ----------------------------------------------------------------------------------------------------------------------
