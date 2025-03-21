# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		mask_utils.py
description:	utility classes for mask injection for training, only span masking for now, 
				but will configure to allow random token masking later for ablation studies
'''
# ----------------------------------------------------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F

class MASK_injection():

	def __init__(self,  mask_type="rand",
						min_mask_pct=0.00,
						max_mask_pct=1.00,
						mean_span=10,
						std_span=5,
						randAA_pct=0.0
					):

		self.mask_type = mask_type
		self.min_mask_pct = min_mask_pct
		self.max_mask_pct = max_mask_pct
		self.mean_span = mean_span
		self.std_span = std_span
		self.randAA_pct = randAA_pct

	def get_span_mask(self, key_padding_mask):
		'''
		computes span masks 
		'''

		# utils
		batch, N = key_padding_mask.shape
		valid = (~key_padding_mask).sum(dim=1)

		# get mask pcts from uniform distribution
		mask_pct = torch.rand((batch, )).clamp( min=self.min_mask_pct, max=self.max_mask_pct)

		# as number of spans (approx). 
		overlap_factor = 1 + (self.mean_span/torch.log2(valid)) # smaller sequences have higher chance of overlap, slightly overshoots, but better than undershooting
		num_spans = ((mask_pct * valid * overlap_factor) / (self.mean_span)).clamp(min=1).long() # div by 2 so can get first set of start idxs then second set

		# sample a span length for each residue from gaussian dist with mean span length and std span length defined above
		span_lengths = torch.round(torch.clamp((torch.randn((batch, N))*self.std_span) + self.mean_span, min=1)).to(torch.int) # Z x N

		# initialize the span mask
		span_mask = torch.zeros_like(key_padding_mask)

		# utils
		seq_idx = torch.arange(N).unsqueeze(0)

		# get rand vals and remove mask positions
		rand_vals = torch.rand(batch, N) # Z x N

		# perform convolution to sharpen edges, i.e. avoid clustering of spand in 1d sequence

		# Define convolution kernel
		kernel_size = (self.mean_span//2) + 1 # make it an odd number so centered properly
		num_supress = (kernel_size - 1)//2
		center_val = torch.tensor([self.mean_span*2])
		suppress_vals = -(1 + torch.arange(num_supress).float())*num_supress
		suppress_vals_rev = torch.flip(suppress_vals, dims=[0])
		kernel = torch.cat([suppress_vals, center_val, suppress_vals_rev], dim=0).view(1, 1, -1)  # out_channels x in_channels x kernel_size

		# Apply 1D convolution with padding to maintain shape
		conv_output = torch.nn.functional.conv1d(rand_vals.unsqueeze(1), kernel, padding=num_supress)  # Shape: (batch, 1, N)

		# Remove channel dimension
		conv_output = conv_output.squeeze(1)

		# sort values and get the number of elements computed to reach mask pct
		conv_sorted, _ = torch.sort(conv_output, dim=1, descending=True)
		conv_thresh = torch.gather(conv_sorted, 1, index=num_spans.unsqueeze(1)-1)
		conv_output.masked_fill_(key_padding_mask, -float("inf"))

		batch_idx, start_idx = torch.nonzero(conv_output >= conv_thresh, as_tuple=True) # numTrue

		# define the end idx by looking up the span length of the start idx
		end_idx = start_idx + span_lengths[batch_idx, start_idx] # numTrue

		# find tokens in the span
		in_span = (seq_idx >= start_idx.unsqueeze(1)) & (seq_idx < end_idx.unsqueeze(1)) & (~key_padding_mask[batch_idx]) # numTrue x N

		# aggregate the span tokens for each batch, from numTrue x N --> Z x N, amax functions as an OR operation, since max is 1 for bool tensor
		span_mask.scatter_reduce_(0, batch_idx.unsqueeze(1).expand(-1, N), in_span, reduce="amax")

		# span mask is true for positions to inject, false otherwise. once inject the mask tokens, invert it to be consistent w/ other masks
		return span_mask

	def get_random_mask(self, key_padding_mask):
		batch, N = key_padding_mask.shape
		mask_pct = torch.rand((batch,1)).clamp(min=self.min_mask_pct, max=self.max_mask_pct)
		mask = torch.rand_like(key_padding_mask, dtype=torch.float) < mask_pct

		return mask

	def inject_mask(self, mask, key_padding_mask, predictions):

		batch, N, num_classes = predictions.shape

		rand_vals = torch.rand_like(mask, dtype=torch.float)

		# inject random tokens into non-masked positions
		# if self.randAA_pct is -1, sample the randAA pct from uniform distribution for each sample, else, it is constant
		if self.randAA_pct == -1:
			randAA_pct = torch.rand((batch, 1), device=mask.device)
		else:
			randAA_pct = self.randAA_pct

		rand_token = torch.where(mask | key_padding_mask, False, rand_vals < randAA_pct)

		# add random token to predictions
		# high in randint is exclusive, so only samples from non-mask tokens
		one_hot_rand_token = torch.nn.functional.one_hot(torch.randint(low=0, high=num_classes-1, size=(batch, N)), num_classes=num_classes)
		
		# for the mask token
		one_hot_mask_token = torch.nn.functional.one_hot(torch.full((batch, N), num_classes-1), num_classes=num_classes)

		# replace these positions in predictions
		predictions = torch.where(rand_token.unsqueeze(2), one_hot_rand_token, predictions)
		predictions = torch.where(mask.unsqueeze(2), one_hot_mask_token, predictions)

		return predictions

	def MASK_tokens(self, batch):

		if self.mask_type == "rand":
			mask = self.get_random_mask(batch.key_padding_mask)
		elif self.mask_type == "span":
			mask = self.get_span_mask(batch.key_padding_mask)
		elif self.mask_type == "noise": # no masking, just replace true aas with random aas
			mask = torch.zeros_like(batch.key_padding_mask)
		else:
			raise ValueError(f"invalid mask injection option: {mask_type}. valid options are 'rand' or 'span'")
		
		predictions = self.inject_mask(mask, batch.key_padding_mask, batch.predictions) # assumes predictions is a one hot representation of labels
		batch.predictions, batch.onehot_mask = predictions, (~mask)