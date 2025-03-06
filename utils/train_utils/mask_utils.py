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

	def __init__(self,  mean_mask_pct=0.5,
						std_mask_pct=0.25,
						min_mask_pct=0.00,
						max_mask_pct=1.00,
						mean_span=10,
						std_span=5,
						randAA_pct=0.0, trueAA_pct=0.0
					):

		self.mean_mask_pct = mean_mask_pct
		self.std_mask_pct = std_mask_pct
		self.min_mask_pct = min_mask_pct
		self.max_mask_pct = max_mask_pct
		self.mean_span = mean_span
		self.std_span = std_span

		self.randAA_pct = randAA_pct
		self.trueAA_pct = trueAA_pct

	def get_span_mask(self, key_padding_mask):
		'''
		computes span masks 
		'''

		# utils
		batch, N = key_padding_mask.shape
		seq_idx = torch.arange(N).unsqueeze(0)
		valid = (~key_padding_mask).sum(dim=1)
		sample_done = lambda span_mask, valid, mask_pct: (((span_mask.sum(dim=1)+mean_span) / (valid+1e-6)) >= mask_pct) & (span_mask.any(dim=1) | (valid==0))
		done = lambda span_mask, valid, mask_pct: sample_done(span_mask, valid, mask_pct).all()

		# define the target percentage of tokens for each sample, just a guidline, as span masking makes it hard to 
		# exactly reach the target while stilll being efficient
		mask_pct = torch.clamp((torch.randn((batch, ))*std_mask_pct) + mean_mask_pct, min=min_mask_pct, max=max_mask_pct)

		# sample a span length for each residue from gaussian dist with mean span length and std span length defined above
		span_lengths = torch.round(torch.clamp((torch.randn((batch, N))*std_span) + mean_span, min=1)).to(torch.int) # Z x N

		# compute the number of spans to select per iteration for each sample, to avoid small updates on large sequence lengths
		# valid samples / mean_span lengths is approx the number of spans that fit (assuming perfect spacing). 
		# multiply by mask pct to get number of spans to reach mask_pct
		num_spans_per_iter = torch.clamp(torch.ceil(mask_pct * (valid / mean_span)), min=1, max=N).long().unsqueeze(1) # Z x 1

		# initialize the span mask
		span_mask = torch.zeros_like(key_padding_mask) # Z x N

		# loop until each sample reaches its mask_pct
		while not done(span_mask, valid, mask_pct):

			# get rand vals
			rand_vals = torch.rand(batch, N) # Z x N

			# get the 1D distance from the nearest span token for each token. use this to increase likelihood of choosing isolated region for next span
			span_batch_idxs, span_N_idxs = torch.nonzero(span_mask, as_tuple=True) # numMask
			dists_raw = (span_N_idxs.unsqueeze(1) - seq_idx).abs() # numMask x N
			dists = torch.full((batch, N), N, dtype=torch.long) # max dist is N, set it to this so amin works properly in next line
			dists.scatter_reduce_(0, span_batch_idxs.unsqueeze(1).expand(-1,N), dists_raw, reduce="amin") # amin gets the minimum distance of each token from all other mask tokens

			# multiply rand vals by dists so isolated regions are more likely to be in topk
			rand_vals *= dists
			rand_vals.masked_fill_(span_mask | key_padding_mask, -float("inf"))

			# sort rand vals and find the kth largest (diff for each sample), use that as threshold to get start_idxs
			rand_vals_sorted, _ = torch.sort(rand_vals, dim=1, descending=True) # Z x N
			rand_val_thresh = torch.gather(rand_vals_sorted, 1, num_spans_per_iter-1)  # Z x 1
			rand_val_thresh.masked_fill_(sample_done(span_mask, valid, mask_pct).unsqueeze(1), float("inf")) # so don't select anything for samples that are done

			# use the thresh to select start idxs
			batch_idx, start_idx = torch.nonzero(rand_vals > rand_val_thresh, as_tuple=True) # numTrue

			# define the end idx by looking up the span length of the start idx
			end_idx = start_idx + span_lengths[batch_idx, start_idx] # numTrue

			# find tokens in the span
			in_span = (seq_idx >= start_idx.unsqueeze(1)) & (seq_idx < end_idx.unsqueeze(1)) & (~key_padding_mask[batch_idx]) # numTrue x N

			# aggregate the span tokens for each sample, from numTrue x N --> Z x N, amax functions as an OR operation, since max is 1 for bool tensor
			span_mask.scatter_reduce_(0, batch_idx.unsqueeze(1).expand(-1, N), in_span, reduce="amax")

			# update the number of spans needed in next iter
			num_spans_per_iter = torch.clamp(torch.ceil(mask_pct * ((valid-span_mask.sum(dim=1)) / mean_span)), min=1, max=N).long().unsqueeze(1) # Z x 1

		# span mask is true for positions to inject, false otherwise. once inject the mask tokens, invert it to be consistent w/ other masks
		return span_mask

	def inject_mask(self, span_mask, predictions):

		batch, N, num_classes = predictions.shape

		rand_vals = torch.where(span_mask, torch.rand_like(span_mask, dtype=torch.float), float("inf")) # dont touch positions not included in span mask

		true_token = rand_vals<self.trueAA_pct
		rand_token = ~true_token & (rand_vals<(self.trueAA_pct+self.randAA_pct) )
		mask_token = ~true_token & ~rand_token & (rand_vals!=float("inf"))

		# predictions is already a onehot representation of labels, so dont need to add true tokens, as those are already present

		# add random token to predictions
		# high in in randint_like is exclusive, so only samples from non-mask tokens
		one_hot_rand_token = torch.nn.functional.one_hot(torch.randint((batch, N), low=0, high=num_classes-1), num_classes=num_classes)
		one_hot_mask_token = torch.nn.functional.one_hot(torch.full((batch, N), num_classes-1), num_classes=num_classes)

		predictions = torch.where(rand_token.unsqueeze(2), one_hot_rand_token, predictions)
		predictions = torch.where(mask_token.unsqueeze(2), one_hot_mask_token, predictions)

		return predictions

	def MASK_tokens(self, batch):

		span_mask = self.get_span_mask(batch.key_padding_mask)
		predictions = self.inject_mask(span_mask, batch.predictions) # assumes predictions is a one hot representation of labels
		
		batch.predictions, batch.onehot_mask = predictions, (~span_mask)
		