'''
title:			distogram_loss.py
author:			jaime cardenas
description:	computes distogram loss inside kernel to avoid materializing the NxNxB matrix. includes forward and bwd
'''

import triton
import triton.language as tl
import torch
import torch.nn as nn
import os
import random

# define configurations for autotuning
configs = [	triton.Config({"BLOCK_I": i, "BLOCK_J": j}, num_warps=w)
			for i in [1, 2, 4, 8, 16, 32]
			for j in [1, 2, 4, 8, 16, 32, 64]
			for w in [1, 2, 4, 8, 16, 32]
		]

# filter out configs that are too big
def keep_fwd(conf):
	autotune = os.environ.get("DIST_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 1) and (BLOCK_J == 1) and (conf.num_warps==1)) # lots of  smem used on bins*dk, only do block sizes of 1

@triton.autotune(list(filter(keep_fwd, configs)),
				 key=['tot_N', 'tot_Z', 'nbins', 'd_k'], # triton will not rerun autotune if these inputs are the same (size of input tensor)
				 restore_value=["out_ptr", "d_features_ptr"]) # make sure autotuning resets the outputs of this function for each configuration
@triton.jit
def _loss_fwd(
	out_ptr, stride_out_Z, stride_out_N,
	features_ptr, stride_features_Z, stride_features_N, stride_features_D,
	d_features_ptr, stride_d_features_Z, stride_d_features_N, stride_d_features_D,
	coords_ptr, stride_coords_Z, stride_coords_S, stride_coords_N,
	bin_proj_ptr, stride_bin_proj_D, stride_bin_proj_B, 
	d_bin_proj_ptr, stride_d_bin_proj_D, stride_d_bin_proj_B, 
	bins_ptr, stride_bins_B,
	mask_ptr, stride_mask_Z, stride_mask_N,

	tot_Z: tl.constexpr, tot_N: tl.constexpr, nbins: tl.constexpr, nbins_pad: tl.constexpr,
	d_k: tl.constexpr, label_smoothing: tl.constexpr,

	BLOCK_I: tl.constexpr, # block sizes, I is 1, for now, J is configurable, probably 16
	BLOCK_J: tl.constexpr
	):

	# ------------------------------------------------------------------------------------------------------------------
	# setup	
	# ------------------------------------------------------------------------------------------------------------------	

	# get block info
	offs_I = BLOCK_I*tl.program_id(0)
	offs_Z = tl.program_id(1)

	# ------------------------------------------------------------------------------------------------------------------
	# setup all the pointers
	# ------------------------------------------------------------------------------------------------------------------

	# init ptr for bins
	bins_block_ptr = tl.make_block_ptr(
		base=bins_ptr,
		shape=(nbins+1, ),
		strides=(stride_bins_B, ),
		offsets=(0, ),
		block_shape=(nbins_pad,),
		order=(0, )
	)

	# init ptr for bin projection
	bin_proj_block_ptr = tl.make_block_ptr(
		base=bin_proj_ptr + (offs_Z*stride_bin_proj_Z),
		shape=(d_k, nbins),
		strides=(stride_bin_proj_D, stride_bin_proj_B),
		offsets=(0, 0),
		block_shape=(d_k, nbins), # assume it is a power of 2 for now
		order=(0, 1)
	)

	# init ptr for coordinates for I rows
	coords_I_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z),
		shape=(3, tot_N),
		strides=(stride_coords_S, stride_coords_N),
		offsets=(0, offs_I),
		block_shape=(4, BLOCK_I), # tensor need to be power of 2, 4th value is masked (x,y,z,mask)
		order=(0, 1)
	)

	# init ptr for coordinates for J columns
	coords_J_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z),
		shape=(3, tot_N),
		strides=(stride_coords_S, stride_coords_N),
		offsets=(0, 0), # loop through J
		block_shape=(4, BLOCK_J),
		order=(0, 1)
	)

	# init ptr for mask for I
	mask_I_ptr = tl.make_block_ptr(
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I,), 
		order=(0, )
	)

	# init ptr for mask for J
	mask_J_ptr = tl.make_block_ptr(
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_J,), 
		order=(0, )
	)

	# init ptr for features of I
	features_I_ptr = tl.make_block_ptr(
		base=features_ptr + (offs_Z*stride_features_Z),
		shape=(tot_N, d_k),
		strides=(stride_features_N, stride_features_D),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, d_k),
		order=(0, 1)
	)

	# init ptr for features of I
	features_J_ptr = tl.make_block_ptr(
		base=features_ptr + (offs_Z*stride_features_Z),
		shape=(tot_N, d_k),
		strides=(stride_features_N, stride_features_D),
		offsets=(0, 0),
		block_shape=(BLOCK_J, d_k),
		order=(0, 1)
	)

	# ------------------------------------------------------------------------------------------------------------------
	# setup CEL info
	# ------------------------------------------------------------------------------------------------------------------

	bins = tl.load(bins_block_ptr, boundary_check=(0,), padding_option="zero")
	bin_min = tl.gather(bins, tl.arange(0, nbins), axis=0)[None, None, :]
	bin_max = tl.gather(bins, tl.arange(1, nbins+1), axis=0)[None, None, :]
	bin_proj = tl.load(bin_proj_block_ptr, boundary_check=(0,1), padding_option="zero")

	# ------------------------------------------------------------------------------------------------------------------
	# load i tensors
	# ------------------------------------------------------------------------------------------------------------------

	coords_I = tl.load(coords_I_ptr, boundary_check=(0,1), padding_option="zero")
	mask_I = tl.load(mask_I_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1)
	features_I = tl.load(features_I_ptr, boundary_check=(0,1), padding_option="zero")

	# ------------------------------------------------------------------------------------------------------------------
	# init smem tensors
	# ------------------------------------------------------------------------------------------------------------------

	out_smem = tl.zeros((BLOCK_I, ), dtype=tl.float32)
	d_features_smem = tl.zeros((BLOCK_I, d_k), dtype=tl.float32)

	# ------------------------------------------------------------------------------------------------------------------
	# loop through pairs
	# ------------------------------------------------------------------------------------------------------------------

	for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1, loop_unroll_factor=1): # no loop unrolling

		# --------------------------------------------------------------------------------------------------------------	
		# load j tensors
		# --------------------------------------------------------------------------------------------------------------	
		
		coords_J = tl.load(coords_J_ptr, boundary_check=(0,1), padding_option="zero")
		mask_J = tl.load(mask_J_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1)
		features_J = tl.load(features_J_ptr, boundary_check=(0,1), padding_option="zero")
		
		# --------------------------------------------------------------------------------------------------------------	
		# fwd computation 
		# --------------------------------------------------------------------------------------------------------------	
		
		# check which bin each pair belongs to and assign true probability to each bin
		dist_raw = coords_I[:, :, None] - coords_J[:, None, :]
		dist = tl.sqrt(tl.sum(dist_raw * dist_raw, axis=0))[:, :, None] # Ni x Nj x 1
		in_bin = (bin_min <= dist) & (bin_max > dist)  # Ni x Nj x B
		valid = mask_I[:, None, None] & mask_J[None, :, None]
		probs_true = valid*((1-label_smoothing)*in_bin + (label_smoothing/nbins))

		# compute predicted probs logits
		probs_logits = tl.dot(features_I[:, None, :] + features_J[None, :, :], bin_proj[None, :, :]) / 2.83 # will make this dependant on dk later# Ni x Nj x B
		
		# tl softmax gives error saying not allowed to specify dim, doing manual safe softmax
		probs_exp = tl.exp(probs_logits - tl.max(probs_logits, axis=2, keep_dims=True))
		probs_exp_sum = tl.sum(probs_exp, axis=2, keep_dims=True)
		probs_pred =  probs_exp / probs_exp_sum

		# compute cross entropy loss
		cel = -tl.sum(tl.sum(probs_true * tl.log(probs_pred), axis=1), axis=1)

		# store in smem
		out_smem += cel

		# --------------------------------------------------------------------------------------------------------------	
		# bwd computation
		# --------------------------------------------------------------------------------------------------------------	
		
		d_probs_logits = valid*(probs_pred - probs_true)
		d_features = tl.dot((d_probs_logits[:, :, :, None] * features_J[None, :, :, :]), axis=1) / 2.83 # sqrt 8
		d_features_smem += 2*d_features # times two, since summing the row and column contribution, only works if symmetric like in linear projection of sum of features

		d_bin_proj = 

		# --------------------------------------------------------------------------------------------------------------	
		# update pointers
		# --------------------------------------------------------------------------------------------------------------	
		
		# advance the J block pointers
		coords_J_ptr = tl.advance(coords_J_ptr, (0, BLOCK_J))
		mask_J_ptr = tl.advance(mask_J_ptr, (BLOCK_J, ))
		features_J_ptr = tl.advance(features_J_ptr, (BLOCK_J, 0, 0))

	# ------------------------------------------------------------------------------------------------------------------
	# setup output pointers
	# ------------------------------------------------------------------------------------------------------------------

	# init ptr for loss, computed for each I seperately to minimize contention, summed after
	out_block_ptr = tl.make_block_ptr(
		base=out_ptr + (offs_Z*stride_out_Z),
		shape=(tot_N, ),
		strides=(stride_out_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I,),
		order=(0, )
	)

	# init ptr for d_features, only done for I
	d_features_block_ptr = tl.make_block_ptr(
		base=d_features_ptr + (offs_Z*stride_d_features_Z),
		shape=(tot_N, d_k),
		strides=(stride_d_features_N, stride_d_features_D),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, d_k),
		order=(0, 1)
	)

	# init ptr for d_bin_proj
	d_bin_proj_block_ptr = d_bin_proj_ptr + (offs_Z*stride_d_bin_proj_Z)


		shape=(tot_N, nbins, d_k),
		strides=(stride_d_bin_proj_N, stride_d_bin_proj_B, stride_d_bin_proj_D),
		offsets=(offs_I, 0, 0),
		block_shape=(BLOCK_I, nbins, d_k),
		order=(0, 1, 2)
	)

	# ------------------------------------------------------------------------------------------------------------------
	# store results
	# ------------------------------------------------------------------------------------------------------------------

	tl.store(out_block_ptr, out_smem, boundary_check=(0,))
	tl.store(d_features_block_ptr, d_features_smem, boundary_check=(0,1))
	tl.atomic_add(d_bin_proj_block_ptr, d_bin_proj_smem, boundary_check=(0,1))

def distogram_loss(features=None, coords=None, bins=None, bin_proj=None, mask=None, label_smoothing=0.0):
	return _distogram_loss.apply(features, coords, bins, bin_proj, mask, label_smoothing)

class _distogram_loss(torch.autograd.Function):
	
	@staticmethod
	def forward(ctx, features, coords, bins, bin_proj, mask, label_smoothing):

		batch, N, d_model = features.shape
		
		features = (features).to(torch.float32).contiguous() # Z x N x B x Dk
		coords = coords.transpose(1, 2).to(torch.float32).contiguous() # Z x 3 x N for coalesced access
		bins = bins.to(torch.float32).contiguous() # B
		bin_proj = bin_proj.to(torch.float32).contiguous() # B
		mask = (~mask).contiguous() # Z x N
		
		d_features = torch.zeros_like(features).contiguous()
		d_bin_proj = torch.zeros_like(bin_proj).contiguous()
		out = torch.zeros_like(mask, dtype=torch.float32).contiguous()

		# define the grid
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"],
								1
							)

		_loss_fwd[grid](
			out, out.stride(0), out.stride(1), 
			features, features.stride(0), features.stride(1), features.stride(2),
			d_features, d_features.stride(0), d_features.stride(1), d_features.stride(2),
			coords, coords.stride(0), coords.stride(1), coords.stride(2),
			bin_proj, bin_proj.stride(0), bin_proj.stride(1), 
			d_bin_proj, d_bin_proj.stride(0), d_bin_proj.stride(1),
			bins, bins.stride(0),
			mask, mask.stride(0), mask.stride(1),
			batch, N, B, triton.next_power_of_2(B+1), d_k, label_smoothing
		)

		# each tokens loss is the average over the j other tokens, recommended to also scale this down outside function
		out = out / ((mask.sum(dim=1, keepdim=True))).clamp(min=1)
		d_features = d_features / ((mask.sum(dim=1))[:, None, None, None]).clamp(min=1) # divide by N**2 to keep same scale as seq cel
		# print(out, out.isnan().any())


		ctx.save_for_backward(d_features)

		return out.sum() 

	@staticmethod
	def backward(ctx, do):
		# precomputed in fwd
		d_features, = ctx.saved_tensors

		# do should just be 1
		return do*d_features, None, None, None, None

class DistogramLoss(nn.Module):
	def __init__(self, bins=64, min_dist=2.0, max_dist=22.0, label_smoothing=0.0):
		super(DistogramLoss, self).__init__()
		'''
		all dists below min dist are put in the lowest bin, all above in the highest bin, in between is linearly spaced
		'''

		# project the features to size dmodel/bins to get features for each bin
		# only learnable param, trying to make the attention heads work the hardest, not this module
		# ensures structure is encoded in the features, this module just has to decide which features correspond to which bin
		self.bins = torch.cat([torch.tensor([0]), torch.linspace(min_dist, max_dist, int(bins-1)), torch.tensor([float("inf")])], dim=0)
		self.label_smoothing = label_smoothing

	def forward(self, features, coords, mask=None):
		return distogram_loss(features, coords, self.bins.to(features.device), mask, self.label_smoothing) # scaler

		