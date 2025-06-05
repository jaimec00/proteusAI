# ----------------------------------------------------------------------------------------------------------------------
'''
author:			jaime cardenas
title:			flash_attn.py
description:	flash attention kernel written in triton, specifically for cross attention, only difference is that q and kv have diff masks
				kernel based on:
					FlashAttention2 paper: https://arxiv.org/abs/2307.08691
					Triton Implementation: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
				Also credits to Umar Jamil (@umarjamilai) for giving a fantastic exlanation and demo:
					YouTube Demo: https://www.youtube.com/watch?v=zy8ChVd_oTM
'''
# ----------------------------------------------------------------------------------------------------------------------

import math
import torch
import triton
import triton.language as tl
import os
import random

# hlper functions for dropout
@triton.jit
def hash5(a, b, c, d, seed):
	# PCG-XSH-RR based hash algorithm used for dropout
    h = seed
    h ^= (a * 0x85ebca6b) + 0x9e3779b9
    h ^= (b * 0xc2b2ae35) + 0x165667b1
    h ^= (c * 0x27d4eb2f) + 0xd6e8feb8
    h ^= (d * 0x85ebca6b) + 0x1b873593
    h = (h ^ (h >> 16)) * 0x85ebca6b
    h ^= (h >> 13)
    h = (h ^ (h >> 16)) * 0xc2b2ae35
    h ^= (h >> 16)
    return h

@triton.jit
def dropout(Z, H, I, J, seed, dropout_prob):
	# uses light hash algorithm for dropout to avoid cuda built in RNG
	# reproducible, but not necessary, since bwd tensor is precomputed in fwd
    h = hash5(Z, H, I, J, seed)
    normalized = h / (0xffffffff) # Convert to [0,1]
    return normalized < dropout_prob # is dropped


# define configurations for autotuning
configs = [	triton.Config({"BLOCK_I": i, "BLOCK_J": j}, num_warps=w)
			for i in [16, 32, 64]
			for j in [16, 32, 64]
			for w in [4, 8, 16]
		]

# filter out configs that are too big
def keep_fwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 64) and (BLOCK_J == 16) and (conf.num_warps==4))

def keep_bwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 32) and (BLOCK_J == 64) and (conf.num_warps==8))


@triton.autotune(list(filter(keep_fwd, configs)),
				 key=['tot_Nq', 'tot_Nkv', 'tot_Z', 'nheads', 'min_d_k'], # triton will not rerun autotune if these inputs are the same (size of input tensor)
				 restore_value=["O_ptr", "L_ptr"]) # make sure autotuning resets the outputs of this function for each configuration
@triton.jit
def _attn_fwd(
	O_ptr, stride_O_Z, stride_O_H, stride_O_N, stride_O_D,
	Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
	K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
	V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,
	L_ptr, stride_L_Z, stride_L_H, stride_L_N,
	mask_ptr, stride_mask_Z, stride_mask_N, 

	tot_Nq: tl.constexpr, tot_Nkv: tl.constexpr, tot_Z: tl.constexpr, 
	nheads: tl.constexpr, d_k: tl.constexpr, min_d_k: tl.constexpr,
	softmax_scale: tl.constexpr, 

	rng_ptr, dropout_p: tl.constexpr,

	BLOCK_I: tl.constexpr, # block sizes
	BLOCK_J: tl.constexpr,
):
	# get block info

	# get start index for query/output rows
	start_I = tl.program_id(0)

	# get the batch and head combo used for this block
	offs_ZH = tl.program_id(1)
	offs_Z = (offs_ZH // nheads)
	offs_H = (offs_ZH % nheads)

	# calculate offset of this block
	offs_I = start_I*BLOCK_I

	# create Q, K, and V block pointers
	Qi_block_ptr = tl.make_block_ptr( # N x d_k
		base=Q_ptr + (offs_Z*stride_Q_Z) + (offs_H*stride_Q_H),
		shape=(tot_Nq, d_k),
		strides=(stride_Q_N, stride_Q_D),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)


def get_neighbors(coords, neighbors=30, mask=None):
	'''wrapper for attn so can call it with kwargs'''

	return _get_neighbors.apply(coords, neighbors, mask)

class _get_neighbors(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coords, neighbors, mask):
		
        batch, N, _ = coords.shape

        coords = coords.contiguous()


		# initialize mask, output, and logsumexp tensors
		mask = torch.ones(batch, N, dtype=torch.bool, device=coords.device).contiguous() if mask is None else ~mask # only mask K
		out = torch.zeros(batch, N, neighbors, dtype=torch.int32, device=coords.device).contiguous() # batch x N x d_model
		
		# define the grid
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"]*args["nheads"],	
								1
							)


		# run the kernel
		_attn_fwd[grid](  	out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x Nkv x d_k
							Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x Nq x d_k
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x Nkv x d_k
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x Nkv x d_k
							L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x Nq
							mask, mask.stride(0), mask.stride(1),
							Nq, Nkv, batch, nheads, d_k, max(d_k, 16), softmax_scale,
							rng, dropout_p
						)

		# for backwards pass
		ctx.save_for_backward(Q, K, V, out, L, mask, rng)
		ctx.softmax_scale = softmax_scale
		ctx.dropout_p = dropout_p

		return out
