# ----------------------------------------------------------------------------------------------------------------------
'''
author:			jaime cardenas
title:			geometric_attn.py
description:	multi-scale geometric flash attention kernel written in triton. 
				kernel based on:
					FlashAttention2 paper: https://arxiv.org/abs/2307.08691
					Triton Implementation: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
				Also credits to Umar Jamil (@umarjamilai) for giving a fantastic exlanation and demo:
					YouTube Demo: https://www.youtube.com/watch?v=zy8ChVd_oTM

				Performs Flash attention, as described in the paper, but includes scaling of attention logits using RBF
				functions based on euclidean distances of alpha carbon pairs. each head uses a distinct spread to compute
				the RBFs, The spreads are learnable, and they interact multiplicitavely with the attention weights, allowing
				direct communication in both the forward and backward passes between RBFs and Q,K.  

				Forward and backward passes are fully implemented, including reproducable dropout masks with RNG seeds 
				for consistent bwd pass
'''
# ----------------------------------------------------------------------------------------------------------------------

import math
import torch
import triton
import triton.language as tl
import os
import random

# define configurations for autotuning
configs = [	triton.Config({"BLOCK_I": i, "BLOCK_J": j}, num_warps=w)
			for i in [16, 32, 64]
			for j in [16, 32, 64]
			for w in [1, 2, 4, 8, 16]
		]

# filter out configs that are too big
def keep_fwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 32) and (BLOCK_J == 16) and (conf.num_warps==2))

def keep_bwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 16) and (BLOCK_J == 32) and (conf.num_warps==2))


@triton.autotune(list(filter(keep_fwd, configs)),
				 key=['tot_N', 'tot_Z', 'nheads', 'min_d_k'], # triton will not rerun autotune if these inputs are the same (size of input tensor)
				 restore_value=["O_ptr", "L_ptr"]) # make sure autotuning resets the outputs of this function for each configuration
@triton.jit
def _attn_fwd(
	O_ptr, stride_O_Z, stride_O_H, stride_O_N, stride_O_D,
	Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
	K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
	V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,
	coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_S,
	spreads_ptr, stride_spreads_H,
	betas_ptr, stride_betas_H,
	L_ptr, stride_L_Z, stride_L_H, stride_L_N,
	mask_ptr, stride_mask_Z, stride_mask_N,

	tot_N: tl.constexpr, tot_Z: tl.constexpr, nheads: tl.constexpr,
	d_k: tl.constexpr, min_d_k: tl.constexpr,
	softmax_scale: tl.constexpr, 
	min_rbf: tl.constexpr, max_rbf: tl.constexpr,

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
		shape=(tot_N, d_k),
		strides=(stride_Q_N, stride_Q_D),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# transpose k when loading directly by flipping N and D,
	KjT_block_ptr = tl.make_block_ptr( # d_k x N
		base=K_ptr + (offs_Z*stride_K_Z) + (offs_H*stride_K_H),
		shape=(d_k, tot_N),
		strides=(stride_K_D, stride_K_N),
		offsets=(0, 0),
		block_shape=(min_d_k, BLOCK_J),
		order=(0, 1)
	)

	Vj_block_ptr = tl.make_block_ptr( # N x d_k
		base=V_ptr + (offs_Z*stride_V_Z) + (offs_H*stride_V_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(0, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# init ptr for coordinates for I rows
	coords_I_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, 4), # tensor need to be power of 2, 4th value is masked (x,y,z,mask)
		order=(0, 1)
	)

	# init coords J ptr, loaded in for loop
	coords_J_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(0, 0),
		block_shape=(BLOCK_J, 4),
		order=(0, 1)
	)

	# create mask pointer for Q/O rows and load
	mask_i_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# create K/V column mask pointer (loaded in the next loop)
	mask_j_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	# get pointers for spreads, min and max distances for this head
	spread_ptr = spreads_ptr + (offs_H*stride_spreads_H)
	beta_ptr = betas_ptr + (offs_H*stride_betas_H)
	rbf_range = max_rbf - min_rbf

	# load the Qi block first, out of bounds values are 0, stays in SRAM throughout
	Qi = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k (fp16)

	# initialize output and statistics block, all in fp32
	Oi = tl.zeros((BLOCK_I, min_d_k), dtype=tl.float32)
	li = tl.zeros((BLOCK_I, ), dtype=tl.float32)
	inf = float("inf") # convenience
	mi = (tl.zeros_like(li) - inf)

	# load spreads, min and max dist, and coords for rbf computation (fp32)
	spread = tl.load(spread_ptr)
	beta = tl.load(beta_ptr)

	coords_I = tl.load(coords_I_ptr, boundary_check=(0,1), padding_option="zero") # N x 4

	# load mask for rows
	mask_i = tl.load(mask_i_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # N x d_k

	# loop through columns of K and V
	for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1, loop_unroll_factor=1): # no loop unrolling

		# compute attn: QK^T/sqrt(d_k). # both in fp16, dot outputs fp32
		Sij = tl.dot(Qi, tl.load(KjT_block_ptr, boundary_check=(0,1), padding_option="zero")) * softmax_scale # N x N

		# load coordinates and compute distances (fp32)
		dists_raw = (coords_I[:, None, :] - tl.load(coords_J_ptr, boundary_check=(0,1), padding_option="zero")[None, :, :]) # N x N x 4
		dists = tl.sqrt(tl.sum(dists_raw * dists_raw, axis=2)) # N x N

		# compute the rbfs
		Rij = tl.exp(-(dists*dists) / (2*spread*spread)) # N x N (fp32)

		# set masked positions to -inf, include out of range dists in mask
		attn_mask = (mask_i[:, None]) & (tl.load(mask_j_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1)[None, :]) & (Rij <= max_rbf) & (Rij >= min_rbf) # N x N

		# scale attention logits by Rij and mask invalid pairs
		Rij_norm = ((2*(Rij-min_rbf)/rbf_range) - 1)
		SRij = tl.where(attn_mask, Sij + beta*Rij_norm, -inf) # N x N (fp32)

		# max of each row
		mij = tl.maximum(mi, tl.max(SRij, axis=1)) # N,  (fp32)

		# compute softmax(SRij - mij) = Pij
		Pij = tl.exp(tl.where(mij[:, None]==-inf, -inf, SRij - mij[:, None])) # N x N (fp32)

		# compute alpha
		alpha = tl.exp(tl.where((mi==-inf) | (mij==-inf), tl.where((mi==-inf) & (mij==-inf), 0, -inf), mi - mij)) # (fp32)

		# update li
		li = alpha*li + tl.sum(Pij, axis=1) # N, (fp32)

		# load Vj compute output. convert Pij to fp16, Vj is already in fp16. output is in fp32
		Oi = tl.dot(Pij.to(tl.float16), tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero"), acc=(Oi*alpha[:, None])) # N x d_k

		# update statistics for next iteration
		mi = mij

		# advance block pointers for columns
		KjT_block_ptr = tl.advance(KjT_block_ptr, (0, BLOCK_J))
		Vj_block_ptr = tl.advance(Vj_block_ptr, (BLOCK_J, 0))
		coords_J_ptr = tl.advance(coords_J_ptr, (BLOCK_J, 0))
		mask_j_ptr = tl.advance(mask_j_ptr, (BLOCK_J, ))

	# epilogue

	# normalize output. li==0 means that all columns in that row are masked out
	Oi = tl.where(li[:, None]!=0, Oi / li[:, None], 0.0)

	# compute log sum exponential (li==0 --> mi + log(li) = -inf)
	mi += tl.log(li)

	# create output block pointer
	Oi_block_ptr = tl.make_block_ptr( # N x d_k
		base=O_ptr + (offs_Z*stride_O_Z) + (offs_H*stride_O_H),
		shape=(tot_N, d_k),
		strides=(stride_O_N, stride_O_D),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# create log sum exp pointer
	Li_block_ptr = tl.make_block_ptr( # N,
		base=L_ptr + (offs_Z*stride_L_Z) + (offs_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# store output and logsum exp
	tl.store(Oi_block_ptr, Oi, boundary_check=(0,1))
	tl.store(Li_block_ptr, mi, boundary_check=(0,))

@triton.autotune(list(filter(keep_bwd, configs)), 
				key=['tot_N', 'tot_Z', 'nheads', 'min_d_k'],
				restore_value=["dQ_ptr", "dK_ptr", "dV_ptr", "d_spreads_ptr"])
@triton.jit
def _attn_bwd(
	Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
	K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
	V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,
	dO_ptr, stride_dO_Z, stride_dO_H, stride_dO_N, stride_dO_D,
	dQ_ptr, stride_dQ_Z, stride_dQ_H, stride_dQ_N, stride_dQ_D,
	dK_ptr, stride_dK_Z, stride_dK_H, stride_dK_N, stride_dK_D,
	dV_ptr, stride_dV_Z, stride_dV_H, stride_dV_N, stride_dV_D,
	D_ptr, stride_D_Z, stride_D_H, stride_D_N,
	L_ptr, stride_L_Z, stride_L_H, stride_L_N,
	coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_S,
	spreads_ptr, stride_spreads_H,
	betas_ptr, stride_betas_H,
	d_spreads_ptr, stride_d_spreads_H,
	d_betas_ptr, stride_d_betas_H,
	mask_ptr, stride_mask_Z, stride_mask_N,
	
	tot_Z: tl.constexpr, tot_N: tl.constexpr, nheads: tl.constexpr, 
	d_k: tl.constexpr, min_d_k: tl.constexpr, softmax_scale: tl.constexpr,
	min_rbf: tl.constexpr, max_rbf: tl.constexpr,

	BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
	# get the column offset that this block will work with
	start_J = tl.program_id(0)
	offs_J = start_J * BLOCK_J

	# get batch head combo that this block will work with
	offs_ZH = tl.program_id(1)
	offs_Z = offs_ZH // nheads
	offs_H = offs_ZH % nheads

	# prep pointers (KjT and Vj stay in SRAM throughout)
	KjT_ptr = tl.make_block_ptr( # d_k x N (transpose on load)
		base=K_ptr + (offs_Z*stride_K_Z) + (offs_H*stride_K_H),
		shape=(d_k, tot_N),
		strides=(stride_K_D, stride_K_N),
		offsets=(0, offs_J),
		block_shape=(min_d_k, BLOCK_J),
		order=(0, 1)
	)

	Vj_ptr = tl.make_block_ptr( # N x d_k
		base=V_ptr + (offs_Z*stride_V_Z) + (offs_H*stride_V_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# load coordinates for J columns
	coords_j_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, 4),
		order=(0, 1)		
	)

	# spread and dist pointers
	spread_ptr = spreads_ptr + (offs_H*stride_spreads_H)
	beta_ptr = betas_ptr + (offs_H*stride_betas_H)

	# initialize mask pointer for j columns 
	mask_j_ptr = tl.make_block_ptr( # N 
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(offs_J, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	# initialize pointers for Qi, dQi, dOi, Li, and Di. only loaded within loop
	Qi_block_ptr = tl.make_block_ptr( # N x d_k
		base=Q_ptr + (offs_Z*stride_Q_Z) + (offs_H*stride_Q_H),
		shape=(tot_N, d_k),
		strides=(stride_Q_N, stride_Q_D),
		offsets=(0, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# perform atomic adds on dQi, which don't support block pointers, so do manual indexing
	dQi_start_ptr = dQ_ptr + (offs_Z*stride_dQ_Z) + (offs_H*stride_dQ_H)
	dQi_block_ptr = dQi_start_ptr + (tl.arange(0,BLOCK_I)[:, None]*stride_dQ_N) + (tl.arange(0,min_d_k)[None, :]*stride_dQ_D)

	dOi_block_ptr = tl.make_block_ptr( # N x d_k
		base=dO_ptr + (offs_Z*stride_dO_Z) + (offs_H*stride_dO_H),
		shape=(tot_N, d_k),
		strides=(stride_dO_N, stride_dO_D),
		offsets=(0, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	Li_block_ptr = tl.make_block_ptr( # N
		base=L_ptr + (offs_Z*stride_L_Z) + (offs_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	Di_block_ptr = tl.make_block_ptr( # N
		base=D_ptr + (offs_Z*stride_D_Z) + (offs_H*stride_D_H),
		shape=(tot_N, ),
		strides=(stride_D_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	coords_i_ptr = tl.make_block_ptr(
		base=coords_ptr + (offs_Z*stride_coords_Z) ,
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(0, 0),
		block_shape=(BLOCK_I, 4),
		order=(0, 1)	
	)

	# initialize mask for i rows
	mask_i_ptr = tl.make_block_ptr( # N 
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# load the spread assigned to this block (based on the head it was assigned), and j coordinates
	spread = tl.load(spread_ptr)
	beta = tl.load(beta_ptr)
	rbf_range = max_rbf - min_rbf
	coords_j = tl.load(coords_j_ptr, boundary_check=(0,1), padding_option="zero")

	# load mask
	mask_j = tl.load(mask_j_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)

	# initialize dKj and dVj and d_spread(scalar), in fp32
	dKj = tl.zeros((BLOCK_J, min_d_k), dtype=tl.float32)
	dVj = tl.zeros((BLOCK_J, min_d_k), dtype=tl.float32)
	d_spread = 0.0
	d_beta = 0.0

	# load KjT and Vj (load as fp16 for matmul)
	KjT = tl.load(KjT_ptr, boundary_check=(0, 1), padding_option="zero")
	Vj = tl.load(Vj_ptr, boundary_check=(0, 1), padding_option="zero")

	inf = float("inf") # convenience
	for i in tl.range(0, triton.cdiv(tot_N, BLOCK_I), 1, loop_unroll_factor=1): # no loop unrolling

		# fp16, N x d_k
		Qi = tl.load(Qi_block_ptr, boundary_check=(0, 1), padding_option="zero")

		# load Qi and compute attn, ie Sij. inputs are fp16, Sij is fp32
		Sij = tl.dot(Qi, KjT) * softmax_scale # N x N

		# load coordinates for i rows and compute distances, in fp32
		dists_raw = (tl.load(coords_i_ptr, boundary_check=(0,1), padding_option="zero"))[:, None, :] - coords_j[None, :, :] # N x N x 4 
		dists = tl.sqrt(tl.sum(dists_raw * dists_raw, axis=2)) # N x N 

		# compute rbfs (fp32)
		Rij = tl.exp(-(dists*dists) / (2.0*spread*spread)) # N x N (fp32)

		# mask out attention that is not relevant to this head
		mask_i = tl.load(mask_i_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)
		attn_mask = (mask_i[:, None]) & (mask_j[None, :]) & (Rij <= max_rbf) & (Rij >= min_rbf) # N x N

		# scale attention logits by RBFs
		Rij_norm = ((2*(Rij-min_rbf)/rbf_range) - 1)
		SRij = tl.where(attn_mask, Sij + beta*Rij_norm, -inf) # N x N (fp32)

		# load log sum exp statistics
		Li = tl.load(Li_block_ptr, boundary_check=(0, ), padding_option="zero") # (fp32)

		# exp(Sij - Lij) = exp(Sij - mi - log(li)) = exp(Sij - mi) / exp(log(li)) 
		# = exp(Sij - mi) / li
		# mi is max for the row pre-softmax (for safe softmax), li is the normalizing term (sum of exponentials for that row)
		Pij = tl.exp(tl.where(attn_mask, SRij - Li[:, None], -inf)) # N x N (fp32)

		# load gradient w.r.t output (fp16)
		dOi = tl.load(dOi_block_ptr, boundary_check=(0, 1), padding_option="zero") # N x d_k

		# compute gradient wrt Vj (dOi already in fp16, out is in fp32)
		dVj += tl.where(mask_j[:, None], tl.dot(tl.permute(Pij, (1,0)).to(tl.float16), dOi), 0.0) # N x d_k

		# compute gradient wrt Pij (dOi and Vj already in fp16)
		dPij = tl.dot(dOi, tl.permute(Vj, (1,0))) # N x N

		# load Di = rowsum(O*dO) to compute gradient wrt SRij (grad of loss wrt Sij*rbf)
		dSRij = Pij * (dPij -  tl.load(Di_block_ptr, boundary_check=(0, ), padding_option="zero")[:, None]) # N x N

		# compute dSij, ie grad wrt Sij. 
		dSij = dSRij # N x N

		# compute gradient wrt rbfs. also direct communication between dRij and Sij
		dRij = dSRij * 2 * beta / rbf_range
		d_beta += dSRij * Rij_norm

		# compute the gradient wrt the spread of this head
		# 		d_rbfs/dspreads  = d/dspreads exp(-(d^2)/(2*sigma^2))
		# 		= [d/dspreads -(d^2)/(2*sigma^2)] * exp(-(d^2)/(2*sigma^2))
		# 		= [d/dspreads -(d^2)/(2*sigma^2)] * rbfs
		# 		= [(d^2)/(sigma^3)] * rbfs
		d_spread_factor = (dists*dists)/(spread*spread*spread)

		# accumulate the gradients for this head's spread
		d_spread += tl.sum(tl.where(attn_mask, dRij * Rij * d_spread_factor , 0.0))

		# compute gradient wrt Qij and perform atomic add to communicate between thread blocks (Kj already in fp16)
		dQi = tl.dot(dSij.to(tl.float16), tl.permute(KjT, (1,0))) * softmax_scale # N x d_k		
		dQi_mask = mask_i[:, None] & (tl.arange(0,min_d_k)[None, :] < d_k)
		tl.atomic_add(dQi_block_ptr, dQi, mask=dQi_mask)

		# compute gradients wrt Kj (Qi already in fp16)
		dKj += tl.where(mask_j[:, None], tl.dot(tl.permute(dSij, (1,0)).to(tl.float16), Qi) * softmax_scale, 0.0)   # N x d_k

		# advance the pointers
		Qi_block_ptr = tl.advance(Qi_block_ptr, (BLOCK_I, 0))
		dQi_block_ptr += BLOCK_I*stride_dQ_N
		dOi_block_ptr = tl.advance(dOi_block_ptr, (BLOCK_I, 0))
		Li_block_ptr = tl.advance(Li_block_ptr, (BLOCK_I, ))
		Di_block_ptr = tl.advance(Di_block_ptr, (BLOCK_I, ))
		coords_i_ptr = tl.advance(coords_i_ptr, (BLOCK_I, 0))
		mask_i_ptr = tl.advance(mask_i_ptr, (BLOCK_I, ))


	# initialize dK and dV pointers to write output
	dKj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dK_ptr + (offs_Z*stride_dK_Z) + (offs_H*stride_dK_H),
		shape=(tot_N, d_k),
		strides=(stride_dK_N, stride_dK_D),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	dVj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dV_ptr + (offs_Z*stride_dV_Z) + (offs_H*stride_dV_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# write dK and dV to HBM, dQ was written to HBM progressively in for loop via atomic adds
	tl.store(dKj_block_ptr, dKj, boundary_check=(0,1))
	tl.store(dVj_block_ptr, dVj, boundary_check=(0,1))

	# store the grad wrt to spread for this head
	d_spread_ptr = d_spreads_ptr + (offs_H*stride_d_spreads_H)
	d_betas_ptr = d_betas_ptr + (offs_H*stride_d_betas_H)
	tl.atomic_add(d_spread_ptr, d_spread)
	tl.atomic_add(d_betas_ptr, d_beta)

def geometric_attn(Q, K, V, coords, spreads, betas, mask=None, min_rbf=0.00, max_rbf=1.00):
	'''wrapper for attn so can call it with kwargs'''

	return _geometric_attn.apply(Q, K, V, coords, spreads, betas, mask, min_rbf, max_rbf)

class _geometric_attn(torch.autograd.Function):

	@staticmethod
	def forward(ctx, Q, K, V, coords, spreads, betas, mask=None, min_rbf=0.01, max_rbf=0.99):
		
		# checks
		assert Q.shape == K.shape == V.shape, f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		batch, nheads, N, d_k = Q.shape
		d_model = nheads*d_k
		softmax_scale = 1/(d_k**0.5)
		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
		assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

		assert spreads.size(0) == nheads, f"number of spreads per batch must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
		assert torch.all(spreads > 0), f"spreads must be a tensor of positive, non-zero floats, not {spreads}"

		# matmults done in fp16
		Q = Q.to(torch.float16).contiguous()
		K = K.to(torch.float16).contiguous()
		V = V.to(torch.float16).contiguous()

		# rbfs in fp32 and make sure everything is contiguous
		coords = coords.to(torch.float32).contiguous()
		spreads = spreads.to(torch.float32).contiguous() # this is now Z x H, need to update everything (forward and back)
		betas = betas.to(torch.float32).contiguous() 

		# initialize mask, output, and logsumexp tensors
		mask = (torch.ones(batch, N, dtype=torch.bool, device=Q.device) if mask is None else ~mask).contiguous() # batch x N		
		out = torch.zeros(batch, nheads, N, d_k, dtype=torch.float32, device=Q.device).contiguous() # batch x N x d_model
		L = torch.zeros(batch, nheads, N, dtype=torch.float32, device=Q.device).contiguous() # batch x nheads x N
		
		# define the grid
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"]*args["nheads"],	
								1
							)

		# run the kernel
		_attn_fwd[grid](  	out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x N x d_k
							Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
							coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
							spreads, spreads.stride(0), # nhead,
							betas, betas.stride(0), # nhead,
							L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x N
							mask, mask.stride(0), mask.stride(1), # batch x N
							N, batch, nheads, d_k, max(d_k, 16), softmax_scale,
							min_rbf, max_rbf
						)

		# for backwards pass
		ctx.save_for_backward(Q, K, V, out, L, coords, spreads, betas, mask)
		ctx.softmax_scale = softmax_scale
		ctx.min_rbf = min_rbf
		ctx.max_rbf = max_rbf

		return out

	@staticmethod
	def backward(ctx, dO):

		# load saved tensors (should all be float32, expect masks). also should all be contiguous from fwd
		Q, K, V, O, L, coords, spreads, betas, mask = ctx.saved_tensors

		# compute D for dSR calculation
		D = torch.sum(O*dO, dim=3).to(torch.float16) # Z x H x N x D -> Z x H x N

		# cast to float16 for matmults
		dO = dO.to(torch.float16).contiguous()

		# checks
		assert Q.stride() == K.stride() == V.stride() == O.stride()
		batch, nheads, N, d_k = Q.shape

		# initialize dQ, dK, and dV, all fp32
		dQ = torch.zeros_like(Q).to(torch.float32).contiguous()
		dK = torch.zeros_like(K).to(torch.float32).contiguous()
		dV = torch.zeros_like(V).to(torch.float32).contiguous()
		d_spreads = torch.zeros_like(spreads).to(torch.float32).contiguous()
		d_betas = torch.zeros_like(betas).to(torch.float32).contiguous()

		# define the grid
		grid = lambda args: (
			triton.cdiv(args["tot_N"], args["BLOCK_J"]), # parralel along J for bwd
			args["tot_Z"]*args["nheads"],
			1
		)

		# run the bwd kernel
		_attn_bwd[grid](	Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3),
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3),
							dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
							dQ, dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
							dK, dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
							dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
							D, D.stride(0), D.stride(1), D.stride(2),
							L, L.stride(0), L.stride(1), L.stride(2),
							coords, coords.stride(0), coords.stride(1), coords.stride(2),
							spreads, spreads.stride(0),
							betas, betas.stride(0),
							d_spreads, d_spreads.stride(0),
							d_betas, d_betas.stride(0),
							mask, mask.stride(0), mask.stride(1),
							rng_seed,
							batch, N, nheads, d_k, max(d_k, 16), ctx.softmax_scale,
							ctx.min_rbf, ctx.max_rbf
						 )

		# return the gradients
		return dQ, dK, dV, None, d_spreads, d_betas, None, None, None
