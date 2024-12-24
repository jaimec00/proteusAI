# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		utils.py
description:	utility functions
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import math
import triton
import triton.language as tl


@triton.jit
def _protein_to_wavefunc_kernel(
		out_ptr,
		stride_out_Z,
		stride_out_N,
		stride_out_D,

		coords_ptr,
		stride_coords_Z,
		stride_coords_N,
		stride_coords_space,

		wavenumber_ptr, 
		stride_wavenumber,

		mask_ptr,
		stride_mask_Z,
		stride_mask_N,

		tot_batch: tl.constexpr,
		tot_N: tl.constexpr,
		d_model:tl.constexpr, 
		num_wn:tl.constexpr,
		pi:tl.constexpr,

		BLOCK_NI: tl.constexpr,
		BLOCK_NJ: tl.constexpr,
		BLOCK_D: tl.constexpr
):

	# get i, j indices
	NI_start = tl.program_id(0)
	NI_offs = (NI_start.to(tl.int64)*BLOCK_NI).to(tl.int32)
	
	NJ_start = tl.program_id(1)
	NJ_offs = (NJ_start.to(tl.int64)*BLOCK_NJ).to(tl.int32)

	# get batch and wavenumber indices
	Z = tl.program_id(2)

	coords_NI_ptr = tl.make_block_ptr( # NI x 3
		base=coords_ptr + (Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_space),
		offsets=(NI_offs, 0),
		block_shape=(BLOCK_NI, 4), # 4th value is masked, tensor needs to be power of two (x,y,z,masked)
		order=(0, 1)
	)
	mask_NI_ptr = tl.make_block_ptr( # NI,
		base=mask_ptr + (Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(NI_offs, ),
		block_shape=(BLOCK_NI, ),
		order=(0, )
	)
	mask_NI = tl.load(mask_NI_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # NI

	
	coords_NJ_ptr = tl.make_block_ptr( # NJ x 3
		base=coords_ptr + (Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_space),
		offsets=(NJ_offs, 0),
		block_shape=(BLOCK_NJ, 4), # 4th value is masked, tensor needs to be power of two (x,y,z,masked)
		order=(0, 1)
	)
	mask_NJ_ptr = tl.make_block_ptr( # NJ,
		base=mask_ptr + (Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(NJ_offs, ),
		block_shape=(BLOCK_NJ, ),
		order=(0, )
	)
	mask_NJ = tl.load(mask_NJ_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # NJ

	mask_IJ = mask_NI[:, None] & mask_NJ[None, :] # NJ x NJ
	mask_IJ = mask_IJ & (  (NI_offs + tl.arange(0, BLOCK_NI)[:, None]) != (NJ_offs + tl.arange(0, BLOCK_NJ)[None, :])   ).to(tl.int1) # NI x NJ

	coords_NI = tl.load(coords_NI_ptr, boundary_check=(0,1), padding_option="zero") # N x 3
	coords_NJ = tl.load(coords_NJ_ptr, boundary_check=(0,1), padding_option="zero") # N x 3

	dists_raw = coords_NI[:, None, :] - coords_NJ[None, :, :] # NI x NJ x 3
	dists = tl.sqrt(tl.sum(dists_raw * dists_raw, axis=2)) # NI x NJ

	wavenumber_block_ptr = tl.make_block_ptr( # D,
		base=wavenumber_ptr,
		shape=(num_wn, ),
		strides=(stride_wavenumber, ),
		offsets=(0, ),
		block_shape=(BLOCK_D, ),
		order=(0, )
	)

	# initilize output pointer. atomic add does not support block pointers
	# also need to add dummy NJ dimension. will be superposing along NJ and writing to 
	# NI, so only use 1st NJ thread per NI to write output and avoid overlapping atomic adds
	NI_out = NI_offs + tl.arange(0, BLOCK_NI)[:, None, None]
	NJ_out = tl.zeros(( 1, BLOCK_NJ, 1), dtype=tl.int32)
	D_out = tl.arange(0, 2*BLOCK_D)[None, None, :]
	out_ptr = out_ptr + Z*stride_out_Z + (NI_out*stride_out_N) + (0*NJ_out) + (D_out*stride_out_D)

	# mask out NJ threads expect one per NI (reducing alogn NJ)
	output_mask = mask_NI[:, None, None] & (((NJ_start + tl.arange(0, BLOCK_NJ)[None, :, None])%BLOCK_NJ)==0)

	for d in tl.range(0, triton.cdiv(num_wn, BLOCK_D), 1):

		# load wavenumbers
		wavenumber = tl.load(wavenumber_block_ptr, boundary_check=(0,), padding_option="zero") # D, 

		# compute phase
		phase = (dists[:, :, None]*wavenumber[None, None, :]) % (2*pi) # NI x NJ x D

		# compute real and imag parts
		real = tl.cos(phase) / tl.where(mask_IJ[:, :, None], dists[:, :, None], float("inf")) # NI x NJ x D
		imag = tl.sin(phase) / tl.where(mask_IJ[:, :, None], dists[:, :, None], float("inf")) # NI x NJ x D

		# superpose real and imag parts in the tile
		real_superposition = tl.sum(real, axis=1) # NI x D 
		imag_superposition = tl.sum(imag, axis=1) # NI x D

		# interleave into features -> N x d_model
		features = tl.interleave(real_superposition, imag_superposition)

		# make sure d_model dimension is less than d_model
		mask_D = ((d*2*BLOCK_D + tl.arange(0, 2*BLOCK_D)) < d_model).to(tl.int1)
		D_output_mask = output_mask & mask_D[None, None, :]
		features = tl.where(D_output_mask, features[:, None, :], 0.0)

		# add features to output tensor
		tl.atomic_add(out_ptr, features, mask=D_output_mask)

		# advance pointers
		wavenumber_block_ptr = tl.advance(wavenumber_block_ptr, (BLOCK_D, ))
		out_ptr += 2*BLOCK_D

class protein_to_wavefunc(torch.autograd.Function):

	@staticmethod # might make wavelengths learnable and make a backward pass, but not focusing on MHA kernel first
	def forward(ctx, coords, d_model, min_wl, max_wl, base, mask=None):

		# checks
		assert (coords.dim() == 3) and (coords.size(2) == 3), f"coords must be of shape (batch x N x 3), not {coords.shape}"
		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model}"
		
		# prepare data
		batch, N, space = coords.shape # input dimensions
		coords = coords.to(torch.float64).contiguous() # double-precision and contiguous

		# prepare the wavenumber values
		num_wl = int(d_model//2) # define the number of wave functions to compute
		wavelengths = (min_wl + (torch.logspace(0, 1, num_wl, base=base, device=coords.device, dtype=torch.float64) - 1) / (base - 1) * (max_wl - min_wl))
		wavenumbers = (2 * torch.pi / wavelengths).contiguous()

		# prepare the mask, triton uses true as compute
		mask = (~mask if mask is not None else torch.ones(batch, N, dtype=torch.bool, device=coords.device)).contiguous()
		
		# prepare output
		out = torch.zeros(batch, N, d_model, dtype=torch.float64, device=coords.device).contiguous()

		# total block size should be less than number of threads per block (approx. 1024)
		BLOCK_NJ = min(1024, triton.next_power_of_2(N))
		BLOCK_NI = 1024 // BLOCK_NJ    # N_i
		BLOCK_D = 1
		# BLOCK_NI x BLOCK_NJ <= 1024

		# define the grid
		grid = lambda args: (	triton.cdiv(args["tot_N"], args["BLOCK_NI"]), 
								triton.cdiv(args["tot_N"], args["BLOCK_NJ"]), 
								args["tot_batch"]
							)

		# run the kernel
		_protein_to_wavefunc_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2),
											coords, coords.stride(0), coords.stride(1), coords.stride(2),
											wavenumbers, wavenumbers.stride(0),
											mask, mask.stride(0), mask.stride(1),
											batch, N, d_model, num_wl, torch.pi,
											BLOCK_NI, BLOCK_NJ, BLOCK_D
										)

		# normalize each feature by the maximum absolute value in the sample
		out.div_(out.abs().max(dim=1, keepdim=True).values)

		return out

@triton.jit
def _triton_attn_kernel(
	O_ptr, # output
	stride_O_Z,
	stride_O_H,
	stride_O_N,
	stride_O_D,

	Q_ptr, # query
	stride_Q_Z,
	stride_Q_H,
	stride_Q_N,
	stride_Q_D,
	
	K_ptr, # key
	stride_K_Z,
	stride_K_H,
	stride_K_N,
	stride_K_D,
	
	V_ptr, # value
	stride_V_Z,
	stride_V_H,
	stride_V_N,
	stride_V_D,

	coords_ptr, # 3d coordinates
	stride_coords_Z,
	stride_coords_N,
	stride_coords_S,

	spread_ptr, # head specific spreads
	stride_spread_H,

	L_ptr, # log sum exponential
	stride_L_Z,
	stride_L_H,
	stride_L_N,

	mask_ptr, # mask
	stride_mask_Z,
	stride_mask_N,

	tot_N: tl.constexpr, # constants
	tot_Z: tl.constexpr,
	nheads: tl.constexpr,
	d_k: tl.constexpr,
	min_d_k: tl.constexpr, # max(16, d_k) bc tl.dot requires dim>=16
	softmax_scale: tl.constexpr,

	BLOCK_I: tl.constexpr, # block sizes
	BLOCK_J: tl.constexpr,
):
	# get block info

	# get start index for query/output rows
	start_I = tl.program_id(0)

	# get the batch and head combo used for this block
	start_ZH = tl.program_id(1) # note that exactly batch*nheads processes launched along y axis, so no need to mask this
	start_Z = (start_ZH // nheads).to(tl.int64)
	start_H = (start_ZH % nheads).to(tl.int64)

	# calculate offset of this block
	I_offs = (start_I.to(tl.int64)*BLOCK_I).to(tl.int32)

	# create Q, K, and V block pointers
	Qi_block_ptr = tl.make_block_ptr( # N x d_k
		base=Q_ptr + (start_Z*stride_Q_Z) + (start_H*stride_Q_H),
		shape=(tot_N, d_k),
		strides=(stride_Q_N, stride_Q_D),
		offsets=(I_offs, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# transpose k when loading directly by flipping N and D, as well as the Nstride and Dstride
	KjT_block_ptr = tl.make_block_ptr( # d_k x N
		base=K_ptr + (start_Z*stride_K_Z) + (start_H*stride_K_H),
		shape=(d_k, tot_N),
		strides=(stride_K_D, stride_K_N),
		offsets=(0, 0),
		block_shape=(min_d_k, BLOCK_J),
		order=(0, 1)
	)

	Vj_block_ptr = tl.make_block_ptr( # N x d_k
		base=V_ptr + (start_Z*stride_V_Z) + (start_H*stride_V_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(0, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# load the Qi block first, out of bounds values are 0, stays in SRAM throughout
	Qi = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k

	# initialize output and statistics block
	Oi = tl.zeros_like(Qi)
	li = tl.zeros((BLOCK_I, ), dtype=tl.float32) #+ 1.0
	inf = float("inf")
	mi = tl.zeros_like(li) - inf

	# create mask pointer for Q/O rows. loading already masks out-of-bounds, 
	# but also make custom mask to set masked vals to -inf in attention mechanism
	mask_i_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(I_offs, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)
	mask_i = tl.load(mask_i_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # N x d_k

	# create K/V column mask pointer (loaded in the next loop)
	mask_j_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	# loop through columns of K and V
	for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1):

		# load K^T and the mask
		KjT = tl.load(KjT_block_ptr, boundary_check=(0,1), padding_option="zero") # d_k x N
		mask_j = tl.load(mask_j_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # N

		# QKT/sqrt(d_k)
		Sij = tl.dot(Qi, KjT) * softmax_scale # N x N

		# set masked positions to -inf
		Sij = tl.where(mask_i[:, None] & mask_j[None, :], Sij, -inf) # N x N

		# max pf each row
		mij = tl.maximum(mi, tl.max(Sij, axis=1)) # N, 

		# compute Pij
		Pij = tl.exp(tl.where(mij[:, None]==-inf, -inf, Sij - mij[:, None])) # N x N
		lij = tl.sum(Pij, axis=1)

		# compute alpha
		alpha = tl.exp(tl.where((mi==-inf) | (mij==-inf), tl.where((mi==-inf) & (mij==-inf), 0, -inf), mi - mij))
		
		# compute lij
		li = alpha*li + lij # N, 

		# update output
		Oi = Oi*alpha[:, None]

		# load Vj
		Vj = tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k
		Oi = tl.dot(Pij, Vj, Oi) # N x d_k

		# update statistics
		mi = mij

		# advance block pointers for columns
		KjT_block_ptr = tl.advance(KjT_block_ptr, (0, BLOCK_J))
		Vj_block_ptr = tl.advance(Vj_block_ptr, (BLOCK_J, 0))
		mask_j_ptr = tl.advance(mask_j_ptr, (BLOCK_J, ))
	
	# epliogue

	# normalize output
	Oi = tl.where(mask_i[:, None], Oi / li[:, None], 0)

	# compute log sum exponential
	mi += tl.log(li)
	mi = tl.where(mask_i, mi, -inf)

	# create output block pointer
	Oi_block_ptr = tl.make_block_ptr( # N x d_k
		base=O_ptr + (start_Z*stride_O_Z) + (start_H*stride_O_H),
		shape=(tot_N, d_k),
		strides=(stride_O_N, stride_O_D),
		offsets=(I_offs, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# create log sum exp pointer
	Li_block_ptr = tl.make_block_ptr( # N,
		base=L_ptr + (start_Z*stride_L_Z) + (start_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(I_offs, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# store output and logsum exp
	tl.store(Oi_block_ptr, Oi, boundary_check=(0,1))
	tl.store(Li_block_ptr, mi, boundary_check=(0,))

class triton_attn(torch.autograd.Function):

	@staticmethod
	def forward(ctx, Q, K, V, coords, spreads, mask=None, dist_factor=3.0):
		
		# checks
		assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		batch, nheads, N, d_k = Q.shape
		d_model = nheads*d_k

		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
		assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

		assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
		assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"

		# initialize mask, output, and logsumexp tensors
		mask = (torch.ones(batch, N, device=Q.device) if mask is None else ~mask).contiguous() # batch x N
		out = torch.zeros(batch, nheads, N, d_k, device=Q.device).contiguous() # batch x N x d_model
		L = torch.zeros(batch, nheads, N, device=Q.device).contiguous() # batch x nheads x N

		# make sure everything is contiguous
		Q = Q.contiguous()
		K = K.contiguous()
		V = V.contiguous()
		coords = coords.contiguous()
		spreads = spreads.contiguous()

		# define block sizes (minimum of 16, as tl.dot needs all dimensions to be >=16)
		BLOCK_I = 32
		BLOCK_J = 32
		
		# define the grid
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"]*args["nheads"],
								1
							)

		# run the kernel
		_triton_attn_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x N x d_k
									Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
									K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
									V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
									coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
									spreads, spreads.stride(0), # nhead, 
									L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x N
									mask, mask.stride(0), mask.stride(1), # batch x N
									N, batch, nheads, d_k, max(d_k, 16), 1/(d_k**0.5),
									BLOCK_I, BLOCK_J
								)

		ctx.save_for_backward(Q, K, V, out, L)

		return out

	@staticmethod
	def backward(ctx, dO):
		Q, K, V, O, L = ctx.saved_tensors
		
def mod_d_model(wf_features, trgt_d_model):

    if wf_features.size(-1) == trgt_d_model:
    	return wf_features
    else:
        raise NotImplementedError
