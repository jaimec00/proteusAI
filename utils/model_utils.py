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
		stride_out_B,
		stride_out_N,
		stride_out_D,

		coords_ptr,
		stride_coords_B,
		stride_coords_N,
		stride_coords_space,

		wavenumber_ptr, 
		stride_wavenumber,

		pad_mask_ptr,
		stride_pad_mask_B,
		stride_pad_mask_N,

		tot_batch: tl.constexpr,
		tot_N: tl.constexpr,
		d_model:tl.constexpr, 
		pi:tl.constexpr,

		BLOCK_NI: tl.constexpr,
		BLOCK_NJ: tl.constexpr,
		BLOCK_BD: tl.constexpr
):

	# get i, j indices
	NI = (tl.program_id(0) * BLOCK_NI) + tl.arange(0, BLOCK_NI)[None, :, None] # 1 x NI x 1
	NJ = (tl.program_id(1) * BLOCK_NJ) + tl.arange(0, BLOCK_NJ)[None, None, :] # 1 x 1 x NJ

	# get batch and wavenumber indices
	BD = (tl.program_id(2) * BLOCK_BD) + tl.arange(0, BLOCK_BD)[:, None, None]
	num_wn = d_model//2
	B = BD // num_wn # BD x 1 x 1
	D = BD % num_wn # BD x 1 x 1

	# get (x, y, z) pointers
	coords_ptr_NI_x = coords_ptr + (B*stride_coords_B) + (NI*stride_coords_N) + (NJ*0) + (0*stride_coords_space) # BD x NI x NJ
	coords_ptr_NI_y = coords_ptr_NI_x + (1*stride_coords_space) # BD x NI x NJ
	coords_ptr_NI_z = coords_ptr_NI_x + (2*stride_coords_space) # BD x NI x NJ

	coords_ptr_NJ_x = coords_ptr + (B*stride_coords_B) + (NJ*stride_coords_N) + (NI*0) + (0*stride_coords_space) # BD x NI x NJ
	coords_ptr_NJ_y = coords_ptr_NJ_x + (1*stride_coords_space) # BD x NI x NJ
	coords_ptr_NJ_z = coords_ptr_NJ_x + (2*stride_coords_space) # BD x NI x NJ

	# create i, j masks (True means compute)
	mask_IJ = ((NI < tot_N) & (NJ < tot_N) & (NI!=NJ) & (B < tot_batch) & (D < num_wn)) # BD x NI x NJ
	
	# get pointers for masks (add 0*N(J/I) to match the shape)
	pad_mask_ptr_NI = pad_mask_ptr + (B*stride_pad_mask_B) + (NI*stride_pad_mask_N) + (NJ*0) # BD x NI x NJ
	pad_mask_ptr_NJ = pad_mask_ptr + (B*stride_pad_mask_B) + (NJ*stride_pad_mask_N) + (NI*0) # BD x NI x NJ

	# load the masks
	pad_mask_NI = tl.load(pad_mask_ptr_NI, mask=mask_IJ, other=0).to(tl.int1) # BD x NI x NJ
	pad_mask_NJ = tl.load(pad_mask_ptr_NJ, mask=mask_IJ, other=0).to(tl.int1) # BD x NI x NJ

	# combine masks and ensure not to compute self distance
	pad_mask_IJ = pad_mask_NI & pad_mask_NJ # BD x NI x NJ

	# load into SRAM
	coords_NI_x = tl.load(coords_ptr_NI_x, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ
	coords_NI_y = tl.load(coords_ptr_NI_y, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ
	coords_NI_z = tl.load(coords_ptr_NI_z, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ

	coords_NJ_x = tl.load(coords_ptr_NJ_x, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ
	coords_NJ_y = tl.load(coords_ptr_NJ_y, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ
	coords_NJ_z = tl.load(coords_ptr_NJ_z, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ

	# compute distances
	dist_x = (coords_NI_x - coords_NJ_x) * (coords_NI_x - coords_NJ_x) # BD x NI x NJ
	dist_y = (coords_NI_y - coords_NJ_y) * (coords_NI_y - coords_NJ_y) # BD x NI x NJ
	dist_z = (coords_NI_z - coords_NJ_z) * (coords_NI_z - coords_NJ_z) # BD x NI x NJ

	dist = (dist_x + dist_y + dist_z).sqrt() # BD x NI x NJ

	# get wavenumbers
	wavenumber_ptr = wavenumber_ptr + (D*stride_wavenumber) + (NI*0) + (NJ*0) # BD x NI x NJ
	wavenumber = tl.load(wavenumber_ptr, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x NI x NJ

	# compute phase
	phase = (wavenumber*dist) % (2*pi) # BD x NI x NJ 

	# compute real and imag parts
	real = tl.cos(phase) / tl.where(pad_mask_IJ, dist, float("inf")) # BD x NI x NJ
	imag = tl.sin(phase) / tl.where(pad_mask_IJ, dist, float("inf")) # BD x NI x NJ

	# superpose real and imag parts in the tile
	real_superposition = tl.sum(real, axis=2, keep_dims=True) + (NJ*0) # BD x NI x NJ 
	imag_superposition = tl.sum(imag, axis=2, keep_dims=True) + (NJ*0) # BD x NI x NJ

	# compute pad mask, check if at least one NJ for each BD x NI
	pad_mask_IJ = (tl.sum(pad_mask_IJ.to(tl.int32), axis=2, keep_dims=True) > 0) # BD x NI x 1 (like torch.any(x, dim=-1))
	
	# only one thread per NI writes to global memory (first NJ thread in the block per BD x NI)
	pad_mask_IJ = (pad_mask_IJ) & ((NJ%BLOCK_NJ)==0) # BD x NI x NJ

	# compute d_model index
	D_real = 1 + (D * 2 - 1) # BD x 1 x 1
	D_imag = 1 + (D * 2) # BD x 1 x 1

	# prepare output pointers (only write to NI)
	out_ptr_real = (out_ptr + (B*stride_out_B) + (D_real*stride_out_D) + (NI*stride_out_N) + (NJ*0)) # BD x NI x NJ
	out_ptr_imag = (out_ptr + (B*stride_out_B) + (D_imag*stride_out_D) + (NI*stride_out_N) + (NJ*0)) # BD x NI x NJ

	# add real and imag parts to output tensor
	tl.atomic_add(out_ptr_real, real_superposition, mask=pad_mask_IJ)
	tl.atomic_add(out_ptr_imag, imag_superposition, mask=pad_mask_IJ)

def protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask=None):
	
	# checks
	assert (coords.dim() == 3) and (coords.size(2) == 3), f"coords must be of shape (batch x N x 3), not {coords.shape}"
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model}"
	
	# prepare data
	batch, N, space = coords.shape # input dimensions
	coords = coords.to(torch.float64).contiguous() # double-precision and contiguous

	# prepare the wavenumber values
	num_wl = d_model//2 # define the number of wave functions to compute
	wavelengths = (min_wl + (torch.logspace(0, 1, num_wl, base=base, device=coords.device, dtype=torch.float64) - 1) / (base - 1) * (max_wl - min_wl))
	wavenumbers = (2 * torch.pi / wavelengths).contiguous()

	# prepare the mask, triton uses true as compute
	mask = (~mask if mask is not None else torch.ones(batch, N, dtype=torch.bool, device=coords.device)).contiguous()
	
	# prepare output
	out = torch.zeros(batch, N, d_model, dtype=torch.float64, device=coords.device).contiguous()

	# total block size should be less than number of threads per block (approx. 1024)
	BLOCK_NI = 2    # N_i
	BLOCK_NJ = 256    # N_j
	BLOCK_BD = 2     # batch x d_model
	# BLOCK_NI x BLOCK_NJ x BLOCK_BD <= 1024

	# compute the grid size
	grid_NI = (N // BLOCK_NI) + 1   # number of NI blocks
	grid_NJ = (N // BLOCK_NJ) + 1   # number of NJ blocks
	grid_BD = (batch * num_wl // BLOCK_BD) + 1      # number of batch x feature blocks

	# define the grid
	grid = (grid_NI, grid_NJ, grid_BD)

	# run the kernel
	_protein_to_wavefunc_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2),
									coords, coords.stride(0), coords.stride(1), coords.stride(2),
									wavenumbers, wavenumbers.stride(0),
									mask, mask.stride(0), mask.stride(1),
									batch, N, d_model, torch.pi,
									BLOCK_NI, BLOCK_NJ, BLOCK_BD
								)

	# normalize each feature by the maximum absolute value in the sample
	out.div_(out.abs().max(dim=1, keepdim=True).values)

	return out


@triton.jit
def _triton_attn_kernel(
	O_ptr,
	stride_O_Z,
	stride_O_H,
	stride_O_N,
	stride_O_D,

	Q_ptr,
	stride_Q_Z,
	stride_Q_H,
	stride_Q_N,
	stride_Q_D,
	
	K_ptr,
	stride_K_Z,
	stride_K_H,
	stride_K_N,
	stride_K_D,
	
	V_ptr,
	stride_V_Z,
	stride_V_H,
	stride_V_N,
	stride_V_D,

	coords_ptr,
	stride_coords_Z,
	stride_coords_N,
	stride_coords_S,

	spread_ptr,
	stride_spread_H,

	L_ptr,
	stride_L_Z,
	stride_L_H,
	stride_L_N,

	mask_ptr,
	stride_mask_Z,
	stride_mask_N,

	tot_N:tl.constexpr,
	tot_Z:tl.constexpr,
	nheads:tl.constexpr,
	d_k:tl.constexpr,
	min_d_k:tl.constexpr,
	softmax_scale:tl.constexpr,

	BLOCK_I:tl.constexpr,
	BLOCK_J:tl.constexpr,
):
	# get block info
	start_I = tl.program_id(0)
	start_ZH = tl.program_id(1) # note that exactly batch*nheads launched along y axis, so no need to mask this
	start_Z = (start_ZH // nheads).to(tl.int64)
	start_H = (start_ZH % nheads).to(tl.int64)

	I_offs = (start_I.to(tl.int64)*BLOCK_I).to(tl.int32)

	# create Q, K, V, and O block pointers
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
		order=(1, 0)
	)

	Vj_block_ptr = tl.make_block_ptr( # N x d_k
		base=V_ptr + (start_Z*stride_V_Z) + (start_H*stride_V_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(0, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# load the Qi block first, out of bounds values are 0
	Qi_block = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k

	# initialize output and statistics block
	Oi_block = tl.zeros_like(Qi_block)
	li_block = tl.zeros((BLOCK_I, ), dtype=tl.float32)
	mi_block = tl.zeros_like(li_block) - float("inf") 

	# make an identity so can perform mat mults from linear arrays easily
	identity_block_i = tl.where(
		tl.arange(0, BLOCK_I)[:, None] == tl.arange(0, BLOCK_I)[None, :],
		1.0,
		0.0
	)

	mask_i_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(I_offs, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)
	mask_i = tl.load(mask_i_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # N x d_k

	mask_j_ptr = tl.make_block_ptr( # N,
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1):

		KjT_block = tl.load(KjT_block_ptr, boundary_check=(0,1), padding_option="zero") # d_k x N
		Vj_block = tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k
		mask_j = tl.load(mask_j_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # N

		Sij = tl.dot(Qi_block, KjT_block) / softmax_scale # N x N
		Sij = tl.where(mask_i[:, None] | mask_j[None, :], Sij, float("-inf")) # N x N

		# join mi and Sij_max to get N x 2, then compute max along axis 1 to get mij of shape N,
		mij = tl.max(tl.join(mi_block, tl.max(Sij, axis=1)), axis=1) # N, 

		Pij = tl.exp(tl.where((Sij==float("-inf")) | (mij[:, None]==float("-inf")), float("-inf"), Sij - mij[:, None])) # N x N

		mi_mij_diff = tl.where((mi_block==float("-inf")) | (mij==float("-inf")), float("-inf"), mi_block - mij)
		lij = tl.exp(mi_mij_diff)*li_block + tl.sum(Pij, axis=1) # N, 

		diag_exp_inv = tl.where(
			(mi_mij_diff)==float("-inf"), 
			0.0, 
			tl.exp(-mi_mij_diff)
		)[:, None] * identity_block_i # N x N 

		Oi_block = tl.dot(diag_exp_inv, Oi_block) + tl.dot(Pij, Vj_block) # N x d_k

		li_block = lij
		mi_block = mij

		KjT_block_ptr = tl.advance(KjT_block_ptr, (0, BLOCK_J))
		Vj_block_ptr = tl.advance(Vj_block_ptr, (BLOCK_J, 0))
		mask_j_ptr = tl.advance(mask_j_ptr, (BLOCK_J, ))
	
	diag_li = identity_block_i*li_block[:, None]
	diag_li_inv = tl.where(diag_li==0, 0.0, 1/diag_li)
	Oi_block = tl.dot(diag_li_inv, Oi_block)
	Li_block = mi_block + tl.log(li_block)

	Oi_block_ptr = tl.make_block_ptr( # N x d_k
		base=O_ptr + (start_Z*stride_O_Z) + (start_H*stride_O_H),
		shape=(tot_N, d_k),
		strides=(stride_O_N, stride_O_D),
		offsets=(I_offs, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	Li_block_ptr = tl.make_block_ptr( # N,
		base=L_ptr + (start_Z*stride_L_Z) + (start_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(I_offs, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	tl.store(Oi_block_ptr, Oi_block, boundary_check=(0,1))
	tl.store(Li_block_ptr, Li_block, boundary_check=(0,))

class triton_attn(torch.autograd.Function):

	@staticmethod
	def forward(Q, K, V, coords, spreads, mask=None, dist_factor=3.0):
		
		assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		batch, nheads, N, d_k = Q.shape
		d_model = nheads*d_k

		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
		assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

		assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
		assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"

		mask = (torch.ones(batch, N, device=Q.device) if mask is None else ~mask).contiguous() # batch x N
		out = torch.zeros(batch, nheads, N, d_k, device=Q.device).contiguous() # batch x N x d_model
		L = torch.zeros(batch, nheads, N, device=Q.device).contiguous() # batch x nheads x N

		Q = Q.contiguous()
		K = K.contiguous()
		V = V.contiguous()
		coords = coords.contiguous()
		spreads = spreads.contiguous()

		BLOCK_I = 16 
		BLOCK_J = BLOCK_I
		
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"]*args["nheads"],
								1)

		_triton_attn_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x N x d_k
									Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
									K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
									V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
									coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
									spreads, spreads.stride(0), # nhead, 
									L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x N
									mask, mask.stride(0), mask.stride(1), # batch x N
									N, batch, nheads, d_k, max(d_k, 16), d_k**0.5,
									BLOCK_I, BLOCK_J
								)

		return out
		
def mod_d_model(wf_features, trgt_d_model):

    if wf_features.size(-1) == trgt_d_model:
    	return wf_features
    else:
        raise NotImplementedError
