# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		attn.py
description:	multi-scale gaussian flash attention kernel written in triton. 
				(in development, currently only forward pass is implemented, bwd is just flashattention2 bwd)
				kernel based on:
					FlashAttention2 paper: https://arxiv.org/abs/2307.08691
					Triton Implementation: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
				
				Performs Flash attention, as described in the paper, but includes scaling of attention logits using RBF
				functions based on euclidean distances of alpha carbon pairs. each head uses a distinct spread to compute
				the RBFs, where the spread corresponds to (roughly) the average wavelength of the feature space it is 
				operating on. wavelength, in this context, refers to the wavelength used to compute the wave function 
				features for a particular feature index, see utils/model_utils/featurization.py). 

				will include a forward and backward pass, currently implementing vanilla FlashAttention2 and make sure 
				it is functioning before adding the gaussian scaling

'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
	O_ptr, stride_O_Z, stride_O_H, stride_O_N, stride_O_D,

	Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
	K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
	V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,

	coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_S,
	spreads_ptr, stride_spreads_H,

	L_ptr, stride_L_Z, stride_L_H, stride_L_N,
	mask_ptr, stride_mask_Z, stride_mask_N,

	tot_N: tl.constexpr, tot_Z: tl.constexpr, nheads: tl.constexpr,
	d_k: tl.constexpr, min_d_k: tl.constexpr, # max(16, d_k) bc tl.dot requires dim>=16
	softmax_scale: tl.constexpr, dist_factor: tl.constexpr,

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

	# load coordinates for I rows
	coords_I_ptr = tl.make_block_ptr(
		base=coords_ptr + (start_Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(I_offs, 0),
		block_shape=(BLOCK_I, 4), # tensor need to be power of 2, 4th value is masked
		order=(0, 1)
	)
	coords_I = tl.load(coords_I_ptr, boundary_check=(0,1), padding_option="zero") # N x 4

	# initialize coords J ptr, used in for loop
	coords_J_ptr = tl.make_block_ptr(
		base=coords_ptr + (start_Z*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_S),
		offsets=(0, 0),
		block_shape=(BLOCK_J, 4), # tensor need to be power of 2, 4th value is masked
		order=(0, 1)
	)

	# load spread for this head, scaler value
	spread_ptr = spreads_ptr + (start_H*stride_spreads_H)
	spread = tl.load(spread_ptr)

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

		# load coordinates and compute distances
		coords_J = tl.load(coords_J_ptr, boundary_check=(0,1), padding_option="zero") # N x 4 (last val masked)
		dists_raw = coords_I[:, None, :] - coords_J[None, :, :] # N x N x 4
		dists = tl.sqrt(tl.sum(dists_raw * dists_raw, axis=2)) # N x N

		# compute the rbfs
		rbfs = tl.exp(-(dists*dists) / (2*spread*spread)) # N x N

		# set masked positions to -inf, include out of range dists in mask
		dists_mask = (dists > spread) & (dists < (dist_factor*spread))
		attn_mask = (mask_i[:, None]) & (mask_j[None, :]) & (dists_mask) # N x N

		# QKT/sqrt(d_k)
		Sij = tl.dot(Qi, KjT) * softmax_scale # N x N

		# scale attention logits by rbf (subtract 1e-5 for Sij < 0, since diagonals rbf will be 1, 
		# this would lead to diagonals w/ negative logits always having attn logits of 0)
		Sij = tl.where(attn_mask, tl.where(Sij < 0, Sij*((1+1e-3)-rbfs), Sij*rbfs), -inf) # N x N

		# max of each row
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
		coords_J_ptr = tl.advance(coords_J_ptr, (BLOCK_J, 0))

	
	# epilogue

	# normalize output
	Oi = tl.where(li[:, None]!=0, Oi / li[:, None], 0.0)

	# compute log sum exponential
	mi += tl.log(li)

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
	mask_ptr, stride_mask_Z, stride_mask_N,

	tot_Z: tl.constexpr, tot_N: tl.constexpr, nheads: tl.constexpr, 
	d_k: tl.constexpr, min_d_k: tl.constexpr, softmax_scale: tl.constexpr,
	BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
	J_start = tl.program_id(0)
	J_offs = J_start * BLOCK_J

	start_ZH = tl.program_id(1)
	start_Z = start_ZH // nheads
	start_H = start_ZH % nheads

	# prep pointers (KjT and Vj stay in SRAM throughout)
	KjT_ptr = tl.make_block_ptr( # d_k x N (transpose on load)
		base=K_ptr + (start_Z*stride_K_Z) + (start_H*stride_K_H),
		shape=(d_k, tot_N),
		strides=(stride_K_D, stride_K_N),
		offsets=(0, J_offs),
		block_shape=(min_d_k, BLOCK_J),
		order=(0, 1)
	)

	Vj_ptr = tl.make_block_ptr( # N x d_k
		base=V_ptr + (start_Z*stride_V_Z) + (start_H*stride_V_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(J_offs, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	# initialize mask for j columns
	mask_j_ptr = tl.make_block_ptr( # N 
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(J_offs, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)
	mask_j = tl.load(mask_j_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)

	# initialize dKj and dVj
	dKj = tl.zeros((BLOCK_J, d_k), dtype=tl.float32)
	dVj = tl.zeros((BLOCK_J, d_k), dtype=tl.float32)

	# load KjT and Vj
	KjT = tl.load(KjT_ptr, boundary_check=(0, 1), padding_option="zero")
	Vj = tl.load(Vj_ptr, boundary_check=(0, 1), padding_option="zero")

	# initialize pointers for Qi, Oi, dQi, dOi, Li, and Di. only loaded within loop
	Qi_block_ptr = tl.make_block_ptr( # N x d_k
		base=Q_ptr + (start_Z*stride_Q_Z) + (start_H*stride_Q_H),
		shape=(tot_N, d_k),
		strides=(stride_Q_N, stride_Q_D),
		offsets=(0, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	# perform atomic adds on dQi, which don't support block pointers, so do manual indexing
	dQi_start_ptr = dQ_ptr + (start_Z*stride_dQ_Z) + (start_H*stride_dQ_H)
	dQi_block_ptr = dQi_start_ptr + (tl.arange(0,BLOCK_I)[:, None]*stride_dQ_N) + (tl.arange(0,min_d_k)[None, :]*stride_dQ_D)

	dOi_block_ptr = tl.make_block_ptr( # N x d_k
		base=dO_ptr + (start_Z*stride_dO_Z) + (start_H*stride_dO_H),
		shape=(tot_N, d_k),
		strides=(stride_dO_N, stride_dO_D),
		offsets=(0, 0),
		block_shape=(BLOCK_I, min_d_k),
		order=(0, 1)
	)

	Li_block_ptr = tl.make_block_ptr( # N
		base=L_ptr + (start_Z*stride_L_Z) + (start_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	Di_block_ptr = tl.make_block_ptr( # N
		base=D_ptr + (start_Z*stride_D_Z) + (start_H*stride_D_H),
		shape=(tot_N, ),
		strides=(stride_D_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# initialize mask for i rows
	mask_i_ptr = tl.make_block_ptr( # N 
		base=mask_ptr + (start_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	inf = float("inf") # convenience
	for i in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1):

		Qi = tl.load(Qi_block_ptr, boundary_check=(0, 1), padding_option="zero")
		Sij = tl.dot(Qi, KjT) * softmax_scale # N x N

		mask_i = tl.load(mask_i_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)
		attn_mask = mask_i[:, None] & mask_j[None, :] # N x N
		Li = tl.load(Li_block_ptr, boundary_check=(0, ), padding_option="zero")
		Pij = tl.exp(tl.where(attn_mask, Sij - Li[:, None], -inf)) # N x N

		dOi = tl.load(dOi_block_ptr, boundary_check=(0, 1), padding_option="zero")
		dVj += tl.where(mask_j[:, None], tl.dot(tl.permute(Pij, (1,0)), dOi), 0.0)

		dPij = tl.dot(dOi, tl.permute(Vj, (1,0)))

		Di = tl.load(Di_block_ptr, boundary_check=(0, ), padding_option="zero")
		dSij = Pij * (dPij - Di[:, None])

		# will do atomic add instead of loading, as other J in parallel leads to race condition
		dQi = tl.dot(dSij, tl.permute(KjT, (1,0))) * softmax_scale
		dQi_mask = mask_i[:, None] & (tl.arange(0,min_d_k)[None, :] < d_k)
		# tl.device_print("", dQi_mask)
		tl.atomic_add(dQi_block_ptr, dQi, mask=dQi_mask)

		dKj += tl.where(mask_j[:, None], tl.dot(tl.permute(dSij, (1,0)), Qi), 0.0) * softmax_scale

		# advance the pointers
		Qi_block_ptr = tl.advance(Qi_block_ptr, (BLOCK_I, 0))
		dQi_block_ptr += BLOCK_I*stride_dQ_N
		dOi_block_ptr = tl.advance(dOi_block_ptr, (BLOCK_I, 0))
		Li_block_ptr = tl.advance(Li_block_ptr, (BLOCK_I, ))
		Di_block_ptr = tl.advance(Di_block_ptr, (BLOCK_I, ))
		mask_i_ptr = tl.advance(mask_i_ptr, (BLOCK_I, ))

	# initialize dK and dV pointers to write output
	dKj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dK_ptr + (start_Z*stride_dK_Z) + (start_H*stride_dK_H),
		shape=(tot_N, d_k),
		strides=(stride_dK_N, stride_dK_D),
		offsets=(J_offs, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	dVj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dV_ptr + (start_Z*stride_dV_Z) + (start_H*stride_dV_H),
		shape=(tot_N, d_k),
		strides=(stride_V_N, stride_V_D),
		offsets=(J_offs, 0),
		block_shape=(BLOCK_J, min_d_k),
		order=(0, 1)
	)

	tl.store(dKj_block_ptr, dKj, boundary_check=(0,1))
	tl.store(dVj_block_ptr, dVj, boundary_check=(0,1))

class attn(torch.autograd.Function):

	@staticmethod
	def forward(ctx, Q, K, V, coords, spreads, mask=None, dist_factor=3.0):
		
		# checks
		assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		batch, nheads, N, d_k = Q.shape
		d_model = nheads*d_k
		softmax_scale = 1/(d_k**0.5)

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
		_attn_fwd[grid](  	out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x N x d_k
							Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
							coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
							spreads, spreads.stride(0), # nhead, 
							L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x N
							mask, mask.stride(0), mask.stride(1), # batch x N
							N, batch, nheads, d_k, max(d_k, 16), softmax_scale, dist_factor,
							BLOCK_I, BLOCK_J
						)

		ctx.save_for_backward(Q, K, V, out, L, coords, spreads, mask)
		ctx.softmax_scale = softmax_scale
		ctx.dist_factor = dist_factor

		return out

	@staticmethod
	def backward(ctx, dO):
		Q, K, V, O, L, coords, spreads, mask = ctx.saved_tensors
		D = torch.sum(O*dO, dim=3) # Z x H x N x D -> Z x H x N
		assert Q.stride() == K.stride() == V.stride() == O.stride()
		batch, nheads, N, d_k = Q.shape 

		Q.contiguous()
		K.contiguous()
		V.contiguous()
		D.contiguous()
		L.contiguous()
		coords.contiguous()
		spreads.contiguous()
		mask.contiguous()

		BLOCK_I = 32
		BLOCK_J = 16

		# initialize mask, output, and logsumexp tensors
		dQ = torch.zeros_like(Q).contiguous()
		dK = torch.zeros_like(K).contiguous()
		dV = torch.zeros_like(V).contiguous()
		
		grid = lambda args: (
			triton.cdiv(args["tot_N"], args["BLOCK_J"]), # parralel along J for bwd
			args["tot_Z"]*args["nheads"],
			1
		)

		_attn_bwd[grid](	Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), 
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), 
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
							dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3), 
							dQ, dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
							dK, dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
							dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
							D, D.stride(0), D.stride(1), D.stride(2),
							L, L.stride(0), L.stride(1), L.stride(2),
							mask, mask.stride(0), mask.stride(1),
							batch, N, nheads, d_k, max(d_k, 16), ctx.softmax_scale, 
							BLOCK_I, BLOCK_J
						 )

		return dQ, dK, dV, None, None, None, None
		# possibly make spreads and dist factor learnable, but we'll see

