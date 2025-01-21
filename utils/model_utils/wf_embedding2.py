# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		wf_embedding.py
description:	converts alpha carbon coordinates to features by modeling each Ca as a point source using Green's
				function solution to the Hemholtz eq. in 3D: 
					nabla^2 psi_k(r) = k^2 * psi_k(r)

				extending this equation to multiple point sources and superposing their wave functions:
					sum_i=0^N( nabla^2 psi_k(r_i) ) = k^2*sum_i=0^N( psi_k(r_i) )

				the solution to this eq. is:
							 				  exp( ik * |r - r_n| )
					psi_k(r) =	sum_n=0^N   -----------------------
												   |r - r_n| 

				since this corresponds to single wavenumber, k=(2pi / wavelength), we can assign a k value, and 
				thus a wavelength to each feature space. note that the output of psi_k(r) is a complex number,
				so each output generates two features, the real part and the imaginary part, i.e. cos and sin terms (euler's formula). 
				so for a particular feature index, its feature is the real/imaginary (depends if odd or even) output of 
				the wavefunction output for its position, r. we assign low index features to small wavelength,
				representing local interactions, and large index features with large wavelengths, representing global interactions.

				formally, the transformation termed WF, takes the feature index and protein coordinates as input,
				the parameters that need to be defined are min_wl, max_wl, and base. these are used to sample
				the wavelengths logarithmically, to emphasize shorter wavelengths, as those are more prone to large
				fluctuations from small changes in wavelength

				WF(r, 2i) = 	lambda <- min_wl + ( (max_wl-min_wl) * (base^(2i/d_model) - 1) / (base-1) )
								k <- 2pi / lambda

												  cos( k * |r - r_n| )
								sum_n=0^N   	-----------------------
													   |r - r_n| 
								
				WF(r, 2i+1) = 	lambda <- min_wl + ( (max_wl-min_wl) * (base^(2i/d_model) - 1) / (base-1) )
								k <- 2pi / lambda	

												  sin( k * |r - r_n| )
								sum_n=0^N   	-----------------------
													   |r - r_n| 

				note that this is reminiscent of positional encoding:
				
											 pos 
				PE(pos, 2i)	= 	sin( ----------------------- )
										10000^(2i/d_model) 

											 pos 
				PE(pos, 2i)	= 	cos( ----------------------- )
										10000^(2i/d_model) 

				this is because the wave function featurization function serves as a generalization of positional encoding
				for irregularly spaced tokens in arbitrary dimensions. this makes the model chain and sequence agnostic,
				relying solely on the wave function features which depend on the spatial arrangement of the Ca coordinates.
				the actual distances are also used in the model, by performing multi-scale gaussian attention on these features,
				where each head uses a distinct spread in its RBF scaling computation, which aligns with the wavelengths
				used to compute the features of the tokens for that head's target feature space. see utils/model_utils/attn.py  

				while this computation would be memory intensive in pytorch, protein_to_wavefunc quickly and efficiently computes the
				exact features by fusing all the required operations into a single triton kernel, with no approximation.

'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import math
import triton
import triton.language as tl
import os

# define configurations for autotuning
# configs = [	triton.Config({"BLOCK_NI": i, "BLOCK_NJ": j}, num_warps=w, num_stages=s)
# 			for j in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# 			for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# 			for w in [4, 8]
# 			for s in [2]
# 		]

# # filter out configs that are too big
# def keep_fwd(conf):
# 	autotune = os.environ.get("WF_AUTOTUNE")
# 	BLOCK_NI = conf.kwargs["BLOCK_NI"]
# 	BLOCK_NJ = conf.kwargs["BLOCK_NJ"]
# 	if autotune == "1":
# 		return (BLOCK_NI * BLOCK_NJ) <= 2048
# 	else:
# 		return ((BLOCK_NI == 2) and (BLOCK_NJ == 1024) and (conf.num_warps==4) and (conf.num_stages==2))

# @triton.autotune(list(filter(keep_fwd, configs)),
# 				 key=['tot_Z', 'tot_N', 'd_model'], # triton will not recompile if these inputs are the same (size of input tensor)
# 				 restore_value=["out_ptr"] # make sure autotuning resets the outputs of this function for each configuration
# ) 

# @triton.heuristics(
# 	values = {
# 		"BLOCK_NI": lambda args: 2048 // min(1024, triton.next_power_of_2(args['tot_N'])),
# 		"BLOCK_NJ": lambda args: min(1024, triton.next_power_of_2(args['tot_N'])),
# 		"num_warps": lambda args: 4
# 	}
# )
@triton.jit
def _wf_embedding_fwd(
		out_ptr, stride_out_Z, stride_out_N, stride_out_D,
		coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_space,
		wavenumber_ptr, stride_wavenumber_K,
		cos_sums_ptr, stride_cos_sums_Z, stride_cos_sums_N, stride_cos_sums_K,
		sin_sums_ptr, stride_sin_sums_Z, stride_sin_sums_N, stride_sin_sums_K,
		mask_ptr, stride_mask_Z, stride_mask_N,

		tot_Z: tl.constexpr,
		tot_N: tl.constexpr,
		d_model:tl.constexpr, 

		BLOCK_NI: tl.constexpr,
		BLOCK_NJ: tl.constexpr,
		# num_warps: tl.constexpr
):

	# get i, j indices
	NI_start = tl.program_id(0)
	NI_offs = NI_start*BLOCK_NI

	NJ_start = tl.program_id(1)
	NJ_offs = NJ_start*BLOCK_NJ

	# get batch and wavenumber indices
	ZK_offs = tl.program_id(2)
	Z_offs = ZK_offs // (d_model//2)
	K_offs = ZK_offs % (d_model//2)

	coords_NI_ptr = tl.make_block_ptr( # NI x 3
		base=coords_ptr + (Z_offs*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_space),
		offsets=(NI_offs, 0),
		block_shape=(BLOCK_NI, 4), # 4th value is masked, tensor needs to be power of two (x,y,z,masked)
		order=(0, 1)
	)
	mask_NI_ptr = tl.make_block_ptr( # NI,
		base=mask_ptr + (Z_offs*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(NI_offs, ),
		block_shape=(BLOCK_NI, ),
		order=(0, )
	)
	
	coords_NJ_ptr = tl.make_block_ptr( # NJ x 3
		base=coords_ptr + (Z_offs*stride_coords_Z),
		shape=(tot_N, 3),
		strides=(stride_coords_N, stride_coords_space),
		offsets=(NJ_offs, 0),
		block_shape=(BLOCK_NJ, 4), # 4th value is masked, tensor needs to be power of two (x,y,z,masked)
		order=(0, 1)
	)
	
	mask_NJ_ptr = tl.make_block_ptr( # NJ,
		base=mask_ptr + (Z_offs*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(NJ_offs, ),
		block_shape=(BLOCK_NJ, ),
		order=(0, )
	)

	# load coords for i and j
	coords_NI = tl.load(coords_NI_ptr, boundary_check=(0,1), padding_option="zero") # N x 4
	coords_NJ = tl.load(coords_NJ_ptr, boundary_check=(0,1), padding_option="zero") # N x 4
	
	# compute dists
	dists_raw = coords_NI[:, None, :] - coords_NJ[None, :, :] # NI x NJ x 4
	dists = tl.sqrt(tl.sum(dists_raw * dists_raw, axis=2)) # NI x NJ
	
	# compute mask
	mask_NI = tl.load(mask_NI_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # NI
	mask_NJ = tl.load(mask_NJ_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # NJ
	mask_IJ = mask_NI[:, None] & mask_NJ[None, :] # NI x NJ
	mask_IJ = mask_IJ & (dists!=0)

	# load wavenumber and compute phase and trig funcs
	phase = dists*tl.load(wavenumber_ptr + (K_offs*stride_wavenumber_K))
	cos = tl.cos(phase)
	sin = tl.sin(phase)

	# accumulate sum of cosines and sins for this wavenumber for bwd pass
	tl.atomic_add(cos_sums_ptr + (Z_offs*stride_cos_sums_Z) + ((NI_offs + tl.arange(0,BLOCK_NI))*stride_cos_sums_N) + (K_offs*stride_cos_sums_K), tl.sum(tl.where(mask_IJ, cos, 0.0), axis=1), mask=mask_NI) # NI
	tl.atomic_add(sin_sums_ptr + (Z_offs*stride_sin_sums_Z) + ((NI_offs + tl.arange(0,BLOCK_NI))*stride_sin_sums_N) + (K_offs*stride_sin_sums_K), tl.sum(tl.where(mask_IJ, sin, 0.0), axis=1), mask=mask_NI) # NI
	
	# compute real and imag parts
	real = cos / tl.where(mask_IJ, dists, float("inf")) # NI x NJ
	imag = sin / tl.where(mask_IJ, dists, float("inf")) # NI x NJ

	# join into features and add
	out_ptr = out_ptr + (Z_offs*stride_out_Z) + ((NI_offs + tl.arange(0,BLOCK_NI)[:, None])*stride_out_N) + ((2*K_offs + tl.arange(0,2)[None, :])*stride_out_D)
	tl.atomic_add(out_ptr, tl.join(tl.sum(real, axis=1), tl.sum(imag, axis=1)), mask=mask_NI[:, None])

def wf_embedding(coords, wavenumbers, mask=None):
	'''wrapper to call protein_to_wavefunc w/ kwargs'''

	return _wf_embedding.apply(coords, wavenumbers, mask)


class _wf_embedding(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coords, wavenumbers, mask=None):
		'''
		converts the alpha carbon coordinates of a protein into a tensor of 
		wavefunction outputs.
		converts a batch x N x 3 tensor to a batch x N x d_model tensor.
		each feature for a Ca is their output of the wave function with a specific wavelength
		each output gets two features, one for the real part, and another for the imaginary part
		the wave function is a superposition of Green's functions, treating each Ca as a point source

		Args:
			coords (torch.Tensor):              tensor containing batches of Ca coords. 
												size = batch x N x 3 
			wavenumbers (torch.Tensor):         tensor containing the wavenumbers to use for each feature idx. 
												size = batch x N x d_model//2
			mask (torch.Tensor):    			tenor containing key padding mask
												size = batch x N 
		
		Returns:
			features (torch.Tensor):    tensor containing batches of token (Ca) features.
										size = batch x N x d_model
		'''


		# checks
		assert (coords.dim() == 3) and (coords.size(2) == 3), f"coords must be of shape (batch x N x 3), not {coords.shape}"
		batch, N, space = coords.shape # input dimensions
		num_wn = wavenumbers.size(0)
		d_model = num_wn * 2
		
		# prepare data
		coords = coords.to(torch.float32).contiguous() #   contiguous

		# prepare the mask, triton uses true as compute
		mask = (~mask if mask is not None else torch.ones(batch, N, dtype=torch.bool, device=coords.device)).contiguous()
		
		# prepare output
		out = torch.zeros(batch, N, d_model, dtype=torch.float32, device=coords.device).contiguous()

		# save sum of cos and sins for bwd pass
		cos_sums = torch.zeros(batch, N, num_wn, dtype=torch.float32, device=coords.device).contiguous()
		sin_sums = torch.zeros(batch, N, num_wn, dtype=torch.float32, device=coords.device).contiguous()

		# define the grid
		grid = lambda args: (	triton.cdiv(args["tot_N"], args["BLOCK_NI"]), 
								triton.cdiv(args["tot_N"], args["BLOCK_NJ"]), 
								args["tot_Z"]*(args["d_model"]//2)
							)

		BLOCK_NI = 2048 // min(1024, triton.next_power_of_2(N))
		BLOCK_NJ = min(1024, triton.next_power_of_2(N))
		# num_warps = 8


		# run the kernel
		_wf_embedding_fwd[grid](  	out, out.stride(0), out.stride(1), out.stride(2),
											coords, coords.stride(0), coords.stride(1), coords.stride(2),
											wavenumbers, wavenumbers.stride(0),
											cos_sums, cos_sums.stride(0), cos_sums.stride(1), cos_sums.stride(2),
											sin_sums, sin_sums.stride(0), sin_sums.stride(1), sin_sums.stride(2),
											mask, mask.stride(0), mask.stride(1),
											batch, N, d_model,
											BLOCK_NI, BLOCK_NJ
										)

		ctx.save_for_backward(cos_sums, sin_sums)

		return out

	@staticmethod
	def backward(ctx, dO):

		# note, masks already applied to O and cos and sin sums, masked vals are zero, 
		# so multiplication ensures non valid positions dont contribute to the gradients
		# i.e. don't need to save mask for bwd

		# load saved tensors from bwd
		cos_sums, sin_sums = ctx.saved_tensors

		# seperate dO into real and imag parts (interleaved, real is first)
		# from Z x N x d_model --> Z x N x d_model//2 
		real_dO = dO[:, :, 0::2]
		imag_dO = dO[:, :, 1::2]

		# compute grad wrt wavenumbers
		# dO_2i=l+1 * sum_j(cos(K|ri-rj|)) - dO_2l * sum(sin(K|ri-rj|))
		# sum the Z dim and N dim, to accumulate gradients, as wavenumbers is a tensor of shape d_model//2
		dk = ((imag_dO*cos_sums) - (real_dO*sin_sums)).sum(dim=1).sum(dim=0) # d_model//2

		return None, dk, None 
