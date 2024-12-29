# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		featurization.py
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

@triton.jit
def _protein_to_wavefunc_kernel(
		out_ptr, stride_out_Z, stride_out_N, stride_out_D,
		coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_space,
		wavenumber_ptr, stride_wavenumber,
		mask_ptr, stride_mask_Z, stride_mask_N,

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
		out_ptr += 2*BLOCK_D * stride_out_D

def protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask=None):
	'''wrapper to call protein_to_wavefunc w/ kwargs'''

	return _protein_to_wavefunc.apply(coords, d_model, min_wl, max_wl, base, mask)


class _protein_to_wavefunc(torch.autograd.Function):

	@staticmethod # might make wavelengths learnable and make a backward pass, but focusing on MHA kernel first
	def forward(ctx, coords, d_model, min_wl, max_wl, base, mask=None):
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
			d_model (int):						features to create. d_model = number_of_wavelengths*2
			min_wl (float):						minimum wavelength to use
			max_wl (float):						maximum wavelength to use
			base (int|float):					wavelengths are sampled logarithmically, chooses the base to use
			mask (torch.Tensor):    			tenor containing key padding mask
												size = batch x N 
		
		Returns:
			features (torch.Tensor):    tensor containing batches of token (Ca) features.
										size = batch x N x d_model
		'''


		# checks
		assert (coords.dim() == 3) and (coords.size(2) == 3), f"coords must be of shape (batch x N x 3), not {coords.shape}"
		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model}"
		
		# prepare data
		batch, N, space = coords.shape # input dimensions
		coords = coords.to(torch.float32).contiguous() #   contiguous

		# prepare the wavenumber values
		num_wl = int(d_model//2) # define the number of wave functions to compute
		wavelengths = (min_wl + (torch.logspace(0, 1, num_wl, base=base, device=coords.device, dtype=torch.float32) - 1) / (base - 1) * (max_wl - min_wl))
		wavenumbers = (2 * torch.pi / wavelengths).contiguous()

		# prepare the mask, triton uses true as compute
		mask = (~mask if mask is not None else torch.ones(batch, N, dtype=torch.bool, device=coords.device)).contiguous()
		
		# prepare output
		out = torch.zeros(batch, N, d_model, dtype=torch.float32, device=coords.device).contiguous()

		# total block size should be less than number of threads per block (approx. 1024)
		BLOCK_NJ = min(1024, triton.next_power_of_2(N))
		BLOCK_NI = triton.cdiv(1024, BLOCK_NJ)
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
		# out.div_(out.abs().max(dim=1, keepdim=True).values)

		return out

def mod_d_model(wf_features, trgt_d_model):

    if wf_features.size(-1) == trgt_d_model:
    	return wf_features
    else:
        raise NotImplementedError