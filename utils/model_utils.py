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
	NI = tl.program_id(0) * BLOCK_NI + tl.arange(0, BLOCK_NI)[None, :, None] # 1 x N x 1
	NJ = tl.program_id(1) * BLOCK_NJ + tl.arange(0, BLOCK_NJ)[None, None, :] # 1 x 1 x N

	# get batch and wavenumber indices
	BD = tl.program_id(2) * BLOCK_BD + tl.arange(0, BLOCK_BD)[:, None, None]
	num_wn = d_model//2
	B = BD // num_wn # BD x 1 x 1
	D = BD % num_wn # BD x 1 x 1

	# get (x, y, z) pointers
	coords_ptr_NI_x = coords_ptr + (B*stride_coords_B) + (NI*stride_coords_N) + (NJ*0) + (0*stride_coords_space) # BD x N x N
	coords_ptr_NI_y = coords_ptr_NI_x + (1*stride_coords_space) # BD x N x N
	coords_ptr_NI_z = coords_ptr_NI_x + (2*stride_coords_space) # BD x N x N

	coords_ptr_NJ_x = coords_ptr + (B*stride_coords_B) + (NJ*stride_coords_N) + (NI*0) + (0*stride_coords_space) # BD x N x N
	coords_ptr_NJ_y = coords_ptr_NJ_x + (1*stride_coords_space) # BD x N x N
	coords_ptr_NJ_z = coords_ptr_NJ_x + (2*stride_coords_space) # BD x N x N

	# create i, j masks (True means compute)
	mask_NI = ((NI < tot_N) & (B < tot_batch) & (D < num_wn)) # BD x N x N
	mask_NJ = ((NJ < tot_N) & (B < tot_batch) & (D < num_wn)) # BD x N x N
	
	# get pointers for masks (add 0*N(J/I) to match the shape)
	pad_mask_ptr_NI = pad_mask_ptr + (B*stride_pad_mask_B) + (NI*stride_pad_mask_N) + (NJ*0) # BD x N x N
	pad_mask_ptr_NJ = pad_mask_ptr + (B*stride_pad_mask_B) + (NJ*stride_pad_mask_N) + (NI*0) # BD x N x N

	# load the masks
	pad_mask_NI = tl.load(pad_mask_ptr_NI, mask=mask_NI, other=0).to(tl.int1) # BD x N x N
	pad_mask_NJ = tl.load(pad_mask_ptr_NJ, mask=mask_NJ, other=0).to(tl.int1) # BD x N x N

	# combine masks and ensure not to compute self distance
	pad_mask_IJ = pad_mask_NI & pad_mask_NJ & (NI!=NJ) # BD x N x N

	# load into SRAM
	coords_NI_x = tl.load(coords_ptr_NI_x, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N
	coords_NI_y = tl.load(coords_ptr_NI_y, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N
	coords_NI_z = tl.load(coords_ptr_NI_z, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N

	coords_NJ_x = tl.load(coords_ptr_NJ_x, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N
	coords_NJ_y = tl.load(coords_ptr_NJ_y, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N
	coords_NJ_z = tl.load(coords_ptr_NJ_z, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N

	# compute distances
	dist_x = (coords_NI_x - coords_NJ_x) * (coords_NI_x - coords_NJ_x) # BD x N x N
	dist_y = (coords_NI_y - coords_NJ_y) * (coords_NI_y - coords_NJ_y) # BD x N x N
	dist_z = (coords_NI_z - coords_NJ_z) * (coords_NI_z - coords_NJ_z) # BD x N x N

	dist = (dist_x + dist_y + dist_z).sqrt() # BD x N x N

	# get wavenumbers
	wavenumber_ptr = wavenumber_ptr + (D*stride_wavenumber) + (NI*0) + (NJ*0)# BD x N x N
	wavenumber = tl.load(wavenumber_ptr, mask=pad_mask_IJ, other=0).to(tl.float64) # BD x N x N

	# compute phase
	phase = (wavenumber*dist) % (2*pi) # BD x N x N 

	# compute real and imag parts
	real = tl.cos(phase) / tl.where(pad_mask_IJ, dist, float("inf")) # BD x N x N
	imag = tl.sin(phase) / tl.where(pad_mask_IJ, dist, float("inf")) # BD x N x N

	# compute d_model index
	D_real = 1 + (D * 2 - 1) # BD x 1 x 1
	D_imag = 1 + (D * 2) # BD x 1 x 1

	# prepare output pointers (only write to NI)
	out_ptr_real = (out_ptr + (B*stride_out_B) + (D_real*stride_out_D) + (NI*stride_out_N) + (NJ*0)) # BD x N x N
	out_ptr_imag = (out_ptr + (B*stride_out_B) + (D_imag*stride_out_D) + (NI*stride_out_N) + (NJ*0)) # BD x N x N

	# add real and imag parts to output tensor
	tl.atomic_add(out_ptr_real, real, mask=pad_mask_IJ)
	tl.atomic_add(out_ptr_imag, imag, mask=pad_mask_IJ)

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
	BLOCK_NI = 16    # N_i
	BLOCK_NJ = 16    # N_j
	BLOCK_BD = 4     # batch x d_model
	# BLOCK_E x BLOCK_B x BLOCK_D <= 1024

	# compute the grid size
	grid_NI = (N // BLOCK_NI) + 1   # number of edge blocks
	grid_NJ = (N // BLOCK_NJ) + 1   # number of edge blocks
	grid_BD = (batch * num_wl // BLOCK_BD) + 1      # number of feature blocks

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

def protein_to_wavefunc_torch(coords, d_model, min_wl, max_wl, base, mask=None, max_splits=16):
	'''
	converts the alpha carbon coordinates of a protein into a tensor of 
	wavefunction outputs.
	converts a batch x N x 3 tensor to a batch x N x d_model tensor.
	each feature for a Ca is their output of the wave function with a specific wavelength
	each output gets two features, one for the real part, and another for the imaginary part
	the wave function is a superposition of Green's functions, treating each Ca as a point source
	note that this function is very memory conscious, planning on computing in portions along the wavelength (d_model)
	dimension if required

	Args:
		coords (torch.Tensor):              tensor containing batches of Ca coords. 
											size = batch x N x 3 
		key_padding_mask (torch.Tensor):    tenor containing key padding mask
											size = batch x N 
		d_model (int):						features to create. d_model = number_of_wavelengths*2
		return_wl (bool): 					whether to return the wavelengths used, useful for plotting 
		min_wl (float):						minimum wavelength to use
		max_wl (float):						maximum wavelength to use
		base (int|float):					wavelengths are sampled logarithmically, chooses the base to use
	
	Returns:
		features (torch.Tensor):    tensor containing batches of token (Ca) features.
									size = batch x N x 512
	'''


	# get shape
	batch, N, _ = coords.shape

	# **GET PAIRWISE DISTANCES**

	# get the euclidean distances ; batch x N x 3 --> batch x N x N 
	pw_dists = torch.sqrt_((coords.unsqueeze(1) - coords.unsqueeze(2)).pow_(2).sum(dim=-1)).to(coords.device)
	
	# diagonal set to 1 to avoid division by zero
	pw_dists += torch.eye(pw_dists.size(1), device=coords.device).unsqueeze(0).expand(batch, -1, -1)

	# set masked values to inf to exclude from wave function calculation (1/inf = 0)
	pw_dist_mask = mask.unsqueeze(1) | mask.unsqueeze(2) # batch x N x N
	pw_dists.masked_fill_(pw_dist_mask, float('inf'))
	
	# **DEFINE WAVELENGTHS**

	# Create a tensor of wavelengths
	wl_tensor = get_wavelengths(min_wl, max_wl, d_model, device=coords.device, base=base) # num_wl, 

	# **COMPUTE GREENS FN

	# split along wavelengths dimension, as these computations are independant from each other
	splits = max(1, int(2**int( math.log( min(1, (N / 10000)) * max_splits, 2 ) )))
	sub_wl = d_model // 2 // splits
	wl_splits = [wl_tensor[step*sub_wl:(1+step)*sub_wl] for step in range(splits)]

	# prepare pw_dists by expanding it to include num_wl
	# batch x N x N --> batch x N x N x num_wl
	pw_dists = pw_dists.unsqueeze(-1).expand(-1, -1, -1, sub_wl)
	pw_dist_mask = pw_dist_mask.unsqueeze(-1).expand(-1, -1, -1, sub_wl)

	del wl_tensor
	torch.cuda.empty_cache()

	real, imag = [], []

	for wavelengths in wl_splits:

		# convert wavelengths to k_values and set up for broadcasting along middle two dimensions (N and N from pw_dists)
		k_values = (2 * torch.pi / wavelengths).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(pw_dists.size(0), pw_dists.size(1), pw_dists.size(1), -1) # num_wl, --> batch x N x N x num_wl

		# compute phase ; batch x N x N x num_wl
		phase = pw_dists.masked_fill(pw_dist_mask, 0.0).mul_(k_values)
		
		# need to compute real and imaginary parts seperately for memory efficiency
		# **REAL PART CALCULATION**

		# compute the Green's function real part
		greens_fn_real = phase.cos_().div_(pw_dists) # batch x N x N x num_wl

		# delete REFERENCE to phase, greens_fn_real still in memory
		del phase
		torch.cuda.empty_cache()

		# take care of padded values and identity positions
		batch, N, _, wl = greens_fn_real.shape
		greens_fn_real.masked_fill_(torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | pw_dist_mask, 0.0) # batch x N x N x num_wl
		
		# superpose all other Ca point sources to each Ca
		superpositions_real = greens_fn_real.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
		real.append(superpositions_real)
		
		del greens_fn_real
		del superpositions_real
		torch.cuda.empty_cache()

		# **IMAGINARY PART CALCULATION**

		phase = pw_dists.masked_fill(pw_dist_mask, 0.0).mul_(k_values) # batch x N x N x num_wl
		greens_fn_imag = phase.sin_().div_(pw_dists) # batch x N x N x num_wl

		# del pw_dists
		del phase
		torch.cuda.empty_cache()

		greens_fn_imag.masked_fill_(torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | pw_dist_mask, 0.0) # batch x N x N x num_wl
		superpositions_imag = greens_fn_imag.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
		imag.append(superpositions_imag)

		del greens_fn_imag	
		del superpositions_imag
		torch.cuda.empty_cache()

	del pw_dist_mask
	del batch, N, _, wl
	torch.cuda.empty_cache()

	# join the k dimensions together | batch x N x num_wl
	real = torch.cat(real, dim=-1)
	imag = torch.cat(imag, dim=-1)

	# **CONCAT INTO FEATURES**

	# every k value gets two features
	# want real and imaginary parts next to each other (for a single k)
	features = torch.stack((real, imag), dim=-1) # batch x N x d_model // 2 x 2

	del real, imag
	torch.cuda.empty_cache()
	
	features = features.view(features.size(0), features.size(1), -1)  # Final shape: batch x N x d_model

	# normalize so that max is one and sign is preserved
	features.div_(features.abs().max(dim=1, keepdim=True).values)

	return features

def get_wavelengths(min_wl=3.7, max_wl=20, d_model=512, base=20, device="cpu"):

	# short range wavelengths get 128 wave functions, medium get 96, and long get 32. each wave function
	# creates two features, one real and one imaginary, for a total of num_wl*2 = 512 features
	# create evenly spaced tensors from 0 to 1 of shape 1,  
	num_wl = (d_model // 2)
	
	log_distribution = (torch.logspace(0, 1, num_wl, base=base, device=device) - 1) / (base - 1) # Scale [1, 2) to [0, 1)
	wavelengths = (min_wl + (log_distribution.mul_(max_wl - min_wl))).to(device) # num_wl,

	return wavelengths

def mod_d_model(wf_features, trgt_d_model):

    if wf_features.size(-1) == trgt_d_model:
    	return wf_features
    else:
        raise NotImplementedError
