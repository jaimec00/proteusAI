

import torch
import math 
import triton
import triton.language as tl

def main():

	# device
	device = torch.device('cuda')

	batch, N, d_model = 4, 10000, 512
	min_wl, max_wl, base = 3.7, 20, 20
	coords = 1000 * torch.randn((batch, N, 3), dtype=torch.float64, device=device)
	mask = (torch.rand((batch, N), device=coords.device) > 1)

	torch.cuda.synchronize()  # Ensure no ongoing GPU operations
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	triton_out = greens_function(coords, d_model, min_wl, max_wl, base, mask)
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	torch_out = greens_function_torch(coords, d_model, min_wl, max_wl, base, mask, 32).to(torch.float64)
	# torch_out = triton_out
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	torch_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	torch_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	error = calculate_error(torch_out, triton_out)

	# print(f"{torch_out=}\n{triton_out=}\n")

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4, equal_nan=True)}")
	print(f"triton percent error: {error:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

def calculate_error(A, B):
	# Ensure the tensors are of the same size
	assert A.size() == B.size(), f"Tensors must have the same size, not {A.shape=} and {B.shape=}"

	# Calculate absolute error and normalize by the ground truth sum
	absolute_error = torch.abs(A - B)
	relative_error = absolute_error / torch.abs(A)  # Element-wise relative error
	error_percentage = (torch.sum(absolute_error) / torch.sum(torch.abs(A))) * 100

	return error_percentage

def greens_function_torch(coords, d_model, min_wl, max_wl, base, mask, max_splits):

	batch, N, space = coords.shape
	dists = torch.cdist(coords, coords)
	mask = mask[:, :, None] | mask[:, None, :] | (torch.eye(N, device=dists.device)[None, :, :] == 1)
	dists = dists.masked_fill_(mask, 1)

	wl_tensor = (min_wl + (torch.logspace(0, 1, d_model//2, base=base, device=dists.device) - 1) / (base - 1) * (max_wl - min_wl))
	# split along wavelengths dimension, as these computations are independant from each other
	splits = max(1, int(2**int( math.log( min(1, batch*N / 10000) * max_splits, 2  ))))
	sub_wl = d_model // 2 // splits
	wl_splits = [wl_tensor[step*sub_wl:(1+step)*sub_wl] for step in range(splits)]

	# prepare dists by expanding it to include num_wl
	# batch x N x N --> batch x N x N x num_wl
	dists = dists.unsqueeze(-1).expand(-1, -1, -1, sub_wl)
	mask = mask.unsqueeze(-1).expand(-1, -1, -1, sub_wl)

	real, imag = [], []


	for wavelengths in wl_splits:

		# convert wavelengths to k_values and set up for broadcasting along middle two dimensions (N and N from dists)
		k_values = (2 * torch.pi / wavelengths).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(dists.size(0), dists.size(1), dists.size(1), -1) # num_wl, --> batch x N x N x num_wl

		# compute phase ; batch x N x N x num_wl
		phase = dists.masked_fill(mask, 0.0).mul_(k_values)
		
		# need to compute real and imaginary parts seperately for memory efficiency
		# **REAL PART CALCULATION**

		# compute the Green's function real part
		greens_fn_real = phase.cos_().div_(dists) # batch x N x N x num_wl

		# delete REFERENCE to phase, greens_fn_real still in memory
		del phase
		torch.cuda.empty_cache()

		# take care of padded values and identity positions
		batch, N, _, wl = greens_fn_real.shape
		greens_fn_real.masked_fill_(torch.eye(N, device=dists.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | mask, 0.0) # batch x N x N x num_wl
		
		# superpose all other Ca point sources to each Ca
		superpositions_real = greens_fn_real.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
		real.append(superpositions_real)
		
		del greens_fn_real
		del superpositions_real
		torch.cuda.empty_cache()

		# **IMAGINARY PART CALCULATION**

		phase = dists.masked_fill(mask, 0.0).mul_(k_values) # batch x N x N x num_wl
		greens_fn_imag = phase.sin_().div_(dists) # batch x N x N x num_wl

		# del dists
		del phase
		torch.cuda.empty_cache()

		greens_fn_imag.masked_fill_(torch.eye(N, device=dists.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | mask, 0.0) # batch x N x N x num_wl
		superpositions_imag = greens_fn_imag.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
		imag.append(superpositions_imag)

		del greens_fn_imag
		del superpositions_imag
		torch.cuda.empty_cache()

	del mask
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


@triton.jit
def _greens_function_kernel(
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

	# get batch, d_model//2 indices
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
	
	# get pointers for masks
	pad_mask_ptr_NI = pad_mask_ptr + (B*stride_pad_mask_B) + (NI*stride_pad_mask_N) + (NJ*0) # BD x N x N
	pad_mask_ptr_NJ = pad_mask_ptr + (B*stride_pad_mask_B) + (NJ*stride_pad_mask_N) + (NI*0) # BD x N x N

	# get masks
	pad_mask_NI = tl.load(pad_mask_ptr_NI, mask=mask_NI, other=0).to(tl.int1) # BD x N x N
	pad_mask_NJ = tl.load(pad_mask_ptr_NJ, mask=mask_NJ, other=0).to(tl.int1) # BD x N x N

	# combine masks
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

def greens_function(coords, d_model, min_wl, max_wl, base, mask=None):
	
	# checks
	assert (coords.dim() == 3) and (coords.size(2) == 3), f"coords must be of shape (batch x N x 3), not {coords.shape}"
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model}"
	
	# prepare data
	batch, N, space = coords.shape
	coords = coords.to(torch.float64).contiguous()
	num_wl = d_model//2
	wavelengths = (min_wl + (torch.logspace(0, 1, num_wl, base=base, device=coords.device, dtype=torch.float64) - 1) / (base - 1) * (max_wl - min_wl))
	wavenumbers = (2 * torch.pi / wavelengths).contiguous()
	mask = (~mask if mask is not None else torch.ones(batch, N, dtype=torch.bool, device=coords.device)).contiguous()
	out = torch.zeros(batch, N, d_model, dtype=torch.float64, device=coords.device).contiguous()

	# total block size should be less than number of threads per block (approx. 1024)
	BLOCK_NI = 16    # edges
	BLOCK_NJ = 16     # batch
	BLOCK_BD = 4     # d_model
	# BLOCK_E x BLOCK_B x BLOCK_D <= 1024

	grid_NI = (N // BLOCK_NI) + 1   # number of edge blocks
	grid_NJ = (N // BLOCK_NJ) + 1   # number of edge blocks
	grid_BD = (batch * num_wl // BLOCK_BD) + 1      # number of feature blocks

	# will tile the edges (will compute lower triangular matrix), batches, and d_model
	grid = (grid_NI, grid_NJ, grid_BD)

	_greens_function_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2),
									coords, coords.stride(0), coords.stride(1), coords.stride(2),
									wavenumbers, wavenumbers.stride(0),
									mask, mask.stride(0), mask.stride(1),
									batch, N, d_model, torch.pi,
									BLOCK_NI, BLOCK_NJ, BLOCK_BD
								)

	out.div_(out.abs().max(dim=1, keepdim=True).values)

	return out

if __name__ == "__main__":
	main()