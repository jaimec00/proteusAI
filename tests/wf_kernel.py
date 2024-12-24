

import torch
from utils.model_utils.featurization import protein_to_wavefunc 

import triton
import triton.language as tl
import math
from utils.test_utils import calculate_error

def main():

	# device
	device = torch.device('cuda')

	batch, N, d_model = 1, 10000, 512
	min_wl, max_wl, base = 3.7, 20, 20
	coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float64, device=device)
	mask = (torch.rand((batch, N), device=device) > 1)

	torch.cuda.synchronize()  # Ensure no ongoing GPU operations
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	triton_out = protein_to_wavefunc.apply(coords, d_model, min_wl, max_wl, base, mask)
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	torch_out = protein_to_wavefunc_torch(coords, d_model, min_wl, max_wl, base, mask, 32).to(torch.float64)
	# torch_out = triton_out
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	torch_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	torch_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	rel_error, abs_error = calculate_error(torch_out, triton_out)

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4, equal_nan=True)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

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



if __name__ == "__main__":
	main()