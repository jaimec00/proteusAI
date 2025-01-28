

import torch
import math
from utils.test_utils import calculate_error, profile_func, profile_bwd 
from utils.model_utils.wf_embedding.cuda.wf_embedding import wf_embedding 

def main():

	# device
	device = torch.device('cuda')
	
	# set rng seed
	torch.manual_seed(0)

	# prepare inputs
	batch, N, d_model = 1, 2048, 512
	min_wl, max_wl, base = 3.7, 20, 20
	coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float32, device=device)
	mask = (torch.rand((batch, N), device=device) > 1)
	wavenumbers = torch.randn((d_model//2,), device=coords.device, dtype=torch.float32, requires_grad=True)

	# to make it easier
	params = [coords, wavenumbers, mask]

	# synchronize device
	torch.cuda.synchronize()  # Ensure no ongoing GPU operations

	# profiling
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 1e-2

	# if intermediate tensor size is too big for torch, just run the kernel twice
	# mem_size = 4*(batch * N * N * d_model/2) / (1024**3)
	# if mem_size > 16: # GB
	# 	print("intermediate tensor size is too big for pytorch comparison, running kernel twice...\n")
	# wf_embedding_torch = wf_embedding
	# else:
	# wf_embedding_torch = wf_embedding_torch

	# profiling section for nsight compute
	torch.cuda.cudart().cudaProfilerStart()

	# first run it a couple times to minimize overhead
	# for i in range(3):
	# 	_, _, _ = profile_func(wf_embedding, params, start_event, end_event)	


	# run the cuda and torch implementations
	cuda_out, cuda_time, cuda_memory = profile_func(wf_embedding, params, start_event, end_event)	

	# end profiling section
	torch.cuda.cudart().cudaProfilerStop()

	torch_out, torch_time, torch_memory = profile_func(wf_embedding_torch, params, start_event, end_event)
	
	# summarize results
	rel_error, abs_error = calculate_error(torch_out, cuda_out)
	print(f"cuda implementation is correct: {torch.allclose(cuda_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"cuda absolute error: {abs_error:.5f}")
	print(f"cuda relative error: {rel_error:.5f}")
	print(f"cuda percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"cuda time: {cuda_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"cuda kernel memory usage: {cuda_memory / (1024 ** 3):.3f} GB\n")

	# now profile bwd, no need for nsight compute as fwd precomputed everything and now it's just torch
	cuda_time, cuda_mem = profile_bwd(cuda_out.sum(), start_event, end_event)

	# keep the gradients then zero them for torch function
	cuda_dk = wavenumbers.grad.clone()
	wavenumbers.grad.zero_()

	# run torch bwd
	torch_time, torch_mem = profile_bwd(torch_out.sum(), start_event, end_event)
	torch_dk = wavenumbers.grad.clone()

	# summarize bwd stats
	rel_error, abs_error = calculate_error(torch_dk, cuda_dk)
	print(f"cuda implementation is correct: {torch.allclose(cuda_dk, torch_dk, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"cuda absolute error: {abs_error:.5f}")
	print(f"cuda relative error: {rel_error:.5f}")
	print(f"cuda percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"cuda time: {cuda_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"cuda kernel memory usage: {cuda_memory / (1024 ** 3):.3f} GB")

	# optional debugging prints
	# print(torch_out)
	print(cuda_out)
	# print(torch_out/cuda_out)
	# print(mask)
	# print(wavenumbers)
	# print(torch_dk)
	# print(cuda_dk)
	# print(torch_dk/cuda_dk)

def wf_embedding_torch(coords, wavenumbers, mask=None):

	# get shape and prepare inputs
	batch, N, _ = coords.shape
	num_wn = wavenumbers.size(0)
	d_model = num_wn*2
	mask = mask if mask is not None else torch.zeros(batch, N, dtype=torch.bool, device=coords.device)

	# distances
	dists = torch.sqrt(torch.sum((coords[:, :, None, :] - coords[:, None, :, :])**2, dim=3)) # Z x N x N
	
	# compute phases
	phases = dists[:, :, :, None] * wavenumbers[None, None, None, :] # Z x N x N x w

	# update mask
	mask = mask[:, :, None] | mask[:, None, :] | (dists==0)

	# get the magnitudes of the wavefunctions
	magnitudes = 1 / torch.where(mask, float("inf"), dists)[:, :, :, None] # Z x N x N x 1

	# compute real and imag parts
	real = magnitudes * torch.cos(phases) # Z x N x N x w
	imag = magnitudes * torch.sin(phases) # Z x N x N x w

	# superpose the wavefunctions
	real_superposition = real.sum(dim=1) # Z x N x w
	imag_superposition = imag.sum(dim=1) # Z x N x w

	# convert to features
	features = torch.stack([real_superposition, imag_superposition], dim=-1).view(batch, N, d_model) # Z x N x d_model

	return features

if __name__ == "__main__":
	main()