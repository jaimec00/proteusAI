

import torch
import math
from utils.test_utils import calculate_error, profile_func, profile_bwd 
from utils.model_utils.wf_embedding.anisotropic.cuda.wf_embedding import wf_embedding 

def main():

	# device
	device = torch.device('cuda')
	
	# set rng seed
	torch.manual_seed(0)

	# prepare inputs
	batch, N, d_model = 1, 2048, 512
	mag_type = 1
	dropout_p = 0.0
	min_wl, max_wl, base = 3.7, 20, 20
	coordsA = max_wl * torch.randn((batch, N, 3), dtype=torch.float32, device=device)
	coordsB = max_wl * torch.randn((batch, N, 3), dtype=torch.float32, device=device)
	mask = (torch.rand((batch, N), device=device) > 1)
	wavenumbers = torch.randn((d_model//2,), device=coordsA.device, dtype=torch.float32, requires_grad=True)

	# to make it easier
	params = [coordsA, coordsB, wavenumbers, mag_type, dropout_p, mask]

	# synchronize device
	torch.cuda.synchronize()  # Ensure no ongoing GPU operations

	# profiling
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 1e-2

	# profiling section for nsight compute
	torch.cuda.cudart().cudaProfilerStart()

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
	# print(cuda_out)
	# print(torch_out/cuda_out)
	# print(mask)
	# print(wavenumbers)
	# print(torch_dk)
	# print(cuda_dk)
	# print(torch_dk/cuda_dk)

def wf_embedding_torch(coordsA, coordsB, wavenumbers, mag_type=1, dropout_p=0.0, mask=None):

	# get shape and prepare inputs
	batch, N, _ = coordsA.shape

	num_wn = wavenumbers.size(0)
	d_model = num_wn*2
	mask = mask if mask is not None else torch.zeros(batch, N, dtype=torch.bool, device=coordsA.device)

	
	R = coordsA[:, :, None, :] - coordsA[:, None, :, :] # Z x N x N x 3
	R_mag = torch.linalg.vector_norm(R, dim=3, keepdim=True) # Z x N x N x 1
	R_norm = R / torch.where(R_mag==0, float("inf"), R_mag) # Z x N x N x 3 

	# distances
	distsA = R_mag.squeeze(3) # Z x N x N x 1 --> Z x N x N
	
	distsB = torch.sum(R_norm * coordsB[:, None, :, :], dim=3) # Z x N x N

	distsAB = distsA - distsB # Z x N x N

	# compute phases
	phases = distsAB[:, :, :, None] * wavenumbers[None, None, None, :] # Z x N x N x w

	# update mask
	mask = mask[:, :, None] | mask[:, None, :] | (distsA==0)

	# get the magnitudes of the wavefunctions
	if mag_type==0:
		magnitudes = torch.where(mask, torch.zeros_like(distsA), torch.ones_like(distsA))[:, :, :, None]
	elif mag_type == 1:
		magnitudes = 1 / torch.where(mask, float("inf"), distsA)[:, :, :, None] # Z x N x N x 1
	elif mag_type == 2: # log2 of 0 is -inf, so 0 dists evaluate to 1/-inf = -0
		magnitudes = 1 / torch.log2(distsA)[:, :, :, None] # Z x N x N x 1
	elif mag_type == 3:
		magnitudes = 1 / torch.where(mask, float("inf"), torch.sqrt(distsA))[:, :, :, None]

	# compute real and imag parts
	real = magnitudes * torch.cos(phases) # Z x N x N x w
	imag = magnitudes * torch.sin(phases) # Z x N x N x w

	# superpose the wavefunctions
	real_superposition = real.sum(dim=2) # Z x N x w
	imag_superposition = imag.sum(dim=2) # Z x N x w

	# convert to features
	features = torch.stack([real_superposition, imag_superposition], dim=-1).view(batch, N, d_model) # Z x N x d_model

	return features

if __name__ == "__main__":
	main()