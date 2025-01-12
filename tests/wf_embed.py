

import torch
from utils.model_utils.wf_embedding import wf_embedding 

import triton
import triton.language as tl
import math
from utils.test_utils import calculate_error, profile_func, profile_bwd 

def main():

	# device
	device = torch.device('cuda')

	batch, N, d_model = 3, 1024, 512
	min_wl, max_wl, base = 3.7, 20, 20
	coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float32, device=device)
	mask = (torch.rand((batch, N), device=device) > 1)

	# prepare the wavenumber values
	# num_wl = int(d_model//2) # define the number of wave functions to compute
	# wavelengths = (min_wl + (torch.logspace(0, 1, num_wl, base=base, device=coords.device, dtype=torch.float32) - 1) / (base - 1) * (max_wl - min_wl))
	# wavenumbers = (2 * torch.pi / wavelengths).contiguous()
	wavenumbers = torch.randint(1,10,(d_model//2,), device=coords.device, dtype=torch.float32, requires_grad=True)
	# torch.linspace(1,3,d_model//2, device=coords.device, dtype=torch.float32, requires_grad=True)

	params = [coords, wavenumbers, mask]

	# for autotuning
	wf_embedding(*params)

	# wavenumbers.grad.zero_()

	torch.cuda.synchronize()  # Ensure no ongoing GPU operations
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-4, 0

	triton_out, triton_time, triton_memory = profile_func(wf_embedding, params, start_event, end_event)
	torch_out, torch_time, torch_memory = profile_func(wf_embedding_torch, params, start_event, end_event)

	rel_error, abs_error = calculate_error(torch_out, triton_out)
	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")



	triton_time, triton_mem = profile_bwd(triton_out.sum(), start_event, end_event)

	triton_dk = wavenumbers.grad.clone()
	wavenumbers.grad.zero_()

	torch_time, torch_mem = profile_bwd(torch_out.sum(), start_event, end_event)

	torch_dk = wavenumbers.grad.clone()

	rel_error, abs_error = calculate_error(torch_dk, triton_dk)
	print(f"triton implementation is correct: {torch.allclose(triton_dk, torch_dk, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")
	# print(triton_dk, torch_dk)


def autotune_wf():
	pass

def wf_embedding_torch(coords, wavenumbers, mask=None):

	# get shape
	batch, N, _ = coords.shape
	num_wn = wavenumbers.size(0)
	d_model = num_wn*2
	mask = mask if mask is not None else torch.zeros(batch, N, dtype=torch.bool, device=coords.device)

	dists = torch.sqrt(torch.sum((coords[:, :, None, :] - coords[:, None, :, :])**2, dim=3)) # Z x N x N
	
	phases = dists[:, :, :, None] * wavenumbers[None, None, None, :] # Z x N x N x w
	mask = mask[:, :, None] | mask[:, None, :] | (dists==0)
	magnitudes = 1 / torch.where(mask, float("inf"), dists)[:, :, :, None] # Z x N x N x 1

	real = magnitudes * torch.cos(phases) # Z x N x N x w
	imag = magnitudes * torch.sin(phases) # Z x N x N x w

	real_superposition = real.sum(dim=1) # Z x N x w
	imag_superposition = imag.sum(dim=1) # Z x N x w


	features = torch.stack([real_superposition, imag_superposition], dim=-1).view(batch, N, d_model) # Z x N x d_model

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