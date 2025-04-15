

import torch
import math
from utils.test_utils import calculate_error, profile_func, profile_bwd 
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.learnable_aa.learnable_wavenumber.cuda.wf_embedding import wf_embedding 

def main():

	# device
	device = torch.device('cuda')
	
	# set rng seed
	torch.manual_seed(0)

	# setup parameters
	batch, N, d_model, num_classes = 2, 1024, 256, 20
	num_wn = d_model // 2
	min_wl, max_wl, base = 3.7, 20, 20 # wavelengths
	
	# setup alpha carbon coords
	coordsA = max_wl * torch.rand((batch, N, 3), dtype=torch.float32, device=device)
	
	# setup beta carbon coords relative to alpha carbons. normalize so magnitudes are 1
	coordsB = torch.randn((batch, N, 3), dtype=torch.float32, device=device)
	coordsB = coordsB / torch.linalg.vector_norm(coordsB, dim=2, keepdim=True)
	
	# assign aa labels and magnitudes, one for each aa, for each wavenumber
	aa_labels = torch.randint(0,num_classes, (batch, N), dtype=torch.int32, device=device)
	aa_magnitudes = torch.rand((num_wn, num_classes), dtype=torch.float32, device=device, requires_grad=True)

	# setup wavenumbers
	wavenumbers = torch.randn((num_wn,), device=coordsA.device, dtype=torch.float32, requires_grad=True)

	# setup mask
	mask = (torch.rand((batch, N), device=device) > 1)

	# put them all into a list
	params = [coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, mask]

	# synchronize device
	torch.cuda.synchronize()  # Ensure no ongoing GPU operations

	# profiling
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 1e-2

	# profiling section for nsight compute
	torch.cuda.cudart().cudaProfilerStart()

	# run the cuda implementation
	cuda_out, cuda_time, cuda_memory = profile_func(wf_embedding, params, start_event, end_event)	

	# end profiling section
	torch.cuda.cudart().cudaProfilerStop()

	# run the torch implementation
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
	cuda_d_aa = aa_magnitudes.grad.clone()
	cuda_d_k = wavenumbers.grad.clone()
	aa_magnitudes.grad.zero_()
	wavenumbers.grad.zero_()

	# run torch bwd
	torch_time, torch_mem = profile_bwd(torch_out.sum(), start_event, end_event)
	torch_d_aa = aa_magnitudes.grad.clone()
	torch_d_k = wavenumbers.grad.clone()

	# summarize bwd stats
	rel_error, abs_error = calculate_error(torch_d_aa, cuda_d_aa)
	print(f"cuda d_aa is correct: {torch.allclose(cuda_d_aa, torch_d_aa, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"cuda d_aa absolute error: {abs_error:.5f}")
	print(f"cuda d_aa relative error: {rel_error:.5f}")
	print(f"cuda d_aa percent error: {rel_error*100:.5f}%\n")

	# summarize bwd stats
	rel_error, abs_error = calculate_error(torch_d_k, cuda_d_k)
	print(f"cuda d_k implementation is correct: {torch.allclose(cuda_d_k, torch_d_k, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"cuda d_k absolute error: {abs_error:.5f}")
	print(f"cuda d_k relative error: {rel_error:.5f}")
	print(f"cuda d_k percent error: {rel_error*100:.5f}%\n")

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
	# print(torch_d_aa)
	# print(cuda_d_aa)
	torch.set_printoptions(threshold=float("inf"), precision=4)
	print((torch_d_aa/cuda_d_aa).mean(dim=1))
	print(wavenumbers)

def wf_embedding_torch(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, mask=None):

	# get shape and prepare inputs
	batch, N, _ = coordsA.shape
	num_wn = wavenumbers.size(0)
	_, num_classes = aa_magnitudes.shape # wavenumbers x k is for coalscing in the kernel

	d_model = num_wn*2
	mask = mask if mask is not None else torch.zeros(batch, N, dtype=torch.bool, device=coordsA.device)
	
	# format is [batch, observer, source, coords]
	R = coordsA[:, :, None, :] - coordsA[:, None, :, :] # Z x N x N x 3

	# compute unit vecotrs of Ca distances
	R_norm = torch.linalg.vector_norm(R, dim=3, keepdim=True) # Z x N x N x 1
	R_unit = R / torch.where(R_norm==0, float("inf"), R_norm) # Z x N x N x 3 

	# distances are just the magnitudes of the raw difference vectors
	distsA = R_norm # Z x N x N x 1
	
	# scale beta carbons by their AA use one hot to keep them differentiable
	# aa_labels : Z x N
	# aa_magnitudes : K x AA
	# coords B should end up being Z x N x K x 3
	
	aa_onehot = torch.nn.functional.one_hot(aa_labels.long(), num_classes=num_classes) # Z x N x A
	coordsB_magnitudes = (aa_onehot[:, :, None, :] * aa_magnitudes[None, None, :, :]).sum(dim=3) # Z x N x 1 x A * 1 x 1 x K x A --> Z x N x K 
	coordsB = coordsB[:, :, None, :] * coordsB_magnitudes[:, :, :, None] # Z x N x K x 3

	distsB = torch.sum(R_unit[:, :, :, None, :] * coordsB[:, None, :, :, :], dim=4) # Z x N x N x K

	distsAB = distsA - distsB # Z x N x N x K

	# compute phases
	phases = distsAB * wavenumbers[None, None, None, :] # Z x N x N x K

	# update mask
	mask = mask[:, :, None, None] | mask[:, None, :, None] | (distsA==0) # Z x N x N x 1

	# get the magnitudes of the wavefunctions
	magnitudes = 1 / torch.where(mask, float("inf"), distsA) # Z x N x N x 1

	# compute real and imag parts
	real = magnitudes * torch.cos(phases) # Z x N x N x K
	imag = magnitudes * torch.sin(phases) # Z x N x N x K

	# superpose the wavefunctions along the sources dim
	real_superposition = real.sum(dim=2) # Z x N x K
	imag_superposition = imag.sum(dim=2) # Z x N x K

	# convert to features
	features = torch.stack([real_superposition, imag_superposition], dim=-1).view(batch, N, d_model) # Z x N x d_model

	return features

if __name__ == "__main__":
	main()