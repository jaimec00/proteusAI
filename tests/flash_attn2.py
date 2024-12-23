

import torch
from utils.test_utils import calculate_error
from utils.model_utils import triton_attn

def main():

	torch.manual_seed(37)
	device = torch.device("cuda")

	batch, nheads, N, d_model = 4, 4, 10000, 512
	assert d_model%2==0 and d_model%nheads==0
	d_k = d_model // nheads
	min_wl, max_wl, base = 3.7, 20, 20

	coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float64, device=device) # batch x N x 3
	spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float32, device=coords.device) - 1) / (base-1) * (max_wl-min_wl) # nheads,
	mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N

	# getting numerical instability with exponential computations, so will apply 
	# layer norm after projections in real model, but for now just initialize a 
	# normal distribution to test
	Q = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32) # batch x nhead x N x d_k 
	K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32) # batch x nhead x N x d_k 
	V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32) # batch x nhead x N x d_k 

	torch.cuda.synchronize()  # Ensure no ongoing GPU operations
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	torch_out = torch_attn(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	torch_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	torch_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	triton_out = triton_attn.forward(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	error = calculate_error(torch_out, triton_out)

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4, equal_nan=True)}")
	print(f"triton percent error: {error:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

def torch_attn(Q, K, V, coords, spreads, mask=None, dist_factor=3.0):

	assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
	batch, nheads, N, d_k = Q.shape
	
	d_model = d_k * nheads
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
	
	assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

	assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
	assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"
	mask = torch.zeros(batch, N) if mask is None else mask # batch x N

	S = torch.matmul(Q, K.transpose(2,3)) / (d_k**0.5) # batch x nheads x N x N
	S = S.masked_fill(mask[:, None, :, None] | mask[:, None, None, :],  float("-inf")) 
	P = torch.softmax(S, dim=-1)
	out = torch.matmul(P, V) # batch x nheads x N x d_k
	
	return out


if __name__ == '__main__':
	main()