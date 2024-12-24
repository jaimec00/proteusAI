

import torch
import torch.nn.functional as F
from utils.test_utils import calculate_error
from utils.model_utils.attn import attn

def main():

	torch.manual_seed(37)
	device = torch.device("cuda")

	batch, nheads, N, d_model = 1, 4, 128, 512
	assert d_model%2==0 and d_model%nheads==0
	d_k = d_model // nheads
	min_wl, max_wl, base = 3.7, 20, 20

	coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float64, device=device) # batch x N x 3
	spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float32, device=coords.device) - 1) / (base-1) * (max_wl-min_wl) # nheads,
	mask = torch.rand((batch, N), device=coords.device) > 0.8 # batch x N

	# getting numerical instability with exponential computations, so will apply 
	# layer norm after projections in real model, but for now just initialize a 
	# normal distribution to test
	Q = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 

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

	triton_out = attn.apply(Q, K, V, coords, spreads, mask) # batch x N x d_model

	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	rel_error, abs_error = calculate_error(torch_out, triton_out)
	atol, rtol = 1e-2, 0

	print("forward pass:\n\n")

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

	print("\nbackward pass:\n\n")

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	# torch
	torch_out.sum().backward()
	torch_dQ, torch_dK, torch_dV = [Q.grad.clone(), K.grad.clone(), V.grad.clone()]

	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	torch_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	torch_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()	

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	# triton
	triton_out.sum().backward()
	triton_dQ, triton_dK, triton_dV = [Q.grad.clone(), K.grad.clone(), V.grad.clone()]

	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	Q_rel_error, Q_abs_error = calculate_error(torch_dQ, triton_dQ)
	print(f"dQ is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=True)}")
	print(f"triton dQ absolute error: {Q_abs_error:.5f}")
	print(f"triton dQ relative error: {Q_rel_error:.5f}")
	print(f"triton dQ percent error: {Q_rel_error*100:.5f}%\n")

	K_rel_error, K_abs_error = calculate_error(torch_dK, triton_dK)
	print(f"dK is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=True)}")
	print(f"triton dK absolute error: {K_abs_error:.5f}")
	print(f"triton dK relative error: {K_rel_error:.5f}")
	print(f"triton dK percent error: {K_rel_error*100:.5f}%\n")

	V_rel_error, V_abs_error = calculate_error(torch_dV, triton_dV)
	print(f"dV is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=True)}")
	print(f"triton dV absolute error: {V_abs_error:.5f}")
	print(f"triton dV relative error: {V_rel_error:.5f}")
	print(f"triton dV percent error: {V_rel_error*100:.5f}%\n")

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
	attn_mask = mask[:, None, :, None] | mask[:, None, None, :]
	S = S.masked_fill(attn_mask,  float("-inf")) 
	S_max = S.max(dim=-1, keepdim=True).values
	P = torch.where(S_max == float("-inf"), 0.0, F.softmax(S-S_max, dim=-1))

	out = torch.matmul(P, V) # batch x nheads x N x d_k
	
	return out


if __name__ == '__main__':
	main()