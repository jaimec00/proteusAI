
import math
import torch
import torch.nn.functional as F
from utils.test_utils import calculate_error, profile_func, profile_bwd
from utils.model_utils.gaussian_attn import attn

def main():

	# setup
	torch.manual_seed(37)
	device = torch.device("cuda")

	# prepare inputs
	batch, nheads, N, d_model = 1, 8, 10000, 512
	assert d_model%2==0 and d_model%nheads==0
	d_k = d_model // nheads
	min_wl, max_wl, base = 3.7, 20, 20
	min_rbf, max_rbf = 0.1, 0.9

	coords = 3.7 * torch.normal(mean=0, std=1, size=(batch, N, 1), dtype=torch.float32, device=device).expand(batch, N, 3) # batch x N x 3
	spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float32, device=coords.device) - 1) / (base-1) * (max_wl-min_wl) # nheads,
	mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N
	context_mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N
	Q = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	params = [Q, K, V, coords, spreads, mask, context_mask, min_rbf, max_rbf]

	# prepare for recording mem and time
	torch.cuda.synchronize()  
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 0

	# autotune triton configs before profiling, allows optimal configs and triton to use cache rather than recompiling
	print("autotuning:\n")

	autotune(attn, params)

	# zero grads
	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()	

	print("forward pass:\n")

	num_floats = batch * nheads * N * d_k  
	num_bytes = num_floats * 4 # assume fp32
	max_bytes = 20 * (1024**3) # 20 GB
	if num_bytes >= max_bytes:
		torch_func = attn # rerun with kernel instead of using torch
	else:
		torch_func = torch_attn

	torch_out, triton_out = test_fwd(torch_func, attn, params, start_event, end_event, atol, rtol)

	print("\nbackward pass:\n")

	test_bwd(torch_out.sum(), triton_out.sum(), Q, K, V, start_event, end_event, atol, rtol)

def autotune(func, params):

	# autotune _attn_fwd
	out = attn(*params)
	
	# autotune _attn_fwd
	out.sum().backwards()

def test_fwd(torch_attn, attn, params, start_event, end_event, atol, rtol):

	triton_out, triton_time, triton_memory = profile_func(attn, params, start_event, end_event)
	torch_out, torch_time, torch_memory = profile_func(torch_attn, params, start_event, end_event)
	rel_error, abs_error = calculate_error(torch_out, triton_out)

	# print(f"{torch_out}\n{triton_out}")

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

	return torch_out, triton_out

def test_bwd(torch_loss, triton_loss, Q, K, V, start_event, end_event, atol, rtol):
	# torch
	torch_time, torch_memory = profile_bwd(torch_loss, start_event, end_event)
	torch_dQ, torch_dK, torch_dV = [Q.grad.clone(), K.grad.clone(), V.grad.clone()]

	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()	

	# triton
	triton_time, triton_memory = profile_bwd(triton_loss, start_event, end_event)
	triton_dQ, triton_dK, triton_dV = [Q.grad.clone(), K.grad.clone(), V.grad.clone()]

	dQ_rel_error, dQ_abs_error = calculate_error(torch_dQ, triton_dQ)
	print(f"dQ is correct: {torch.allclose(triton_dQ, torch_dQ, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton dQ absolute error: {dQ_abs_error:.5f}")
	print(f"triton dQ relative error: {dQ_rel_error:.5f}")
	print(f"triton dQ percent error: {dQ_rel_error*100:.5f}%\n")

	dK_rel_error, dK_abs_error = calculate_error(torch_dK, triton_dK)
	print(f"dK is correct: {torch.allclose(triton_dK, torch_dK, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton dK absolute error: {dK_abs_error:.5f}")
	print(f"triton dK relative error: {dK_rel_error:.5f}")
	print(f"triton dK percent error: {dK_rel_error*100:.5f}%\n")

	dV_rel_error, dV_abs_error = calculate_error(torch_dV, triton_dV)
	print(f"dV is correct: {torch.allclose(triton_dV, torch_dV, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton dV absolute error: {dV_abs_error:.5f}")
	print(f"triton dV relative error: {dV_rel_error:.5f}")
	print(f"triton dV percent error: {dV_rel_error*100:.5f}%\n")

	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")


def torch_attn(Q, K, V, coords, spreads, mask=None, context_mask=None, min_rbf=0.1, max_rbf=0.9):

	assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
	batch, nheads, N, d_k = Q.shape
	
	d_model = d_k * nheads
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
	
	assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

	assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
	assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"
	mask = torch.zeros(batch, N) if mask is None else mask # batch x N
	context_mask = mask if context_mask is None else context_mask # batch x N
	min_dists = torch.sqrt(2*(spreads**2)*math.log(1/max_rbf)).contiguous()
	max_dists = torch.sqrt(2*(spreads**2)*math.log(1/min_rbf)).contiguous()

	Q = Q.contiguous()
	K = K.contiguous()
	V = V.contiguous()
	coords = coords.contiguous()
	spreads = spreads.contiguous()
	mask = mask.contiguous()
	context_mask = context_mask.contiguous()

	S = torch.matmul(Q, K.transpose(2,3)) / (d_k**0.5) # batch x nheads x N x N

	dists = torch.sqrt(torch.sum((coords[:, :, None, :] - coords[:, None, :, :])**2, axis=3))[:, None, :, :]
	dists_mask = dists > (max_dists[None, :, None, None])
	rbfs = torch.exp(-(dists**2)/(2*(spreads[None, :, None, None]**2)))
	rbfs = torch.where(dists <= min_dists[None, :, None, None], 1.0, rbfs) # clamp mins to one
	rbfs = torch.where(S<0, (1+min_rbf)-rbfs, rbfs)

	print( 1 - (dists_mask.sum(-1).sum(-1)/(N**2)))

	attn_mask = mask[:, None, :, None] | context_mask[:, None, None, :] | dists_mask
	S = torch.where(attn_mask, float("-inf"), S*rbfs)

	S_max = S.max(dim=-1, keepdim=True).values
	P = torch.where(S_max == float("-inf"), 0.0, F.softmax(S-S_max, dim=-1) ) 
	# P = torch.softmax(S, dim=-1)

	out = torch.matmul(P, V) # batch x nheads x N x d_k
	
	return out


if __name__ == '__main__':
	main()