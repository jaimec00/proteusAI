
import math
import torch
import torch.nn.functional as F
from utils.test_utils import calculate_error, profile_func, profile_bwd
from utils.model_utils.geometric_attn.geometric_attn import geometric_attn
import os

def main():

	# setup
	torch.manual_seed(37)
	device = torch.device("cuda")

	# prepare inputs
	batch, nheads, N, d_model = 1, 4, 8192, 512
	assert d_model%2==0 and d_model%nheads==0
	d_k = d_model // nheads
	min_wl, max_wl, base = 3.7, 20, 20

	coords = 3.7 * torch.normal(mean=0, std=1, size=(batch, N, 1), dtype=torch.float32, device=device).expand(batch, N, 3) # batch x N x 3
	# spreads = torch.logspace(0, 1, nheads, base, dtype=torch.float32, device=coords.device, requires_grad=True).unsqueeze(0).expand(batch, -1)

	spreads = torch.full((nheads, ), 3, device=coords.device, dtype=torch.float32, requires_grad=True)

	mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N
	Q = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	
	params = [Q, K, V, coords, spreads, mask]

	# prepare for recording mem and time
	torch.cuda.synchronize()  
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 0

	# autotune triton configs before profiling, allows optimal configs and triton to use cache rather than recompiling
	print("autotuning:\n")

	# log autotuning
	# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

	# autotune(geometric_attn, params)

	# # zero grads
	# Q.grad.zero_()
	# K.grad.zero_()
	# V.grad.zero_()
	# spreads.grad.zero_()

	print("forward pass:\n")

	# max number of floats is the intermediate attn tensor 
	num_floats = batch * nheads * N * N  * 2
	num_bytes = num_floats * 4 # assume fp32
	max_bytes = 20 * (1024**3) # 20 GB
	if num_bytes >= max_bytes:
		print("skipping torch implementation because intermediate tensor is too large")
		torch_func = geometric_attn # rerun with kernel instead of using torch
	else:
		torch_func = torch_attn

	torch_out, triton_out = test_fwd(torch_func, geometric_attn, params, start_event, end_event, atol, rtol)
	print("\nbackward pass:\n")

	test_bwd(torch_out.sum(), triton_out.sum(), Q, K, V, spreads, start_event, end_event, atol, rtol)

	# Q.grad.zero_()
	# K.grad.zero_()
	# V.grad.zero_()
	# spreads.grad.zero_()

	# print("\ntesting with dropout: \n")

	# # need seperate test for dropout, simply to ensure the mask is reproducible. not the best test but just run the kernel
	# # twice and see if get the same results
	# dropout = 0.1 # note that for practice rng seed is dynamically generated, make sure you hard code it in the fwd pass to test this
	# dropout_params = params + [dropout]
	
	# print("dropout forward: \n")
	# run1_fwd, run2_fwd = test_fwd(geometric_attn, geometric_attn, dropout_params, start_event, end_event, atol, rtol)
	# print("\ndropout bwd: \n")

	# test_bwd(run1_fwd.sum(), run2_fwd.sum(), Q, K, V, spreads, start_event, end_event, atol, rtol)

def autotune(func, params):

	# autotune _attn_fwd
	out = func(*params)
	
	# autotune _attn_bwd
	out.sum().backward(retain_graph=False)

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

def test_bwd(torch_loss, triton_loss, Q, K, V, spreads_p, start_event, end_event, atol, rtol):

	# torch
	torch_time, torch_memory = profile_bwd(torch_loss, start_event, end_event)
	torch_dQ, torch_dK, torch_dV, torch_d_spreads = [i.grad.clone() for i in [Q, K, V, spreads_p]]

	# # zero grads
	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()
	spreads_p.grad.zero_()

	# triton
	triton_time, triton_memory = profile_bwd(triton_loss, start_event, end_event)
	triton_dQ, triton_dK, triton_dV, triton_d_spreads = [i.grad.clone() for i in [Q, K, V, spreads_p]]

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

	d_spreads_rel_error, d_spreads_abs_error = calculate_error(torch_d_spreads, triton_d_spreads)
	print(f"d_spreads is correct: {torch.allclose(triton_d_spreads, torch_d_spreads, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton d_spreads absolute error: {d_spreads_abs_error:.5f}")
	print(f"triton d_spreads relative error: {d_spreads_rel_error:.5f}")
	print(f"triton d_spreads percent error: {d_spreads_rel_error*100:.5f}%\n")

	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

def test_dropout():
	pass

def torch_attn(Q, K, V, coords, spreads, mask=None):

	assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
	batch, nheads, N, d_k = Q.shape
	
	d_model = d_k * nheads
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
	
	assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

	assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
	assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"
	mask = torch.zeros(batch, N) if mask is None else mask # batch x N

	Q = Q.contiguous()
	K = K.contiguous()
	V = V.contiguous()
	coords = coords.contiguous()
	spreads = spreads.contiguous()
	mask = mask.contiguous()

	S = torch.matmul(Q, K.transpose(2,3)) / (2*(d_k**0.5)) # batch x nheads x N x N

	dists = torch.sqrt(torch.sum((coords[:, :, None, :] - coords[:, None, :, :])**2, axis=3))[:, None, :, :]

	rbfs = torch.exp(-(dists**2)/(2*(spreads[None, :, None, None]**2)))
	dist_mask = (rbfs<=0.01)
	rbfs = torch.where(S<0, 2-rbfs, rbfs + 1)

	attn_mask = mask[:, None, :, None] | mask[:, None, None, :] | dist_mask
	S = torch.where(attn_mask, float("-inf"), S*rbfs)

	P = torch.softmax(S, dim=-1)

	out = torch.matmul(P, V) # batch x nheads x N x d_k
	
	return out


if __name__ == '__main__':
	main()