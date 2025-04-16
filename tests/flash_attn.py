
import math
import torch
import torch.nn.functional as F
from utils.test_utils import calculate_error, profile_func, profile_bwd
from utils.model_utils.attn.cross_attn import flash_attn
import os

def main():

	# setup
	#torch.manual_seed(37)
	device = torch.device("cuda")

	# prepare inputs
	batch, nheads, N, d_model = 1, 4, 4096, 512
	assert d_model%2==0 and d_model%nheads==0
	d_k = d_model // nheads

	mask = torch.rand((batch, N), device=device) > 1 # batch x N
	Q = torch.normal(mean=0, std=1, size=(batch, nheads, N//2, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # batch x nhead x N x d_k 
	
	params = [Q, K, V, mask]

	# prepare for recording mem and time
	torch.cuda.synchronize()  
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 0

	# autotune triton configs before profiling, allows optimal configs and triton to use cache rather than recompiling
	# print("autotuning:\n")

	# # log autotuning
	# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

	# autotune(flash_attn, params)

	# # zero grads
	# Q.grad.zero_()
	# K.grad.zero_()
	# V.grad.zero_()

	print("forward pass:\n")

	torch_out, triton_out = test_fwd(torch_attn, flash_attn, params, start_event, end_event, atol, rtol)
	print("\nbackward pass:\n")

	test_bwd(torch_out.sum(), triton_out.sum(), Q, K, V, start_event, end_event, atol, rtol)

def autotune(func, params):

	# autotune _attn_fwd
	out = func(*params)
	
	# autotune _attn_bwd
	out.sum().backward(retain_graph=False)

def test_fwd(torch_attn, attn, params, start_event, end_event, atol, rtol):

	triton_out, triton_time, triton_memory = profile_func(attn, params, start_event, end_event)
	torch_out, torch_time, torch_memory = profile_func(torch_attn, params, start_event, end_event)
	rel_error, abs_error = calculate_error(torch_out, triton_out)

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
	torch_dQ, torch_dK, torch_dV = [i.grad.clone() for i in [Q, K, V]]

	# # zero grads
	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()

	# triton
	triton_time, triton_memory = profile_bwd(triton_loss, start_event, end_event)
	triton_dQ, triton_dK, triton_dV = [i.grad.clone() for i in [Q, K, V]]

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

def test_dropout():
	pass

def torch_attn(Q, K, V, mask=None):

	batch, nheads, N, d_k = Q.shape
	
	d_model = d_k * nheads
	assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
	
	mask = torch.zeros(batch, N, device=Q.device) if mask is None else mask # batch x N

	Q = Q.contiguous()
	K = K.contiguous()
	V = V.contiguous()

	S = torch.matmul(Q, K.transpose(2,3)) / (d_k**0.5) # batch x nheads x N x N

	S = torch.where(mask[:, None, None, :], float("-inf"), S) 
	P = torch.softmax(S, dim=-1)
	out = torch.matmul(P, V) # batch x nheads x N x d_k

	return out

if __name__ == '__main__':
	main()
