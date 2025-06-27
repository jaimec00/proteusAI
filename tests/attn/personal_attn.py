
import math
import torch
import torch.nn.functional as F
from utils.test_utils import calculate_error, profile_func, profile_bwd
from utils.model_utils.attn.personal_attn import personal_attn
import os

def main():

	# setup
	#torch.manual_seed(37)
	device = torch.device("cuda")

	torch.manual_seed(0)
	torch.set_printoptions(threshold=float("inf"), precision=4)

	# prepare inputs
	Z, H, N, d_model = 1, 4, 500, 256
	assert d_model%2==0 and d_model%H==0
	d_k = d_model // H
	min_spread, max_spread, num_spreads = 2,22,15 # numspreads must be 2**n - 1, where n is any positive integer
	d_g = 4*num_spreads+2

	# init Ca
	Ca = 100 * torch.randn((Z, N, 3), dtype=torch.float32, device=device)# Z x N x 3
	Cb = torch.randn((Z, N, 3), dtype=torch.float32, device=device) # Z x N x 3
	Cb = Cb / torch.linalg.vector_norm(Cb, dim=2, keepdim=True) # make Cb into unit vectors 
	
	# init spreads
	spreads = torch.linspace(min_spread, max_spread, num_spreads, dtype=torch.float32, device=Ca.device)

	# mask
	mask = torch.rand((Z, N), device=Ca.device) > 1 # Z x N


	# Q K V and Wg
	Q = torch.randn((Z, H, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # Z x nhead x N x d_k 
	K = torch.randn((Z, H, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # Z x nhead x N x d_k 
	V = torch.randn((Z, H, N, d_k), device=device, dtype=torch.float32, requires_grad=True) # Z x nhead x N x d_k 
	Wg = torch.randn((H, d_g, d_k), device=device, dtype=torch.float32, requires_grad=True) # H x d_g x d_k

	# 
	params = [Q, K, V, Wg, Ca, Cb, spreads, mask]

	# prepare for recording mem and time
	torch.cuda.synchronize()  
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 0

	doautotune = False
	if doautotune:

		# autotune triton configs before profiling, allows optimal configs and triton to use cache rather than recompiling
		print("autotuning:\n")

		# log autotuning
		os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

		# for i in range(2):
		autotune(personal_attn, params)

	# for i in range(2):
	# 	tmp, _, _ = profile_func(personal_attn, params, start_event, end_event)
	# 	profile_bwd(tmp.sum(), start_event, end_event)

	# zero grads
	# Q.grad.zero_()
	# K.grad.zero_()
	# V.grad.zero_()
	# Wg.grad.zero_()
	
	print("forward pass:\n")

	torch_out, triton_out = test_fwd(torch_attn, personal_attn, params, start_event, end_event, atol, rtol)

	print("\nbackward pass:\n")

	test_bwd(torch_out.sum(), triton_out.sum(), Q, K, V, Wg, start_event, end_event, atol, rtol)


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

def test_bwd(torch_loss, triton_loss, Q, K, V, Wg, start_event, end_event, atol, rtol):

	# torch
	torch_time, torch_memory = profile_bwd(torch_loss, start_event, end_event)
	torch_dQ, torch_dK, torch_dV, torch_dWg = [i.grad.clone() for i in [Q, K, V, Wg]]

	# # zero grads
	Q.grad.zero_()
	K.grad.zero_()
	V.grad.zero_()
	Wg.grad.zero_()

	# triton
	triton_time, triton_memory = profile_bwd(triton_loss, start_event, end_event)
	triton_dQ, triton_dK, triton_dV, triton_dWg = [i.grad.clone() for i in [Q, K, V, Wg]]

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

	dWg_rel_error, dWg_abs_error = calculate_error(torch_dWg, triton_dWg)
	print(f"dWg is correct: {torch.allclose(triton_dWg, torch_dWg, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton dWg absolute error: {dWg_abs_error:.5f}")
	print(f"triton dWg relative error: {dWg_rel_error:.5f}")
	print(f"triton dWg percent error: {dWg_rel_error*100:.5f}%\n")

	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

def torch_attn(Q, K, V, Wg, Ca, Cb, spreads, mask=None):

	Z, H, N, d_k = Q.shape	
	d_model = d_k * H
	mask = torch.zeros(Z, N) if mask is None else mask # Z x N

	Q = Q.contiguous()
	K = K.contiguous()
	V = V.contiguous()
	Wg = Wg.contiguous()
	Ca = Ca.contiguous()
	Cb = Cb.contiguous()
	spreads = spreads.contiguous()
	mask = mask.contiguous()

	num_spreads = spreads.size(0)
	Dg = 4*num_spreads + 2

	# get dists first
	CaCa = torch.sqrt(torch.sum((Ca[:, :, None, :] - Ca[:, None, :, :])**2, axis=3))
	CaCb = torch.sqrt(torch.sum((Ca[:, :, None, :] - (Ca[:, None, :, :] + 1.53*Cb[:, None, :, :]))**2, axis=3))
	CbCa = torch.sqrt(torch.sum(((Ca[:, :, None, :] + 1.53*Cb[:, :, None, :]) - Ca[:, None, :, :])**2, axis=3))
	CbCb = torch.sqrt(torch.sum(((Ca[:, :, None, :] + 1.53*Cb[:, :, None, :]) - (Ca[:, None, :, :] + 1.53*Cb[:, None, :, :]))**2, axis=3))

	# now rbfs
	CaCa_rbf = 2*torch.exp(-(CaCa[:, :, :, None]**2) / (2*(spreads[None, None, None, :]**2))) - 1
	CaCb_rbf = 2*torch.exp(-(CaCb[:, :, :, None]**2) / (2*(spreads[None, None, None, :]**2))) - 1
	CbCa_rbf = 2*torch.exp(-(CbCa[:, :, :, None]**2) / (2*(spreads[None, None, None, :]**2))) - 1
	CbCb_rbf = 2*torch.exp(-(CbCb[:, :, :, None]**2) / (2*(spreads[None, None, None, :]**2))) - 1

	# now dots
	CaCa_raw = Ca[:, None, :, :] - Ca[:, :, None, :] 
	CaCa_mag = torch.linalg.vector_norm(CaCa_raw, dim=3, keepdim=True)
	CbCa_dot = torch.sum(Cb[:, :, None, :] * CaCa_raw / torch.where(CaCa_mag==0, 1, CaCa_mag ), dim=3)[:, :, :, None]
	CbCb_dot = torch.sum(Cb[:, :, None, :] * Cb[:, None, :, :], dim=3)[:, :, :, None]

	g = torch.stack([torch.stack([torch.arange(0,16, dtype=torch.float32)[None, None, None, :], torch.arange(16, 32, dtype=torch.float32)[None, None, None, :]], dim=4).view(1, 1, 1, -1), torch.stack([torch.arange(32,48, dtype=torch.float32)[None, None, None, :], torch.arange(48,64, dtype=torch.float32)[None, None, None, :]], dim=4).view(1, 1, 1, -1)], dim=4).view(1,1,1, -1)[:, :, :, :Dg]


	# compute g, interleave the same way as in triton for reroducibility, Z x N x N x Dg
	g = torch.stack([torch.stack([torch.cat([CaCa_rbf, CbCa_dot], dim=3), torch.cat([CaCb_rbf, torch.zeros(Z, N, N, 1, device=CaCb.device)], dim=3)], dim=4).view(Z, N, N, -1), torch.stack([torch.cat([CbCa_rbf, CbCb_dot], dim=3), torch.cat([CbCb_rbf, torch.zeros(Z, N, N, 1, device=CaCb.device)], dim=3)], dim=4).view(Z, N, N, -1)], dim=4).view(Z, N, N, -1)[:, :, :, :Dg]
	G = torch.matmul(g[:, None, :, :, :], Wg[None, :, None, :, :]) #  Z x 1 x N x N x Dg @ 1 x H x 1 x Dg x Dk --> Z x H x N x N x Dk
	KG = K[:, :, None, :, :] + G

	S = torch.matmul(Q[:, :, :, None, :], KG.transpose(3,4)).squeeze(3) / ((d_k**0.5)) # Z x H x N x N

	attn_mask = mask[:, None, None, :]  

	S = torch.where(attn_mask, -float("inf"), S) 
	P = torch.softmax(S, dim=-1)

	out = torch.matmul(P, V) # Z x H x N x d_k

	return out

if __name__ == '__main__':
	main()
