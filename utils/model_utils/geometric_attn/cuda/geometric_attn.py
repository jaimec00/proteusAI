import torch
from random import randint
# from utils.model_utils.geometric_attn.cuda.attn_fwd import attn_fwd_kernel
# from utils.model_utils.geometric_attn.cuda.attn_bwd import attn_bwd_kernel

# for testing and development

from torch.utils.cpp_extension import load
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# dynamically compile and load the extension
attn_fwd_kernel = load(
	name="attn_fwd_kernel",
	sources=[os.path.join(base_dir, "attn_fwd/attn_fwd_if.cpp"), os.path.join(base_dir, "attn_fwd/attn_fwd_kernel.cu")],
	verbose=True  # Verbose output for debugging
)

def geometric_attn(Q, K, V, coords, spreads, mask=None, dropout=0.0):
	return _geometric_attn.apply(Q, K, V, coords, spreads, mask, dropout)

class _geometric_attn(torch.autograd.Function):

	@staticmethod
	def forward(ctx, Q, K, V, coords, spreads, mask, dropout):
		
		# checks
		assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		batch, nheads, N, d_k = Q.shape
		d_model = nheads*d_k
		softmax_scale = 1/((d_k**0.5)*2) # divide by 2 bc rbfs scale logits by two at most
		assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
		assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 
		assert spreads.size(0) == nheads, f"number of spreads per batch must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
		assert torch.all(spreads > 0), f"spreads must be a tensor of positive, non-zero floats, not {spreads}"

		# matmults done in fp16
		Q = Q.to(torch.float16).contiguous()
		K = K.to(torch.float16).contiguous()
		V = V.to(torch.float16).contiguous()

		# rbfs in fp32 and make sure everything is contiguous
		coords = torch.where(mask.unsqueeze(2), 12345, coords) # bake mask into coords
		coords = coords.transpose(1,2).to(torch.float32).contiguous()
		spreads = spreads.to(torch.float32).contiguous() 

		# initialize mask, output, and logsumexp tensors
		out = torch.zeros(batch, nheads, N, d_k, dtype=torch.float32, device=Q.device).contiguous() # batch x N x d_model
		L = torch.zeros(batch, nheads, N, dtype=torch.float32, device=Q.device).contiguous() # batch x nheads x N
		
		# generate a rng seed for each batch and head
		rng_seed = np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)

		attn_fwd_kernel.fwd(
			Q, K, V,
			coords, spreads,
			L, out,
			softmax_scale,
			dropout,
			rng_seed
		)

		# for backwards pass
		ctx.save_for_backward(Q, K, V, out, L, coords, spreads, rng_seed)
		ctx.softmax_scale = softmax_scale
		ctx.dropout = dropout

		return out

	# @staticmethod
	# def backward(ctx, dO):

	# 	# load saved tensors (should all be float32, expect masks). also should all be contiguous from fwd
	# 	Q, K, V, O, L, coords, spreads, mask, rng_seed = ctx.saved_tensors

	# 	# compute D for dSR calculation
	# 	D = torch.sum(O*dO, dim=3).to(torch.float16) # Z x H x N x D -> Z x H x N

	# 	# cast to float16 for matmults
	# 	dO = dO.to(torch.float16).contiguous()

	# 	# checks
	# 	assert Q.stride() == K.stride() == V.stride() == O.stride()
	# 	batch, nheads, N, d_k = Q.shape 

	# 	# initialize dQ, dK, and dV, all fp32
	# 	dQ = torch.zeros_like(Q).to(torch.float32).contiguous()
	# 	dK = torch.zeros_like(K).to(torch.float32).contiguous()
	# 	dV = torch.zeros_like(V).to(torch.float32).contiguous()
	# 	d_spreads = torch.zeros_like(spreads).to(torch.float32).contiguous()
		
	# 	# define the grid
	# 	grid = lambda args: (
	# 		triton.cdiv(args["tot_N"], args["BLOCK_J"]), # parralel along J for bwd
	# 		args["tot_Z"]*args["nheads"],
	# 		1
	# 	)

	# 	# kernel run

	# 	dQ = dQ
	# 	dK = dK
	# 	dV = dV
	# 	d_spreads = d_spreads
		
	# 	# return the gradients
	# 	return dQ, dK, dV, None, d_spreads, None, None