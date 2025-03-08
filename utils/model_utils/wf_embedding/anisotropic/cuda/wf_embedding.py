import torch
import numpy as np
from utils.model_utils.wf_embedding.anisotropic.cuda import wf_embedding_kernel

# # for testing and development
# from torch.utils.cpp_extension import load
# import os
# base_dir = os.path.dirname(os.path.abspath(__file__))
# # dynamically compile and load the extension
# wf_embedding_kernel = load(
# 	name="wf_embedding_kernel",
# 	sources=[os.path.join(base_dir, "wf_embedding_if.cpp"), os.path.join(base_dir, "wf_embedding_kernel.cu")],
# 	# extra_include_paths=["/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/include"],
# 	verbose=True  # Verbose output for debugging
# )

def wf_embedding(coords_alpha, coords_beta, wavenumbers, magnitude_type=1, dropout_p=0.0, mask=None):
	return _wf_embedding.apply(coords_alpha, coords_beta, wavenumbers, magnitude_type, dropout_p, mask)

class _wf_embedding(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coords_alpha, coords_beta, wavenumbers, magnitude_type, dropout_p, mask):

		assert magnitude_type in [0,1,2,3] # mag=1, mag=1/|R|, mag=1/log2(|R|), mag=1/sqrt(|R|)

		# bake the mask into coords w/ arbitrary val. less likely to give NaNs than using inf
		coords_alpha = torch.where(mask.unsqueeze(2), 12345, coords_alpha) # only need to bake the mask into one of the coords tensors

		# convert dtypes and make contiguous. everything in fp32
		coords_alpha = coords_alpha.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel
		coords_beta = coords_beta.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel

		# deal w/ wavenumbers
		wavenumbers = wavenumbers.to(torch.float32).contiguous()

		# get tensor sizes
		batch, space, N = coords_alpha.shape
		d_model = 2 * wavenumbers.shape[0]

		# instantiate the output tensor
		out = torch.zeros(batch, N, d_model, dtype=coords_alpha.dtype, device=coords_alpha.device).contiguous()

		# for bwd pass
		d_imag = torch.zeros(batch, N, d_model//2, dtype=coords_alpha.dtype, device=coords_alpha.device).contiguous()
		d_real = torch.zeros_like(d_imag).contiguous()

		rng = np.random.default_rng()  # Create a new random generator
		rng_seed = rng.integers(0, 2**32, dtype=np.uint32)

		# call the kernel
		wf_embedding_kernel.forward(    coords_alpha, coords_beta, 
										wavenumbers, 
										out,
										d_imag, d_real,
										magnitude_type, dropout_p, rng_seed, # havent tested and compiled yet
								)

		# save for the backward
		ctx.save_for_backward(d_imag, d_real)

		return out

	@staticmethod
	def backward(ctx, dO):

		# note, masks already applied to O and cos and sin sums, masked vals are zero, 
		# so multiplication ensures non valid positions dont contribute to the gradients
		# i.e. don't need to save mask for bwd

		# load saved tensors from bwd
		d_imag, d_real = ctx.saved_tensors

		# seperate dO into real and imag parts (interleaved, real is first)
		# from Z x N x d_model --> Z x N x d_model//2 
		real_dO = dO[:, :, 0::2]
		imag_dO = dO[:, :, 1::2]

		# compute grad wrt wavenumbers
		# dO_2i=l+1 * sum_j(cos(K|ri-rj|)) - dO_2l * sum(sin(K|ri-rj|))
		# sum the Z dim and N dim, to accumulate gradients, as wavenumbers is a tensor of shape d_model//2
		dk = ((imag_dO*d_imag) - (real_dO*d_real)).sum(dim=(0,1)) # d_model//2

		return None, None, dk, None, None, None
