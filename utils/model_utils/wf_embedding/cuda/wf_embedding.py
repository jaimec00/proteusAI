import torch
# from utils.model_utils.wf_embedding.cuda import wf_embedding_kernel
from torch.utils.cpp_extension import load
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# dynamically compile and load the extension
wf_embedding_kernel = load(
	name="wf_embedding_kernel",
	sources=[os.path.join(base_dir, "wf_embedding_if.cpp"), os.path.join(base_dir, "wf_embedding_kernel.cu")],
	verbose=True  # Verbose output for debugging
)

def wf_embedding(coords, wavenumbers, mask=None):
	return _wf_embedding.apply(coords, wavenumbers, mask)

class _wf_embedding(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coords, wavenumbers, mask):
		
		# bake the mask into coords
		coords = torch.where(mask.unsqueeze(2), 12345.6789, coords)

		# convert dtypes and make contiguous. everything in fp32
		coords = coords.transpose(1, 2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel
		
		# deal w/ wavenumbers
		wavenumbers = wavenumbers.to(torch.float32).contiguous()

		# get tensor sizes
		batch, space, N = coords.shape
		d_model = 2 * wavenumbers.shape[0]

		# instantiate the output tensor
		out = torch.zeros(batch, N, d_model, dtype=coords.dtype, device=coords.device).contiguous()

		# for bwd pass
		cos_sums = torch.zeros(batch, N, d_model//2, dtype=coords.dtype, device=coords.device).contiguous()
		sin_sums = torch.zeros_like(cos_sums).contiguous()

		# call the kernel
		wf_embedding_kernel.forward(    coords, wavenumbers, 
										out,
										cos_sums, sin_sums
								)
		

		# save for the backward
		ctx.save_for_backward(cos_sums, sin_sums)

		return out

	@staticmethod
	def backward(ctx, dO):

		# note, masks already applied to O and cos and sin sums, masked vals are zero, 
		# so multiplication ensures non valid positions dont contribute to the gradients
		# i.e. don't need to save mask for bwd

		# load saved tensors from bwd
		cos_sums, sin_sums = ctx.saved_tensors

		# seperate dO into real and imag parts (interleaved, real is first)
		# from Z x N x d_model --> Z x N x d_model//2 
		real_dO = dO[:, :, 0::2]
		imag_dO = dO[:, :, 1::2]

		# compute grad wrt wavenumbers
		# dO_2i=l+1 * sum_j(cos(K|ri-rj|)) - dO_2l * sum(sin(K|ri-rj|))
		# sum the Z dim and N dim, to accumulate gradients, as wavenumbers is a tensor of shape d_model//2
		dk = ((imag_dO*cos_sums) - (real_dO*sin_sums)).sum(dim=1).sum(dim=0) # d_model//2

		return None, dk, None 