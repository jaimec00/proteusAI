import torch
import numpy as np
from utils.model_utils.wf_embedding.anisotropic.cb_scaling.learnable_wavenumbers.cuda import wf_embedding_kernel

# # for testing and development
# from torch.utils.cpp_extension import load
# import os
# base_dir = os.path.dirname(os.path.abspath(__file__))
# # dynamically compile and load the extension
# wf_embedding_kernel = load(
# 	name="wf_embedding_kernel",
# 	sources=[os.path.join(base_dir, "wf_embedding_if.cpp"), os.path.join(base_dir, "wf_embedding_kernel.cu")],
# 	verbose=True  # Verbose output for debugging
# )

def wf_embedding(coordsA, coordsB, cb_magnitudes, wavenumbers, mask=None):
	return _wf_embedding.apply(coordsA, coordsB, cb_magnitudes, wavenumbers, mask)

class _wf_embedding(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coordsA, coordsB, cb_magnitudes, wavenumbers, mask):

		# get tensor sizes
		batch, N, space = coordsA.shape
		num_wn = cb_magnitudes.size(0)
		d_model = 2 * num_wn


		# gather the dtypes so can convert the output to this. coords A is what is used to determine O dtype, and cb_magnitudes dtype for d_aa
		o_dtype = coordsA.dtype
		d_cb_dtype = cb_magnitudes.dtype
		d_k_dtype = wavenumbers.dtype

		# bake the mask into coords w/ arbitrary val. less likely to give NaNs than using inf
		coordsA = torch.where(mask.unsqueeze(2), 12345, coordsA) # only need to bake the mask into one of the coords tensors

		# convert dtypes and make contiguous. everything in fp16
		coordsA = coordsA.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel
		
		# same w/ coords b, expected to be unit vectors, might explicitly do this later
		coordsB = coordsB.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel

		# deal w/ wavenumbers and cb scales
		wavenumbers = wavenumbers.to(torch.float32).contiguous()
		cb_magnitudes = cb_magnitudes.to(torch.float32).contiguous() # compromise to fp16 for better occupancy

		# instantiate the output tensor
		out = torch.zeros(batch, N, d_model, dtype=torch.float32, device=coordsA.device).contiguous()

		# for bwd pass
		d_cb = torch.zeros(batch, N, d_model, dtype=torch.float32, device=cb_magnitudes.device).contiguous()
		d_k = torch.zeros(batch, N, d_model, dtype=torch.float32, device=wavenumbers.device).contiguous()

		# call the kernel
		wf_embedding_kernel.forward(    coordsA, coordsB, 
										cb_magnitudes,
										wavenumbers, 
										out, d_cb, d_k
									)

		# save for the backward, and recast to input dtype
		ctx.save_for_backward(d_cb.to(d_cb_dtype), d_k.to(d_k_dtype))

		return out.to(o_dtype)

	@staticmethod
	def backward(ctx, dO):

		# load saved tensors from bwd
		d_cb, d_k = ctx.saved_tensors

		d_cb = (dO * d_cb).sum(dim=(0,1)) # mult w/ dO and sum batch and N dims; Z x N x D x 1 * Z x N x D x A --> D x A
		d_cb = d_cb[::2] + d_cb[1::2] # sum real and imag parts; K x A

		d_k = (dO*d_k).sum(dim=(0,1)) # Z x N x D * Z x N x D --> D
		d_k = d_k[::2] + d_k[1::2] # sum the real and imag parts; K

		return None, None, d_cb, d_k, None
