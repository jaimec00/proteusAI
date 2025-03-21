import torch
import numpy as np
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.static_aa.cuda import wf_embedding_kernel

# for testing and development
# from torch.utils.cpp_extension import load
# import os
# base_dir = os.path.dirname(os.path.abspath(__file__))
# # dynamically compile and load the extension
# wf_embedding_kernel = load(
# 	name="wf_embedding_kernel",
# 	sources=[os.path.join(base_dir, "wf_embedding_if.cpp"), os.path.join(base_dir, "wf_embedding_kernel.cu")],
# 	verbose=True  # Verbose output for debugging
# )

def wf_embedding(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, dropout_p=0.0, mask=None):
	return _wf_embedding.apply(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, dropout_p, mask)

class _wf_embedding(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, dropout_p, mask):

		# get tensor sizes
		batch, N, space = coordsA.shape
		num_wn, num_aa = aa_magnitudes.shape
		d_model = 2 * num_wn

		assert num_aa<32, "num_aa should be less than WarpSize (32). kernel was designed and optimized for num_aa=20 though"

		# gather the dtypes so can convert the output to this. coords A is what is used to determine O dtype, and aa_magnitudes dtype for d_aa
		o_dtype = coordsA.dtype

		# bake the mask into coords w/ arbitrary val. less likely to give NaNs than using inf
		coordsA = torch.where(mask.unsqueeze(2), 12345, coordsA) # only need to bake the mask into one of the coords tensors

		# convert dtypes and make contiguous. everything in fp16
		coordsA = coordsA.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel
		
		# same w/ coords b, expected to be unit vectors, might explicitly do this later
		coordsB = coordsB.transpose(1,2).to(torch.float32).contiguous() # transpose to make memory access more efficient in the kernel

		# deal w/ wavenumbers
		wavenumbers = wavenumbers.to(torch.float32).contiguous()

		# aas
		aa_labels = aa_labels.to(torch.int16).contiguous() # int 16, max idx is 19, so this precision is overkill anyways
		aa_magnitudes = aa_magnitudes.to(torch.float32).contiguous() # compromise to fp16 for better occupancy

		# instantiate the output tensor
		out = torch.zeros(batch, N, d_model, dtype=torch.float32, device=coordsA.device).contiguous()

		rng = np.random.default_rng()  # Create a new random generator
		rng_seed = rng.integers(0, (2**32)-1, dtype=np.uint32)

		# call the kernel
		wf_embedding_kernel.forward(    coordsA, coordsB, 
										aa_labels, aa_magnitudes,
										wavenumbers, out,
										dropout_p, rng_seed
								)

		return out.to(o_dtype)

	@staticmethod
	def backward(ctx, dO):
		return None, None, None, None, None, None, None
