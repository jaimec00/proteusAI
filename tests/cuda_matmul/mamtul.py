import torch
from random import randint
from torch.utils.cpp_extension import load
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# dynamically compile and load the extension
matmul_kernel = load(
	name="matmul_kernel",
	sources=[os.path.join(base_dir, "matmul_if.cpp"), os.path.join(base_dir, "matmul_kernel.cu")],
	verbose=True  # Verbose output for debugging
)

torch.manual_seed(0)
device = torch.device("cuda")

def main():
	M, N, K = 1024, 256, 128
	A = torch.rand(M, K, device=device, dtype=torch.float16).contiguous()
	B = torch.rand(K, N, device=device, dtype=torch.float16).contiguous()
	C = torch.rand(M, N, device=device, dtype=torch.float16).contiguous()
	D = torch.zeros_like(C).contiguous()
	
	# run the kernel to accumulate result to D. AB not changed
	matmul_kernel.fwd(A, B, C, D)

	# see  if it worked
	out_real = torch.matmul(A, B, C)
	is_close = torch.allclose(D, out_real)
	print(is_close)

if __name__ == "__main__":
	main()