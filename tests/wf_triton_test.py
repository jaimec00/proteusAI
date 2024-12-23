

import torch
# from utils.model_utils import protein_to_wavefunc, protein_to_wavefunc_torch 

def main():

	# device
	device = torch.device('cuda')

	batch, N, d_model = 1, 40000, 512
	min_wl, max_wl, base = 3.7, 20, 20
	coords = torch.tensor([[[i,i,i] for i in range(N)] for j in range(batch)], dtype=torch.float64, device=device)
	mask = (torch.rand((batch, N), device=coords.device) > 1)

	torch.cuda.synchronize()  # Ensure no ongoing GPU operations
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	triton_out = protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask)
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	triton_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	triton_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	torch.cuda.empty_cache()  # Clear the cache for consistent results
	torch.cuda.reset_peak_memory_stats()
	start_event.record()
	# torch_out = protein_to_wavefunc_torch(coords, d_model, min_wl, max_wl, base, mask, 32).to(torch.float64)
	torch_out = triton_out
	end_event.record()
	torch.cuda.synchronize()  # Wait for all GPU work to finish
	torch_time = start_event.elapsed_time(end_event)  # Time in milliseconds
	torch_memory = torch.cuda.max_memory_allocated()  # Peak memory in bytes

	error = calculate_error(torch_out, triton_out)

	# print(f"{torch_out=}\n{triton_out=}\n")
	# print(f"{torch_out/triton_out}\n")
	# print(f"{torch_out-triton_out}\n")

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=1e-4, rtol=1e-4, equal_nan=True)}")
	print(f"triton percent error: {error:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

def calculate_error(A, B):
	# Ensure the tensors are of the same size
	assert A.size() == B.size(), f"Tensors must have the same size, not {A.shape=} and {B.shape=}"

	# Calculate absolute error and normalize by the ground truth sum
	absolute_error = torch.abs(A - B)
	relative_error = absolute_error / torch.abs(A)  # Element-wise relative error
	error_percentage = (torch.sum(absolute_error) / torch.sum(torch.abs(A))) * 100

	return error_percentage

if __name__ == "__main__":
	main()