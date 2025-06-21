import torch

def calculate_error(A, B):
	# Ensure the tensors are of the same size
	assert A.size() == B.size(), f"Tensors must have the same size, not {A.shape=} and {B.shape=}"

	# Calculate absolute error and normalize by the ground truth sum

	abs_error_l2 = torch.norm(A - B, p=2)  # Absolute (L2 norm)
	rel_error_l2 = abs_error_l2 / torch.norm(A, p=2)  # Relative (L2 norm)


	return rel_error_l2, abs_error_l2

def profile_func(func, args, start_event, end_event):

	torch.cuda.empty_cache() 
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	out = func(*args)

	end_event.record()
	torch.cuda.synchronize()
	func_time = start_event.elapsed_time(end_event)  
	func_mem = torch.cuda.max_memory_allocated()   

	return out, func_time, func_mem

def profile_bwd(loss, start_event, end_event):
	torch.cuda.empty_cache() 
	torch.cuda.reset_peak_memory_stats()
	start_event.record()

	loss.backward(retain_graph=False)

	end_event.record()
	torch.cuda.synchronize()
	func_time = start_event.elapsed_time(end_event)  
	func_mem = torch.cuda.max_memory_allocated()   

	return func_time, func_mem
