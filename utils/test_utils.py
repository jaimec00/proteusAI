import torch

def calculate_error(A, B):
	# Ensure the tensors are of the same size
	assert A.size() == B.size(), f"Tensors must have the same size, not {A.shape=} and {B.shape=}"

	# Calculate absolute error and normalize by the ground truth sum
	absolute_error = torch.abs(A - B)
	relative_error = absolute_error / torch.abs(A)  # Element-wise relative error
	error_percentage = (torch.sum(absolute_error) / torch.sum(torch.abs(A))) * 100

	return error_percentage
