import torch

def calculate_error(A, B):
	# Ensure the tensors are of the same size
	assert A.size() == B.size(), f"Tensors must have the same size, not {A.shape=} and {B.shape=}"

	# Calculate absolute error and normalize by the ground truth sum

	abs_error_l2 = torch.norm(A - B, p=2)  # Absolute (L2 norm)
	rel_error_l2 = abs_error_l2 / torch.norm(A, p=2)  # Relative (L2 norm)


	return rel_error_l2, abs_error_l2
