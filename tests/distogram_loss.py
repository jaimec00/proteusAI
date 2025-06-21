import torch
from utils.test_utils import calculate_error, profile_func, profile_bwd
from utils.model_utils.base_modules.distogram_loss import distogram_loss

def main():

	device = "cuda"
	torch.manual_seed(1)

	batch, N, bins, d_model = 1, 4096, 32, 128
	min_d, max_d = 2., 22.
	label_smoothing = 0.0

	coords = torch.rand((batch, N, 3), device=device) * 20
	features = torch.rand((batch, N, d_model), device=device, requires_grad=True) 
	bin_proj = torch.rand((d_model, bins), device=device, requires_grad=True)
	mask = torch.rand((batch, N), device=device) > 1.0
	bins = torch.cat([torch.tensor([0], device=device), torch.linspace(min_d, max_d, bins-1, device=device), torch.tensor([float("inf")], device=device)])

	params = [features, coords, bins, bin_proj, mask, label_smoothing]

	# prepare for recording mem and time
	torch.cuda.synchronize()  
	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)
	atol, rtol = 1e-2, 0

	# fwd
	print("\nforward pass:\n")

	for i in range(3): # warmup, first run also times the compilation which idc about
		triton_out, triton_time, triton_memory = profile_func(distogram_loss, params, start_event, end_event)
	torch_out, torch_time, torch_memory = profile_func(distogram_torch, params, start_event, end_event)
	rel_error, abs_error = calculate_error(torch_out, triton_out)

	print(f"triton implementation is correct: {torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton absolute error: {abs_error:.5f}")
	print(f"triton relative error: {rel_error:.5f}")
	print(f"triton percent error: {rel_error*100:.5f}%")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB")

	# bwd
	print("\nbackward pass:\n")

	# torch
	torch_time, torch_memory = profile_bwd(torch_out, start_event, end_event)
	torch_d_features = features.grad.clone()

	# zero grads
	features.grad.zero_()

	# triton
	triton_time, triton_memory = profile_bwd(triton_out, start_event, end_event)
	triton_d_features = features.grad.clone()

	d_features_rel_error, d_features_abs_error = calculate_error(torch_d_features, triton_d_features)
	print(f"d_features is correct: {torch.allclose(triton_d_features, torch_d_features, atol=atol, rtol=rtol, equal_nan=False)}")
	print(f"triton d_features absolute error: {d_features_abs_error:.5f}")
	print(f"triton d_features relative error: {d_features_rel_error:.5f}")
	print(f"triton d_features percent error: {d_features_rel_error*100:.5f}%\n")
	print(f"torch time: {torch_time:.3f} ms")
	print(f"triton time: {triton_time:.3f} ms")
	print(f"torch memory usage: {torch_memory / (1024 ** 3):.3f} GB")
	print(f"triton kernel memory usage: {triton_memory / (1024 ** 3):.3f} GB\n")
	# print(triton_d_features)


def distogram_torch(features, coords, bins, bin_proj, mask, label_smoothing):

	# compute dist and the true bin
	dists = torch.linalg.vector_norm(coords[:, :, None, :] - coords[:, None, :, :], dim=3) # Z x N x N
	bin_min = bins[:-1] # B
	bin_max = bins[1:] # B
	in_bin = (dists[:, :, :, None] >= bin_min[None, None, None, :]) & (dists[:, :, :, None] < bin_max[None, None, None, :]) # Z x N x N x B

	# prep cel input
	CEL = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing, reduction="none")
	labels = in_bin.to(torch.int32).argmax(dim=3) # Z x N x N
	labels = torch.where(mask[:, :, None] | mask[:, None, :], -1, labels)
	labels_flat = labels.view(-1)

	# get the bin prediction probabilities, by dotting the DK dim for each B
	pred = torch.matmul(features[:, :, None :] + features[:, None, :, :], bin_proj[None, None, :, :]) # Z x N x N x D @ 1 x 1 x D x B --> Z x N x N x B
	p_pred_flat = pred.view(-1, pred.size(3))

	# compute cel
	batch, N, _ = coords.shape
	cel = (CEL(p_pred_flat, labels_flat).view(batch, N, N) / (2*(~mask).sum(dim=1)[:, None, None].clamp(min=1))).sum()
	return cel

if __name__ == "__main__":
	main()