# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		utils.py
description:	utility functions
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
from pathlib import Path
import torch.nn.functional as F
import gc

def protein_to_wavefunc(coords: torch.Tensor, key_padding_mask: torch.Tensor, d_model: int=512, return_wl=False, min_wl=3.7, max_wl=20, base=20):
	'''
	converts the alpha carbon coordinates of a protein into a tensor of 
	wavefunction outputs.
	converts a batch x N x 3 tensor to a batch x N x d_model tensor.
	each feature for a Ca is their output of the wave function with a specific wavelength
	each output gets two features, one for the real part, and another for the imaginary part
	the wave function is a superposition of Green's functions, treating each Ca as a point source
	note that this function is very memory conscious, planning on computing in portions along the wavelength (d_model)
	dimension if required

	Args:
		coords (torch.Tensor):              tensor containing batches of Ca coords. 
											size = batch x N x 3 
		key_padding_mask (torch.Tensor):    tenor containing key padding mask
											size = batch x N 
		d_model (int):						features to create. d_model = number_of_wavelengths*2
		return_wl (bool): 					whether to return the wavelengths used, useful for plotting 
		min_wl (float):						minimum wavelength to use
		max_wl (float):						maximum wavelength to use
		base (int|float):					wavelengths are sampled logarithmically, chooses the base to use
	
	Returns:
		features (torch.Tensor):    tensor containing batches of token (Ca) features.
									size = batch x N x 512
	'''

	# specify device
	device = coords.device

	# **GET PAIRWISE DISTANCES**

	# get the euclidean distances ; batch x N x 3 --> batch x N x N 
	pw_dists = torch.sqrt_((coords.unsqueeze(1) - coords.unsqueeze(2)).pow_(2).sum(dim=-1)).to(device)
	
	# diagonal set to 1 to avoid division by zero
	pw_dists += torch.eye(pw_dists.size(1), device=device).unsqueeze(0).expand(pw_dists.size(0), -1, -1)

	# set masked values to inf to exclude from wave function calculation (1/inf = 0)
	pw_dist_mask = key_padding_mask.unsqueeze(1) | key_padding_mask.unsqueeze(2) # batch x N x N
	pw_dists.masked_fill_(pw_dist_mask, float('inf'))
	
	# **DEFINE WAVELENGTHS**

	# min wl slightly below avg Ca neighbor dist 
	min_wl = torch.tensor([min_wl], device=device).expand(pw_dists.size(0), d_model//2) # Angstroms
	max_wl = torch.tensor([max_wl], device=device).expand(pw_dists.size(0), d_model//2)

	# Create a tensor of wavelengths
	wavelengths = get_wavelengths(min_wl, max_wl, d_model, device=device, base=base)

	# remove unecessary variables asap
	del min_wl, max_wl
	torch.cuda.empty_cache()

	# **COMPUTE GREENS FN

	# convert wavelengths to k_values and set up for broadcasting along middle two dimensions (N and N from pw_dists)
	k_values = (2 * torch.pi / wavelengths).unsqueeze(1).unsqueeze(1).expand(-1, pw_dists.size(1), pw_dists.size(1), -1) # batch x num_wl --> batch x N x N x num_wl
	
	# only remove from memory if not needed
	if not return_wl:
		del wavelengths
		torch.cuda.empty_cache()

	# prepare pw_dists by expanding it to include num_wl
	# batch x N x N --> batch x N x N x num_wl
	pw_dists = pw_dists.unsqueeze(-1).expand(-1, -1, -1, k_values.size(-1))
	pw_dist_mask = pw_dist_mask.unsqueeze(-1).expand(-1, -1, -1, k_values.size(-1))

	# compute phase ; batch x N x N x num_wl
	phase = pw_dists.masked_fill(pw_dist_mask, 0.0).mul_(k_values)
	
	# need to compute real and imaginary parts seperately for memory efficiency
	# **REAL PART CALCULATION**

	# compute the Green's function real part
	greens_fn_real = phase.cos_().div_(pw_dists) # batch x N x N x num_wl

	# delete REFERENCE to phase, greens_fn_real still in memory
	del phase
	torch.cuda.empty_cache()

	# take care of padded values and identity positions
	batch, N, _, wl = greens_fn_real.shape
	greens_fn_real.masked_fill_(torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | pw_dist_mask, 0.0) # batch x N x N x num_wl
	
	# superpose all other Ca point sources to each Ca
	superpositions_real = greens_fn_real.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
	
	del greens_fn_real
	torch.cuda.empty_cache()

	# **IMAGINARY PART CALCULATION**

	phase = pw_dists.masked_fill(pw_dist_mask, 0.0).mul_(k_values) # batch x N x N x num_wl
	greens_fn_imag = phase.sin_().div_(pw_dists) # batch x N x N x num_wl

	del pw_dists
	del phase
	torch.cuda.empty_cache()

	greens_fn_imag.masked_fill_(torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, wl) | pw_dist_mask, 0.0) # batch x N x N x num_wl
	superpositions_imag = greens_fn_imag.sum(dim=2)  # sum over the third dimension ; batch x N x num_wl
	
	del greens_fn_imag	
	del pw_dist_mask
	del batch, N, _, wl
	torch.cuda.empty_cache()

	# **CONCAT INTO FEATURES**

	# every k value gets two features
	# want real and imaginary parts next to each other (for a single k)
	features = torch.stack((superpositions_real, superpositions_imag), dim=-1)
	
	del superpositions_real, superpositions_imag
	torch.cuda.empty_cache()
	
	features = features.view(features.size(0), features.size(1), -1)  # Final shape: batch x N x d_model

	# normalize by number of Ca
	features.div_(torch.sum(~key_padding_mask, dim=1, keepdim=True))

	if return_wl:
		return features, wavelengths
	else:
		return features


def get_wavelengths(min_wl, max_wl, d_model, base=20, device="cpu"):

	# short range wavelengths get 128 wave functions, medium get 96, and long get 32. each wave function
	# creates two features, one real and one imaginary, for a total of num_wl*2 = 512 features
	# create evenly spaced tensors from 0 to 1 of shape 1,  
	num_wl = (d_model // 2)
	num_short_wl = int(num_wl * 4/8)
	num_med_wl = int(num_wl * 2/8)
	num_long_wl = int(num_wl * 2/8)
	
	log_distribution = (torch.logspace(0, 1, num_wl, base=base, device=device) - 1) / (base - 1) # Scale [1, 2) to [0, 1)
	log_distribution = log_distribution.expand(min_wl.size(0), num_wl).to(device)
	wavelengths = (min_wl + (log_distribution.mul_(max_wl - min_wl))).to(device) # 1 x 1

	return wavelengths

def check_gpu_memory(threshold=0.8):
	"""Check if GPU memory usage exceeds a given threshold."""
	if torch.cuda.is_available():
		total_memory = torch.cuda.get_device_properties(0).total_memory
		allocated_memory = torch.cuda.memory_allocated(0)
		reserved_memory = torch.cuda.memory_reserved(0)
		used_memory = max(allocated_memory, reserved_memory)
		
		# Return True if memory usage exceeds threshold, False otherwise
		return (used_memory / total_memory) > threshold
	return False

def smooth_and_noise_labels(label_batch, mean_lbl_smooth, stdev_lbl_smooth, stdev_noise, num_classes=20):
	"""
	Apply label smoothing and noise on a per-position basis for each prediction in the batch.

	Params:
		label_batch (torch.Tensor): A batch of correct labels with shape (batch_size, N), where each entry is the index of the correct label.
		mean_lbl_smooth (float): The mean for the Gaussian distribution of label smoothing factors.
		stdev_lbl_smooth (float): The stdev for the Gaussian distribution of label smoothing factors.
		stdev_noise (float): The stdev for the Gaussian distribution of noise to add to each position.

	Returns:
		noised_labels (torch.Tensor): A batch of predictions with label smoothing and noise added, with shape (batch_size, N, 20).
	"""
	
	# Sample label smoothing factors for each position from a Gaussian distribution
	lbl_smooth_factors = torch.clamp(torch.normal(mean_lbl_smooth, stdev_lbl_smooth, size=label_batch.shape, device=label_batch.device), min=0, max=(num_classes - 1)/num_classes)

	# Apply label smoothing by assigning the smoothed factor to the true label positions
	one_hot_labels = F.one_hot(torch.clamp(label_batch, min=0), num_classes=num_classes).float().to(label_batch.device)  # Shape: (batch_size, N, num_classes)
	smoothed_labels = (one_hot_labels * (1 - lbl_smooth_factors.unsqueeze(-1))) + \
						((1 - one_hot_labels) * (lbl_smooth_factors).unsqueeze(-1) / (num_classes - 1))

	# Generate noise and add it to the smoothed predictions
	mean_noise = 0 # to make sure we subtract and add sometimes
	noise_values = torch.normal(mean_noise, stdev_noise, size=smoothed_labels.shape, device=label_batch.device)
	noised_labels = smoothed_labels + noise_values

	# Clamp negative values to zero
	noised_labels = torch.clamp(noised_labels, min=0)

	# Zero out padded tokens and renormalize each row to sum to 1
	noised_labels = noised_labels.masked_fill(label_batch.unsqueeze(-1) == -1, 0)  # Ignore padding
	
	noised_labels = F.normalize(noised_labels, p=1, dim=-1)  # Ensure probabilities sum to 1

	return noised_labels

def noise_uniform_distribution(prediction, key_padding_mask, stdev_noise):
	# Generate noise and add it to the smoothed predictions
	mean_noise = 0 # to make sure we subtract and add sometimes
	noise_values = torch.normal(mean_noise, stdev_noise, size=prediction.shape, device=prediction.device)
	noised_prediction = prediction + noise_values

	# Clamp negative values to zero
	noised_prediction = torch.clamp(noised_prediction, min=0)

	# Zero out padded tokens and renormalize each row to sum to 1
	noised_prediction = noised_prediction.masked_fill(key_padding_mask.unsqueeze(-1), 0)  # Ignore padding
	
	noised_prediction = F.normalize(noised_prediction, p=1, dim=-1)  # Ensure probabilities sum to 1

	return noised_prediction

def compute_cel_and_seq_sim(label_batch, noised_labels, num_classes=20):
	"""
	Compute the mean cross-entropy loss (CEL) and sequence similarity (accuracy) between
	the original label batch and the noised & smoothed label batch.

	Params:
		label_batch (torch.Tensor): A batch of correct labels with shape (batch_size, N), where each entry is the index of the correct label.
		noised_labels (torch.Tensor): A batch of smoothed and noised label predictions with shape (batch_size, N, num_classes).
		key_padding_mask (torch.Tensor): A mask with shape (batch_size, N), where True indicates padding positions.

	Returns:
		mean_cel (float): The mean cross-entropy loss over all valid positions.
		mean_seq_sim (float): The mean sequence similarity (accuracy) over all valid positions.
	"""

	# Flatten the tensor to simplify indexing for CEL calculation
	noised_labels_flat = noised_labels.view(-1, num_classes)
	label_batch_flat = label_batch.view(-1)
	key_padding_mask_flat = label_batch_flat == -1

	# Compute CEL: Use the log of the predicted probability for the true class
	# Gather the predicted probability for each true class in label_batch
	true_label_probs = noised_labels_flat[torch.arange(len(label_batch_flat)), label_batch_flat]
	log_probs = torch.log(true_label_probs + 1e-12)  # Avoid log(0) by adding a small epsilon
	cel = -log_probs  # Cross-entropy loss for each position

	# Mask out padded tokens (where key_padding_mask is True)
	valid_cel = cel[~key_padding_mask_flat]
	mean_cel = valid_cel.mean().item() if len(valid_cel) > 0 else 0.0

	# Compute Sequence Similarity (accuracy): Check if the predicted class matches the true label
	predicted_classes = noised_labels_flat.argmax(dim=-1)
	valid_positions = (~key_padding_mask_flat).float().sum()
	correct_predictions = (predicted_classes == label_batch_flat).float().sum() / valid_positions if valid_positions > 0 else 0.0  # Ignore padded positions
	seq_sim = (correct_predictions.float().item()) * 100  # Compute accuracy over all valid positions

	return mean_cel, seq_sim


# ----------------------------------------------------------------------------------------------------------------------
# old

def fourier_feature_mapping(batch: torch.Tensor, freq_magnitudes: torch.Tensor, freqs_per_layer: int=32):

	# get the dimension
	batch_size, N, dim = batch.shape # batch, number of positions, dimensions (3d coords)
	
	# generate L spherical layers of n isotropic frequency vectors into a num_frequencies x 3 tensor
	B = fibonacci_sampling(freq_magnitudes, freqs_per_layer) # ( (num_freqs_per_layer * layers) x 3 )

	# batch x N x 3 @ 3 x n*L --> batch x N x n*L
	Bx = torch.matmul(batch, B.T) * 2 * torch.pi # batch x num_positions x (layers*freqs_per_layer)

	# compute the sin and cos terms, this doubles the dimensionality of n*L
	sin_Bx = torch.sin(Bx)
	cos_Bx = torch.cos(Bx)

	# interleave the sin and cos terms so that the same frequencies are next to each other
	mapped_batch = torch.cat((cos_Bx, sin_Bx), dim=-1)
	
	# print(mapped_batch)
	return mapped_batch

def fibonacci_sampling(freq_magnitudes: torch.Tensor, freqs_per_layer: int = 32):
	"""
	Perform fibonacci sampling of frequency vectors with differentiable PyTorch operations.

	Args:
		freq_magnitudes (torch.Tensor): Tensor of magnitudes for the frequency vectors. Should be learnable.
		freqs_per_layer (int): Number of frequency vectors per layer.
	
	Returns:
		torch.Tensor: A tensor of shape (num_layers * freqs_per_layer, 3) with sampled frequency vectors.
	"""


	# Golden angle in radians (constant scalar)
	phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, device=freq_magnitudes.device)))

	# Create a range of indices for freq_vec (for all layers)
	freq_vec = torch.arange(freqs_per_layer, dtype=torch.float32, device=freq_magnitudes.device)

	# Compute the y values (scaling from 1 to -1, for spherical coordinates)
	y = 1 - 2 * ((freq_vec + 1) / (freqs_per_layer))  # Shape: (freqs_per_layer,)

	# Broadcast freq_magnitudes to match dimensions for vectorized computation
	y = y.unsqueeze(0) * freq_magnitudes.unsqueeze(1)  # Shape: (num_layers, freqs_per_layer)

	# Compute the radius at each y (for the circle at height y)
	radius_at_y = torch.sqrt(1 - torch.pow(y / freq_magnitudes.unsqueeze(1), 2) + 1e-3) * freq_magnitudes.unsqueeze(1)

	# Compute the theta values (angle increments) for each freq_vec
	theta = phi * freq_vec  # Shape: (freqs_per_layer,)

	# Now compute the x and z values using cos and sin of theta, and radius
	x = torch.cos(theta).unsqueeze(0) * radius_at_y  # Shape: (num_layers, freqs_per_layer)
	z = torch.sin(theta).unsqueeze(0) * radius_at_y  # Shape: (num_layers, freqs_per_layer)

	# Combine x, y, and z into a single tensor for the frequency vectors
	freq_vectors = torch.stack([x, y, z], dim=-1)  # Shape: (num_layers, freqs_per_layer, 3)

	# Reshape to (num_layers * freqs_per_layer, 3)
	freq_vectors = freq_vectors.view(-1, 3)



	return freq_vectors

