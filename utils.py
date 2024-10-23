# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		utils.py
description:	utility functions
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
from pathlib import Path

def protein_to_wavefunc(coords: torch.Tensor, key_padding_mask: torch.Tensor):
    '''
    converts the alpha carbon coordinates of a protein into a tensor of 
    wavefunction outputs.
    converts a batch x N x 3 tensor to a batch x N x 512 tensor.
    each feature for a Ca is their output of the wave function with a specific wavelength
    each output gets two features, one for the real part, and another for the imaginary part
    the wave function is a superposition of Green's functions, treating each Ca as a point source

    Args:
        coords (torch.Tensor):      tensor containing batches of Ca coords. 
                                    size = batch x N x 3 
    
    Returns:
        features (torch.Tensor):    tensor containing batches of token (Ca) features.
                                    size = batch x N x 512
    '''

    # get the euclidean distances ; batch x N x 3 --> batch x N x N 
    pw_dists = torch.sqrt(torch.sum((coords[:, :, None, :] - coords[:, None, :, :]) ** 2, dim=-1)) 
    
    # define an identity to avoid division by zero later ; batch x N x N 
    identity = torch.eye(pw_dists.size(1)).expand(pw_dists.size(0), -1, -1)
    
    # diagonal set to 100 to avoid division by zero and to exclude it from min()
    pw_dists = pw_dists + identity * 100

    # masked values pw_dists set to inf
    pw_dist_mask = key_padding_mask[:, :, None] | key_padding_mask[:, None, :] # batch x N x N
    pw_dists = pw_dists.masked_fill(pw_dist_mask, 100)

    # define wavelengths

    # get each ca's minimum ; batch x N x N --> batch x N
    min_wls = torch.min(pw_dists, dim=-1)[0]

    # average the minimum over each batch while allowing for broadcasting 
    # along last dimension ; batch x N --> batch x 1 
    min_wl = (torch.sum(min_wls * ~key_padding_mask, dim=1) / torch.sum(~key_padding_mask, dim=1)).view(min_wls.size(0), 1)

    # subtract 999 from diagonals to keep them reasonable and non-zero and to exclude it from maximum calculation 
    pw_dists = pw_dists - identity * 99

    # masked values set to -inf
    pw_dists = pw_dists.masked_fill(pw_dist_mask, 0)

    # get each ca's maxiumm ; batch x N x N --> batch x N
    max_wls = torch.max(pw_dists, dim=-1)[0]

    # average the maxiumm over each batch while allowing for broadcasting 
    # along last dimension ; batch x N --> batch, 1 
    max_wl = (torch.sum(max_wls * ~key_padding_mask, dim=1) / torch.sum(~key_padding_mask, dim=1)).view(max_wls.size(0), 1)

    # set masked values back to inf
    pw_dists = pw_dists.masked_fill(pw_dist_mask, float('inf'))

    # Create a tensor of wavelengths

    # short range wavelengths get 128 wave functions, medium get 96, and long get 32. each wave function
    # creates two features, one real and one imaginary, for a total of 256*2 = 512 features
    # create evenly spaced tensors from 0 to 1 of shape 1,  
    features_short = torch.linspace(0, 1, 128).expand(min_wl.size(0), 128) # batch x 128
    features_medium = torch.linspace(0, 1, 96).expand(min_wl.size(0), 96) # batch x 96
    features_long = torch.linspace(0, 1, 32).expand(min_wl.size(0), 32) # batch x 32

    # create the wavelength tensors for each range
    wavelengths_short = min_wl + features_short * (8 - min_wl) # batch x 128
    wavelengths_medium = 8.1 + features_medium * (20 - 8.1) # batch x 96
    wavelengths_long = 20.5 + features_long * (max_wl - 20.5) # batch x 32
    
    # concat all the features into a single feature tensor
    all_wl = [wavelengths_short, wavelengths_medium, wavelengths_long]
    wavelengths = torch.cat(all_wl, dim = 1) # batch x 256

    # convert wavelengths to k_values and set up for broadcasting along middle two dimensions (N and N from pw_dists)
    k_values = 2 * torch.pi / wavelengths[:, None, None, :] # batch x 256 --> batch x 1 x 1 x 256
    pw_dists = pw_dists[:, :, :, None] # set up for broadcasting along last dim ; batch x N x N --> batch x N x N x 1

    # for phase calculation, make masked values zero, to avoid problems with large exponentials
    phase_pw_dists = pw_dists.masked_fill(pw_dist_mask[:, :, :, None], 0.0) # batch x N x N x 1
    phase =  phase_pw_dists * k_values # batch x N x N x 1 * batch x 1 x 1 x 256 --> batch x N x N x 256

    # compute the Green's function for all wavelengths, for all batches
    greens_fn = (torch.cos(phase) + 1.0j * torch.sin(phase)) / (pw_dists * 4 * torch.pi) # batch x N x N x 256
    # note that for masked values, the phase is 0.0, and the magnitude is 1/inf = 0, so when superpose the functions, masked values have no bearing
    # actually get nan+nanj for masked values for some reason, so just set them to 0
    greens_fn = greens_fn.masked_fill(pw_dist_mask[:, :, :, None], 0.0)

    # subtract the diagonal terms that were there to prevent division by zero
    greens_fn_identity = torch.eye(greens_fn.size(1))[None, :, :, None] * greens_fn # batch x N x N x 256
    greens_fn = greens_fn - greens_fn_identity # batch x N x N x 256

    # superpose the wave functions for all wavelengths
    superpositions = torch.sum(greens_fn, dim=2)  # sum over the third dimension ; batch x N x 256

    # extract the real and imaginary parts
    real_sup = superpositions.real # batch x N x 256
    imag_sup = superpositions.imag # batch x N x 256

    # this is not learnable, so can break the computational chain and do index assignment
    # every k value gets two features
    # want real and imaginary parts next to each other (for a single k)
    features = torch.zeros(superpositions.size(0), superpositions.size(1), superpositions.size(2)*2) # batch x N x 512
    features[:, :, 0::2] = real_sup # batch x N x 512
    features[:, :, 1::2] = imag_sup # batch x N x 512

    return features

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

