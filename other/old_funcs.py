# old

import torch

class PositionalEncoding(nn.Module):
	def __init__(self, N, d_model=512):
		super(PositionalEncoding, self).__init__()
		self.positional_encoding = torch.zeros(N, d_model) # N x d_model
		
		position = torch.arange(0, N, dtype=torch.float32).unsqueeze(1) # N x 1
		div_term = torch.pow(10000, (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)) # sin and cos terms for each div term, so has length d_model/2 ; d_model/2,
		
		self.positional_encoding[:, 0::2] = torch.sin(position / div_term) # N x d_model
		self.positional_encoding[:, 1::2] = torch.cos(position / div_term) # N x d_model
		
		# Register as buffer so it's part of the model but not trainable
		# self.register_buffer('positional_encoding', positional_encoding)
	def forward(self, x):
		pe = self.positional_encoding[None, :, :].to(x.device) # batch x N x d_model
		x = x + pe # batch x N x d_model
		return x

class CrossFeatureNorm(nn.Module):
	'''
	normalizes each feature independantly across the sequence. it is independant of batches (not batch norm)
	this is helpful because each feature for a given token (Ca atom) is the output of that token for the global 
	superposed wavefunction at a particular k (wavelength). thus, each feature in a given token is only relevant
	RELATIVE to the CORRESPONDING features of all other tokens in the sequence. 
	This essentially normalizes each wavefunction's (psi_k) output. Note that this normalizes the real part and 
	the imaginary part independantly 
	'''
	def __init__(self, d_model, eps=1e-5):
		super(CrossFeatureNorm, self).__init__()

		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(1,1,d_model))
		self.beta = nn.Parameter(torch.zeros(1,1,d_model))

	def forward(self, x, key_padding_mask=None):

		if key_padding_mask is not None:
			# key_padding_mask is of shape (batch, N) - we invert it to use as a valid mask
			valid_mask = ~key_padding_mask.unsqueeze(-1)  # batch x N --> batch x N x 1
			
			# Mask invalid (padded) positions
			x_masked = x * valid_mask  # Zero out padded positions in x ; batch x N x d_model

			# Compute the mean and variance only for valid positions
			sum_valid = torch.sum(x_masked, dim=1) # batch x d_model
			num_valid = valid_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
			mean = sum_valid / num_valid  # shape (batch, d_model)

			# Subtract the mean for valid positions
			mean_expanded = mean.unsqueeze(1)  # shape (batch, 1, d_model)
			x_centered = (x - mean_expanded) * valid_mask  # Zero out padded positions again after centering

			# Compute variance only for valid positions
			variance = torch.sum(x_centered ** 2, dim=1) / num_valid  # shape (batch, d_model)
			variance_expanded = variance.unsqueeze(1)
			std = torch.sqrt(variance_expanded + self.eps)

			# Normalize the valid positions
			x_norm = (x_centered / std) * valid_mask 

		else:
			# compute mean and variance ; batch x N x d_model --> batch x 1 x d_model
			mean = x.mean(dim=1, keepdim=True)
			var = x.var(dim=1, keepdim=True, unbiased=False)

			# normalize each feature independently across the sequence ; batch x N x d_model
			x_norm = (x - mean) / torch.sqrt(var + self.eps)


		# apply learnable scaling (gamma) and shifting (beta) to each feature
		x = self.gamma * x_norm + self.beta

		return x

class StaticLayerNorm(nn.Module):
	def __init__(self, normalized_shape, eps=1e-5):
		super(StaticLayerNorm, self).__init__()
		self.normalized_shape = normalized_shape
		self.eps = eps  # Small epsilon to prevent division by zero

	def forward(self, x):
		# Calculate the mean and variance along the feature dimension
		mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, N, 1)
		var = x.var(dim=-1, keepdim=True, unbiased=False)  # Shape: (batch_size, N, 1)

		# Normalize to zero mean and unit variance without learned parameters
		x_normalized = (x - mean) / torch.sqrt(var + self.eps)
		return x_normalized

def pt_to_data(pts: Path, all_bb: int=0, device="cpu", features=False, num_inputs=512, max_size=10000):
	all_tensors = []
	all_labels = []
	for idx, pt_dir in enumerate(pts.iterdir()):

		if idx >= num_inputs: break

		if not features:
			ca_pt = pt_dir / f"{pt_dir.name}_ca.pt"
			ca_pt = torch.load(ca_pt, map_location=device, weights_only=True)
		else:
			ca_pt = pt_dir / f"{pt_dir.name}_features.pt"
			ca_pt = torch.load(ca_pt, map_location=device, weights_only=True).squeeze(0)

		aa_pt = pt_dir / f"{pt_dir.name}_aa.pt"
		aa_pt = torch.load(aa_pt, map_location=device, weights_only=True)

		all_tensors.append(ca_pt)
		all_labels.append(aa_pt)

	data = Data(all_tensors, all_labels, device)

	return data

def pdbs_to_data(pdbs: list[Path], all_bb: int=0):

	parser = PDBParser(QUIET=True)
	amino_acids = "ACDEFGHIKLMNPQRSTVWY"

	all_bb_coords, all_labels = [], []
	
	for pdb in pdbs:
	
		pdb_id = pdb.name.rstrip(".pdb")
		structure = parser.get_structure(pdb_id, pdb)
		try: 
			model = structure[0]
		except KeyError:
			continue

		sequence = ""
		bb_coords = []

		for chain_idx, chain in enumerate(model):
	
			for position, resi in enumerate(chain): # this assumes all residues modeled in the pdb, need to filter input pdbs from rcsb for this 

				Ca_bb = resi['CA'].coord
				N_bb = resi['N'].coord
				C_bb = resi['C'].coord
				O_bb = resi['O'].coord

				pos_bb_coords = [list(coords) for coords in [Ca_bb, N_bb, C_bb, O_bb]] if all_bb else list(Ca_bb)
				bb_coords.append(pos_bb_coords)

				three_letter = resi.get_resname() 
				aa = protein_letters_3to1[three_letter[0].upper() + three_letter[1:].lower()]
				sequence += aa

			break # only working with one chain for now

		bb_coords = torch.tensor(bb_coords)
		bb_coords = translate_origin_to_COM(bb_coords)
		bb_coords = rotate_with_PCA(bb_coords)

		label = torch.zeros(len(sequence), 20)
		for pos, aa in enumerate(sequence):
			label[pos, amino_acids.index(aa)] = 1

		assert bb_coords.size(0) == label.size(0)
		label = torch.argmax(label, dim=-1)

		all_bb_coords.append(bb_coords)
		all_labels.append(label)

	data = Data(all_bb_coords, all_labels)

	return data



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

