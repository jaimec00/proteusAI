# old

import torch



class AminoAcidEmbedding(nn.Module):

	def __init__(self, num_aas=21, d_model=512, esm2_weights_path="utils/model_utils/esm2/esm2_t33_650M_UR50D.pt", d_hidden_aa=1024, hidden_layers_aa=0, learnable_esm=False, dropout=0.0):
		super(AminoAcidEmbedding, self).__init__()

		self.use_esm = esm2_weights_path != ""

		if self.use_esm:
			if not esm2_weights_path.endswith(".pt"): # download the chosen model
				esm2_weights = get_esm_weights(esm2_weights_path)
			else: # load from precomputed file, note that this should be created w/ main func of utils/model_utils/esm2/get_esm_weights.py for proper mapping to proteusAI alphabet
				try:
					esm2_weights = torch.load(esm2_weights_path, weights_only=True)
				except FileNotFoundError as e:
					raise e(f"could not find ESM2 weights at {esm2_weights_path}")

			# initialize esm2 weights
			aa_d_model = esm2_weights["esm2_linear_nobias.weight"].size(1)
			self.aa_linear_nobias = nn.Linear(in_features=num_aas, out_features=aa_d_model, bias=False)
			self.aa_linear_nobias.weight.data = esm2_weights["esm2_linear_nobias.weight"].T

			self.aa_layernorm = nn.LayerNorm(normalized_shape=aa_d_model)
			self.aa_layernorm.weight.data = esm2_weights["esm2_layernorm.weight"]
			self.aa_layernorm.bias.data = esm2_weights["esm2_layernorm.bias"]

			if not learnable_esm:
				self.esm2_linear_nobias.weight.requires_grad = False
				self.esm2_layernorm.weight.requires_grad = False
				self.esm2_layernorm.bias.requires_grad = False
				
		else:
			aa_d_model = d_model
			self.aa_linear_nobias = nn.Linear(num_aas, aa_d_model, bias=False)
			self.aa_layernorm = nn.LayerNorm(aa_d_model)

		self.ffn = MLP(aa_d_model, d_model, d_hidden_aa, hidden_layers_aa, dropout)
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, aas):

		aas = self.aa_linear_nobias(aas)
		aas = self.aa_layernorm(aas)
		aas = self.norm(self.dropout(self.ffn(aas)))

		return aas



class Decoder(nn.Module):
	'''
	bidirectional decoder

	seq updates itself based on struct info and struct_embeddings
	ie seq as Q, struct as KV
	then structure queries sequence, to see how to update itself via struct embeddings
	i.e. structure is QV, sequence is K

	testing, sequence is the variable portion (lots of different masking combos)
	so want a way for sequence to act as a soft suggestion on how to update structure
	also allows the sequence to adapt itself to the structure before acting as the key 

	while not common to use a different modality for K and V (seq and struct), this is minimized
	by first doing standard cross attention, where seq is Q and struct is KV. this allows the seq
	to update itself based on the struct, reducing the difference in modalities for the following unorthodox cross attn layer
	'''

	def __init__(self, d_model=512, d_hidden=1024, hidden_layers=0, nhead=8, min_spread=1, max_spread=6, base=20, num_spread=8, min_rbf=0.01, max_rbf=0.99, beta=2.0, learnable_spreads=False, dropout=0.0, attn_dropout=0.0):
		super(Decoder, self).__init__()

		# seq cross-attention layers
		self.seq_attn = GeoAttention(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, dropout=attn_dropout)
		self.seq_attn_norm = nn.LayerNorm(d_model)
		self.seq_attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.seq_attn_ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.seq_attn_ffn_norm = nn.LayerNorm(d_model)
		self.seq_attn_ffn_dropout = nn.Dropout(dropout)

		# struct cross-attention layers
		self.struct_attn = GeoAttention(d_model, nhead, min_spread=min_spread, max_spread=max_spread, base=base, num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, dropout=attn_dropout)
		self.struct_attn_norm = nn.LayerNorm(d_model)
		self.struct_attn_dropout = nn.Dropout(dropout)

		# Feed-forward network
		self.struct_attn_ffn = MLP(d_model, d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)
		self.struct_attn_ffn_norm = nn.LayerNorm(d_model)
		self.struct_attn_ffn_dropout = nn.Dropout(dropout)

	def forward(self, wf, aas, coords, key_padding_mask=None):

		# seq queries struct to update itself via struct embeddings
		aas2 = self.seq_attn(	aas, wf, wf,
							coords=coords,
							key_padding_mask=key_padding_mask
						)

		# residual connection with dropout
		aas = self.seq_attn_norm(aas + self.seq_attn_dropout(aas2))

		# Feed-forward network for aas
		aas = self.seq_attn_ffn_norm(aas + self.seq_attn_ffn_dropout(self.seq_attn_ffn(aas)))

		# struct queries seq to update itself, via struct embeddings
		wf2 = self.struct_attn(	wf, aas, wf,
							coords=coords,
							key_padding_mask=key_padding_mask
						)

		# residual connection with dropout
		wf = self.struct_attn_norm(wf + self.struct_attn_dropout(wf2))

		# Feed-forward network for wavefunction
		wf = self.struct_attn_ffn_norm(wf + self.struct_attn_ffn_dropout(self.struct_attn_ffn(wf)))

		return wf, aas

		
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

