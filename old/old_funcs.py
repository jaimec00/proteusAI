# old

import torch


class VAELosses():
	'''
	class to store losses
	'''
	def __init__(self): 

		# saved for logging
		self.kl_div = [] # kl div to match gaussian prior
		self.reconstruction = [] # reconstruction of the wf

		self.kl_div_no_aa = []
		self.reconstruction_no_aa = []

		# actual losses
		self.all_losses = [] # full scaled loss, contains kldiv, reconstruction, and cel

		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token per cha

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''

		valid_toks = self.valid_toks.item()
		avg_kl_div = sum(kl_div.item() for kl_div in self.kl_div if kl_div) / valid_toks
		avg_reconstruction = sum(reconstruction.item() for reconstruction in self.reconstruction if reconstruction) / valid_toks
		avg_kl_div_no_aa = sum(kl_div.item() for kl_div in self.kl_div_no_aa if kl_div) / valid_toks
		avg_reconstruction_no_aa = sum(reconstruction.item() for reconstruction in self.reconstruction_no_aa if reconstruction) / valid_toks
		avg_loss = sum(loss.item() for loss in self.all_losses if loss) / valid_toks
		
		return avg_kl_div, avg_reconstruction, avg_kl_div_no_aa, avg_reconstruction_no_aa, avg_loss

	def add_losses(self, kl_div, reconstruction, kl_div_no_aa, reconstruction_no_aa, full_loss, valid_toks=1):
		self.kl_div.append(kl_div)
		self.reconstruction.append(reconstruction)
		self.kl_div_no_aa.append(kl_div_no_aa)
		self.reconstruction_no_aa.append(reconstruction_no_aa)
		self.all_losses.append(full_loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.kl_div.extend(other.kl_div)
		self.reconstruction.extend(other.reconstruction)
		self.kl_div_no_aa.extend(kl_div_no_aa)
		self.reconstruction_no_aa.extend(reconstruction_no_aa)
		self.all_losses.extend(other.all_losses)
		self.valid_toks += valid_toks

	def clear_losses(self):
		self.kl_div = []
		self.reconstruction = []
		self.kl_div_no_aa = []
		self.reconstruction_no_aa = []
		self.all_losses = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.all_losses[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.kl_div = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.kl_div]
		self.reconstruction = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.reconstruction]
		self.kl_div_no_aa = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.kl_div_no_aa]
		self.reconstruction_no_aa = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.reconstruction_no_aa]
		self.all_losses = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.all_losses]

	def __len__(self):
		return len(self.all_losses)

class DiffusionLosses():
	def __init__(self):
		self.squared_errors = []
		# self.nll = []
		self.total_loss = [] # squared_err + gamma*nll
		self.valid_toks = 0

	def get_avg(self, is_inference=False):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		valid_toks = self.valid_toks.item()
		if is_inference: # store the seq sims in squared errors list, instead of dealing w seperate lists the whole run
			return 100*sum(match.item() for match in self.squared_errors if match) / valid_toks
		else:
			squared_err = sum(loss.item() for loss in self.squared_errors if loss) / valid_toks
			# nll = sum(loss.item() for loss in self.nll if loss) / valid_toks
			total_loss = sum(loss.item() for loss in self.total_loss if loss) / valid_toks

			return squared_err, total_loss
		
	def add_losses(self, squared_error, total_loss, valid_toks=1):
		self.squared_errors.append(squared_error)
		# self.nll.append(nll)
		self.total_loss.append(total_loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		self.squared_errors.extend(other.squared_errors)
		# self.nll.extend(other.nll)
		self.total_loss.extend(other.total_loss)
		self.valid_toks += other.valid_toks

	def clear_losses(self):
		self.squared_errors = []
		# self.nll = []
		self.total_loss = []
		self.valid_toks = 0

	def get_last_loss(self):
		return self.total_loss[-1]

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.squared_errors = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.squared_errors]
		# self.nll = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.nll]
		self.total_loss = [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else np.array([loss]) for loss in self.total_loss]

	def __len__(self):
		return len(self.total_loss)


class VAELossFunction(nn.Module):

	def __init__(self, beta=1.0, kappa=1.0, midpoint=4000, anneal=True):
		super(VAELossFunction, self).__init__()
		self.beta = beta
		self.kappa = kappa
		self.midpoint = midpoint
		self.anneal = anneal
		self.kl_annealing_step = 0 # for kl annealing

	def kl_div(self, prior_mean_pred, prior_log_var_pred, mask):
		kl_div = -0.5*torch.sum(1 + prior_log_var_pred - prior_mean_pred.pow(2) - torch.exp(prior_log_var_pred), dim=2) # Z x N
		
		return (kl_div*(~mask)).sum() 

	def reconstruction(self, reconstruct_mean_pred, reconstruct_mean_true, mask):
		return ((reconstruct_mean_true - reconstruct_mean_pred).pow(2) * (~mask).unsqueeze(2)).sum()

	def full_loss(self, kl_div, reconstruction):# cel is typically larger than mse and kldiv, so scale it down so vae focuses on wf reconstruction more

		# beta starts small and gradualy increases	
		beta = self.beta if not self.anneal else self.beta/(1+math.exp(-self.kappa*(self.kl_annealing_step-self.midpoint)))
		return ( beta * kl_div) + reconstruction 

	def forward(self, 	prior_mean_pred, prior_log_var_pred,
						reconstruct_mean_pred, 
						prior_mean_pred_no_aa, prior_log_var_pred_no_aa,
						reconstruct_mean_pred_no_aa, 
						reconstruct_mean_true, mask
				):
		kl_div = self.kl_div(prior_mean_pred, prior_log_var_pred, mask)
		reconstruction = self.reconstruction(reconstruct_mean_pred, reconstruct_mean_true, mask)
		kl_div_no_aa = self.kl_div(prior_mean_pred_no_aa, prior_log_var_pred_no_aa, mask)
		reconstruction_no_aa = self.reconstruction(reconstruct_mean_pred_no_aa, reconstruct_mean_true, mask) 
		full_loss = self.full_loss(kl_div, reconstruction) + self.full_loss(kl_div_no_aa, reconstruction_no_aa)

		return kl_div, reconstruction, kl_div_no_aa, reconstruction_no_aa, full_loss # return all for logging, only full loss used for backprop

class DiffusionLossFunction(nn.Module):
	def __init__(self, gamma=1.0): # gamma scales the nll term
		super(DiffusionLossFunction, self).__init__()
		self.gamma = gamma

	def forward(self, noise_pred, noise_true, abar, mask):
		'''sum of squared errors plus an NLL term to evaluate the probability of the estimated x0 under encoders mean and var'''

		squared_err = ((noise_true - noise_pred).pow(2)*(~mask)).sum()
		loss = squared_err #+ self.gamma*nll # testing if nll implicitly improves squared err
		return squared_err, loss

# ----------------------------------------------------------------------------------------------------------------------

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


class DiTEncoder(nn.Module):
	'''
	encoder used in DiT w/ timestep conditioning via adaLN(Zero)
	'''
	def __init__(self, 	d_model=512, heads=8, 
						d_hidden=2048, hidden_layers=0, dropout=0.0,
						bias=False, min_rbf=0.000,
						d_in_t=512, d_hidden_t=2048, hidden_layers_t=512
					):
		super(DiTEncoder, self).__init__()

		# Self-attention layers
		if bias:
			self.attn = GeoAttention(d_model=d_model, heads=heads, min_rbf=min_rbf)
		else:
			self.attn = Attention(d_model=d_model, d_other=d_model, heads=heads)

		# adaptive layernorm
		self.static_norm = StaticLayerNorm(d_model)
		self.attn_adaLN = adaLN(d_in=d_in_t, d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t, dropout=dropout)
		self.ffn_adaLN = adaLN(d_in=d_in_t,  d_model=d_model, d_hidden=d_hidden_t, hidden_layers=hidden_layers_t, dropout=dropout)

		# feed forward network
		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)

		# dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, t, coords, key_padding_mask=None):
	
		# get the ada ln
		gamma1, beta1, alpha1 = self.attn_adaLN(t)
		gamma2, beta2, alpha2 = self.ffn_adaLN(t)

		# attn
		x2 = gamma1*self.static_norm(x) + beta1
		x2 = self.attn(x2, x2, x2, coords, mask=key_padding_mask)
		x = x + self.dropout(x2*alpha1)

		# ffn
		x2 = gamma2*self.static_norm(x) + beta2
		x = x + self.dropout(self.ffn(x2)*alpha2)

		return x


class WaveFunctionDecoding(nn.Module):
	def __init__(self,  d_model=256, d_latent=32, d_proj=512,
						d_hidden_pre=2048, hidden_layers_pre=0, 
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=4, heads=8,
						use_bias=False, min_rbf=0.000,
						d_hidden_attn=1024, hidden_layers_attn=0,
						dropout=0.10,
					):

		super(WaveFunctionDecoding, self).__init__()
		
		self.dropout = nn.Dropout(dropout)

		self.d_proj = nn.Linear(d_latent, d_proj)
		self.norm = nn.LayerNorm(d_proj)
		self.space_enc = MLP(d_in=d_model, d_out=d_proj, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout) 

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_proj, d_other=d_proj, heads=heads, 
													bias=use_bias, min_rbf=min_rbf,
													d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													dropout=dropout
												)
												for _ in range(encoder_layers)
											])

		self.mlp_post = MLP(d_in=d_proj, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=0.0)

	def forward(self, wf, wf_no_aa, key_padding_mask=None): # forward generates the mean and stds, so can use them for loss, 
		
		# pre-process the wf
		wf = self.d_proj(wf)
		wf = self.norm(wf + self.dropout(self.space_enc(wf_no_aa))) # spatial encoding on aa ambiguous wf (cb scale is mean of all aas for each k)

		# self attention on updated wf
		for encoder in self.encoders:
			wf = encoder(wf, wf, wf, mask=key_padding_mask)

		# post-process to get final wf
		wf = self.mlp_post(wf)

		return wf

class WaveFunctionEncoding(nn.Module):
	def __init__(self,  d_model=256, d_latent=32, d_proj=512,
						d_hidden_pre=1024, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=1,
						encoder_layers=4, heads=8,
						use_bias=False, min_rbf=0.000,
						d_hidden_attn=1024, hidden_layers_attn=0,
						dropout=0.10
				):
		super(WaveFunctionEncoding, self).__init__()

		self.dropout = nn.Dropout(dropout)

		# pre preocess
		self.d_proj = nn.Linear(d_model, d_proj)
		self.mlp_pre = MLP(d_in=d_proj, d_out=d_proj, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout)
		self.norm_pre = nn.LayerNorm(d_proj)

		# self attention on wf
		self.encoders = nn.ModuleList([ 	Encoder(d_model=d_proj, d_other=d_proj, heads=heads, 
													min_rbf=min_rbf, bias=use_bias,
													d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													dropout=dropout)
											for _ in range(encoder_layers)
										])

		# post process to get mean and log vars
		self.mlp_post = MLP(d_in=d_proj, d_out=2*d_latent, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=0.0) # no dropout on this mlp
		
	def forward(self, wf, key_padding_mask=None, a=2.0): # forward generates the mean and log vars, so can use them for loss, use self.encode to directly sample from latent 
		
		# preprocess
		wf = self.d_proj(wf)
		wf = self.norm_pre(wf + self.dropout(self.mlp_pre(wf)))

		# wf encoders
		for encoder in self.encoders:
			wf = encoder(wf, wf, wf, mask=key_padding_mask)

		# get means and logvars
		latent_stats = self.mlp_post(wf)
		latent_mean, latent_log_var = torch.chunk(latent_stats, chunks=2, dim=2)

		# map the log var to [-a,a], for numerical stability
		# latent_log_var = a*torch.tanh(latent_log_var/a)

		return latent_mean, latent_log_var

	def sample(self, latent_mean, latent_log_var):
		return latent_mean + torch.exp(latent_log_var*0.5)*torch.randn_like(latent_log_var)

	def encode(self, wf, key_padding_mask=None):
		latent_mean, latent_log_var = self.forward(wf, key_padding_mask=key_padding_mask)
		protein_latent = self.sample(latent_mean, latent_log_var)
		return protein_latent


def inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, cycles=10, diffusion_iters=1, temp=1e-6):

	# prep
	batch, N = aas.shape
	t_max = self.wf_diffusion.noise_scheduler.t_max

	# fixed position have an AA label, positions to predict are -1, so they are set to random aa. doesnt matter too much, bc starts at white noise
	fixed_aas = aas!=-1
	aas = torch.where(fixed_aas, aas, torch.randint_like(aas, 0, self.num_aas))

	wf_no_aa = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=True)

	# multiple embedding + encoding + diffusion + decoding + extraction runs, each giving a slightly better guess and thus using less noise
	for t_fwd in range(t_max, 0, -t_max//cycles):

		# perform embedding
		wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

		# all of the following modules do not use coords alpha, since replaced geo attn with regular attn + spatial encoding, but leaving it until i am sure geo attn is dead

		# encode from wf space to latent space
		protein = self.wf_encoding.encode(wf, key_padding_mask=key_padding_mask)
		
		for diffusion_iter in range(diffusion_iters): # slight nudge, then denoise, then repeat, nudging latent towards manifold, rather than starting from full gaussian
			
			# add gaussian noise to latent space
			protein_noised, _ = self.wf_diffusion.noise(protein, t_fwd)

			# remove the noise
			protein = self.wf_diffusion.denoise(protein_noised, t_fwd, key_padding_mask=key_padding_mask)

		# decode from latent space to wf space
		wf_pred = self.wf_decoding(protein, wf_no_aa, key_padding_mask=key_padding_mask)

		# extract sequence from wf
		aa_pred = self.wf_extraction.extract(wf_pred, wf_no_aa, key_padding_mask=key_padding_mask)

		# keeping fixed positions as they were for next iteration, 
		aas = torch.where(fixed_aas, aas, aa_pred)

	return aas

def mlm_inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, entropy_increment=0.1, temp=1e-6):

	entropy_threshold = 0.1 # entropy threshold
	entropies = torch.zeros_like(aas, dtype=torch.float32) + float("inf") # will return this later so user knows confidence
	fixed = (aas!=-1) | key_padding_mask # if not -1, then should be fixed as context
	predicted = fixed # to keep track of predicted pos as loop through
	aas.masked_fill_(~fixed, 20) # add the mask token to non fixed tokens

	while not predicted.all(): # already deals w/ masking

		# get wf
		wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

		# predict sequence
		aa_logits = self.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask)
		pred_aa = self.wf_extraction.sample(aa_logits, temp)

		# compute entropy
		probs = aa_logits.softmax(dim=2)
		entropy = torch.sum(-probs*torch.log(probs), dim=2)
		
		# update
		update_aa = (entropy < entropy_threshold) & (entropies == float("inf")) & (~predicted.all(dim=1, keepdim=True)) # dont update already finished samples
		entropies[update_aa] = entropy[update_aa]
		aas[update_aa] = pred_aa[update_aa]
		predicted |= update_aa

		# increment entropy threshold
		entropy_threshold += entropy_increment

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


	def run_vae_training(self):
		
		# get wf
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)
		wf_no_aa = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True, no_aa=True)
		
		# predict mean and log var
		wf_encoded_mean, wf_encoded_log_var = self.epoch_parent.training_run_parent.model(wf=wf, key_padding_mask=self.key_padding_mask, encoding=True)

		# sample from the latent space given the mean and var
		wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.sample(wf_encoded_mean, wf_encoded_log_var)
		
		# decode from latent space to wf space, only computes mean to make reconstruction loss a simple squared error
		wf_decoded = self.epoch_parent.training_run_parent.model(latent=wf_encoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, decoding=True)

		# now for aa ambiguous wf
		
		# predict mean and log var
		wf_encoded_mean_no_aa, wf_encoded_log_var_no_aa = self.epoch_parent.training_run_parent.model(wf=wf_no_aa, key_padding_mask=self.key_padding_mask, encoding=True)

		# sample from the latent space given the mean and var
		wf_encoded_no_aa = self.epoch_parent.training_run_parent.model.wf_encoding.sample(wf_encoded_mean_no_aa, wf_encoded_log_var_no_aa)
		
		# decode from latent space to wf space, only computes mean to make reconstruction loss a simple squared error
		wf_decoded_no_aa = self.epoch_parent.training_run_parent.model(latent=wf_encoded_no_aa, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, decoding=True)

		# convert to output object
		return VAEOutput(self, wf_encoded_mean, wf_encoded_log_var, wf_decoded, wf_encoded_mean_no_aa, wf_encoded_log_var_no_aa, wf_decoded_no_aa, wf)

	def run_extraction_finetune_training(self):
		
		# get wf
		wf = self.epoch_parent.training_run_parent.model(self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)
		wf_no_aa = self.epoch_parent.training_run_parent.model(self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True, no_aa=True)

		# make into latent representation
		wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.encode(wf, key_padding_mask=self.key_padding_mask)
		
		# decode from latent space to wf space, only computes mean to make reconstruction loss a simple squared error
		wf_decoded = self.epoch_parent.training_run_parent.model(latent=wf_encoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, decoding=True)

		# extract sequence
		seq_pred = self.epoch_parent.training_run_parent.model(wf=wf_decoded, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, extraction=True)

		# convert to output object
		return ExtractionOutput(self, seq_pred)

	def run_diffusion_training(self):
		
		# get clean wavefunction
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)

		# get timesteps from uniform distribution, as well as abars for reconstructing x0 for nll loss
		timesteps = self.epoch_parent.training_run_parent.model.wf_diffusion.get_random_timesteps(wf.size(0), wf.device)
		abars, _ = self.epoch_parent.training_run_parent.model.wf_diffusion.noise_scheduler(timesteps.unsqueeze(1).unsqueeze(2)) # for loss scaling

		# noise = wf - wf_no_aa # completely shatters assumption of gaussian noise, not score matching anymore, but nothing is working
		# noised_wf = (abars**0.5)*wf + ((1-abars)**0.5)*wf_no_aa # hope is to define a linear path between no aa and correct aa wf, literally just making shit up at this point
		# what the fuck, its working i think. see if it can predict wfaa just from base in one step
		# noised_wf = wf_no_aa
		# timesteps = torch.ones(wf.size(0), device=wf.device) # all at t=1 for consistent signal

		# encode the wf in latent space
		# wf_latent_mean, wf_latent_logvar = self.epoch_parent.training_run_parent.model(wf=wf, wf_no_aa=wf_no_aa, key_padding_mask=self.key_padding_mask, encoding=True)
		# wf_encoded = self.epoch_parent.training_run_parent.model.wf_encoding.sample(wf_latent_mean, wf_latent_logvar)


		# add noise
		noised_wf, noise = self.epoch_parent.training_run_parent.model.wf_diffusion.noise(wf, timesteps)
		# noise = (abars**0.5)*noise - ((1-abars)**0.5)*wf # v param

		# predict noise
		noise_pred = self.epoch_parent.training_run_parent.model(latent=noised_wf, t=timesteps, coords_alpha=self.coords_alpha, key_padding_mask=self.key_padding_mask, diffusion=True)

		# convert to output object
		return DiffusionOutput(self, noise_pred, noise, abars)

	def run_mlm_training(self):
		''' 
		fucccccccccccck diffusion isnt working. doing mlm so i can at least get something for my thesis. but will go back to diffusion right after. the idea is too fucking good to fail.
		have a special mask token embedded in wf representation. plan for inference is to update most confident token(s) and rerun with update until convergence
		also allow the model to predict non masked residues, as wf embedding 
		distributes aa info to all tokens, so basically non masked aas still are obfuscated by the mask token
		also allows updates during inference as the model gains more context

		probably going to do autoregressive inference, need to run a few tests, but always overfits when use true labels, so training
		it on its own predictions as context. 
		similar technique to recycles, except that always run no context first with no grad. 
		'''

		# first run it with no aa info, all mask tokens
		# all_mask = torch.full_like(self.aas, 20)
		# with torch.no_grad():
		# 	wf = self.epoch_parent.training_run_parent.model.module(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=all_mask, key_padding_mask=self.key_padding_mask, embedding=True)
		# 	seq_pred = self.epoch_parent.training_run_parent.model.module(wf=wf, coords_alpha=self.coords_alpha, key_padding_mask=self.key_padding_mask, extraction=True)
		# 	self.aas = seq_pred.argmax(dim=2)

		# get random aa masking percentages
		rand_aa_pct = torch.rand((self.size(0), 1), device=self.coords.device) * 0.75 # about 25% is the max context it gets

		# apply the random aas (20 is mask token)
		rand_vals = torch.rand_like(self.aas, dtype=torch.float32)
		is_mask = rand_vals > rand_aa_pct
		self.aas = torch.where(is_mask, 20, self.aas)

		# only predict masked vals
		self.labels = torch.where(is_mask, self.labels, -1)

		# run wf embedding (slow, learnable aa)
		wf = self.epoch_parent.training_run_parent.model.module(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, embedding=True)

		# run wf extraction
		seq_pred = self.epoch_parent.training_run_parent.model.module(wf=wf, coords_alpha=self.coords_alpha, key_padding_mask=self.key_padding_mask, extraction=True)

		return ExtractionOutput(self, seq_pred, None, None) # no distogram for now
	
	def run_mlm_inference(self):
		''' 
		fucccccccccccck diffusion isnt working. doing mlm so i can at least get something for my thesis. but will go back to diffusion right after. the idea is too fucking good to fail.
		have a special mask token embedded in wf representation. plan for inference is to update most confident token(s) and rerun with update until convergence
		also allow the model to predict non masked residues, as wf embedding 
		distributes aa info to all tokens, so basically non masked aas still are obfuscated by the mask token
		also allows updates during inference as the model gains more context
		'''


		# apply the random aas (20 is mask token)
		is_mask = ~self.chain_mask # give the model sequence info of non representative chains like in pmpnn
		self.aas = torch.where(is_mask, 20, self.aas) 

		# only predict masked vals
		self.labels = torch.where(is_mask, self.labels, -1)

		# run wf embedding (slow, learnable aa)
		wf = self.epoch_parent.training_run_parent.model(coords_alpha=self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask, temp=self.temp, inference=True)

		# run wf extraction
		seq_pred = self.epoch_parent.training_run_parent.model(wf=wf, coords_alpha=self.coords_alpha, key_padding_mask=self.key_padding_mask, extraction=True)

		return ExtractionOutput(self, seq_pred)

	def run_full_inference(self):

		# inference uses random aas to test predictions from only structure, seq sim only computed for representative chain. model gets full structure info, but no sequence info
		self.aas = -torch.ones_like(self.aas) # no sequence info at all. diffusion starts by making all non fixed (eq. -1) positions to random AA before running
		t = torch.full((self.coords_alpha.size(0),), self.epoch_parent.training_run_parent.hyper_parameters.diffusion.scheduler.t_max)

		seq_pred = self.epoch_parent.training_run_parent.model(	self.coords_alpha, coords_beta=self.coords_beta, aas=self.aas, key_padding_mask=self.key_padding_mask,
																inference=True, t=t,
																cycles=self.epoch_parent.training_run_parent.training_parameters.inference.cycles, 
																temp=self.epoch_parent.training_run_parent.training_parameters.inference.temperature
															)

		return InferenceOutput(self, seq_pred)


class VAEOutput():
	'''
	computes loss for true wf, and the aa ambiguous function, with the goal of forcing the encoder to map similar structures to similar latent spaces
	'''
	def __init__(self, 	batch_parent, 
						latent_mean_pred, latent_log_var_pred, wf_mean_pred, 
						latent_mean_pred_no_aa, latent_log_var_pred_no_aa, wf_mean_pred_no_aa,
						wf_mean_true
				):
		
		# batch parent
		self.batch_parent = batch_parent 

		# predictions
		# encoder outputs, loss is computed by comparing to gaussian, so dont need a true
		self.latent_mean_pred = latent_mean_pred # gaussian prior mean
		self.latent_log_var_pred = latent_log_var_pred # gaussian prior log var
		self.latent_mean_pred_no_aa = latent_mean_pred_no_aa # also do aa ambiguous
		self.latent_log_var_pred_no_aa = latent_log_var_pred_no_aa 

		# decoder outputs
		self.wf_mean_pred = wf_mean_pred # wf prediction mean
		self.wf_mean_pred_no_aa = wf_mean_pred_no_aa # wf prediction mean
		self.wf_mean_true = wf_mean_true # true

		# valid tokens for averaging
		self.valid_toks = (batch_parent.labels!=-1).sum()

	def compute_losses(self):
		return self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(	self.latent_mean_pred, self.latent_log_var_pred, 
																						self.wf_mean_pred, 
																						self.latent_mean_pred_no_aa, self.latent_log_var_pred_no_aa, self.wf_mean_pred_no_aa, 
																						self.wf_mean_true, 
																						self.batch_parent.labels==-1, 
																					)

class DiffusionOutput():
	def __init__(self, batch_parent, noise_pred, true_noise, abars):
		self.batch_parent = batch_parent
		self.noise_pred = noise_pred
		self.true_noise = true_noise
		self.mask = batch_parent.labels.unsqueeze(2)==-1
		self.valid_toks = (~self.mask).sum()
		self.abars = abars
	def compute_losses(self):
		return self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(self.noise_pred, self.true_noise, self.abars, self.mask)

class InferenceOutput():
	def __init__(self, batch_parent, seq_pred):
		self.batch_parent = batch_parent 
		self.seq_pred = seq_pred
		self.valid_toks = (batch_parent.labels!=-1).sum()
		self.valid_samples = (batch_parent.labels!=-1).any(dim=1).sum()

	def compute_matches(self):
		'''greedy selection, computed seq sim here for simplicity, will do it with other losses later '''
		
		prediction_flat = self.seq_pred.view(-1) # batch*N
		labels_flat = self.batch_parent.labels.view(-1) # batch x N --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		matches = ((prediction_flat == labels_flat) & (valid_mask)).sum() # 1, 
		
		return matches 

	def compute_losses(self):
		matches = self.compute_matches()

		return [matches]

def mlm_inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, entropy_increment=0.1, temp=1e-6):

	entropy_threshold = 0.1 # entropy threshold
	entropies = torch.zeros_like(aas, dtype=torch.float32) + float("inf") # will return this later so user knows confidence
	fixed = (aas!=-1) | key_padding_mask # if not -1, then should be fixed as context
	predicted = fixed # to keep track of predicted pos as loop through
	aas.masked_fill_(~fixed, 20) # add the mask token to non fixed tokens

	while not predicted.all(): # already deals w/ masking

		# get wf
		wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

		# predict sequence
		aa_logits = self.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask)
		pred_aa = self.wf_extraction.sample(aa_logits, temp)

		# compute entropy
		probs = aa_logits.softmax(dim=2)
		entropy = torch.sum(-probs*torch.log(probs), dim=2)
		
		# update
		update_aa = (entropy < entropy_threshold) & (entropies == float("inf")) & (~predicted.all(dim=1, keepdim=True)) # dont update already finished samples
		entropies[update_aa] = entropy[update_aa]
		aas[update_aa] = pred_aa[update_aa]
		predicted |= update_aa

		# increment entropy threshold
		entropy_threshold += entropy_increment

	return aas

def inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, cycles=10, diffusion_iters=1, temp=1e-6):

	# prep
	batch, N = aas.shape
	t_max = self.wf_diffusion.noise_scheduler.t_max

	# fixed position have an AA label, positions to predict are -1, so they are set to random aa. doesnt matter too much, bc starts at white noise
	fixed_aas = aas!=-1
	aas = torch.where(fixed_aas, aas, torch.randint_like(aas, 0, self.num_aas))

	wf_no_aa = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=True)

	# multiple embedding + encoding + diffusion + decoding + extraction runs, each giving a slightly better guess and thus using less noise
	for t_fwd in range(t_max, 0, -t_max//cycles):

		# perform embedding
		wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

		# all of the following modules do not use coords alpha, since replaced geo attn with regular attn + spatial encoding, but leaving it until i am sure geo attn is dead

		# encode from wf space to latent space
		protein = self.wf_encoding.encode(wf, key_padding_mask=key_padding_mask)
		
		for diffusion_iter in range(diffusion_iters): # slight nudge, then denoise, then repeat, nudging latent towards manifold, rather than starting from full gaussian
			
			# add gaussian noise to latent space
			protein_noised, _ = self.wf_diffusion.noise(protein, t_fwd)

			# remove the noise
			protein = self.wf_diffusion.denoise(protein_noised, t_fwd, key_padding_mask=key_padding_mask)

		# decode from latent space to wf space
		wf_pred = self.wf_decoding(protein, wf_no_aa, key_padding_mask=key_padding_mask)

		# extract sequence from wf
		aa_pred = self.wf_extraction.extract(wf_pred, wf_no_aa, key_padding_mask=key_padding_mask)

		# keeping fixed positions as they were for next iteration, 
		aas = torch.where(fixed_aas, aas, aa_pred)

	return aas

class CrossFeatureNorm(nn.Module):
	'''
	normalizes each feature independantly across the sequence. it is independant of batches (not batch norm)
	this is helpful because each feature for a given token (Ca atom) is the output of that token for the global 
	superposed wavefunction at a particular wavelength. thus, each feature in a given token is only relevant
	RELATIVE to the CORRESPONDING features of all other tokens in the sequence. 
	This essentially normalizes each wavefunction's (psi_k) output to have mean of 0 and std of 1. 
	Note that this normalizes the real part and the imaginary part independantly 
	the resulting features are then scaled by 1/sqrt(d_model), so that the variance of the whole wf is 1
	'''
	def __init__(self, d_model):
		super(CrossFeatureNorm, self).__init__()

	def forward(self, x, mask=None):

		batch, N, d_model = x.shape

		mask = mask if mask is not None else torch.ones(batch, N, device=x.device, dtype=torch.bool) # Z x N
		valid = (~mask).sum(dim=1, keepdim=True).unsqueeze(2).clamp(min=1) # Z x 1 x 1
		mean = (x*(~mask).unsqueeze(2)).sum(dim=1, keepdim=True) / valid # Z x 1 x D
		x = x - mean # Z x N x D
		std = torch.sqrt(torch.where(x.sum(dim=1, keepdim=True)==0, 1, x.pow(2)).sum(dim=1, keepdim=True)/valid) # Z x 1 x D
		x = x/std # Z x N x D
		# x = x/(d_model**0.5) # Z x N x D

		return x

class PositionalEncoding(nn.Module):
	def __init__(self, d_model=512):
		super(PositionalEncoding, self).__init__()

		self.d_model = d_model
		self.register_buffer('wavenumbers', torch.pow(10000, -(torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)))

	def forward(self, pos):

		batch, N = pos.shape

		phase = pos[:, :, None] * self.wavenumbers[None, None, :]

		pe = torch.stack([torch.sin(phase), torch.cos(phase)], dim=3).view(batch, N, self.d_model) # N x d_model
		
		return pe

class WaveFunctionExtraction(nn.Module):
	
	def __init__(self, 	d_model=512, d_wf=128, num_aas=20, # model dimension
						d_hidden_pre=2048, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=4, heads=8, 
						use_bias=False, learn_spreads=True, min_rbf=0.001,
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10
				):

		super(WaveFunctionExtraction, self).__init__()

		
		identity = lambda x: x # identity functino to skip computation
		zero = lambda x: torch.zeros_like(x) # returns zeros tensor, for funcs that are included in skip connection

		self.dropout = nn.Dropout(dropout)

		self.proj_pre = nn.Linear(d_wf, d_model) if d_wf!=d_model else identity # skip linear if same dims
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout) if hidden_layers_pre!=-1 else zero
		self.norm_pre = nn.LayerNorm(d_model) if hidden_layers_pre!=-1 else identity
		self.mlp_post = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=dropout) if hidden_layers_post!=-1 else zero
		self.norm_post = nn.LayerNorm(d_model) if hidden_layers_post!=-1 else identity

		# self.convs = nn.ModuleList([ConvFormer((3,5,7), d_model=d_model, dropout=dropout) for _ in range(encoder_layers+1)])

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_other=d_model, heads=heads, 
													min_rbf=min_rbf, bias=use_bias, learn_spreads=learn_spreads,
													d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, personal=True,
													dropout=dropout
												) 
										for _ in range(encoder_layers)
									])

		# map to aa prob logits
		self.out_proj = nn.Linear(d_model, num_aas)
		init_xavier(self.out_proj)

	def forward(self, wf, coords, coords_beta, chain_idxs, key_padding_mask=None):

		# linear projection
		wf = self.proj_pre(wf)


		# non linear tranformation for more intricate features
		wf = self.norm_pre(wf + self.mlp_pre(wf))

		# geometric/vanilla attn encoders
		for encoder in self.encoders:
			# wf = conv(wf, chain_idxs)
			# wf = encoder(wf, wf, wf, coords, mask=key_padding_mask)
			wf = encoder(wf, wf, wf, coords, coords_beta, mask=key_padding_mask)

		# wf = self.convs[-1](wf, chain_idxs)

		# post process
		wf = self.norm_post(wf + self.dropout(self.mlp_post(wf)))

		# map to probability logits
		aa_logits = self.out_proj(wf)

		return aa_logits

	def sample(self, aa_probs, temp=1e-6):

		batch, N, num_aas = aa_probs.shape

		# sample from the distributions
		aa_labels = torch.multinomial(aa_probs.view(batch*N, num_aas), num_samples=1, replacement=False).view(batch, N)

		return aa_labels

	def get_probs(self, wf, coords_alpha, key_padding_mask=None, temp=1e-6):

		# perform extraction
		aa_logits = self.forward(wf, coords_alpha, key_padding_mask)

		# softmax on temp scaled logits to get AA probs
		aa_probs = torch.softmax(aa_logits/temp, dim=2)

		return aa_probs

	def extract(self, wf, coords_alpha, key_padding_mask=None, temp=1e-6):

		# get temp scaled probs
		aa_probs = self.get_probs(wf, coords_alpha, key_padding_mask, temp) 

		# sample from distribution
		aas = self.sample(aa_logits, temp)

		return aas


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


class ConvFormer(nn.Module):
	def __init__(self, kernel_sizes=(3,5,7), d_model=256, dropout=0.0):
		super(ConvFormer, self).__init__()
		self.conv_weights = nn.ParameterList([nn.Parameter(torch.randn((d_model, d_model, kernel_size))) for kernel_size in kernel_sizes])
		self.conv_biases = nn.ParameterList([nn.Parameter(torch.randn((d_model))) for kernel_size in kernel_sizes])
		self.mlp = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, dropout=dropout) 
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, chain_idxs):

		B, L, _ = x.shape

		chain_tensor  = torch.zeros(x.size(0), x.size(1), dtype=torch.int32, device=x.device)
		for sample_idx, sample in enumerate(chain_idxs):
			for sample_chain_idx, (chain_start, chain_stop) in enumerate(sample, start=1): # start at one so that padded positions have idx=0, ie not mixed with nonpadded
				chain_tensor[sample_idx, chain_start:chain_stop] = sample_chain_idx

		x1 = x.permute(0,2,1)
		for conv_weight, conv_bias in zip(self.conv_weights, self.conv_biases):

			cin, cout, K = conv_weight.shape

			# Compute output length O = ceil(L / stride)
			O = L
			# Compute total padding
			P_total = K-1
			if P_total < 0:
				P_total = 0
			pad_left = P_total // 2
			pad_right = P_total - pad_left

			# Pad x along length
			# F.pad works for 3D input: pad=(pad_left, pad_right) on last dim
			x_padded = torch.nn.functional.pad(x1, (pad_left, pad_right))

			# Pad idxs along length with pad_value
			# Build a new tensor of shape (B, L + pad_left + pad_right)
			L_padded = L + pad_left + pad_right
			device = x.device
			dtype_idxs = chain_tensor.dtype
			idxs_padded = torch.full((B, L_padded), 0, dtype=dtype_idxs, device=device)
			# copy original idxs into center
			idxs_padded[:, pad_left: pad_left + L] = chain_tensor

			# Extract windows via unfold
			# x_padded.unfold -> shape (B, cin, O, K)
			Xw = x_padded.unfold(dimension=2, size=K, step=1)
			# idxs_padded.unfold -> shape (B, O, K)
			idxs_windows = idxs_padded.unfold(dimension=1, size=K, step=1)

			# Build mask: keep positions where idx == center idx
			center = K // 2
			center_idxs = idxs_windows[:, :, center]        # shape (B, O)
			mask = (idxs_windows == center_idxs.unsqueeze(2))  # shape (B, O, K), bool

			# Apply mask: broadcast over cin
			# Xw: (B, cin, O, K); mask.unsqueeze(1): (B,1,O,K) -> broadcast to (B,cin,O,K)
			Xw_masked = Xw * mask.unsqueeze(1)

			# Contract with weight via einsum
			# weight shape: (cout, cin, K)
			# out[b, o, i] = sum_c,k Xw_masked[b, c, i, k] * weight[o, c, k]
			out = torch.einsum('b c i k, o c k -> b o i', Xw_masked, conv_weight)
			# Add bias if provided
			out = out + conv_bias.view(1, cout, 1)

			x1 = out

		x = self.norm1(x + self.dropout(x1.permute(0,2,1)))
		x = self.norm2(x + self.dropout(self.mlp(x)))

		return x


class FiLM(nn.Module):
	def __init__(self, d_model=512, d_hidden=1024, hidden_layers=0, dropout=0.1):
		super(FiLM, self).__init__()

		# single mlp that outputs gamma and beta, manually split in fwd
		self.gamma_beta = MLP(d_in=d_model, d_out=2*d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout)

	def forward(self, e_t, x): # assumes e_t is Z x 1 x d_model
		gamma_beta = self.gamma_beta(e_t)
		gamma, beta = torch.split(gamma_beta, dim=-1, split_size_or_sections=gamma_beta.shape[-1] // 2)
		return gamma*x + beta

