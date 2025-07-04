
import torch
import torch.nn as nn

from utils.model_utils.base_modules.base_modules import MLP, adaLN, MPNN, init_xavier

class SequenceDenoiser(nn.Module):	
	def __init__(self, 	d_model=128, t_max=1000,
						layers=3, dropout=0.00
				):

		super(SequenceDenoiser, self).__init__()

		# compute wavenumbers for sinusoidal embeddings of timesteps
		self.register_buffer("wavenumbers", 10000**(-torch.arange(0, d_model, 2) / d_model))
		self.noise_scheduler = NoiseScheduler(t_max=t_max)

		# integrate sequence into the node embeddings
		self.sequence_messenger = MLP(d_in=2*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu")
		self.sequence_messenger_norm = adaLN(d_model)

		self.denoisers = nn.ModuleList([MPNN(d_model, dropout=dropout, use_adaLN=True) for _ in range(layers)])

		self.noise_proj = nn.Linear(d_model, d_model, bias=False)

		self.dropout = nn.Dropout(dropout)

		init_xavier(self.noise_proj)

	def forward(self, V, E, K, Z, t, edge_mask=None):

		# featurize the timestep w/ frequency embedding
		t_features = self.featurize_t(t)

		# combine structure node embedding and sequence latent embeddings, conditioned on timestep
		VZ = torch.cat([V, Z], dim=2)
		V = self.sequence_messenger_norm(V + self.dropout(self.sequence_messenger(VZ)), t_features)

		# perform message passing conditioned on timestep to denoise the latents
		for denoiser in self.denoisers:
			V, E = denoiser(V, E, K, edge_mask, t=t_features)
			
		# linear projection for noise pred, no bias
		noise = self.noise_proj(V)

		# return noise
		return noise

	def featurize_t(self, t):

		# once in the latent space, add token embedding info
		# featurize the timestep (shape: batch, -> batch x 1 x d_latent) with  sinusoidal embedding
		phase = self.wavenumbers.unsqueeze(0).unsqueeze(1)*t.unsqueeze(1).unsqueeze(2)
		sine = torch.sin(phase) # Z x 1 x K
		cosine = torch.cos(phase) # Z x 1 x K
		t_features = torch.stack([sine, cosine], dim=3).view(t.size(0), 1, self.wavenumbers.size(0)*2) # Z x 1 x d_latent

		return t_features

	def get_random_timesteps(self, batch_size, device):
		return torch.randint(1, self.noise_scheduler.t_max+1, (batch_size,), device=device)

	def noise(self, Z, t):

		if isinstance(t, int):
			t = torch.full((Z.size(0), 1, 1), t, device=Z.device)
		elif isinstance(t, torch.Tensor):
			t = t.unsqueeze(1).unsqueeze(2)
		alpha_bar_t, _ = self.noise_scheduler(t) 
		noise = torch.randn_like(Z)
		Z_noised = (alpha_bar_t**0.5)*Z + ((1-alpha_bar_t)**0.5)*noise

		return Z_noised, noise

	def denoise(self, V, E, K, Z, t_start, edge_mask=None): # meant to operate on same t for all samples in batch during inference

		# convert to tensor
		t_bwd = torch.full((Z.size(0), 1, 1), t_start, device=Z.device)
			
		# perform diffusion
		while (t_bwd>=1).any():

			# compute alpha_bar for t and t-1
			alpha_bar_t, alpha_bar_tminus1 = self.noise_scheduler(t_bwd) 
			
			# predict the noise
			noise_pred = self.forward(V, E, K, Z, t_bwd.squeeze(1,2), edge_mask)

			# update Z, use ode flow to deterministically move the Z towards high prob denisty manifold. non-markovian denoising
			pred_Z_0 = (Z - ((1-alpha_bar_t)**0.5)*noise_pred)/(alpha_bar_t**0.5)

			pred_Z_grad_t = ((1 - alpha_bar_tminus1)**0.5) * noise_pred

			Z = (alpha_bar_tminus1**0.5)*pred_Z_0 + pred_Z_grad_t

			# update t
			t_bwd -= 1

		return Z


class NoiseScheduler(nn.Module):
	def __init__(self, t_max=1000):
		super(NoiseScheduler, self).__init__()

		self.t_max = t_max

	def forward(self, t, s=0.008): # Z x 1 x 1
		
		f = lambda t_in: torch.cos((torch.pi/2)*(t_in/self.t_max + s)/(1 + s))**2
		f_t = f(t)
		f_tminus1 = f(t-1) 
		f_0 = f(torch.zeros_like(t))
		alpha_bar_t = f_t / f_0
		alpha_bar_tminus1 = f_tminus1 / f_0

		return alpha_bar_t, alpha_bar_tminus1