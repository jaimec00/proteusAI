# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		wf_diffusion.py
description:	performs diffusion on wavefunction representation of protein
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.base_modules.base_modules import init_xavier
from utils.model_utils.base_modules.encoder import Encoder

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionDiffusion(nn.Module):
	
	def __init__(self, 	d_model=512,
						alpha_bar_min=0.0, noise_schedule_type="cosine", t_max=1000,
						d_in_timestep=512, d_hidden_timestep=2048, hidden_layers_timestep=0,
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=8, heads=8,
						min_spread=3.0, min_rbf=0.001, max_rbf=0.85, 
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10
				):

		super(WaveFunctionDiffusion, self).__init__()

		# compute wavenumbers for sinusoidal embeddings of timesteps
		self.register_buffer("wavenumbers", 10000**(-torch.arange(0, d_in_timestep, 2) / d_model))
		self.noise_scheduler = NoiseScheduler(alpha_bar_min=alpha_bar_min, noise_schedule_type=noise_schedule_type, t_max=t_max)
		self.dropout = nn.Dropout(dropout)

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread,
													min_rbf=min_rbf, max_rbf=max_rbf, 
													dropout=dropout,
													use_adaLN=True, d_in_t=d_in_timestep, d_hidden_t=d_hidden_timestep, hidden_layers_t=hidden_layers_timestep
												) 
										for _ in range(encoder_layers)
									])

		self.out_norm = nn.LayerNorm(d_model)
		self.noise_proj = nn.Linear(d_model)
		init_xavier(self.noise_proj)

	# wf is the noised wf, context is unnoised, but with no aa info, used as kv in cross attention, defaults to self-attention if context is none
	def forward(self, wf, coords_alpha, t, key_padding_mask=None, context=None):

		t_features = self.featurize_t(t)

		# geometric attention encoders
		for encoder in self.encoders:
			wf = self_encoder(wf, coords_alpha, key_padding_mask=key_padding_mask, t=t_features)

		# post processing mlp
		noise = self.noise_proj(self.out_norm(wf))

		return noise

	def featurize_t(self, t):

		# once in the latent space, add token embedding info
		# featurize the timestep (shape: batch, -> batch x 1 x d_model) with  sinusoidal embedding
		phase = self.wavenumbers.unsqueeze(0).unsqueeze(1)*t.unsqueeze(1).unsqueeze(2)
		sine = torch.sin(phase) # Z x 1 x K
		cosine = torch.cos(phase) # Z x 1 x K
		t_features = torch.stack([sine, cosine], dim=3).view(t.size(0), 1, self.wavenumbers.size(0)*2) # Z x 1 x d_model

		return t_features

	def get_random_timesteps(self, batch_size, device):
		return torch.randint(1, self.noise_scheduler.t_max+1, (batch_size,), device=device)

	def noise(self, wf, t):

		if isinstance(t, int):
			t = torch.full((wf.size(0), 1, 1), t, device=wf.device)
		elif isinstance(t, torch.Tensor):
			t = t.unsqueeze(1).unsqueeze(2)
		alpha_bar_t, _ = self.noise_scheduler(t) 
		noise = torch.randn_like(wf)
		wf = (alpha_bar_t**0.5)*wf + ((1-alpha_bar_t)**0.5)*noise

		return wf, noise

	def denoise(self, wf, coords_alpha, t_start, key_padding_mask=None, context=None): # meant to operate on same t for all samples in batch during inference

		# convert to tensor
		t_bwd = torch.full((coords_alpha.size(0), 1, 1), t_start, device=coords_alpha.device)
			
		# perform diffusion
		while (t_bwd>=1).any():

			# compute alpha_bar for t and t-1
			alpha_bar_t, alpha_bar_tminus1 = self.noise_scheduler(t_bwd) 
			
			# predict the noise
			noise_pred = self.forward(wf, coords_alpha, t_bwd.squeeze(1,2), key_padding_mask=key_padding_mask, context=context)

			# update wf, use ode flow to deterministically move the wf towards high prob denisty manifold. non-markovian denoising
			pred_wf_0 = (wf - ((1-alpha_bar_t)**0.5)*noise_pred)/(alpha_bar_t**0.5)
			pred_wf_grad_t = ((1 - alpha_bar_tminus1)**0.5) * noise_pred
			wf = (alpha_bar_tminus1**0.5)*pred_wf_0 + pred_wf_grad_t

			# update t
			t_bwd -= 1

		return wf


class NoiseScheduler(nn.Module):
	def __init__(self, alpha_bar_min=0.0, noise_schedule_type="cosine", t_max=100):
		super(NoiseScheduler, self).__init__()

		self.alpha_bar_min = alpha_bar_min # 0.0 is full noise , no signal
		self.noise_scheduler = self.get_scheduler(noise_schedule_type)
		self.t_max = t_max

	def forward(self, t: torch.Tensor): # Z x 1 x 1

		return self.noise_scheduler(t) # Z x 1 x 1

	def cosine_scheduler(self, t: torch.Tensor, s=0.008): # s is approx pixel bin width, no direct equivilant for my data, so just start with this and tune
		f = lambda t_in: self.alpha_bar_min + (1 - self.alpha_bar_min)*(torch.cos((torch.pi/2)*(t_in/self.t_max + s)/(1 + s))**2)
		f_t = f(t)
		f_tminus1 = f(t-1) 
		f_0 = f(torch.zeros_like(t))
		alpha_bar_t = f_t / f_0
		alpha_bar_tminus1 = f_tminus1 / f_0

		return alpha_bar_t, alpha_bar_tminus1

	def get_scheduler(self, schedule_type):

		schedules = {	
						"cosine": self.cosine_scheduler,
					}
		try:
			return schedules[schedule_type]
		except KeyValueError as e:
			e(f"invalid noise scheduler chosen. valid options are {schedules.keys()}")
