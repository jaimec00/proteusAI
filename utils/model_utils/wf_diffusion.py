# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		wf_diffusion.py
description:	performs diffusion on wavefunction representation of protein
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.base_modules.base_modules import MLP
from utils.model_utils.base_modules.encoder import Encoder

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionDiffusion(nn.Module):
	
	def __init__(self, 	d_model=512,
						beta_min=1e-4, beta_max=0.02, beta_schedule_type="linear", t_max=100,
						min_wl=0.001, max_wl=100, # for sinusoidal timesteps
						mlp_timestep=False, d_hidden_timestep=2048, hidden_layers_timestep=0, norm_timestep=False,
						mlp_pre=False, d_hidden_pre=2048, hidden_layers_pre=0, norm_pre=False,
						mlp_post=False, d_hidden_post=2048, hidden_layers_post=0, norm_post=False,
						encoder_layers=8, heads=8, learnable_spreads=True,
						min_spread=3.0, max_spread=15.0, base_spreads=1.0, num_spread=32,
						min_rbf=0.001, max_rbf=0.85, beta=2.0,
						d_hidden_attn=2048, hidden_layers_attn=0,
						min_freq=0.01, max_freq=1000, # for timestep embedding
						dropout=0.10, attn_dropout=0.00,
				):

		super(WaveFunctionDiffusion, self).__init__()

		# compute wavenumbers for sinusoidal embeddings of timesteps
		self.wavenumbers = 2*torch.pi / (min_wl-1 + torch.logspace(0, -1, d_model//2, max_wl-min_wl+1))

		self.beta_scheduler = BetaScheduler(beta_min=beta_min, beta_max=beta_max, beta_schedule_type=beta_schedule_type, t_max=t_max)

		self.dropout = nn.Dropout(dropout)
		self.mlp_timestep = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_timestep, hidden_layers=hidden_layers_timestep) if mlp_timestep else None
		self.norm_timestep = nn.LayerNorm(d_model) if norm_timestep else None 
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre) if mlp_pre else None
		self.norm_pre = nn.LayerNorm(d_model) if norm_pre else None
		self.mlp_post = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post) if mlp_post else None
		self.norm_post = nn.LayerNorm(d_model) if norm_post else None

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread, max_spread=max_spread, base_spread=base_spreads, 
													num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, 
													dropout=dropout, attn_dropout=attn_dropout
												) 
										for _ in range(encoder_layers)
									])



	def forward(self, wf, coords_alpha, t, key_padding_mask=None):

		batch, N, d_model = wf.shape

		# pre processing mlp
		if self.mlp_pre is not None:
			wf = wf + self.dropout(self.mlp_pre(wf))
		if self.norm_pre is not None:
			wf = self.norm_pre(wf)

		# once in the latent space, add token embedding info
		# featurize the timestep (shape: batch, -> batch x 1 x d_model) with  sinusoidal embedding
		phase = self.wavenumbers.unsqueeze(0).unsqueeze(1)*t.unsqueeze(1).unsqueeze(2)
		sine = torch.sin(phase) # Z x 1 x K
		cosine = torch.cos(phase) # Z x 1 x K
		t_features = torch.stack(sine, cosine, dim=3).view(batch, 1, d_model) # Z x 1 x d_model

		# mlp on just timestep embeddings, can't imagine this will help much, but just leave the option
		if self.mlp_timestep is not None:
			t_features = t_features + self.dropout(self.mlp_timestep(t_features))
		if self.norm_timestep is not None:
			t_features = self.norm_timestep(t_features)

		# simple addition for first stage of testing
		wf = wf + t_features

        # geometric attention encoders
		for encoder in self.encoders:
			wf = encoder(wf, coords_alpha, key_padding_mask)

		# post processing mlp
		if self.mlp_post is not None:
			wf = wf + self.dropout(self.mlp_post(wf))
		if self.norm_post is not None:
			wf = self.norm_post(wf)

		return wf

	def get_random_timesteps(self, batch_size, device):
		return torch.randint(0, self.beta_scheduler.t_max, batch_size, device=device)

	def noise(self, wf, t):

		t = t.unsqueeze(1).unsqueeze(2)
		_, _, alpha_bar_t = self.beta_scheduler(t)
		noise = torch.randn_like(wf)
		wf = (alpha_bar_t**0.5)*wf + ((1-alpha_bar_t)**0.5)*noise

		return wf, noise

	def denoise(self, wf, coords_alpha, t_start, key_padding_mask=None):

		# perform diffusion
		for t_bwd in range(t_start, 0, -1):

			# compute beta, alpha, and alpha_bar for t
			t_bwd = t_bwd.unsqueeze(1).unsqueeze(2)
			beta_t, alpha_t, alpha_bar_t = self.beta_scheduler(t_bwd) 

			# compute sigma_t and z to maintain stochastic nature in reverse process
			sigma_t = beta_t**0.5
			z = torch.randn_like(wf) * (t_bwd!=1) # set to 0 on final step

			# predict the noise
			noise_pred = self.forward(wf, coords_alpha, t_bwd.squeeze(1,2), key_padding_mask)

			# update wf
			wf = (1/(alpha_t**0.5)) * (wf - (beta_t / ((1-alpha_bar_t)**0.5))*noise_pred) + sigma_t*z

		return wf


class BetaScheduler(nn.Module):
	def __init__(self, beta_min=1e-4, beta_max=0.02, beta_schedule_type="linear", t_max=100):
		super(BetaScheduler, self).__init__()

		self.beta_min = beta_min
		self.beta_max = beta_max
		self.beta_schedule = self.get_scheduler(beta_schedule_type)
		self.t_max = t_max

	def forward(self, t: torch.Tensor):

		beta_t = self.beta_scheduler(t)
		alpha_t = 1 - beta_t
		alpha_bar_t = 1
		for s in range(1, t+1):
			alpha_bar_t *= 1 - self.beta_scheduler(s)

		return beta_t, alpha_t, alpha_bar_t

	def linear_scheduler(self, t: torch.Tensor):
		beta = self.beta_min + (self.beta_max - self.beta_min)*(t-1)/(self.t_max-1)
		return beta

	def get_scheduler(self, schedule_type):
		if schedule_type=="linear":
			return self.linear_scheduler
		else:
			raise NotImplementedError
