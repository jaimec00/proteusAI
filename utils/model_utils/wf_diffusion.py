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
						alpha_bar_min=0.0, beta_schedule_type="cosine", t_max=1000,
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
		self.register_buffer("wavenumbers", 2*torch.pi / (min_wl-1 + torch.logspace(0, -1, d_model//2, max_wl-min_wl+1)))

		self.beta_scheduler = BetaScheduler(alpha_bar_min=alpha_bar_min, beta_schedule_type=beta_schedule_type, t_max=t_max)

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
		t_features = torch.stack([sine, cosine], dim=3).view(batch, 1, d_model) # Z x 1 x d_model

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
		return torch.randint(1, self.beta_scheduler.t_max+1, (batch_size,), device=device)

	def noise(self, wf, t):

		if isinstance(t, int):
			t = torch.full((wf.size(0), 1, 1), t, device=wf.device)
		elif isinstance(t, torch.Tensor):
			t = t.unsqueeze(1).unsqueeze(2)
		alpha_bar_t, _ = self.beta_scheduler(t) 
		noise = torch.randn_like(wf)
		wf = (alpha_bar_t**0.5)*wf + ((1-alpha_bar_t)**0.5)*noise

		return wf, noise

	def denoise(self, wf, coords_alpha, t_start, key_padding_mask=None): # meant to operate on same t for all samples in batch during inference

		# convert to tensor
		t_bwd = torch.full((coords_alpha.size(0), 1, 1), t_start, device=coords_alpha.device)
			
		# perform diffusion
		while (t_bwd>=1).any():

			# compute alpha_bar for t and t-1
			alpha_bar_t, alpha_bar_tminus1 = self.beta_scheduler(t_bwd) 
			
			# predict the noise
			noise_pred = self.forward(wf, coords_alpha, t_bwd.squeeze(1,2), key_padding_mask)

			# update wf, use ode flow to deterministically move the wf towards high prob denisty manifold. non-markovian denoising
			pred_wf_0 = (wf - ((1-alpha_bar_t)**0.5)*noise_pred)/(alpha_bar_t**0.5)
			pred_wf_grad_t = ((1 - alpha_bar_tminus1)**0.5) * noise_pred
			wf = (alpha_bar_tminus1**0.5)*pred_wf_0 + pred_wf_grad_t

			# update t
			t_bwd -= 1

		return wf


class BetaScheduler(nn.Module):
	def __init__(self, alpha_bar_min=0.0, beta_schedule_type="cosine", t_max=100):
		super(BetaScheduler, self).__init__()

		self.alpha_bar_min = alpha_bar_min # 0.0 is full noise , no signal
		self.beta_scheduler = self.get_scheduler(beta_schedule_type)
		self.t_max = t_max

	def forward(self, t: torch.Tensor): # Z x 1 x 1

		return self.beta_scheduler(t) # Z x 1 x 1

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
						"cosine": self.cosine_scheduler
					}
		try:
			return schedules[schedule_type]
		except KeyValueError as e:
			e(f"invalid noise scheduler chosen. valid options are {schedules.keys()}")
