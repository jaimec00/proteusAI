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
from utils.model_utils.base_modules.base_modules import FiLM, StaticLayerNorm
from utils.model_utils.base_modules.encoder import Encoder

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionDiffusion(nn.Module):
	
	def __init__(self, 	d_model=512,
						alpha_bar_min=0.0, beta_schedule_type="cosine", t_max=1000,
						d_hidden_timestep=2048, hidden_layers_timestep=0,
						mlp_pre=False, d_hidden_pre=2048, hidden_layers_pre=0, norm_pre=False,
						mlp_post=False, d_hidden_post=2048, hidden_layers_post=0, norm_post=False,
						encoder_layers=8, decoder_layers=8, heads=8, learnable_spreads=True, # share params between enc and dec for now
						min_spread=3.0, max_spread=15.0, base_spreads=1.0, num_spread=32,
						min_rbf=0.001, max_rbf=0.85, beta=2.0,
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10, attn_dropout=0.00,
				):

		super(WaveFunctionDiffusion, self).__init__()

		# compute wavenumbers for sinusoidal embeddings of timesteps
		self.register_buffer("wavenumbers", 10000**(-torch.arange(0, d_model, 2) / d_model))

		self.beta_scheduler = BetaScheduler(alpha_bar_min=alpha_bar_min, beta_schedule_type=beta_schedule_type, t_max=t_max)

		self.dropout = nn.Dropout(dropout)

		self.film_timestep_pre = FiLM(d_model=d_model, d_hidden=d_hidden_timestep, hidden_layers=hidden_layers_timestep, dropout=0.0) # no dropout in FiLM, since only depends on t
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre) if mlp_pre else None
		self.norm_pre = nn.LayerNorm(d_model) if norm_pre else None

		# for the kv latent
		self.film_timestep_pre_base = FiLM(d_model=d_model, d_hidden=d_hidden_timestep, hidden_layers=hidden_layers_timestep, dropout=0.0)
		self.mlp_pre_base = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre) if mlp_pre else None
		self.norm_pre_base = nn.LayerNorm(d_model) if norm_pre else None

		self.mlp_post = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post) if mlp_post else None
		self.norm_post = nn.LayerNorm(d_model) if norm_post else None

		self.film_timesteps_enc = nn.ModuleList([FiLM(d_model=d_model, d_hidden=d_hidden_timestep, hidden_layers=hidden_layers_timestep, dropout=0.0) for _ in range(encoder_layers)])
		self.film_timesteps_dec = nn.ModuleList([FiLM(d_model=d_model, d_hidden=d_hidden_timestep, hidden_layers=hidden_layers_timestep, dropout=0.0) for _ in range(decoder_layers)])

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread, max_spread=max_spread, base_spread=base_spreads, 
													num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, 
													dropout=dropout, attn_dropout=attn_dropout
												) 
										for _ in range(encoder_layers)
									])

		self.decoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread, max_spread=max_spread, base_spread=base_spreads, 
													num_spread=num_spread, min_rbf=min_rbf, max_rbf=max_rbf, beta=beta, learnable_spreads=learnable_spreads, 
													dropout=dropout, attn_dropout=attn_dropout
												) 
										for _ in range(decoder_layers)
									])

	# wf is the noised wf, context is unnoised, but with no aa info, used as kv in cross attention, defaults to self-attention if context is none
	def forward(self, wf, coords_alpha, t, key_padding_mask=None, context=None):

		batch, N, d_model = wf.shape

		# once in the latent space, add token embedding info
		# featurize the timestep (shape: batch, -> batch x 1 x d_model) with  sinusoidal embedding
		phase = self.wavenumbers.unsqueeze(0).unsqueeze(1)*t.unsqueeze(1).unsqueeze(2)
		sine = torch.sin(phase) # Z x 1 x K
		cosine = torch.cos(phase) # Z x 1 x K
		t_features = torch.stack([sine, cosine], dim=3).view(batch, 1, d_model) # Z x 1 x d_model

		# norm wf and scale based on timestep
		wf = self.film_timestep_pre(t_features, wf)

		# norm context and scale based on timestep
		# context = self.film_timestep_pre_base(t_features, context)

		# pre processing mlp, not using this
		if self.mlp_pre is not None:
			wf = wf + self.dropout(self.mlp_pre(wf))
			# context = context + self.dropout(self.mlp_pre_base(context))
		if self.norm_pre is not None:
			wf = self.norm_pre(wf)
			# context = self.norm_pre_base(context)

		# geometric attention decoders, not actually decoders, just uses cross attention, do this first to give the model an idea of what the wf should look like
		# for decoder, film_timestep in zip(self.decoders, self.film_timesteps_dec):
		# 	wf = decoder(wf, coords_alpha, key_padding_mask=key_padding_mask, context=context, t=t_features, film=film_timestep)
		# skip cross attention for now, see how it does

		# geometric attention encoders, just self attention
		for encoder, film_timestep in zip(self.encoders, self.film_timesteps_enc):
			wf = encoder(wf, coords_alpha, key_padding_mask=key_padding_mask, t=t_features, film=film_timestep)

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

	def denoise(self, wf, coords_alpha, t_start, key_padding_mask=None, context=None): # meant to operate on same t for all samples in batch during inference

		# convert to tensor
		t_bwd = torch.full((coords_alpha.size(0), 1, 1), t_start, device=coords_alpha.device)
			
		# perform diffusion
		while (t_bwd>=1).any():

			# compute alpha_bar for t and t-1
			alpha_bar_t, alpha_bar_tminus1 = self.beta_scheduler(t_bwd) 
			
			# predict the noise
			noise_pred = self.forward(wf, coords_alpha, t_bwd.squeeze(1,2), key_padding_mask=key_padding_mask, context=context)

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

	def cosine_scheduler(self, t: torch.Tensor, s=0.000): # s is approx pixel bin width, no direct equivilant for my data, so just start with this and tune
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
