'''
variational auto encoder
'''

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import MLP

class WaveFunctionEncoding(nn.Module):
	def __init__(self,  d_model=512, d_latent=512,
						d_hidden_pre=2048, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=1,
						encoder_layers=8, heads=8,
						min_spread=3.0, min_rbf=0.001, max_rbf=0.85, 
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10
				):
		super(WaveFunctionEncoding, self).__init__()

		self.dropout = nn.Dropout(dropout)
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre)
		self.norm_pre = nn.LayerNorm(d_model)
		self.mlp_post = MLP(d_in=d_model, d_out=2*d_latent, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post)
		
		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread, min_rbf=min_rbf, max_rbf=max_rbf, 
													dropout=dropout
												) 
										for _ in range(encoder_layers)
							])

	def forward(self, wf, coords_alpha, key_padding_mask=None): # forward generates the mean and stds, so can use them for loss, 
		
		wf = wf + self.dropout(self.mlp_pre(wf))
		wf = self.norm_pre(wf)

		# encoders
		for encoder in self.encoders:
			wf = encoder(wf, coords_alpha, key_padding_mask=key_padding_mask)

		wf = self.mlp_post(wf)

		latent_mean, latent_log_var = torch.chunk(wf, chunks=2, dim=-1)

		return latent_mean, latent_log_var

	def sample(self, latent_mean, latent_log_var):
		return latent_mean + torch.exp(latent_log_var*0.5)*torch.randn_like(latent_log_var)

	def encode(self, wf, coords_alpha, key_padding_mask=None):
		latent_mean, latent_log_var = self.forward(wf, coords_alpha, key_padding_mask=key_padding_mask)
		wf_latent = self.sample(latent_mean, latent_log_var)
		return wf_latent