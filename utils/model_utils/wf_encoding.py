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
						encoder_layers=8, heads=8, use_bias=False,
						min_spread=3.0, min_rbf=0.001, max_rbf=0.85, 
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10
				):
		super(WaveFunctionEncoding, self).__init__()

		self.dropout = nn.Dropout(dropout)
		self.space_enc = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre) # spatial encoding, uses the same as preprocess for now
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre)
		self.norm_pre = nn.LayerNorm(d_model)
		self.mlp_post = MLP(d_in=d_model, d_out=2*d_latent, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post)
		
		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, use_bias=use_bias, min_spread=min_spread, min_rbf=min_rbf, max_rbf=max_rbf, 
													dropout=dropout
												) 
										for _ in range(encoder_layers)
							])

	def forward(self, wf, coords_alpha, key_padding_mask=None, a=2.0, wf_no_aa=None): # forward generates the mean and log vars, so can use them for loss, use self.encode to directly sample from latent 
		
		space_enc = wf_no_aa + self.dropout(self.space_enc(wf_no_aa)) # spatial encoding on aa ambiguous wf
		wf = wf + self.dropout(self.mlp_pre(wf)) 
		wf = self.norm_pre(wf + space_enc)

		# encoders
		for encoder in self.encoders:
			wf = encoder(wf, coords_alpha, key_padding_mask=key_padding_mask)

		wf = self.mlp_post(wf)

		latent_mean, latent_log_var = torch.chunk(wf, chunks=2, dim=-1)

		# a decides the value at which the log var saturates, 
		# the problem (i think) is that reconstruction error makes log var very negative, and exacerbates gradients
		# this does soft clipping on log var from -2 to +2. ensures vars range from exp(-2) to exp(2) = 0.1353 to 7.3891
		latent_log_var = a*torch.tanh(latent_log_var/a)

		return latent_mean, latent_log_var

	def sample(self, latent_mean, latent_log_var):
		return latent_mean + torch.exp(latent_log_var*0.5)*torch.randn_like(latent_log_var)

	def encode(self, wf, coords_alpha, key_padding_mask=None, wf_no_aa=None):
		latent_mean, latent_log_var = self.forward(wf, coords_alpha, key_padding_mask=key_padding_mask, wf_no_aa=wf_no_aa)
		wf_latent = self.sample(latent_mean, latent_log_var)
		return wf_latent