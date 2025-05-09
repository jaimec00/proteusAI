'''
variational auto encoder
maps rich residue level info to compressed protein latent space of fixed size
'''

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import MLP

class WaveFunctionEncoding(nn.Module):
	def __init__(self,  d_model=256, d_latent=32, d_proj=512,
						d_hidden_pre=1024, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=1,
						encoder_layers=4, heads=8,
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