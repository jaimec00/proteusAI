'''
variational auto encoder
maps rich residue level info to compressed protein latent space of fixed size
'''

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import MLP

class WaveFunctionEncoding(nn.Module):
	def __init__(self,  d_model=512, d_latent=1024, N_latent=64,
						d_hidden_pre=1024, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=1,
						encoder_self_layers=4, self_heads=8,
						self_d_hidden_attn=1024, self_hidden_layers_attn=0,
						encoder_cross_layers=4, cross_heads=8,
						cross_d_hidden_attn=1024, cross_hidden_layers_attn=0,
						dropout=0.10
				):
		super(WaveFunctionEncoding, self).__init__()

		self.dropout = nn.Dropout(dropout)

		# pre preocess
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout)
		self.norm_pre = nn.LayerNorm(d_model)

		# self attention on wf
		self.wf_encoders = nn.ModuleList([ 	Encoder(d_model=d_model, d_other=d_model, heads=self_heads, 
													d_hidden=self_d_hidden_attn, hidden_layers=self_hidden_layers_attn, 
													dropout=dropout)
											for _ in range(encoder_self_layers)
										])

		# this is the starting point of the "global protein" representation. serves as Q in cross attn, w/ wf as KV to aggregate context
		self.init_protein = nn.Parameter(torch.randn(N_latent, d_latent))
		# cross attn
		self.protein_encoders = nn.ModuleList([ Encoder(	d_model=d_latent, d_other=d_model, heads=cross_heads,
															d_hidden=cross_d_hidden_attn, hidden_layers=cross_hidden_layers_attn, 
															dropout=dropout)
												for _ in range(encoder_cross_layers)
											])

		# post process to get mean and log vars
		self.mlp_post = MLP(d_in=d_latent, d_out=2*d_latent, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=0.0) # no dropout on this mlp
		
	def forward(self, wf, key_padding_mask=None, a=2): # forward generates the mean and log vars, so can use them for loss, use self.encode to directly sample from latent 
		
		# preprocess
		wf = self.norm_pre(wf + self.dropout(self.mlp_pre(wf)))

		# wf encoders
		# for encoder in self.wf_encoders:
		# 	wf = encoder(wf, wf, wf, mask=key_padding_mask)

		# protein encoders
		protein = self.init_protein.unsqueeze(0).expand(wf.size(0), -1, -1) # first iter uses the learned protein repre, after that use the updated one
		for encoder in self.protein_encoders:
			protein = encoder(protein, wf, wf, mask=key_padding_mask) # only need mask for wf, not protein

		# get means and logvars
		latent_stats = self.mlp_post(protein)
		latent_mean, latent_log_var = torch.chunk(latent_stats, chunks=2, dim=2)

		# map the log var to [-a,a]
		latent_log_var = a*torch.tanh(latent_log_var/a)

		return latent_mean, latent_log_var

	def sample(self, latent_mean, latent_log_var):
		return latent_mean + torch.exp(latent_log_var*0.5)*torch.randn_like(latent_log_var)

	def encode(self, wf, key_padding_mask=None):
		latent_mean, latent_log_var = self.forward(wf, key_padding_mask=key_padding_mask)
		protein_latent = self.sample(latent_mean, latent_log_var)
		return protein_latent