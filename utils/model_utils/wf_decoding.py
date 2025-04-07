'''
variational auto encoder, decoder part
same as encoder for now, but make seperate class in case i end up changing it
'''

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import init_xavier, MLP

class WaveFunctionDecoding(nn.Module):
	def __init__(self,  d_model=512, d_latent=512,
						mlp_pre=False, d_hidden_pre=2048, hidden_layers_pre=0, norm_pre=False,
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=8, heads=8, learnable_spreads=True,
						min_spread=3.0, max_spread=15.0, base_spreads=1.0, num_spread=32,
						min_rbf=0.001, max_rbf=0.85, beta=2.0,
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10, attn_dropout=0.00,
					):

		super(WaveFunctionDecoding, self).__init__()
		
		self.dropout = nn.Dropout(dropout)
		self.mlp_pre = MLP(d_in=d_latent, d_out=d_latent, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre) 
		self.norm_pre = nn.LayerNorm(d_latent)
		self.mlp_post = MLP(d_in=d_latent, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post)

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_latent, d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													heads=heads, min_spread=min_spread, min_rbf=min_rbf, max_rbf=max_rbf,  
													dropout=dropout
												) 
										for _ in range(encoder_layers)
							])

	def forward(self, wf_latent, coords_alpha, key_padding_mask=None): # forward generates the mean and stds, so can use them for loss, 
		
		wf_latent = wf_latent + self.dropout(self.mlp_pre(wf_latent))
		wf_latent = self.norm_pre(wf_latent)

		# encoders
		for encoder in self.encoders:
			wf_latent = encoder(wf_latent, coords_alpha, key_padding_mask=key_padding_mask)

		wf = self.mlp_post(wf_latent)

		return wf
