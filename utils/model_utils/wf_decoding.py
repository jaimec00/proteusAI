'''
variational auto encoder, decoder part
same as encoder for now, but make seperate class in case i end up changing it
'''

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import init_xavier, MLP

class WaveFunctionDecoding(nn.Module):
	def __init__(self,  d_model=256, d_latent=32, d_proj=512,
						d_hidden_pre=2048, hidden_layers_pre=0, 
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=4, heads=8,
						use_bias=False, min_rbf=0.000,
						d_hidden_attn=1024, hidden_layers_attn=0,
						dropout=0.10,
					):

		super(WaveFunctionDecoding, self).__init__()
		
		self.dropout = nn.Dropout(dropout)

		self.d_proj = nn.Linear(d_latent, d_proj)
		self.norm = nn.LayerNorm(d_proj)
		self.space_enc = MLP(d_in=d_model, d_out=d_proj, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout) 

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_proj, d_other=d_proj, heads=heads, 
													bias=use_bias, min_rbf=min_rbf,
													d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													dropout=dropout
												)
												for _ in range(encoder_layers)
											])

		self.mlp_post = MLP(d_in=d_proj, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=0.0)

	def forward(self, wf, wf_no_aa, key_padding_mask=None): # forward generates the mean and stds, so can use them for loss, 
		
		# pre-process the wf
		wf = self.d_proj(wf)
		wf = self.norm(wf + self.dropout(self.space_enc(wf_no_aa))) # spatial encoding on aa ambiguous wf (cb scale is mean of all aas for each k)

		# self attention on updated wf
		for encoder in self.encoders:
			wf = encoder(wf, wf, wf, mask=key_padding_mask)

		# post-process to get final wf
		wf = self.mlp_post(wf)

		return wf
