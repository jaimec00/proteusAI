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
						d_hidden_pre=2048, hidden_layers_pre=0, 
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_self_layers=4, self_heads=8,
						self_d_hidden_attn=1024, self_hidden_layers_attn=0,
						encoder_cross_layers=4, cross_heads=8,
						cross_d_hidden_attn=1024, cross_hidden_layers_attn=0,
						dropout=0.10,
					):

		super(WaveFunctionDecoding, self).__init__()
		
		self.dropout = nn.Dropout(dropout)
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout) 
		self.norm_pre = nn.LayerNorm(d_model)
		self.mlp_post = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=0.0)
		self.residue_encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_other=d_latent, heads=cross_heads, 
															d_hidden=cross_d_hidden_attn, hidden_layers=cross_hidden_layers_attn, 
															dropout=dropout
												)
												for _ in range(encoder_cross_layers)
											])
		self.wf_encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_other=d_model, heads=self_heads, 
													d_hidden=self_d_hidden_attn, hidden_layers=self_hidden_layers_attn, 
													dropout=dropout
												)
												for _ in range(encoder_self_layers)
											])
	def forward(self, wf, protein, key_padding_mask=None): # forward generates the mean and stds, so can use them for loss, 
		
		# pre-process the wf
		wf = self.norm_pre(wf + self.dropout(self.mlp_pre(wf))) # spatial encoding on aa ambiguous wf (cb scale is mean of all aas for each k)

		# cross attn, wf is query, protein is kv
		for encoder in self.residue_encoders:
			wf = encoder(wf, protein, protein, mask=None)

		# self attention on updated wf
		for encoder in self.wf_encoders:
			wf = encoder(wf, wf, wf, mask=key_padding_mask)

		# post-process to get final wf
		wf = wf + self.mlp_post(wf)

		return wf
