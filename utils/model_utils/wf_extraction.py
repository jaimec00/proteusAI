# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		wf_extraction.py
description:	extracts sequence information from wavefunction representation of protein
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.base_modules.encoder import Encoder
from utils.model_utils.base_modules.base_modules import init_xavier, MLP

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionExtraction(nn.Module):
	
	def __init__(self, 	d_model=512, d_wf=128, num_aas=20, # model dimension
						bins=32, dk=8, # auxiliary task of predicting distograms
						d_hidden_pre=2048, hidden_layers_pre=0,
						d_hidden_post=2048, hidden_layers_post=0,
						encoder_layers=4, heads=8, min_rbf=0.001,
						d_hidden_attn=2048, hidden_layers_attn=0,
						dropout=0.10
				):

		super(WaveFunctionExtraction, self).__init__()

		self.dropout = nn.Dropout(dropout)
		# self.proj_pre = nn.Linear(d_wf, d_model)
		self.mlp_pre = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_pre, hidden_layers=hidden_layers_pre, dropout=dropout)
		self.norm_pre = nn.LayerNorm(d_model)
		self.mlp_post = MLP(d_in=d_model, d_out=d_model, d_hidden=d_hidden_post, hidden_layers=hidden_layers_post, dropout=dropout)
		self.norm_post = nn.LayerNorm(d_model)

		self.encoders = nn.ModuleList([ Encoder(	d_model=d_model, d_other=d_model, heads=heads, min_rbf=min_rbf,
													d_hidden=d_hidden_attn, hidden_layers=hidden_layers_attn, 
													dropout=dropout
												) 
										for _ in range(encoder_layers)
									])

		# map to aa prob logits
		self.out_proj = nn.Linear(d_model, num_aas)
		# self.dist_proj = nn.Parameter(torch.rand((d_model, bins))) # project i+j from Nxd to Nxb
		init_xavier(self.out_proj)

	def forward(self, wf, coords, key_padding_mask=None, distogram=False):

		# non linear tranformation for more intricate features
		wf = self.norm_pre(wf + self.mlp_pre(wf))

		# # geometric attn encoders
		for encoder in self.encoders:
			wf = encoder(wf, wf, wf, coords, mask=key_padding_mask)

		# # post process #SKIPPING POST MLP, IF OLD DOESNT WORK, add this back
		# wf = self.norm_post(wf + self.dropout(self.mlp_post(wf)))

		# map to probability logits
		aa_logits = self.out_proj(wf)

		# if distogram: # predicts features for distogram computation
		# 	space = torch.matmul(wf.unsqueeze(1), self.dist_proj.unsqueeze(0)).permute(0,2,1,3) / (wf.size(2)**0.5)
		# 	return aa_logits, space

		return aa_logits#, None

	def sample(self, aa_logits, temp=1e-6):

		batch, N, num_aas = aa_logits.shape

		# softmax on temp scaled logits to get AA probs
		aa_probs = torch.softmax(aa_logits/temp, dim=2)

		# sample from the distributions
		aa_labels = torch.multinomial(aa_probs.view(batch*N, num_aas), num_samples=1, replacement=False).view(batch, N)

		return aa_labels

	def extract(self, wf, coords_alpha, key_padding_mask=None, temp=1e-6):

		# perform extraction
		aa_logits = self.forward(wf, coords_alpha, key_padding_mask=key_padding_mask)

		# sample from distribution
		aas = self.sample(aa_logits, temp)

		return aas

