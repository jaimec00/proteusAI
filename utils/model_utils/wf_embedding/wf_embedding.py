# ----------------------------------------------------------------------------------------------------------------------

'''
author: 		jaime cardenas
title:  		wf_embeds.py
description:	embeds the structure + sequence into single representation via wavefunction embedding
				WF embedding's job is simply to map the residues to the physical wavefunction space
				WF encoding maps the wf representation to latent space
'''

# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.learnable_aa.cuda.wf_embedding import wf_embedding as wf_embedding_learnAA
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.static_aa.cuda.wf_embedding import wf_embedding as wf_embedding_staticAA
from utils.model_utils.base_modules.Cb_utils import get_coords
from utils.model_utils.base_modules.base_modules import StaticLayerNorm

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionEmbedding(nn.Module):

	def __init__(self, 	d_model=512, num_aas=20,
						min_wl=3.7, max_wl=20, base_wl=20, 
						learnable_aa=True, dropout=0.0
				):

		super(WaveFunctionEmbedding, self).__init__()

		# compute wavenumbers
		self.register_buffer("wavenumbers", 2 * torch.pi / (min_wl + (max_wl - min_wl) * (torch.logspace(0,1,d_model//2, base_wl) - 1) / (base_wl - 1)))
		
		# initialize aa magnitudes (overwritten by proteusAI if specified)
		self.aa_magnitudes = nn.Parameter(torch.ones(d_model//2, num_aas), requires_grad=learnable_aa)

		# additional layers
		self.dropout = dropout # not implemented for learnable rn bc of register pressure, might take the hit and add it though since just for pre-training

		# norm, only want to center data and make var 1, no learning or affine transformation
		self.norm = StaticLayerNorm(d_model)

	def forward(self, coords_alpha, coords_beta, aa_labels, key_padding_mask=None, no_aa=False):

		dropout = self.dropout if self.training else 0.0

		if no_aa: # compute the anisotropic wavefunction, but using the mean magnitude for each wavenumber so that the aa info is ambiguous. plan to use this as kv in cross attn in diffusion
			aa_magnitudes = self.aa_magnitudes.mean(dim=1, keepdim=True).expand(self.aa_magnitudes.shape)
			wf = wf_embedding_staticAA(coords_alpha, coords_beta, aa_labels, aa_magnitudes, self.wavenumbers, dropout_p=dropout, mask=key_padding_mask)
		else:
			if self.aa_magnitudes.requires_grad and torch.is_grad_enabled(): # learnable AA is much slower (4.5x), avoid it if not tracking grads of AAs
				wf = wf_embedding_learnAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.wavenumbers, dropout_p=dropout, mask=key_padding_mask)
			else:
				wf = wf_embedding_staticAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.wavenumbers, dropout_p=dropout, mask=key_padding_mask)

		wf = self.norm(wf)

		return wf

	def get_CaCb_coords(self, coords, chain_idxs=None):
		return get_coords(coords, chain_idxs)

# ----------------------------------------------------------------------------------------------------------------------
