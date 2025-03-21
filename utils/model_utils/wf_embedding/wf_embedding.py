# ----------------------------------------------------------------------------------------------------------------------

'''
author: 		jaime cardenas
title:  		wf_embeds.py
description:	embeds the structure + sequence into single representation via wavefunction embedding
'''

# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.learnable_aa.cuda.wf_embedding import wf_embedding as wf_embedding_learnAA
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.static_aa.cuda.wf_embedding import wf_embedding as wf_embedding_staticAA
from utils.model_utils.base_modules.base_modules import CrossFeatureNorm

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
		self.norm = CrossFeatureNorm(d_model) # normalize each feature across the sequence independantly, s.t. mean is zero and std is 1, then scale by 1/sqrt(dmodel) so whole wf is normed
		self.dropout = dropout # not implemented for learnable rn bc of register pressure, might take the hit and add it though since just for pre-training

	def forward(self, coords_alpha, coords_beta, aa_labels, key_padding_mask=None):

		dropout = self.dropout if self.training else 0.0

		if self.aa_magnitudes.requires_grad and torch.is_grad_enabled(): # learnable AA is much slower (4.5x), avoid it if not tracking grads of AAs
			wf = wf_embedding_learnAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.wavenumbers, dropout_p=dropout, mask=key_padding_mask)
		else:
			wf = wf_embedding_staticAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.wavenumbers, dropout_p=dropout, mask=key_padding_mask)

		wf = self.norm(wf)

		return wf

# ----------------------------------------------------------------------------------------------------------------------
