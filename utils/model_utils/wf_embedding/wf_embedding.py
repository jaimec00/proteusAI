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
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.learnable_aa.learnable_wavenumber.cuda.wf_embedding import wf_embedding as wf_embedding_learnAA
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.static_aa.cuda.wf_embedding import wf_embedding as wf_embedding_staticAA
from utils.model_utils.base_modules.Cb_utils import get_coords

from data.constants import alphabet

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionEmbedding(nn.Module):

	def __init__(self, d_wf=512, min_wl=4.0, max_wl=35.0, base_wl=20.0, learn_wl=False):

		super(WaveFunctionEmbedding, self).__init__()

		wl = min_wl + (max_wl-min_wl)*((torch.logspace(0,1,d_wf//2, base_wl) - 1)/(base_wl-1))
		wn = torch.log(torch.pi*2/wl)
		self.wavenumbers = nn.Parameter(wn, requires_grad=learn_wl) # learns log of wavenumbers, initialized so all dmodel dims start w/ corresponding wavelength of 2pi
		self.aa_magnitudes = nn.Parameter(torch.ones(d_wf//2, len(alphabet)))

	def get_wavenumbers(self):
		return torch.exp(self.wavenumbers) # learns log of wavenumbers, ie log(2pi/lambda) = log(2pi) - log(lambda), log(2pi) is constant, so learning -log(lambda)

	def forward(self, coords_alpha, coords_beta, labels, key_padding_mask=None):

		if self.training & self.aa_magnitudes.requires_grad & torch.is_grad_enabled():
			wf = wf_embedding_learnAA(coords_alpha, coords_beta, labels, self.aa_magnitudes, self.get_wavenumbers(), mask=key_padding_mask)
		else:
			wf = wf_embedding_staticAA(coords_alpha, coords_beta,  labels, self.aa_magnitudes, self.get_wavenumbers(), dropout_p=0.0, mask=key_padding_mask)

		return wf

	def get_CaCb_coords(self, coords, chain_idxs=None, norm=False):
		return get_coords(coords, chain_idxs, norm)

# ----------------------------------------------------------------------------------------------------------------------
