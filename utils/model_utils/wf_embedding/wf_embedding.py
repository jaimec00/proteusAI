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
from utils.model_utils.wf_embedding.isotropic.cuda.wf_embedding import wf_embedding as wf_embedding_iso
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.learnable_aa.learnable_wavenumber.cuda.wf_embedding import wf_embedding as wf_embedding_learnAA
from utils.model_utils.wf_embedding.anisotropic.aa_scaling.static_aa.cuda.wf_embedding import wf_embedding as wf_embedding_staticAA
from utils.model_utils.wf_embedding.anisotropic.cb_scaling.learnable_wavenumbers.cuda.wf_embedding import wf_embedding as wf_embedding_learnCB
from utils.model_utils.base_modules.Cb_utils import get_coords
from utils.model_utils.base_modules.base_modules import StaticLayerNorm

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionEmbedding(nn.Module):

	def __init__(self, d_wf=512, min_wl=4.0, max_wl=35.0, base_wl=20.0, anisotropic=True, learn_wl=False, learn_aa=False):

		super(WaveFunctionEmbedding, self).__init__()

		wl = min_wl + (max_wl-min_wl)*((torch.logspace(0,1,d_wf//2, base_wl) - 1)/(base_wl-1))
		wn = torch.log(torch.pi*2/wl)
		self.wavenumbers = nn.Parameter(wn, requires_grad=learn_wl) # learns log of wavenumbers, initialized so all dmodel dims start w/ corresponding wavelength of 2pi
		self.aa_magnitudes = nn.Parameter(torch.ones(d_wf//2), requires_grad=learn_aa)
		self.aniso = anisotropic

	def get_wavenumbers(self):
		return torch.exp(self.wavenumbers) # learns log of wavenumbers, ie log(2pi/lambda) = log(2pi) - log(lambda), log(2pi) is constant, so learning -log(lambda)

	def forward(self, coords_alpha, coords_beta, key_padding_mask=None):

		if self.aniso: 
			wf = wf_embedding_learnCB(coords_alpha, coords_beta, self.aa_magnitudes, self.get_wavenumbers(), mask=key_padding_mask)
		else:
			wf = wf_embedding_iso(coords_alpha, self.get_wavenumbers(), magnitude_type=1, dropout_p=0.0, mask=key_padding_mask)

		return wf

	def get_CaCb_coords(self, coords, chain_idxs=None, norm=False):
		return get_coords(coords, chain_idxs, norm)

# ----------------------------------------------------------------------------------------------------------------------
