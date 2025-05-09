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
from utils.model_utils.wf_embedding.anisotropic.cb_scaling.learnable_wavenumbers.cuda.wf_embedding import wf_embedding as wf_embedding_learnCB
from utils.model_utils.base_modules.Cb_utils import get_coords
from utils.model_utils.base_modules.base_modules import StaticLayerNorm

# ----------------------------------------------------------------------------------------------------------------------

class WaveFunctionEmbedding(nn.Module):

	def __init__(self, 	d_model=512, num_aas=20, old=False):

		super(WaveFunctionEmbedding, self).__init__()

		# wl = 4.0 + 16*((torch.logspace(0,1,d_model//2, 30) - 1)/29)
		# wn = torch.log(torch.pi*2/wl)

		self.wavenumbers = nn.Parameter(torch.zeros(d_model//2)) # learns log of wavenumbers, initialized so all dmodel dims start w/ corresponding wavelength of 2pi
		self.num_aas = num_aas

		# initialize aa magnitudes (overwritten by proteusAI if specified)
		self.old = old # whether running old model w/ no aa info
		if old: # learn scaling factors independant of aa
			self.aa_magnitudes = nn.Parameter(torch.ones(d_model//2))
		else: # aa specific
			self.aa_magnitudes = nn.Parameter(torch.ones(d_model//2, num_aas))

		# additional layers
		self.norm = StaticLayerNorm(d_model)

	def get_wavenumbers(self):
		return torch.exp(self.wavenumbers) # learns log of wavenumbers, ie log(2pi/lambda) = log(2pi) - log(lambda), log(2pi) is constant, so learning -log(lambda)

	def forward(self, coords_alpha, coords_beta, aa_labels, key_padding_mask=None, no_aa=False):

		if self.old: 
			return self.norm(wf_embedding_learnCB(coords_alpha, coords_beta, self.aa_magnitudes, self.get_wavenumbers(), mask=key_padding_mask))
			# return wf_embedding_learnCB(coords_alpha, coords_beta, self.aa_magnitudes, self.get_wavenumbers(), mask=key_padding_mask)

		if no_aa: # compute the anisotropic wavefunction, but using the mean magnitude for each wavenumber so that the aa info is ambiguous. 
			aa_magnitudes = self.aa_magnitudes.mean(dim=1, keepdim=True).repeat(1, self.aa_magnitudes.size(1))
			wf = wf_embedding_staticAA(coords_alpha, coords_beta, aa_labels, aa_magnitudes, self.get_wavenumbers(), dropout_p=0.0, mask=key_padding_mask)
		else:
			if self.aa_magnitudes.requires_grad and torch.is_grad_enabled() and self.training: # learnable AA is much slower (4.5x), avoid it if not tracking grads of AAs
				wf = wf_embedding_learnAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.get_wavenumbers(), mask=key_padding_mask)
			else:
				wf = wf_embedding_staticAA(coords_alpha, coords_beta, aa_labels, self.aa_magnitudes, self.get_wavenumbers(), dropout_p=0.0, mask=key_padding_mask)

		# norming improvves performance, norms each tokens features, no affine transformation, so not layernorm
		wf = self.norm(wf)

		return wf

	def get_CaCb_coords(self, coords, chain_idxs=None):
		return get_coords(coords, chain_idxs)

# ----------------------------------------------------------------------------------------------------------------------
