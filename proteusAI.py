# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on backbone coordinates. 
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from utils.model_utils.gnn.featurizer import FeaturizeProtein
from utils.model_utils.gnn.structure_encoder import StructureEncoder
from utils.model_utils.gnn.vae import SequenceEncoder, SequenceDecoder
from utils.model_utils.gnn.diffusion import SequenceDenoiser

from data.constants import canonical_aas

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	def __init__(self,  K=30, d_model=128, # model dims
						min_wl=3.5, max_wl=25.0, base_wl=20.0, anisotropic=True, learn_wl=True, learn_aa=False, # node embedding
						min_rbf=2.0, max_rbf=22.0, num_rbfs=16, # edge embedding
						struct_enc_layers=3, seq_enc_layers=3, seq_diff_layers=3, seq_dec_layers=3, 
						t_max=1000, dropout=0.00 # general
					):

		super(proteusAI, self).__init__()

		# to featurize the protiein
		self.featurizer = FeaturizeProtein( K=K, d_model=d_model, # model dims
											min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, anisotropic=anisotropic, learn_wl=learn_wl, learn_aa=learn_aa, # node features (wf embedding)
											min_rbf=min_rbf, max_rbf=max_rbf, num_rbfs=num_rbfs # edge features
										)

		# encode structure via nodes and edges
		self.structure_encoder = StructureEncoder(d_model=d_model, layers=struct_enc_layers, dropout=dropout)

		# encode sequence into latent vectors, conditioned on struct, so no one to one mapping of aa, the latent vectors should be highly contetualized
		# will prob not add kl div term, since i am thinking that the MSE of the diffusion module should serve to structure the latent space by itself,
		# so that small changes in latent dim correspond to small changes in seq
		self.sequence_encoder = SequenceEncoder(d_model=d_model, layers=seq_enc_layers, dropout=dropout) 

		# denoise the latent vectors, also option to do inpainting, ie use unnoised latent vectors as context. will also add option for interpolation
		self.sequence_denoiser = SequenceDenoiser(d_model=d_model, layers=seq_diff_layers, t_max=t_max, dropout=dropout)
		
		# decode the sequence, which in this case takes the denoised latents and reconstructs the sequence. 
		# not using MSE on original seq embeddings, rather predicting prob distribution of 20 dim aa vector and reconstruction loss is CEL
		self.sequence_decoder = SequenceDecoder(d_model=d_model, layers=seq_dec_layers, dropout=dropout) 


	def forward(self, C, L, chain_idxs, t=None, node_mask=None, inference=False, temp=1e-6):
		'''
		C is coords
		L is labels (-1 means to be decoded, unless nodemask is true at that position)
		node mask is essentially key_padding_mask
		inpaint_mask is Z x N of bools, where 1 means do inpaint (0 means keeo fixed). will prob not use at first, ie same noise for all aas
		'''

		# featurize protein
		V, E, K, S, edge_mask = self.featurizer(C, L, chain_idxs, node_mask)

		# encode structure
		V, E = self.structure_encoder(V, E, K, edge_mask) 

		# encode sequence conditioned on struct into latent params (sampling also done in forward)
		Z, Z_mu, Z_logvar = self.sequence_encoder(V, E, K, S, edge_mask)

		# sample t if it is None
		if t is None:
			if inference: # full noise if in inference
				t = self.sequence_denoiser.noise_scheduler.t_max
			else: # random timestep if not
				t = self.sequence_denoiser.get_random_timesteps(V.size(0), V.device)

		# noise the latent
		Z_noised, noise = self.sequence_denoiser.noise(Z, t)

		if inference:

			# denoise the latent through all timesteps
			Z_denoised = self.sequence_denoiser.denoise(V, E, K, Z_noised, t, edge_mask) # do full denoising if in inference
			noise_pred = 0.0 # for consistent reutrn values

			# predict aa labels w/ temp sampling from denoised latent
			S_pred = self.sequence_decoder(V, E, K, Z_denoised, edge_mask, sample=True, temp=temp)

		else:

			# predict the noise from only this timestep
			noise_pred = self.sequence_denoiser(V, E, K, Z_noised, t, edge_mask) # do full denoising if in inference

			# reconstruct prob distribution from original latents
			S_pred = self.sequence_decoder(V, E, K, Z, edge_mask, sample=False)

		# return all for loss calculation, Z_mu/logvar prob not used for kldiv, but in case i add that later
		return Z_mu, Z_logvar, noise, noise_pred, S_pred
