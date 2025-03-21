# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on alpha carbon coordinates. 
				uses wave function embedding, diffusion, and extraction 
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding
from utils.model_utils.wf_diffusion import WaveFunctionDiffusion
from utils.model_utils.wf_extraction import WaveFunctionExtraction
from utils.model_utils.base_modules.Cb_utils import get_coords

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	'''
	proteusAI. holds the three models, embedding, diffusion, and extraction. for inference, set inference=True in forward method
	'''
	
	def __init__(self, 	# model dimension and number of amino acid classes 
						d_model=512, num_aas=20, 

						# wf embedding params
						embedding_min_wl=2, embedding_max_wl=10, embedding_base_wl=25, embedding_learnable_aa=True,

						# wf diffusion params
						diffusion_beta_min=1e-4, diffusion_beta_max=0.02, diffusion_beta_schedule_type="linear", diffusion_t_max=100,
						diffusion_min_wl=0.001, diffusion_max_wl=100, # for sinusoidal timestep embedding
						diffusion_mlp_timestep=False, diffusion_d_hidden_timestep=2048, diffusion_hidden_layers_timestep=0, diffusion_norm_timestep=False,
						diffusion_mlp_pre=False, diffusion_d_hidden_pre=2048, diffusion_hidden_layers_pre=0, diffusion_norm_pre=False,
						diffusion_mlp_post=False, diffusion_d_hidden_post=2048, diffusion_hidden_layers_post=0, diffusion_norm_post=False,
						diffusion_encoder_layers=8, diffusion_heads=8, diffusion_learnable_spreads=True,
						diffusion_min_spread=3.0, diffusion_max_spread=15.0, diffusion_base_spreads=1.0, diffusion_num_spread=32,
						diffusion_min_rbf=0.001, diffusion_max_rbf=0.85, diffusion_beta=2.0,
						diffusion_d_hidden_attn=2048, diffusion_hidden_layers_attn=0,

						# wf extraction params
						extraction_mlp_pre=False, extraction_d_hidden_pre=2048, extraction_hidden_layers_pre=0, extraction_norm_pre=False,
						extraction_mlp_post=False, extraction_d_hidden_post=2048, extraction_hidden_layers_post=0, extraction_norm_post=False,
						extraction_encoder_layers=8, extraction_heads=8, extraction_learnable_spreads=True,
						extraction_min_spread=3.0, extraction_max_spread=15.0, extraction_base_spreads=1.0, extraction_num_spread=32,
						extraction_min_rbf=0.001, extraction_max_rbf=0.85, extraction_beta=2.0,
						extraction_d_hidden_attn=2048, extraction_hidden_layers_attn=0,

						# dropout params
						dropout=0.10, attn_dropout=0.00, wf_dropout=0.00,
				):

		super(proteusAI, self).__init__()

		self.num_aas = num_aas

		self.wf_embedding = WaveFunctionEmbedding(	d_model=d_model, num_aas=num_aas, 
													min_wl=embedding_min_wl, max_wl=embedding_max_wl, base_wl=embedding_base_wl,
													learnable_aa=embedding_learnable_aa, dropout=wf_dropout
												)

		self.wf_diffusion = WaveFunctionDiffusion(	d_model=d_model, 
													beta_min=diffusion_beta_min, beta_max=diffusion_beta_max, beta_schedule_type=diffusion_beta_schedule_type, t_max=diffusion_t_max,
													min_wl=diffusion_min_wl, max_wl=diffusion_max_wl,
													mlp_pre=diffusion_mlp_pre, d_hidden_pre=diffusion_d_hidden_pre, hidden_layers_pre=diffusion_hidden_layers_pre, norm_pre=diffusion_norm_pre, 
													mlp_post=diffusion_mlp_post, d_hidden_post=diffusion_d_hidden_post, hidden_layers_post=diffusion_hidden_layers_post, norm_post=diffusion_norm_post,							
													encoder_layers=diffusion_encoder_layers, heads=diffusion_heads, learnable_spreads=diffusion_learnable_spreads,
													min_spread=diffusion_min_spread, max_spread=diffusion_max_spread, base_spreads=diffusion_base_spreads, num_spread=diffusion_num_spread,
													min_rbf=diffusion_min_rbf, max_rbf=diffusion_max_rbf, beta=diffusion_beta,
													d_hidden_attn=diffusion_d_hidden_attn, hidden_layers_attn=diffusion_hidden_layers_attn,
													dropout=dropout, attn_dropout=attn_dropout
												)

		self.wf_extraction = WaveFunctionExtraction(	d_model=d_model, num_aas=num_aas, 
														mlp_pre=extraction_mlp_pre, d_hidden_pre=extraction_d_hidden_pre, hidden_layers_pre=extraction_hidden_layers_pre, norm_pre=extraction_norm_pre,
														mlp_post=extraction_mlp_post, d_hidden_post=extraction_d_hidden_post, hidden_layers_post=extraction_hidden_layers_post, norm_post=extraction_norm_post,
														encoder_layers=extraction_encoder_layers, heads=extraction_heads, learnable_spreads=extraction_learnable_spreads,
														min_spread=extraction_min_spread, max_spread=extraction_max_spread, base_spreads=extraction_base_spreads, num_spread=extraction_num_spread,
														min_rbf=extraction_min_rbf, max_rbf=extraction_max_rbf, beta=extraction_beta,
														d_hidden_attn=extraction_d_hidden_attn, hidden_layers_attn=extraction_hidden_layers_attn,
														dropout=dropout, attn_dropout=attn_dropout
													)

	def forward(self, 	coords_alpha, coords_beta=None, aas=None, wf=None, chain_idxs=None, key_padding_mask=None, 
						embedding=False, diffusion=False, extraction=False, t=None,
						inference=False, cycles=10, temp=1e-6 # last row is for inference only
				):

		if (coords_beta is None) and (inference | embedding):
			coords_alpha, coords_beta = self.get_CaCb_coords(coords_alpha, chain_idxs) 

		if inference:
			assert aas is not None, "aas cannot be NoneType if running inference, must be a tensor of labels of size batch x N"
			with torch.no_grad:
				return self.inference(coords_alpha, coords_beta, aas, key_padding_mask, cycles, temp)

		assert embedding ^ (diffusion ^ extraction), 	"one of embedding, diffusion, OR extraction must be selected for a single\
														forward pass, if looking to run full inference pipeline, set inference=True"

		if embedding: # encode the structure + sequence via wave function embedding
			assert aas is not None, "aas cannot be NoneType if running embedding, must be a tensor of labels of size batch x N"
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask)
		elif diffusion: # run diffusion
			assert wf is not None, "wf cannot be NoneType if running diffusion, must be a tensor of size batch x N x d_model"
			assert t is not None, "t (timestep) cannot be NoneType if running diffusion, must be a tensor of size batch,"
			wf = self.wf_diffusion(wf, coords_alpha, t, key_padding_mask)
		elif extraction: # run extraction
			assert wf is not None, "wf cannot be NoneType if running extraction, must be a tensor of size batch x N x d_model"
			wf = self.wf_extraction(wf, coords_alpha, key_padding_mask)

		return wf

	def inference(self, aas, coords_alpha, coords_beta, key_padding_mask=None, cycles=10, temp=1e-6):

		# prep
		batch, N = aas.shape
		t_max = self.wf_diffusion.beta_scheduler.t_max

		# fixed position have an AA label, positions to predict are -1, so they are set to random aa
		fixed_aas = aas!=-1
		aas = torch.where(fixed_aas, aas, torch.randint(0, self.num_aas, aas.shape))

		# multiple embedding + diffusion + extraction runs, each giving a slightly better guess and thus using less noise
		for cycle, t_fwd in zip(cycles, range(t_max, 0, -t_max//cycles)):

			# perform embedding
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask)
			
			# add gaussian noise
			wf, _ = self.wf_diffusion.noise(wf, t_fwd)

			# remove the noise
			wf = self.wf_diffusion.denoise(wf, coords_alpha, t_fwd, key_padding_mask)

			# perform extraction
			new_aas = self.wf_extraction.extract(wf, coords_alpha, key_padding_mask)

			# keeping fixed positions as they were for next iteration, 
			aas = torch.where(fixed_aas, aas, new_aas)

		return aas
	
	def get_CaCb_coords(self, coords, chain_idxs=None):
		return get_coords(coords, chain_idxs)

	def freeze_WFEmbedding_weights(self):
		for param in self.wf_embedding.parameters():
			param.requires_grad = False

	def freeze_WFDiffusion_weights(self):
		for param in self.wf_diffusion.parameters():
			param.requires_grad = False

	def freeze_WFExtraction_weights(self):
		for param in self.wf_extraction.parameters():
			param.requires_grad = False

	def load_WFEmbedding_weights(self, weights_path, device="cuda"):
		embedding_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_embedding.load_state_dict(embedding_weights)

	def load_WFDiffusion_weights(self, weights_path, device="cuda"):
		diffusion_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_diffusion.load_state_dict(diffusion_weights)
	
	def load_WFExtraction_weights(self, weights_path, device="cuda"):
		extraction_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_extraction.load_state_dict(extraction_weights)

	def save_WFEmbedding_weights(self, weights_path):
		torch.save(self.wf_embedding.state_dict(), weights_path)

	def save_WFDiffusion_weights(self, weights_path):
		torch.save(self.wf_diffusion.state_dict(), weights_path)

	def save_WFExtraction_weights(self, weights_path):
		torch.save(self.wf_extraction.state_dict(), weights_path)