# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on alpha carbon coordinates. 
				uses wave function embedding, encoding, diffusion, decoding, and extraction 

				embedding maps Ca in cartesian space to wf space
				encoding maps Ca in wf space to latent space
				diffusion moves imperfect latent space representation closer to manifold of viable proteins in the same latent space
				decoding maps latent space to wf space 
				extraction maps wf space to discrete sequence space (sequence prediction)
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding
from utils.model_utils.wf_encoding import WaveFunctionEncoding
from utils.model_utils.wf_diffusion import WaveFunctionDiffusion
from utils.model_utils.wf_decoding import WaveFunctionDecoding
from utils.model_utils.wf_extraction import WaveFunctionExtraction

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	'''
	proteusAI. holds the three models, embedding, diffusion, and extraction. for inference, set inference=True in forward method
	'''
	
	def __init__(self, 	# model dimension and number of amino acid classes 
						d_model=512, d_latent=512, num_aas=20, 

						# wf embedding params
						embedding_min_wl=2, embedding_max_wl=10, embedding_base_wl=25, embedding_learnable_aa=True,

						# wf encoder params
						encoding_mlp_pre=False, encoding_d_hidden_pre=2048, encoding_hidden_layers_pre=0, encoding_norm_pre=False,
						encoding_d_hidden_post=2048, encoding_hidden_layers_post=0, 
						encoding_encoder_layers=8, encoding_heads=8, encoding_learnable_spreads=True,
						encoding_min_spread=3.0, encoding_max_spread=15.0, encoding_base_spreads=1.0, encoding_num_spread=32,
						encoding_min_rbf=0.001, encoding_max_rbf=0.85, encoding_beta=2.0,
						encoding_d_hidden_attn=2048, encoding_hidden_layers_attn=0,

						# wf diffusion params
						diffusion_alpha_bar_min=0.0, diffusion_beta_schedule_type="cosine", diffusion_t_max=100,
						diffusion_d_hidden_timestep=2048, diffusion_hidden_layers_timestep=0,
						diffusion_mlp_pre=False, diffusion_d_hidden_pre=2048, diffusion_hidden_layers_pre=0, diffusion_norm_pre=False,
						diffusion_mlp_post=False, diffusion_d_hidden_post=2048, diffusion_hidden_layers_post=0, diffusion_norm_post=False,
						diffusion_encoder_layers=8, diffusion_decoder_layers=8, diffusion_heads=8, diffusion_learnable_spreads=True,
						diffusion_min_spread=3.0, diffusion_max_spread=15.0, diffusion_base_spreads=1.0, diffusion_num_spread=32,
						diffusion_min_rbf=0.001, diffusion_max_rbf=0.85, diffusion_beta=2.0,
						diffusion_d_hidden_attn=2048, diffusion_hidden_layers_attn=0,

						# wf decoder params
						decoding_mlp_pre=False, decoding_d_hidden_pre=2048, decoding_hidden_layers_pre=0, decoding_norm_pre=False,
						decoding_d_hidden_post=2048, decoding_hidden_layers_post=0,
						decoding_encoder_layers=8, decoding_heads=8, decoding_learnable_spreads=True,
						decoding_min_spread=3.0, decoding_max_spread=15.0, decoding_base_spreads=1.0, decoding_num_spread=32,
						decoding_min_rbf=0.001, decoding_max_rbf=0.85, decoding_beta=2.0,
						decoding_d_hidden_attn=2048, decoding_hidden_layers_attn=0,

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

		self.wf_encoding = WaveFunctionEncoding(	d_model=d_model, d_latent=d_latent,
													mlp_pre=encoding_mlp_pre, d_hidden_pre=encoding_d_hidden_pre, hidden_layers_pre=encoding_hidden_layers_pre, norm_pre=encoding_norm_pre, 
													d_hidden_post=encoding_d_hidden_post, hidden_layers_post=encoding_hidden_layers_post,						
													encoder_layers=encoding_encoder_layers, heads=encoding_heads, learnable_spreads=encoding_learnable_spreads,
													min_spread=encoding_min_spread, max_spread=encoding_max_spread, base_spreads=encoding_base_spreads, num_spread=encoding_num_spread,
													min_rbf=encoding_min_rbf, max_rbf=encoding_max_rbf, beta=encoding_beta,
													d_hidden_attn=encoding_d_hidden_attn, hidden_layers_attn=encoding_hidden_layers_attn,
													dropout=dropout, attn_dropout=attn_dropout
												)										

		self.wf_diffusion = WaveFunctionDiffusion(	d_model=d_latent, 
													alpha_bar_min=diffusion_alpha_bar_min, beta_schedule_type=diffusion_beta_schedule_type, t_max=diffusion_t_max,
													d_hidden_timestep=diffusion_d_hidden_timestep, hidden_layers_timestep=diffusion_hidden_layers_timestep,
													mlp_pre=diffusion_mlp_pre, d_hidden_pre=diffusion_d_hidden_pre, hidden_layers_pre=diffusion_hidden_layers_pre, norm_pre=diffusion_norm_pre, 
													mlp_post=diffusion_mlp_post, d_hidden_post=diffusion_d_hidden_post, hidden_layers_post=diffusion_hidden_layers_post, norm_post=diffusion_norm_post,							
													encoder_layers=diffusion_encoder_layers, decoder_layers=diffusion_decoder_layers, heads=diffusion_heads, learnable_spreads=diffusion_learnable_spreads,
													min_spread=diffusion_min_spread, max_spread=diffusion_max_spread, base_spreads=diffusion_base_spreads, num_spread=diffusion_num_spread,
													min_rbf=diffusion_min_rbf, max_rbf=diffusion_max_rbf, beta=diffusion_beta,
													d_hidden_attn=diffusion_d_hidden_attn, hidden_layers_attn=diffusion_hidden_layers_attn,
													dropout=dropout, attn_dropout=attn_dropout
												)

		self.wf_decoding = WaveFunctionDecoding(	d_model=d_model, d_latent=d_latent,
													mlp_pre=decoding_mlp_pre, d_hidden_pre=decoding_d_hidden_pre, hidden_layers_pre=decoding_hidden_layers_pre, norm_pre=decoding_norm_pre, 
													d_hidden_post=decoding_d_hidden_post, hidden_layers_post=decoding_hidden_layers_post,						
													encoder_layers=decoding_encoder_layers, heads=decoding_heads, learnable_spreads=decoding_learnable_spreads,
													min_spread=decoding_min_spread, max_spread=decoding_max_spread, base_spreads=decoding_base_spreads, num_spread=decoding_num_spread,
													min_rbf=decoding_min_rbf, max_rbf=decoding_max_rbf, beta=decoding_beta,
													d_hidden_attn=decoding_d_hidden_attn, hidden_layers_attn=decoding_hidden_layers_attn,
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

	def forward(self, 	coords_alpha, coords_beta=None, aas=None, wf=None, context=None, chain_idxs=None, key_padding_mask=None, 
						embedding=False, encoding=False, diffusion=False, decoding=False, extraction=False, no_aa=False, t=None,
						inference=False, cycles=10, temp=1e-6 # last row is for inference only
				):

		if (coords_beta is None) and (embedding or inference):
			coords_alpha, coords_beta = self.wf_embedding.get_CaCb_coords(coords_alpha, chain_idxs) 

		if inference:
			assert aas is not None, "aas cannot be NoneType if running inference, must be a tensor of labels of size batch x N"
			with torch.no_grad():
				return self.inference(coords_alpha, coords_beta, aas, key_padding_mask, cycles, temp)

		assert ((embedding ^ encoding) ^ decoding) ^ (diffusion ^ extraction), 	"one of embedding, encoding, diffusion, decoding, OR extraction must be selected for a single\
														forward pass, if looking to run full inference pipeline, set inference=True"

		if embedding: # encode the structure + sequence via wave function embedding
			assert aas is not None, "aas cannot be NoneType if running embedding, must be a tensor of labels of size batch x N"
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=no_aa) # iso gives option to compute isotropic version
		elif encoding:
			assert wf is not None, "wf cannot be NoneType if running encoding, must be a tensor of size batch x N x d_model"
			wf = self.wf_encoding(wf, coords_alpha, key_padding_mask=key_padding_mask)
		elif diffusion: # run diffusion
			assert wf is not None, "wf cannot be NoneType if running diffusion, must be a tensor of size batch x N x d_latent"
			assert t is not None, "t (timestep) cannot be NoneType if running diffusion, must be a tensor of size batch,"
			wf = self.wf_diffusion(wf, coords_alpha, t, key_padding_mask=key_padding_mask, context=context)
		elif decoding:
			assert wf is not None, "wf cannot be NoneType if running decoding, must be a tensor of size batch x N x d_model"
			wf = self.wf_decoding(wf, coords_alpha, key_padding_mask=key_padding_mask)
		elif extraction: # run extraction
			assert wf is not None, "wf cannot be NoneType if running extraction, must be a tensor of size batch x N x d_model"
			wf = self.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask)

		return wf

	def inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, cycles=10, temp=1e-6):

		# prep
		batch, N = aas.shape
		t_max = self.wf_diffusion.beta_scheduler.t_max

		# fixed position have an AA label, positions to predict are -1, so they are set to random aa. doesnt matter too much, bc starts at white noise
		fixed_aas = aas!=-1
		aas = torch.where(fixed_aas, aas, torch.randint_like(aas, 0, self.num_aas))

		# no aa context wf, to give diffusion a starting point
		wf_base = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=True)
		wf_base_encoded = self.wf_encoding.encode(wf_base, coords_alpha, key_padding_mask)

		# multiple embedding + diffusion + extraction runs, each giving a slightly better guess and thus using less noise
		for t_fwd in range(t_max, 0, -t_max//cycles):

			# perform embedding
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

			# encode from wf space to latent space
			latent_wf = self.wf_encoding.encode(wf, coords_alpha, key_padding_mask)
			
			# add gaussian noise to latent space
			latent_wf_noised, _ = self.wf_diffusion.noise(wf, t_fwd)

			# remove the noise
			latent_wf_denoised = self.wf_diffusion.denoise(latent_wf_noised, coords_alpha, t_fwd, key_padding_mask=key_padding_mask, context=wf_base_encoded)

			# decode from latent space to wf space
			wf_pred = self.wf_decoding(latent_wf_denoised, coords_alpha, key_padding_mask)

			# extract sequence from wf
			aa_pred = self.wf_extraction.extract(wf_pred, coords_alpha, key_padding_mask)

			# keeping fixed positions as they were for next iteration, 
			aas = torch.where(fixed_aas, aas, aa_pred)

		return aas

	# easier to deal with weights for seperate modules individually and partition in the training run script, also allows people to freeze certain modules by editing the code if interested

	def freeze_WFEmbedding_weights(self):
		for param in self.wf_embedding.parameters():
			param.requires_grad = False

	def freeze_WFEncoding_weights(self):
		for param in self.wf_encoding.parameters():
			param.requires_grad = False
			
	def freeze_WFDiffusion_weights(self):
		for param in self.wf_diffusion.parameters():
			param.requires_grad = False

	def freeze_WFDecoding_weights(self):
		for param in self.wf_decoding.parameters():
			param.requires_grad = False

	def freeze_WFExtraction_weights(self):
		for param in self.wf_extraction.parameters():
			param.requires_grad = False

	def load_WFEmbedding_weights(self, weights_path, device="cuda"):
		embedding_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_embedding.load_state_dict(embedding_weights)

	def load_WFEncoding_weights(self, weights_path, device="cuda"):
		encoding_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_encoding.load_state_dict(encoding_weights)

	def load_WFDiffusion_weights(self, weights_path, device="cuda"):
		diffusion_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_diffusion.load_state_dict(diffusion_weights)
	
	def load_WFDecoding_weights(self, weights_path, device="cuda"):
		decoding_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_decoding.load_state_dict(decoding_weights)
	
	def load_WFExtraction_weights(self, weights_path, device="cuda"):
		extraction_weights = torch.load(weights_path, map_location=device, weights_only=True)
		self.wf_extraction.load_state_dict(extraction_weights)

	def save_WFEmbedding_weights(self, weights_path):
		torch.save(self.wf_embedding.state_dict(), weights_path)

	def save_WFEncoding_weights(self, weights_path):
		torch.save(self.wf_encoding.state_dict(), weights_path)

	def save_WFDiffusion_weights(self, weights_path):
		torch.save(self.wf_diffusion.state_dict(), weights_path)

	def save_WFDecoding_weights(self, weights_path):
		torch.save(self.wf_decoding.state_dict(), weights_path)

	def save_WFExtraction_weights(self, weights_path):
		torch.save(self.wf_extraction.state_dict(), weights_path)