# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on alpha carbon coordinates. 
				uses wave function embedding, encoding, diffusion, decoding, and extraction 

				embedding maps Ca in cartesian space to wf space
				encoding maps Ca in wf space to fixed size protein latent space
				diffusion moves imperfect latent space representation closer to manifold of viable proteins in the same latent space
				decoding maps latent space to wf space, uses cross attention between aa ambiguous residues and the protein latent, then multiple self attention layers
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
	proteusAI. holds the five models, embedding, encoding, diffusion, decoding, and extraction. for inference, set inference=True in forward method
	'''
	
	def __init__(self, 	# model dimension and number of amino acid classes, plus option to run old model, 
						# which is just embedding with no aa info + extraction 
						d_model=256, d_latent=256, d_wf=256, num_aas=20, 
						
						# legacy, these only load embedding and extraction modules. if both false, all modules loaded
						old=False,
						mlm=False,

						# wf embedding params, everything is learnable, so not included
						embedding_min_wl=4.0, embedding_max_wl=35.0, embedding_base_wl=20.0, embedding_learn_wl=True, embedding_learn_aa=True,

						# wf encoder params
						encoding_d_hidden_pre=1024, encoding_hidden_layers_pre=0,
						encoding_d_hidden_post=2048, encoding_hidden_layers_post=1, 
						encoding_encoder_layers=4, encoding_heads=8, 
						encoding_use_bias=False, encoding_min_rbf=0.000,
						encoding_d_hidden_attn=1024, encoding_hidden_layers_attn=0,

						# wf diffusion params
						diffusion_alpha_bar_min=0.0,
						diffusion_noise_schedule_type="linear", diffusion_t_max=100,
						diffusion_d_in_timestep=256, diffusion_d_hidden_timestep=1024, diffusion_hidden_layers_timestep=0,
						diffusion_d_hidden_post=1024, diffusion_hidden_layers_post=0,
						diffusion_encoder_layers=4, diffusion_heads=8, 
						diffusion_use_bias=False, diffusion_min_rbf=0.000,
						diffusion_d_hidden_attn=1024, diffusion_hidden_layers_attn=0,

						# wf decoder params
						decoding_d_hidden_pre=1024, decoding_hidden_layers_pre=0,
						decoding_d_hidden_post=1024, decoding_hidden_layers_post=0,
						decoding_encoder_layers=4, decoding_heads=8, 
						decoding_use_bias=False, decoding_min_rbf=0.000,
						decoding_d_hidden_attn=1024, decoding_hidden_layers_attn=0,
						
						# wf extraction params
						extraction_bins=32, extraction_dk=8,  # for distogram auxiliary loss
						extraction_d_hidden_pre=1024, extraction_hidden_layers_pre=0, 
						extraction_d_hidden_post=1024, extraction_hidden_layers_post=0,
						extraction_encoder_layers=4, extraction_heads=8, 
						extraction_use_bias=False, extraction_min_rbf=0.001,
						extraction_d_hidden_attn=1024, extraction_hidden_layers_attn=0,

						# dropout params
						dropout=0.10,
						attn_dropout=0.10
				):

		super(proteusAI, self).__init__()

		self.num_aas = num_aas
		self.mlm = mlm

		# have num_aa + 1 for mask token in MLM
		self.wf_embedding = WaveFunctionEmbedding(	d_model=d_wf, 
													min_wl=embedding_min_wl, max_wl=embedding_max_wl, 
													base_wl=embedding_base_wl, learn_wl=embedding_learn_wl, learn_aa=embedding_learn_aa,
													num_aas=num_aas + mlm, old=old
												)

		if not (old or mlm):
			
			self.wf_encoding = WaveFunctionEncoding(	d_model=d_model, d_latent=d_latent, d_proj=d_latent,
														d_hidden_pre=encoding_d_hidden_pre, hidden_layers_pre=encoding_hidden_layers_pre, 
														d_hidden_post=encoding_d_hidden_post, hidden_layers_post=encoding_hidden_layers_post,						
														encoder_layers=encoding_encoder_layers, heads=encoding_heads, 
														use_bias=encoding_use_bias, min_rbf=encoding_min_rbf,
														d_hidden_attn=encoding_d_hidden_attn, hidden_layers_attn=encoding_hidden_layers_attn,
														dropout=dropout
													)										

			self.wf_diffusion = WaveFunctionDiffusion(	d_latent=d_latent, d_proj=d_latent,
														alpha_bar_min=diffusion_alpha_bar_min, noise_schedule_type=diffusion_noise_schedule_type, t_max=diffusion_t_max,
														d_in_timestep=diffusion_d_in_timestep, d_hidden_timestep=diffusion_d_hidden_timestep, hidden_layers_timestep=diffusion_hidden_layers_timestep,
														d_hidden_post=diffusion_d_hidden_post, hidden_layers_post=diffusion_hidden_layers_post,					
														encoder_layers=diffusion_encoder_layers, heads=diffusion_heads, 
														use_bias=diffusion_use_bias, min_rbf=diffusion_min_rbf,
														d_hidden_attn=diffusion_d_hidden_attn, hidden_layers_attn=diffusion_hidden_layers_attn,
														dropout=dropout
													)

			self.wf_decoding = WaveFunctionDecoding(	d_model=d_model, d_latent=d_latent, d_proj=d_latent,
														d_hidden_pre=decoding_d_hidden_pre, hidden_layers_pre=decoding_hidden_layers_pre, 
														d_hidden_post=decoding_d_hidden_post, hidden_layers_post=decoding_hidden_layers_post,						
														encoder_layers=decoding_encoder_layers, heads=decoding_heads, 
														use_bias=decoding_use_bias, min_rbf=decoding_min_rbf,
														d_hidden_attn=decoding_d_hidden_attn, hidden_layers_attn=decoding_hidden_layers_attn,
														dropout=dropout
													)

		self.wf_extraction = WaveFunctionExtraction(	d_model=d_model, d_wf=d_wf, num_aas=num_aas,
														bins=extraction_bins, dk=extraction_dk, # auxiliary loss of predicting distograms
														d_hidden_pre=extraction_d_hidden_pre, hidden_layers_pre=extraction_hidden_layers_pre,
														d_hidden_post=extraction_d_hidden_post, hidden_layers_post=extraction_hidden_layers_post,
														encoder_layers=extraction_encoder_layers, heads=extraction_heads, 
														use_bias=extraction_use_bias, min_rbf=extraction_min_rbf,
														d_hidden_attn=extraction_d_hidden_attn, hidden_layers_attn=extraction_hidden_layers_attn,
														dropout=dropout, attn_dropout=attn_dropout
													)

	def forward(self, 	coords_alpha=None, coords_beta=None, aas=None, wf=None, wf_no_aa=None, latent=None, t=None, chain_idxs=None, key_padding_mask=None, 
						embedding=False, encoding=False, diffusion=False, decoding=False, extraction=False, no_aa=False, distogram=False, 
						inference=False, cycles=10, temp=1e-6 # last row is for inference only
				):

		if (coords_beta is None) and (embedding or inference):
			coords_alpha, coords_beta = self.wf_embedding.get_CaCb_coords(coords_alpha, chain_idxs) 

		if inference:
			assert aas is not None, "aas cannot be NoneType if running inference, must be a tensor of labels of size batch x N"
			
			with torch.no_grad():
				if self.mlm:
					return self.mlm_inference(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, temp=temp)
				return self.inference(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, cycles=cycles, temp=temp)

		assert ((embedding ^ encoding) ^ decoding) ^ (diffusion ^ extraction), 	"one of embedding, encoding, diffusion, decoding, OR extraction must be selected for a single\
														forward pass, if looking to run full inference pipeline, set inference=True"

		if embedding: # encode the structure + sequence via wave function embedding
			assert aas is not None, "aas cannot be NoneType if running embedding, must be a tensor of labels of size batch x N"
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=no_aa) # no_aa allows you to compute wf from mean of aa for each wavenumber, essentially just encoding struct w/ no aa info for spatial encoding
		elif encoding:
			assert wf is not None, "wf cannot be NoneType if running encoding, must be a tensor of size batch x N x d_model"
			wf = self.wf_encoding(wf, key_padding_mask=key_padding_mask)
		elif diffusion: # run diffusion
			assert latent is not None, "wf cannot be NoneType if running diffusion, must be a tensor of size batch x N x d_latent"
			assert t is not None, "t (timestep) cannot be NoneType if running diffusion, must be a tensor of size batch,"
			wf = self.wf_diffusion(latent, t, coords_alpha, key_padding_mask=key_padding_mask)
		elif decoding:
			assert latent is not None, "wf cannot be NoneType if running decoding, must be a tensor of size batch x N x d_model"
			wf = self.wf_decoding(latent, wf_no_aa, key_padding_mask=key_padding_mask)
		elif extraction: # run extraction
			assert wf is not None, "wf cannot be NoneType if running extraction, must be a tensor of size batch x N x d_model"
			wf = self.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask, distogram=distogram)

		return wf

	def inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, cycles=10, diffusion_iters=1, temp=1e-6):

		# prep
		batch, N = aas.shape
		t_max = self.wf_diffusion.noise_scheduler.t_max

		# fixed position have an AA label, positions to predict are -1, so they are set to random aa. doesnt matter too much, bc starts at white noise
		fixed_aas = aas!=-1
		aas = torch.where(fixed_aas, aas, torch.randint_like(aas, 0, self.num_aas))

		wf_no_aa = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask, no_aa=True)

		# multiple embedding + encoding + diffusion + decoding + extraction runs, each giving a slightly better guess and thus using less noise
		for t_fwd in range(t_max, 0, -t_max//cycles):

			# perform embedding
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

			# all of the following modules do not use coords alpha, since replaced geo attn with regular attn + spatial encoding, but leaving it until i am sure geo attn is dead

			# encode from wf space to latent space
			protein = self.wf_encoding.encode(wf, key_padding_mask=key_padding_mask)
			
			for diffusion_iter in range(diffusion_iters): # slight nudge, then denoise, then repeat, nudging latent towards manifold, rather than starting from full gaussian
				
				# add gaussian noise to latent space
				protein_noised, _ = self.wf_diffusion.noise(protein, t_fwd)

				# remove the noise
				protein = self.wf_diffusion.denoise(protein_noised, t_fwd, key_padding_mask=key_padding_mask)

			# decode from latent space to wf space
			wf_pred = self.wf_decoding(protein, wf_no_aa, key_padding_mask=key_padding_mask)

			# extract sequence from wf
			aa_pred = self.wf_extraction.extract(wf_pred, wf_no_aa, key_padding_mask=key_padding_mask)

			# keeping fixed positions as they were for next iteration, 
			aas = torch.where(fixed_aas, aas, aa_pred)

		return aas

	def mlm_inference(self, coords_alpha, coords_beta, aas, key_padding_mask=None, entropy_increment=0.1, temp=1e-6):

		entropy_threshold = 0.1 # entropy threshold
		entropies = torch.zeros_like(aas, dtype=torch.float32) + float("inf") # will return this later so user knows confidence
		fixed = (aas!=-1) | key_padding_mask # if not -1, then should be fixed as context
		predicted = fixed # to keep track of predicted pos as loop through
		aas.masked_fill_(~fixed, 20) # add the mask token to non fixed tokens

		while not predicted.all(): # already deals w/ masking

			# get wf
			wf = self.wf_embedding(coords_alpha, coords_beta, aas, key_padding_mask=key_padding_mask)

			# predict sequence
			aa_logits = self.wf_extraction(wf, coords_alpha, key_padding_mask=key_padding_mask)
			pred_aa = self.wf_extraction.sample(aa_logits, temp)

			# compute entropy
			probs = aa_logits.softmax(dim=2)
			entropy = torch.sum(-probs*torch.log(probs), dim=2)
			
			# update
			update_aa = (entropy < entropy_threshold) & (entropies == float("inf")) & (~predicted.all(dim=1, keepdim=True)) # dont update already finished samples
			entropies[update_aa] = entropy[update_aa]
			aas[update_aa] = pred_aa[update_aa]
			predicted |= update_aa

			# increment entropy threshold
			entropy_threshold += entropy_increment

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
		self.wf_extraction.load_state_dict(extraction_weights, strict=False)

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

	def turn_off_bias(self):
		'''
		i think it would be better for the initial wf embedding to be learned without any bias
		then, once reach a certain seq sim, freeze wf embedding, and allow the bias to be learnable
		i see worse performance when training these end to end, so i think that the geo attn relies too
		much on the bias at the begginging, but then when the features become expressive, it overrides them 
		with the bias, which stalls learning
		whenever i start with pretrained wf embedding and fresh encoders, performance is much better,
		so i am pretty convinced this is the issue
		'''
		with torch.no_grad():
			for encoder in self.wf_extraction.encoders:
				encoder.attn.beta_weights.fill_(-float('inf')) # learns log of betas, so beta = exp(-inf) = 0. now just keeping it frozen
				encoder.attn.beta_weights.requires_grad = False

	def turn_on_bias(self):
		with torch.no_grad():
			for encoder in self.wf_extraction.encoders:
				encoder.attn.beta_weights.fill_(0) # init the betas to exp(0) = 1
				encoder.attn.beta_weights.requires_grad = True
