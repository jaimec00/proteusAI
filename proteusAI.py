# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on alpha carbon coordinates. 
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from utils.model_utils.wf_embedding.wf_embedding import WaveFunctionEmbedding
from utils.model_utils.wf_extraction import WaveFunctionExtraction

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	'''
	proteusAI. holds the five models, embedding, encoding, diffusion, decoding, and extraction. for inference, set inference=True in forward method
	'''
	
	def __init__(self, 	# model dimension and number of amino acid classes
						d_model=256, d_wf=256, num_aas=20, 

						# wf embedding params
						min_wl=4.0, max_wl=35.0, base_wl=20.0, anisotropic=True, learn_wl=True, learn_aa=True,
						
						# wf extraction params
						d_hidden_pre=1024, hidden_layers_pre=-1, 
						d_hidden_post=1024, hidden_layers_post=-1,
						encoder_layers=4, heads=8,
						use_bias=False, learn_spreads=False, min_rbf=0.001,
						d_hidden_attn=1024, hidden_layers_attn=0,

						# dropout params
						dropout=0.10,
				):

		super(proteusAI, self).__init__()

		self.num_aas = num_aas

		# have num_aa + 1 for mask token in MLM
		self.wf_embedding = WaveFunctionEmbedding(	d_wf=d_wf, min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, 
													anisotropic=anisotropic, learn_wl=learn_wl, learn_aa=learn_aa,
												)

		self.wf_extraction = WaveFunctionExtraction(	d_model=d_model, d_wf=d_wf, num_aas=num_aas,
														d_hidden_pre=d_hidden_pre, hidden_layers_pre=hidden_layers_pre,
														d_hidden_post=d_hidden_post, hidden_layers_post=hidden_layers_post,
														encoder_layers=encoder_layers, heads=heads, 
														use_bias=use_bias, learn_spreads=learn_spreads, min_rbf=min_rbf,
														d_hidden_attn=d_hidden_attn, hidden_layers_attn=hidden_layers_attn,
														dropout=dropout
													)

	def forward(self, 	coords_alpha=None, wf=None, chain_idxs=None, key_padding_mask=None, 
						embedding=False, extraction=False,
						inference=False, temp=1e-6 # last row is for inference only
				):

		if embedding or inference:
			coords_alpha, coords_beta = self.wf_embedding.get_CaCb_coords(coords_alpha, chain_idxs) 

		if inference:			
			with torch.no_grad():
				return self.inference(coords_alpha, coords_beta, key_padding_mask=key_padding_mask, temp=temp)

		if embedding: # encode the structure + sequence via wave function embedding
			wf = self.wf_embedding(coords_alpha, coords_beta, key_padding_mask=key_padding_mask)
		if extraction: # run extraction
			wf = self.wf_extraction(wf, coords_alpha, coords_beta, chain_idxs, key_padding_mask=key_padding_mask)

		return wf

	def inference(self, coords_alpha, coords_beta, key_padding_mask=None, temp=1.0):
		wf = self.wf_embedding(coords_alpha, coords_beta, key_padding_mask=key_padding_mask)
		aas = self.wf_extraction.extract(wf, coords_alpha, key_padding_mask=key_padding_mask)

		return aas

	# easier to deal with weights for seperate modules individually and partition in the training run script, also allows people to freeze certain modules by editing the code if interested

	def freeze_WFEmbedding_weights(self):
		for param in self.wf_embedding.parameters():
			param.requires_grad = False

	def freeze_WFExtraction_weights(self):
		for param in self.wf_extraction.parameters():
			param.requires_grad = False

	def load_weights(self, model_weights, embedding=True, extraction=True):
		embedding_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_embedding")}
		extraction_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_extraction")}

		if embedding:
			self.wf_embedding.load_state_dict(embedding_weights)
		if extraction:
			self.wf_extraction.load_state_dict(extraction_weights)

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
