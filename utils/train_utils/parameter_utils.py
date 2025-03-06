# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		parameter_utils.py
description:	utility classes for parameter management for training 
'''
# ----------------------------------------------------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------	

class HyperParameters():

	def __init__(self, 	d_model,
						freeze_structure_weights,
						freeze_sequence_weights,
						cp_struct_enc_2_seq_enc,
						learnable_wavelengths,
						wf_type, anisotropic_wf,
						min_wl, max_wl, base_wl, 
						d_hidden_we, hidden_layers_we, 
						use_aa,
						d_hidden_aa, hidden_layers_aa, 
						esm2_weights_path, learnable_esm,
						struct_encoder_layers, seq_encoder_layers, num_heads,
						learnable_spreads,
						min_spread, max_spread, base_spread, num_spread,
						min_rbf, max_rbf, beta,
						d_hidden_attn, hidden_layers_attn, 
						temperature, use_model ):
		self.d_model = d_model
		self.freeze_structure_weights = freeze_structure_weights
		self.freeze_sequence_weights = freeze_sequence_weights
		self.cp_struct_enc_2_seq_enc = cp_struct_enc_2_seq_enc
		self.learnable_wavelengths = learnable_wavelengths
		self.wf_type = wf_type
		self.anisotropic_wf = anisotropic_wf
		self.min_wl = min_wl
		self.max_wl = max_wl
		self.base_wl = base_wl 
		self.d_hidden_we = d_hidden_we
		self.hidden_layers_we = hidden_layers_we 
		self.use_aa = use_aa
		self.d_hidden_aa = d_hidden_aa
		self.hidden_layers_aa = hidden_layers_aa
		self.esm2_weights_path = esm2_weights_path
		self.learnable_esm = learnable_esm
		self.struct_encoder_layers = struct_encoder_layers
		self.seq_encoder_layers = seq_encoder_layers
		self.num_heads = num_heads
		self.learnable_spreads = learnable_spreads
		self.min_spread = min_spread
		self.max_spread = max_spread 
		self.beta = beta
		self.base_spread = base_spread 
		self.num_spread = num_spread 
		self.min_rbf = min_rbf
		self.max_rbf = max_rbf
		self.d_hidden_attn = d_hidden_attn
		self.hidden_layers_attn = hidden_layers_attn 
		self.temperature = temperature
		self.use_model = use_model

class TrainingParameters():

	def __init__(self, 	epochs,
						accumulation_steps, 
						lr_type, # cyclic, attn or plataeu
						warmup_steps, # for attn
						lr_initial_min, lr_initial_max, lr_final_min, lr_final_max, lr_cycle_length, # for cyclic
						lr_scale, lr_patience, lr_step, # for plataeu
						beta1, beta2, epsilon, # for adam optim
						dropout, attn_dropout, wf_dropout,
						label_smoothing,
						loss_type, grad_clip_norm, 
						use_amp, use_chain_mask,
						noise_coords_std,
						early_stopping_thresh,
						early_stopping_tolerance
				):
		self.epochs = epochs
		self.accumulation_steps = accumulation_steps 
		self.lr_type = lr_type 
		self.warmup_steps = warmup_steps
		self.lr_initial_min = lr_initial_min
		self.lr_initial_max = lr_initial_max
		self.lr_final_min = lr_final_min
		self.lr_final_max = lr_final_max
		self.lr_cycle_length = lr_cycle_length 
		self.lr_scale = lr_scale 
		self.lr_patience = lr_patience
		self.lr_step = lr_step
		self.beta1 = beta1 
		self.beta2 = beta2
		self.epsilon = epsilon 
		self.dropout = dropout 
		self.attn_dropout = attn_dropout 
		self.wf_dropout = wf_dropout
		self.label_smoothing = label_smoothing
		self.loss_type = loss_type 
		self.grad_clip_norm = grad_clip_norm
		self.use_amp = use_amp 
		self.use_chain_mask = use_chain_mask
		self.noise_coords_std = noise_coords_std
		self.early_stopping_thresh = early_stopping_thresh
		self.early_stopping_tolerance = early_stopping_tolerance

# ----------------------------------------------------------------------------------------------------------------------
