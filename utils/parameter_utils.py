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

class InputPerturbationParameters():

	def __init__(self,  
						initial_max_one_hot_injection_mean, final_max_one_hot_injection_mean, 
						min_one_hot_injection_mean, one_hot_injection_stdev, 
						one_hot_injection_cycle_length ,
					):

		# input label smoothing
		self.initial_min_lbl_smooth_mean = initial_min_lbl_smooth_mean
		self.final_min_lbl_smooth_mean = final_min_lbl_smooth_mean
		self.max_lbl_smooth_mean = max_lbl_smooth_mean

		self.min_lbl_smooth_stdev = min_lbl_smooth_stdev
		self.max_lbl_smooth_stdev = max_lbl_smooth_stdev
		
		# input noise
		self.min_noise_stdev = min_noise_stdev
		self.initial_max_noise_stdev = initial_max_noise_stdev
		self.final_max_noise_stdev = final_max_noise_stdev
		
		# label smooth and noise cycle length (oscillate with phase shift of pi; in sync)
		self.lbl_smooth_noise_cycle_length = lbl_smooth_noise_cycle_length
		
		# input one-hot injection
		self.min_one_hot_injection_mean = min_one_hot_injection_mean
		self.initial_max_one_hot_injection_mean = initial_max_one_hot_injection_mean
		self.final_max_one_hot_injection_mean = final_max_one_hot_injection_mean
		
		self.one_hot_injection_stdev = one_hot_injection_stdev
		
		# cycle length of one-hot injection
		self.one_hot_injection_cycle_length = one_hot_injection_cycle_length


	def calculate_stage(self, epoch, cycle_length, min_value, max_value, phase_shift=0):
		"""
		Calculate a stage value oscillating between min_value and max_value
		based on the current epoch, total epochs, and frequency.

		Args:
			epoch (int): The current epoch number (0-indexed).
			total_epochs (int): The total number of epochs.
			frequency (float): The frequency of the oscillation (number of cycles over all epochs).
			min_value (float): The minimum value of the stage.
			max_value (float): The maximum value of the stage.

		Returns:
			float: The calculated stage value for the current epoch.
		"""
		# Calculate the amplitude (half of the difference between max and min)
		amplitude = (max_value - min_value) / 2
		
		# Calculate the midpoint (average of max and min)
		midpoint = (max_value + min_value) / 2
		
		# Calculate the stage value using a sinusoidal function
		stage_value = midpoint + amplitude * math.sin(2 * math.pi * (1/cycle_length) * epoch + phase_shift)
		
		return stage_value

	def get_onehot_params(self, epoch):
		'''
		get the mean and standard deviations of one-hot injection depending on the stage

		Args:
			initial_max_one_hot_injection_mean (float):
			final_max_one_hot_injection_mean (float):
			min_one_hot_injection_mean (float):
			one_hot_injection_stdev (float):
			stage (float):
			one_hot_injection_cycle_length (float):

		Returns:
			mean_onehot (float):
			stdev_onehot (float):

		'''

		# compute one-hot injection means and stdevs
		decayed_max_onehot = self.initial_max_one_hot_injection_mean + epoch.stage * (self.final_max_one_hot_injection_mean - self.initial_max_one_hot_injection_mean) # slowsly increase the max one-hot injection, so the model first learns mostly from raw environments, and progressively gets introduced to AA modulated envs
		mean_onehot = self.calculate_stage(epoch.epoch, self.one_hot_injection_cycle_length, self.min_one_hot_injection_mean, decayed_max_onehot, phase_shift=-math.pi/2) 	# start with low one hot injection, to give the model a sense of the raw environments without AA modulation
		stdev_onehot = self.one_hot_injection_stdev

		return mean_onehot, stdev_onehot

	def get_input_perturbations(self, epoch):

		mean_onehot, stdev_onehot = self.get_onehot_params(epoch)
		mean_lbl_smooth, stdev_lbl_smooth = self.get_lbl_smooth_params(epoch)
		stdev_noise = self.get_noise_params(epoch)

		input_perturbations = InputPerturbations(mean_onehot, stdev_onehot, stdev_noise,
												mean_lbl_smooth, stdev_lbl_smooth,
												)

		return input_perturbations


class InputPerturbations():

	def __init__(self,  MASK_injection_mean, MASK_injection_stdev):
		
		self.onehot_injection_mean = onehot_injection_mean
		self.onehot_injection_stdev = onehot_injection_stdev 

		self.apply_onehot = None in [onehot_injection_mean, onehot_injection_stdev]
		self.apply_lbl_smooth = None in [lbl_smooth_mean, lbl_smooth_stdev]

	def MASK_injection(self, prediction, labels, key_padding_mask, max_pct=0.90):
		'''
		injects one hot labels into a prediction batch. each sample within a batch gets
		assigned a percent value of label smoothing, e.g. if the assigned value is 0.5,
		half of the tokens will be one hot encoded with the correct label. each sample's
		percentage of one-hot tokens is sampled from a gaussian distribution, with the
		specified mean and stdev, both of which are constant for a given batch

		Args:
			prediction (torch.Tensor): 		prediction batch, either smoothed and noised labels
											or noised uniform distribution. batch x N x 20
			labels (torch.Tensor): 			label batch, contains correct classes for each position within a batch
											batch x N
			mean_onehot (float):			mean of the gaussian distribution to be sampled from
			stdev_onehot (float):			stdev of the gaussian distribution to be sampled from
			max_pct (float):				the maximum percentage of positions to one-hot label, since we do not want all
											positions to be one-hot encoded

		Returns:
			one_hot_pred (torch.Tensor):	prediction tensor with a percentage of the positions 
											(depending on the percentage sampled on a per batch basis) 
											being one hot encoded
			onehot_mask (torch.Tensor): 	mask containing True for positions with one-hot vectors, and invalid positions
											Valid positions that have not been one-hot labeled (those that the loss will
											be computed for) are False 

		'''

		batch, N, num_classes = prediction.shape
		valid_pos = ~key_padding_mask

		# get a percentage of one-hot labeling for each sample in the batch, sampled from gaussian distribution on per sample basis 
		# batch, 
		one_hot_pct = torch.clamp(
			torch.normal(self.onehot_injection_mean, self.onehot_injection_stdev, size=(batch,), device=labels.device),
			min=0.01, max=max_pct
		)

		# select positions to one-hot label
		# need to make sure that the percentage is not of N, but of valid positions for each sample

		# neat trick: multiply valid positions (boolean tensor) by random values between 0 and 1. False (non-valid) will be zero
		# and True will be non-zero. subtract this from one so non-valid positions are 1 and valid are less than 1. 
		# then select positions that are less than the one_hot_pct threshold at corresponding sample to one-hot label
		# non-valid positions will never be less than 1 (they equal 1), and valid positions will be sampled if they are less than
		# the corresponding percentage, effectively getting that percent of valid positions specified from one_hot_pct
		random_vals = 1 - (valid_pos * torch.rand(valid_pos.shape, device=valid_pos.device)) # batch x N
		onehot_mask = random_vals < one_hot_pct.unsqueeze(-1) # true for positions to add one-hot label ; batch x N

		# ensure at least one position per sample is not one-hot labeled
		# true for samples with all one-hot or masked
		all_one_hot = ~torch.any(~(onehot_mask | key_padding_mask), dim=-1) # batch x N --> batch,

		# for samples with all one-hot vectors, randomly choose one valid position to leave not one-hot encoded
		# first need to redetermine valid positions for those samples
		all_one_hot_and_valid = valid_pos & all_one_hot.unsqueeze(-1) # batch x N

		# multiply valid positions with all one hot by rand number and get the maximum value (may be more than one value, which is ok)
		all_one_hot_and_valid = all_one_hot_and_valid * torch.rand(all_one_hot_and_valid.shape, device=all_one_hot_and_valid.device) # batch x N
		no_one_hot = (all_one_hot_and_valid != 0) & (all_one_hot_and_valid == torch.max(all_one_hot_and_valid, dim=-1).values.unsqueeze(-1))  # batch x N
		
		# set the selected position(s) in one hot mask to false, so that position is not one hot encoded
		onehot_mask = torch.where(no_one_hot, False, onehot_mask)

		# define the one-hot labels
		# class can't be -1, set to zero then mask it later
		labels = torch.where(key_padding_mask, 0.0, labels).long()
		one_hot_labels = F.one_hot(labels, num_classes=num_classes).float() # batch x N x 20
		one_hot_labels = torch.where(key_padding_mask.unsqueeze(-1), 0.0, one_hot_labels)

		# prediction
		one_hot_pred = torch.where(onehot_mask.unsqueeze(-1), one_hot_labels, prediction)

		return one_hot_pred, onehot_mask

	def apply_perturbations(self, batch):
		
		if self.apply_onehot:
			batch.predictions, batch.onehot_mask = self.one_hot_injection(batch.predictions, batch.labels, batch.key_padding_mask)
		# make a percentage of the masked positions have an incorrect class
		# hard code it for testing, lets say 
		MASK_tokens = batch.predictions[:, :, 20] == 1 # Z x N
		selected_MASK_tokens = MASK_tokens & (torch.rand(MASK_tokens.shape) < 0.15) # Z x N
		rand_tokens = F.one_hot(torch.randint(0,20, MASK_tokens.shape), num_classes=21) # Z x N x 21
		batch.predictions = torch.where(selected_MASK_tokens.unsqueeze(2),rand_tokens, batch.predictions)

class HyperParameters():

	def __init__(self, 	d_model,
						min_wl, max_wl, base_wl, 
						d_hidden_wl, hidden_layers_wl, 
						d_hidden_aa, hidden_layers_aa,
						dualcoder_layers, num_heads,
						min_spread, max_spread, base_spread,
						min_rbf, max_rbf, 
						d_hidden_attn, hidden_layers_attn, 
						temperature, use_model ):
		self.d_model = d_model
		self.min_wl = min_wl
		self.max_wl = max_wl
		self.base_wl = base_wl 
		self.d_hidden_wl = d_hidden_wl
		self.hidden_layers_wl = hidden_layers_wl 
		self.d_hidden_aa = d_hidden_aa
		self.hidden_layers_aa = hidden_layers_aa
		self.dualcoder_layers = dualcoder_layers
		self.num_heads = num_heads
		self.min_spread = min_spread
		self.max_spread = max_spread 
		self.base_spread = base_spread 
		self.min_rbf = min_rbf
		self.max_rbf = max_rbf 
		self.d_hidden_attn = d_hidden_attn
		self.hidden_layers_attn = hidden_layers_attn 
		self.temperature = temperature
		self.use_model = use_model

class TrainingParameters():

	def __init__(self, 	epochs, batch_sizes, seq_sizes, batch_tokens, 
						accumulation_steps, learning_step, 
						lr_type, lr_initial_min, lr_initial_max, lr_final_min, lr_final_max, lr_cycle_length,
						beta1, beta2, epsilon, 
						dropout, label_smoothing, include_ncaa, 
						loss_type, loss_sum_norm, lr_scale, lr_patience, 
						use_amp, use_chain_mask
				):
		self.epochs = epochs
		self.batch_sizes = batch_sizes
		self.seq_sizes = seq_sizes
		self.batch_tokens = batch_tokens
		self.accumulation_steps = accumulation_steps
		self.learning_step = learning_step
		self.lr_type = lr_type 
		self.lr_initial_min = lr_initial_min
		self.lr_initial_max = lr_initial_max
		self.lr_final_min = lr_final_min
		self.lr_final_max = lr_final_max
		self.lr_cycle_length = lr_cycle_length
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.dropout = dropout
		self.label_smoothing = label_smoothing
		self.include_ncaa = include_ncaa
		self.loss_type = loss_type
		self.loss_sum_norm = loss_sum_norm
		self.lr_scale = lr_scale
		self.lr_patience = lr_patience
		self.use_amp = use_amp
		self.use_chain_mask = use_chain_mask

# ----------------------------------------------------------------------------------------------------------------------
