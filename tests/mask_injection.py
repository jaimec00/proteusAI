


import torch
import math

class MASK_injection():

	def __init__(self,  initial_min_MASK_injection_mean, initial_max_MASK_injection_mean,
						final_min_MASK_injection_mean, final_max_MASK_injection_mean, 
						MASK_injection_stdev, MASK_injection_cycle_length, 
						randAA_pct=0.0, trueAA_pct=0.0
					):

		self.initial_min_MASK_injection_mean = initial_min_MASK_injection_mean
		self.initial_max_MASK_injection_mean = initial_max_MASK_injection_mean 
		
		self.final_min_MASK_injection_mean = final_min_MASK_injection_mean
		self.final_max_MASK_injection_mean = final_max_MASK_injection_mean
		
		self.MASK_injection_stdev = MASK_injection_stdev
		
		self.MASK_injection_cycle_length = MASK_injection_cycle_length
		
		# of the MASK tokens, what pct to make random AAs and what percent the ground truth
		# makes sure the model also predicts tokens that already have predictions, might be 
		# useful to minimize the effect of early incorrect predictions by allowing the model to 
		# self correct in later stages during auto-regressive-like inference
		self.randAA_pct = randAA_pct
		self.trueAA_pct = trueAA_pct

	def calc_mean_MASK(self, epoch, stage):



		# compute one-hot injection means and stdevs
		min_mean = self.initial_min_MASK_injection_mean + stage * (self.final_min_MASK_injection_mean - self.initial_min_MASK_injection_mean)
		max_mean = self.initial_max_MASK_injection_mean + stage * (self.final_max_MASK_injection_mean - self.initial_max_MASK_injection_mean)
		mean_MASK = self.calculate_stage(epoch, self.MASK_injection_cycle_length, min_mean, max_mean, phase_shift=-math.pi/2) # start with low MASK injection, to give the model as much context as possible to start, only allowed to predict a few positions and slowly increase difficulty

		# for easy logging
		self.MASK_injection_mean = mean_MASK

		return mean_MASK
		
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

	def inject_MASK(self, prediction, key_padding_mask, MASK_injection_mean, MASK_injection_stdev, min_pct=0.01):
		'''
		initial predictions are assumed to be one hot tensor of ground truth labels
		'''

		batch, N, num_classes = prediction.shape
		valid_pos = ~key_padding_mask

		# sampled from gaussian distribution on per sample basis 
		# do min pct to increase the probability that at least a few tookens are MASK, but will explicitly 
		# add MASK tokens to samples with no MASK token later anyways
		MASK_pct = torch.clamp(
			torch.normal(MASK_injection_mean, MASK_injection_stdev, size=(batch,), device=prediction.device),
			min=min_pct, max=1.0
		)

		# select positions to apply MASK token
		# need to make sure that the percentage is not of N, but of valid positions for each sample

		# neat trick: multiply valid positions (boolean tensor) by random values between 0 and 1. False (non-valid) will be zero
		# and True will be non-zero. subtract this from one so non-valid positions are 1 and valid are less than 1. 
		# then select positions that are less than the one_hot_pct threshold at corresponding sample to one-hot label
		# non-valid positions will never be less than 1 (they equal 1), and valid positions will be sampled if they are less than
		# the corresponding percentage, effectively getting that percent of valid positions specified from MASK_pct
		random_vals = 1 - (valid_pos * torch.rand(valid_pos.shape, device=valid_pos.device)) # batch x N
		MASK_mask = random_vals < MASK_pct.unsqueeze(1) # true for positions to add MASK token to ; batch x N

		# ensure at least one position per sample is MASK labeled
		# true for samples with all one-hot or masked
		no_MASK = ~torch.any(MASK_mask & valid_pos, dim=1) # batch x N --> batch,

		# for samples with no MASK tokens, randomly choose one valid position to MASK
		# first need to redetermine valid positions for those samples
		no_MASK_and_valid = valid_pos & no_MASK.unsqueeze(-1) # batch x N

		# multiply valid positions with all one hot by rand number and get the maximum value (may be more than one value, which is ok)
		no_MASK_and_valid = no_MASK_and_valid * torch.rand(no_MASK_and_valid.shape, device=no_MASK_and_valid.device) # batch x N
		add_MASK = (no_MASK_and_valid != 0) & (no_MASK_and_valid == torch.max(no_MASK_and_valid, dim=-1).values.unsqueeze(-1))  # batch x N
		
		# set the selected position(s) in MASK_mask to True, so that position is MASK labeled
		MASK_mask = torch.where(add_MASK, True, MASK_mask)

		# define the one-hot labels
		# class can't be -1, set to zero then mask it later

		# define a tensor for the MASK token, where idx 20 is 1
		MASK_tokens = torch.zeros(prediction.shape, device=prediction.device)
		MASK_tokens[:, :, 20] = 1

		# inject the MASK token
		prediction = torch.where(MASK_mask.unsqueeze(2), MASK_tokens, prediction)

		# invert the MASK_mask so 1 corresponds to non MASK positions
		return prediction, ~MASK_mask

	def MASK_tokens(self, predictions, mask):
		
		predictions, onehot_mask = self.inject_MASK(predictions, mask, self.MASK_injection_mean, self.MASK_injection_stdev)
		
		return predictions, onehot_mask

		# skip rand token injection for now
		# make a percentage of the masked positions have an incorrect class
		# hard code it for testing, lets say 
		# MASK_tokens = batch.predictions[:, :, 20] == 1 # Z x N
		# selected_MASK_tokens = MASK_tokens & (torch.rand(MASK_tokens.shape) < self.randAA_pct) # Z x N
		# rand_tokens = F.one_hot(torch.randint(0,20, MASK_tokens.shape), num_classes=21) # Z x N x 21
		# batch.predictions = torch.where(selected_MASK_tokens.unsqueeze(2),rand_tokens, batch.predictions)


def main():
	initial_min_MASK_injection_mean, initial_max_MASK_injection_mean = 0.0, 0.2
	final_min_MASK_injection_mean, final_max_MASK_injection_mean = 0.2, 0.4
	MASK_injection_stdev, MASK_injection_cycle_length = 0.05, 10.0
	randAA_pct, trueAA_pct = 0.0, 0.0

	mask_injection = MASK_injection(initial_min_MASK_injection_mean, initial_max_MASK_injection_mean,
									final_min_MASK_injection_mean, final_max_MASK_injection_mean, 
									MASK_injection_stdev, MASK_injection_cycle_length, 
									randAA_pct=0.0, trueAA_pct=0.0)

	batch, N, tokens = 1, 4, 21
	predictions = torch.zeros(batch, N, tokens)
	mask = torch.zeros(batch, N, dtype=torch.bool)

	epochs = 20

	for epoch in range(epochs):
		stage = epoch/epochs
		mask_injection.calc_mean_MASK(epoch, stage)

		print(mask_injection.MASK_injection_mean)

		new_pred, onehot = mask_injection.MASK_tokens(predictions, mask)

		print(new_pred, onehot)
		print()



if __name__ == "__main__":
	main() 