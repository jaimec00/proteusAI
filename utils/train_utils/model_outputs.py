
import torch

# ----------------------------------------------------------------------------------------------------------------------
# outputs

class ModelOutputs():
	'''
	'''
	def __init__(self, batch_parent, seq_pred):
		
		# batch parent
		self.batch_parent = batch_parent 

		# predictions
		self.seq_pred = seq_pred # seq pred from wf output

		# valid tokens for averaging
		self.valid_toks = (batch_parent.labels!=-1).sum().item()

	def get_losses(self):
		losses = self.batch_parent.epoch_parent.training_run_parent.losses.loss_function(self.seq_pred, self.batch_parent.labels, inference=self.batch_parent.inference)
		self.batch_parent.epoch_parent.training_run_parent.losses.tmp.add_losses(*losses, valid_toks=self.valid_toks)
		self.batch_parent.epoch_parent.training_run_parent.toks_processed += self.valid_toks

