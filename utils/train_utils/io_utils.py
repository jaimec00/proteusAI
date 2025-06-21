# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		io_utils.py
description:	utility classes for input/output operations during training 
'''
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import logging
import torch
import math
import sys

# ----------------------------------------------------------------------------------------------------------------------

class Output():

	def __init__(self, out_path, model_checkpoints=10, rank=0, world_size=1):\

		self.rank = rank # deal with the loggin logic here
		self.world_size = world_size

		self.out_path = Path(out_path)
		self.out_path.mkdir(parents=True, exist_ok=True)

		self.cel_plot = self.out_path / Path("cel_plot.png")
		self.seq_plot = self.out_path / Path("seq_plot.png")
		self.test_plot = self.out_path / Path("test_plot.png")
		self.weights_path = self.out_path / Path("model_parameters.pth")
		self.adam_path = self.out_path / Path("adam.pth")
		self.log = self.setup_logging(self.out_path / Path("log.txt"))
		self.model_checkpoints = model_checkpoints

	def setup_logging(self, log_file):

		logger = logging.getLogger("proteusAI_log")
		logger.setLevel(logging.DEBUG)

		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

		file_handler = logging.FileHandler(log_file, mode="w")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(logging.DEBUG)
		console_handler.setFormatter(formatter)

		logger.addHandler(file_handler)
		logger.addHandler(console_handler)

		return logger

	def log_trainingrun(self, training_parameters, hyper_parameters, data):
		'''basically just prints the config file w/ a little more info'''

		log = 	textwrap.dedent(f'''

		training the {"Ca only model" if training_parameters.ca_only_model else "full backbone model"}

		wave function embedding parameters: {training_parameters.num_embedding_params:,}
		wave function extraction parameters: {training_parameters.num_extraction_params:,}
		total parameters: {training_parameters.num_params:,}
		
		model hyper-parameters:

			d_model: {hyper_parameters.d_model} 
			d_wf: {hyper_parameters.d_wf} 
			num_aa: {hyper_parameters.num_aa}
		
			wave function embedding:
				min_wavelength: {hyper_parameters.embedding.min_wl}
				max_wavelength: {hyper_parameters.embedding.max_wl}
				base_wavelength: {hyper_parameters.embedding.base_wl}
				anisotropic: {hyper_parameters.embedding.anisotropic}
				learnable_wavelengths: {hyper_parameters.embedding.learn_wl}
				learnable_amino_acids: {hyper_parameters.embedding.learn_aa}
			
			wave function extraction:
				wf preprocessing:
					d_hidden: {hyper_parameters.extraction.pre_process.d_hidden}
					hidden_layers: {hyper_parameters.extraction.pre_process.hidden_layers}
				wf post_process:
					d_hidden: {hyper_parameters.extraction.post_process.d_hidden}
					hidden_layers: {hyper_parameters.extraction.post_process.hidden_layers}
				encoder:
					encoder_layers: {hyper_parameters.extraction.encoders.layers}
					heads: {hyper_parameters.extraction.encoders.heads}
					min_rbf: {hyper_parameters.extraction.encoders.min_rbf}
					d_hidden_attn: {hyper_parameters.extraction.encoders.d_hidden_attn}
					hidden_layers_attn: {hyper_parameters.extraction.encoders.hidden_layers_attn}
					use_bias: {hyper_parameters.extraction.encoders.use_bias}
					min_rbf (N/A if use_bias is False): {hyper_parameters.extraction.encoders.min_rbf}

		data:
  			data_path: {data.data_path}
			dataset split ({data.num_train + data.num_val + data.num_test:,} clusters total): 
				train clusters: {data.num_train:,}
				validation clusters: {data.num_val:,}
				test clusters: {data.num_test:,}
			batch size (tokens): {data.batch_tokens:,}
			max batch size (samples): {data.max_batch_size:,}
			min sequence length (tokens): {data.min_seq_size:,}
			max sequence length (tokens): {data.max_seq_size:,}
			effective batch size (tokens): {training_parameters.loss.accumulation_steps if training_parameters.loss.token_based_step else data.batch_tokens * training_parameters.loss.accumulation_steps}

		training-parameters:
			epochs: {training_parameters.epochs}
			weights:
				freeze embedding weigths after sequence similarity >= {training_parameters.weights.embedding.freeze_at_seq_sim}%
				turn geometric attention bias off to start: {training_parameters.weights.geo_attn.init_bias_off}
				turn geometric attention bias on after sequence similarity >= {training_parameters.weights.geo_attn.turn_bias_on_at_seq_sim}%
			checkpoint:
				checkpoint_path: {training_parameters.checkpoint.path}
				use_model: {training_parameters.checkpoint.use_model}
				use_embedding_weights: {training_parameters.checkpoint.use_embedding_weights}
				use_extraction_weights: {training_parameters.checkpoint.use_extraction_weights}
				use_adam: {training_parameters.checkpoint.use_adam}
				use_scheduler: {training_parameters.checkpoint.use_scheduler}
			inference:
				temperature: {training_parameters.inference.temperature}
			early_stopping:
				thresh: {training_parameters.early_stopping.thresh} 
				tolerance: {training_parameters.early_stopping.tolerance}
			adam:
				beta1: {training_parameters.adam.beta1}
				beta2: {training_parameters.adam.beta2}
				epsilon: {training_parameters.adam.epsilon}
			regularization:
				dropout: {training_parameters.regularization.dropout}
				noise_coords_std: {training_parameters.regularization.noise_coords_std}
				use_chain_mask: {training_parameters.regularization.use_chain_mask}
			loss:
				accumulation_steps: {training_parameters.loss.accumulation_steps} 
				label_smoothing: {training_parameters.loss.cel.label_smoothing}
				grad_clip_norm: {training_parameters.loss.grad_clip_norm}
			lr:
				lr_step: {training_parameters.lr.lr_step}
				warmup_steps: {training_parameters.lr.warmup_steps}
		
		output directory: {self.out_path}
		''')

		if self.rank==0:
			self.log.info(log)

	def log_epoch(self, epoch, step, current_lr):

		if self.rank==0:
			self.log.info(textwrap.dedent(f'''
			
				{'-'*80}
				epoch {epoch}, step {step:,}: 
				{'-'*80}
				
				current learning rate: {current_lr}
			''')
			)

	def log_losses(self, losses, mode):

		# workers pickle their loss objects, send to master, master extends the loss
		if self.rank == 0:
			loss_list = [None for _ in range(self.world_size)]
		else:
			loss_list = None


		torch.distributed.gather_object(
			obj=losses.tmp,
			object_gather_list=loss_list,
			dst=0,
			group=None
		)

		if self.rank!=0: 
			return # workers are done after gather

		for worker_loss in loss_list[1:]: # exclude the master loss object, as that is what we are extending
			losses.tmp.extend_losses(worker_loss) 

		tmp_losses = []

		cel, seq_sim1, seq_sim3, seq_sim5 = losses.tmp.get_avg()
		self.log.info(f"{mode} cross entropy loss per token: {str(cel)}")
		self.log.info(f"{mode} top1 accuracy per token: {str(seq_sim1)}")	
		self.log.info(f"{mode} top3 accuracy per token: {str(seq_sim3)}")	
		self.log.info(f"{mode} top5 accuracy per token: {str(seq_sim5)}\n")	
		tmp_losses.extend([cel, seq_sim1, seq_sim3, seq_sim5])
		
		if mode == "train":
			losses.train.add_losses(*tmp_losses)
		elif mode == "validation":	
			losses.val.add_losses(*tmp_losses)
		else: # testing
			losses.test.extend_losses(losses.tmp)


	def log_epoch_losses(self, losses):
		self.log_losses(losses, "train")

	def log_val_losses(self, losses):
		self.log_losses(losses, "validation")

	def log_test_losses(self, losses):
		self.log_losses(losses, "test")

	def plot_training(self, losses, training_type):

		# convert to numpy arrays
		losses.to_numpy()

		epochs = [i + 1 for i in range(len(losses.train))]
		self.plot_extraction(losses, epochs)

	def plot_extraction(self, losses, epochs):

		# Create the plot, only plotting final loss, will add functionality to plot kldiv, cel, and mse seperately later
		plt.plot(epochs, losses.train.cel, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.cel, marker='o', color='blue', label="Validation")

		# Adding title and labels
		plt.title('Loss vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Cross Entropy Loss')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.cel_plot)
		self.log.info(f"graph of loss vs. epoch saved to {self.cel_plot}")

		plt.figure()

		plt.plot(epochs, losses.train.matches1, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.train.matches3, marker='x', color='red', label="Training")
		plt.plot(epochs, losses.train.matches5, marker='-', color='red', label="Training")
		plt.plot(epochs, losses.val.matches1, marker='o', color='blue', label="Validation")
		plt.plot(epochs, losses.val.matches3, marker='x', color='blue', label="Validation")
		plt.plot(epochs, losses.val.matches5, marker='-', color='blue', label="Validation")
		
		# Adding title and labels
		plt.title('Sequence Similarity vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Sequence Similarity (%)')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.seq_plot)
		self.log.info(f"graph of seq_similarity vs. epoch saved to {self.seq_plot}")

	def plot_testing(self, losses): # not implemented

		losses.test.to_numpy()
		
		# Create a histogram with 30 bins and add a black edge to each bin for clarity
		plt.hist(losses.test.matches, bins=100, edgecolor='black')

		# Label the axes and add a title
		plt.xlabel('Sequence Similarity')
		plt.ylabel('Frequency')
		plt.title('Histogram of Sequence Similarities')
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.test_plot)
		self.log.info(f"histogram of test seq_sims saved to {self.test_plot}")

	def save_checkpoint(self, model, adam=None, scheduler=None, appended_str=""):

		checkpoint_path = self.weights_path.parent / Path(f"checkpoint_{appended_str}.pth")
		checkpoint = {"model": model.module.state_dict(), "adam": adam.state_dict(), "scheduler": scheduler.state_dict()}
		torch.save(checkpoint, checkpoint_path)
		self.log.info(f"checkpoint saved to {checkpoint_path}")

# ----------------------------------------------------------------------------------------------------------------------
