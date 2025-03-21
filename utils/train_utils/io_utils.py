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

	def __init__(self, out_path, loss_plot=Path("loss_plot.png"), seq_plot=Path("seq_sim_plot.png"), weights_path=Path("model_parameters.pth"), model_checkpoints=10):

		self.out_path = Path(out_path)
		self.out_path.mkdir(parents=True, exist_ok=True)
		self.loss_plot = self.out_path / Path(loss_plot)
		self.seq_plot = self.out_path / Path(seq_plot)
		self.weights_path = self.out_path / Path(weights_path)
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

	def log_hyperparameters(self, training_parameters, hyper_parameters, data):
		'''basically just prints the config file w/ a little more info'''

		log = 	textwrap.dedent(f'''

		wave function embedding parameters: {training_parameters.num_embedding_params:,}
		wave function diffusion parameters: {training_parameters.num_diffusion_params:,}
		wave function extraction parameters: {training_parameters.num_extraction_params:,}
		total parameters: {training_parameters.num_params:,}
		
		model hyper-parameters:

			d_model: {hyper_parameters.d_model} 
			num_aa: {hyper_parameters.num_aa}

			wave function embedding:
				min_wl: {hyper_parameters.embedding.min_wl}
				max_wl: {hyper_parameters.embedding.max_wl}
				base_wl: {hyper_parameters.embedding.base_wl}
				learnable_aa: {hyper_parameters.embedding.learnable_aa}

			wave function diffusion:
				scheduler:
					beta_min: {hyper_parameters.diffusion.scheduler.beta_min} 
					beta_max: {hyper_parameters.diffusion.scheduler.beta_max}
					beta_schedule_type: {hyper_parameters.diffusion.scheduler.beta_schedule_type}
					t_max: {hyper_parameters.diffusion.scheduler.t_max}
				timestep:
					min_wl: {hyper_parameters.diffusion.timestep.min_wl}
					max_wl: {hyper_parameters.diffusion.timestep.max_wl}
					use_mlp: {hyper_parameters.diffusion.timestep.use_mlp}
					d_hidden: {hyper_parameters.diffusion.timestep.d_hidden}
					hidden_layers: {hyper_parameters.diffusion.timestep.hidden_layers} 
					use_norm: {hyper_parameters.diffusion.timestep.use_norm}
				wf preprocessing:
					use_mlp: {hyper_parameters.diffusion.pre_process.use_mlp} 
					d_hidden: {hyper_parameters.diffusion.pre_process.d_hidden}
					hidden_layers: {hyper_parameters.diffusion.pre_process.hidden_layers}
					use_norm: {hyper_parameters.diffusion.pre_process.use_norm}
				wf post_process:
					use_mlp: {hyper_parameters.diffusion.post_process.use_mlp}
					d_hidden: {hyper_parameters.diffusion.post_process.d_hidden}
					hidden_layers: {hyper_parameters.diffusion.post_process.hidden_layers}
					use_norm: {hyper_parameters.diffusion.post_process.use_norm}
				encoders:
					encoder_layers: {hyper_parameters.diffusion.encoders.layers}
					heads: {hyper_parameters.diffusion.encoders.heads}
					learnable_spreads: {hyper_parameters.diffusion.encoders.learnable_spreads}
					min_spread: {hyper_parameters.diffusion.encoders.min_spread}
					max_spread: {hyper_parameters.diffusion.encoders.max_spread}
					base_spreads: {hyper_parameters.diffusion.encoders.base_spreads}
					num_spread: {hyper_parameters.diffusion.encoders.num_spread}
					min_rbf: {hyper_parameters.diffusion.encoders.min_rbf}
					max_rbf: {hyper_parameters.diffusion.encoders.max_rbf}
					beta: {hyper_parameters.diffusion.encoders.beta}
					d_hidden_attn: {hyper_parameters.diffusion.encoders.d_hidden_attn}
					hidden_layers_attn: {hyper_parameters.diffusion.encoders.hidden_layers_attn}

			wave function extraction:
				wf preprocessing:
					use_mlp: {hyper_parameters.extraction.pre_process.use_mlp} 
					d_hidden: {hyper_parameters.extraction.pre_process.d_hidden}
					hidden_layers: {hyper_parameters.extraction.pre_process.hidden_layers}
					use_norm: {hyper_parameters.extraction.pre_process.use_norm}
				wf post_process:
					use_mlp: {hyper_parameters.extraction.post_process.use_mlp}
					d_hidden: {hyper_parameters.extraction.post_process.d_hidden}
					hidden_layers: {hyper_parameters.extraction.post_process.hidden_layers}
					use_norm: {hyper_parameters.extraction.post_process.use_norm}
				encoder:
					encoder_layers: {hyper_parameters.extraction.encoders.layers}
					heads: {hyper_parameters.extraction.encoders.heads}
					learnable_spreads: {hyper_parameters.extraction.encoders.learnable_spreads}
					min_spread: {hyper_parameters.extraction.encoders.min_spread}
					max_spread: {hyper_parameters.extraction.encoders.max_spread}
					base_spreads: {hyper_parameters.extraction.encoders.base_spreads}
					num_spread: {hyper_parameters.extraction.encoders.num_spread}
					min_rbf: {hyper_parameters.extraction.encoders.min_rbf}
					max_rbf: {hyper_parameters.extraction.encoders.max_rbf}
					beta: {hyper_parameters.extraction.encoders.beta}
					d_hidden_attn: {hyper_parameters.extraction.encoders.d_hidden_attn}
					hidden_layers_attn: {hyper_parameters.extraction.encoders.hidden_layers_attn}

		data:
  			data_path: {data.data_path}
			dataset split ({data.num_train + data.num_val + data.num_test} clusters total): 
				train clusters: {data.num_train}
				validation clusters: {data.num_val}
				test clusters: {data.num_test}
			batch size (tokens): {data.batch_tokens}
			max batch size (samples): {data.max_batch_size}
			min sequence length (tokens): {data.min_seq_size}
			max sequence length (tokens): {data.max_seq_size}
			effective batch size (tokens): {data.batch_tokens * training_parameters.loss.accumulation_steps}

		training-parameters:
			train_type: {training_parameters.train_type}
			epochs: {training_parameters.epochs}
			weights:
				use_model: {training_parameters.weights.use_model}
				use_embedding_weights: {training_parameters.weights.use_embedding_weights}
				use_diffusion_weights: {training_parameters.weights.use_diffusion_weights}
				use_extraction_weights: {training_parameters.weights.use_extraction_weights}
			inference:
				temperature: {training_parameters.inference.temperature}
				cycles: {training_parameters.inference.cycles}
			early_stopping:
				thresh: {training_parameters.early_stopping.thresh} 
				tolerance: {training_parameters.early_stopping.tolerance}
			adam:
				beta1: {training_parameters.adam.beta1}
				beta2: {training_parameters.adam.beta2}
				epsilon: {training_parameters.adam.epsilon}
			regularization:
				dropout: {training_parameters.regularization.dropout}
				wf_dropout: {training_parameters.regularization.wf_dropout}
				attn_dropout: {training_parameters.regularization.attn_dropout}
				label_smoothing: {training_parameters.regularization.label_smoothing}
				noise_coords_std: {training_parameters.regularization.noise_coords_std}
				use_chain_mask: {training_parameters.regularization.use_chain_mask}
			loss:
				accumulation_steps: {training_parameters.loss.accumulation_steps} 
				grad_clip_norm: {training_parameters.loss.grad_clip_norm}
			lr:
				lr_step: {training_parameters.lr.lr_step}
				warmup_steps: {training_parameters.lr.warmup_steps}
		
		output directory: {self.out_path}
		''')

		self.log.info(log)

	def log_epoch(self, epoch, current_lr):

		self.log.info(textwrap.dedent(f'''
		
			{'-'*80}
			epoch {epoch}: 
			{'-'*80}
			
			current learning rate: {current_lr}
		''')
		)

	def log_epoch_losses(self, losses):
		loss, seq_sim = losses.tmp.get_avg()
		self.log.info(f"train loss per token: {str(loss.item())}")
		self.log.info(f"train seq_sim per token: {str(seq_sim.item())}\n")		
		losses.train.add_losses(loss, seq_sim)

	def log_val_losses(self, losses):
		loss, seq_sim = losses.tmp.get_avg()
		self.log.info(f"validation loss per token: {str(loss.item())}")
		self.log.info(f"validation seq_sim per token: {str(seq_sim.item())}\n")
		losses.val.add_losses(loss, seq_sim)

	def log_test_losses(self, losses):
		test_loss, test_seq_sim = losses.tmp.get_avg()
		self.log.info(f"testing loss per token: {test_loss.item()}")
		self.log.info(f"test sequence similarity per token: {test_seq_sim.item()}")
		losses.test.extend_losses(losses.tmp) # include all test losses to make a histogram

	def plot_training(self, losses, training_type):

		# convert to numpy arrays
		losses.to_numpy()

		if training_type in ["extraction", "extraction_denoised"]:
			self.plot_extraction(losses)
		elif training_type == "diffusion":
			self.plot_diffusion(losses)

	def plot_diffusion(self, losses):

		# Create the plot
		plt.plot([i + 1 for i in range(len(losses.train))], losses.train.losses, marker='o', color='red', label="Training")
		plt.plot([i + 1 for i in range(len(losses.val))], losses.val.losses, marker='o', color='blue', label="Validation")

		# Adding title and labels
		plt.title('Mean Squared Error vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Mean Squared Error')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.loss_plot)
		self.log.info(f"graph of loss vs. epoch saved to {self.loss_plot}")

	def plot_extraction(self, losses):

		# Create the plot
		plt.plot([i + 1 for i in range(len(losses.train))], losses.train.losses, marker='o', color='red', label="Training")
		plt.plot([i + 1 for i in range(len(losses.val))], losses.val.losses, marker='o', color='blue', label="Validation")

		# Adding title and labels
		plt.title('Cross Entropy Loss vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Cross Entropy Loss')
		
		# Adding a horizontal line at the base loss (2.99 for random AA sequence prediction)
		plt.axhline(y=2.99, color='r', linestyle='--', label="Base Loss (Random Prediction)")

		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.loss_plot)
		self.log.info(f"graph of loss vs. epoch saved to {self.loss_plot}")

		plt.figure()

		plt.plot([i + 1 for i in range(len(losses.train))], losses.train.matches, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(losses.val))], losses.val.matches, marker='o', color='blue', label="Validation Output (no context)")
		
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

	def save_model(self, model, train_type="", appended_str=""):
		if appended_str:
			weights_path = self.weights_path.parent / Path(f"{'.'.join(self.weights_path.name.split(".")[:-1])}_{appended_str}.{self.weights_path.name.split(".")[-1]}") 
		else:
			weights_path = self.weights_path
		torch.save(model.state_dict(), weights_path)
		self.log.info(f"weights saved to {weights_path}")

		weights_path = lambda train_type_str: self.weights_path.parent / Path(f"{'.'.join(self.weights_path.name.split(".")[:-1])}_{train_type_str}.{self.weights_path.name.split(".")[-1]}")
		if train_type == "extraction":
			model.save_WFEmbedding_weights(weights_path=weights_path("embedding"))
			model.save_WFExtraction_weights(weights_path=weights_path("extraction"))
		elif train_type == "diffusion":
			model.save_WFDiffusion_weights(weights_path=weights_path("diffusion"))
		elif train_type == "extraction_denoised":
			model.save_WFExtraction_weights(weights_path=weights_path("extraction"))

	def write_new_clusters(self, cluster_info, val_clusters, test_clusters):

		# seperate training, validation, and testing
		val_pdbs = cluster_info.loc[cluster_info.CLUSTER.isin(val_clusters), :]
		test_pdbs =cluster_info.loc[cluster_info.CLUSTER.isin(test_clusters), :]

		# get lists of unique clusters
		val_clusters = "\n".join(str(i) for i in val_pdbs.CLUSTER.unique().tolist())
		test_clusters = "\n".join(str(i) for i in test_pdbs.CLUSTER.unique().tolist())

		with    open(   self.out_path / Path("test_clusters.txt"),   "w") as t, \
				open(   self.out_path / Path("valid_clusters.txt"),  "w") as v:
				v.write(val_clusters)
				t.write(test_clusters)

		# save training pdbs
		cluster_info.to_csv(self.out_path / Path("list.csv"), index=False)

# ----------------------------------------------------------------------------------------------------------------------
