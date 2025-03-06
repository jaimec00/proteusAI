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

	def __init__(self, out_path, loss_plot=Path("loss_plot.png"), seq_plot=Path("seq_sim_plot.png"), weights_path=Path("model_parameters.pth"), model_checkpoints=20):

		out_path.mkdir(parents=True, exist_ok=True)
		self.out_path = out_path
		self.loss_plot = self.out_path / loss_plot
		self.seq_plot = self.out_path / seq_plot
		self.weights_path = self.out_path / weights_path
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

	def log_hyperparameters(self, training_parameters, hyper_parameters, MASK_injection, data):

		MASK_info = "" if not hyper_parameters.use_aa else textwrap.dedent(f'''
		mean_mask_pct: {MASK_injection.mean_mask_pct}
		std_mask_pct: {MASK_injection.std_mask_pct}
		min_mask_pct: {MASK_injection.min_mask_pct} 
		max_mask_pct: {MASK_injection.max_mask_pct} 
		mean_span: {MASK_injection.mean_span}
		std_span: {MASK_injection.std_span}
		''')

		wf_types = {0: "none", 1: "inverse", 2: "log2", 3: "sqrt"}

		log = 	textwrap.dedent(f'''

		model hyper-parameters:
			model parameters: {training_parameters.num_params:,}
			d_model: {hyper_parameters.d_model}
			
			structure weights are{" " if hyper_parameters.freeze_structure_weights else " NOT "}frozen
			sequence weights are{" " if hyper_parameters.freeze_sequence_weights else " NOT "}frozen
			structure encoder weights are{" " if hyper_parameters.cp_struct_enc_2_seq_enc else " NOT "}copied to sequence encoder
			
			using {"anisotropic" if hyper_parameters.anisotropic_wf else "isotropic"} wave function embedding
			wave function embedding normalization: {wf_types[hyper_parameters.wf_type]}
			learnable_wavelengths: {hyper_parameters.learnable_wavelengths}
			min_wl: {hyper_parameters.min_wl} 
			max_wl: {hyper_parameters.max_wl} 
			base_wl: {hyper_parameters.base_wl}
			d_hidden_we: {hyper_parameters.d_hidden_we}
			hidden_layers_we: {hyper_parameters.hidden_layers_we}

			d_hidden_aa: {hyper_parameters.d_hidden_aa}
			hidden_layers_aa: {hyper_parameters.hidden_layers_aa}
			ESM2 weights: {hyper_parameters.esm2_weights_path}
			learnable_ESM2_weights: {hyper_parameters.learnable_esm}

			number of structure encoders: {hyper_parameters.struct_encoder_layers}
			number of sequence encoders: {hyper_parameters.seq_encoder_layers}
			number of attention heads: {hyper_parameters.num_heads}
			learnable_spreads: {hyper_parameters.learnable_spreads}
			min_spread: {hyper_parameters.min_spread} 
			max_spread: {hyper_parameters.max_spread} 
			base_spread: {hyper_parameters.base_spread}
			min_rbf: {hyper_parameters.min_rbf} 
			max_rbf: {hyper_parameters.max_rbf}
			beta: {hyper_parameters.beta}
			d_hidden_attn: {hyper_parameters.d_hidden_attn}
			hidden_layers_attn: {hyper_parameters.hidden_layers_attn}
			temperature: {hyper_parameters.temperature}

			dataset split ({data.num_train + data.num_val + data.num_test} clusters total): 
				train clusters: {data.num_train}
				validation clusters: {data.num_val}
				test clusters: {data.num_test}
			batch size (tokens): {data.batch_tokens}
			max batch size (samples): {data.max_batch_size}
			min sequence length (tokens): {data.min_seq_size}
			max sequence length (tokens): {data.max_seq_size}
			effective batch size (tokens): {data.batch_tokens * training_parameters.accumulation_steps}

			epochs: {training_parameters.epochs}
			learning rate: {training_parameters.lr_step}
			learning rate plateau scaling factor: {training_parameters.lr_scale}
			learning rate plateau patience: {training_parameters.lr_patience}
			dropout: {training_parameters.dropout}
			dropout_attention: {training_parameters.attn_dropout}
			label-smoothing: {training_parameters.label_smoothing}
			coordinate_noise_stdev: {training_parameters.noise_coords_std} A
			
			{MASK_info}

			output directory: {self.out_path}
		''')

		self.log.info(log)

	def log_epoch(self, epoch, current_lr):

		self.log.info(textwrap.dedent(f'''
		
			{'-'*80}
			epoch {epoch.epoch}: 
			{'-'*80}
			
			current learning rate: {current_lr}
		''')
		)

	def log_epoch_losses(self, epoch, losses):

		loss, seq_sim = epoch.losses.get_avg()
		self.log.info(f"train loss per token: {str(loss)}")
		self.log.info(f"train seq_sim per token: {str(seq_sim)}\n")		
		losses.add_losses(loss, seq_sim)

	def log_val_losses(self, losses, all_losses):
		
		# compute epoch loss
		loss, seq_sim = losses.get_avg()
		self.log.info(f"validation loss per token: {str(loss)}")
		self.log.info(f"validation seq_sim per token: {str(seq_sim)}\n")
		all_losses.add_losses(loss, seq_sim)

	def log_test_losses(self, test_losses, test_ar_losses):
		test_loss, test_seq_sim = test_losses.get_avg()
		_, test_ar_seq_sim = test_ar_losses.get_avg()
		self.log.info(f"testing loss per token: {test_loss}")
		self.log.info(f"test sequence similarity per token: {test_seq_sim}")
		self.log.info(f"test auto-regressive sequence similarity per token: {test_ar_seq_sim}")
	
	def plot_training(self, train_losses, val_losses, val_losses_context):

		# convert to numpy arrays
		for losses in [train_losses, val_losses, val_losses_context]:
			if losses is not None:
				losses.to_numpy()

		# Create the plot
		plt.plot([i + 1 for i in range(len(train_losses.losses))], train_losses.losses, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.losses))], val_losses.losses, marker='o', color='blue', label="Validation Output (no context)")
		if val_losses_context is not None:
			plt.plot([i + 1 for i in range(len(val_losses_context.losses))], val_losses_context.losses, marker='o', color='orange', label="Validation Output (w/ context)")
		
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

		plt.plot([i + 1 for i in range(len(train_losses.matches))], train_losses.matches, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.matches))], val_losses.matches, marker='o', color='blue', label="Validation Output (no context)")
		if val_losses_context is not None:
			plt.plot([i + 1 for i in range(len(val_losses_context.matches))], val_losses_context.matches, marker='o', color='orange', label="Validation Output (w/ context)")
		
		# Adding title and labels
		plt.title('Mean Sequence Similarity vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Mean Sequence Similarity (%)')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.seq_plot)
		self.log.info(f"graph of seq_similarity vs. epoch saved to {self.seq_plot}")

	def save_model(self, model, appended_str=""):
		if appended_str:
			weights_path = self.weights_path.parent / Path(f"{'.'.join(self.weights_path.name.split(".")[:-1])}_{appended_str}.{self.weights_path.name.split(".")[-1]}") 
		else:
			weights_path = self.weights_path
		torch.save(model.state_dict(), weights_path)
		self.log.info(f"weights saved to {weights_path}")

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
