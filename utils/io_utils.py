# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		io_utils.py
description:	utility classes for input/output operations during training 
'''
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from torchviz import make_dot
from pathlib import Path
import textwrap
import logging
import torch
import math
import sys

# ----------------------------------------------------------------------------------------------------------------------

class Output():

	def __init__(self, out_path, loss_plot=Path("loss_plot.png"), seq_plot=Path("seq_sim_plot.png"), weights_path=Path("model_parameters.pth"), write_dot=False):

		out_path.mkdir(parents=True, exist_ok=True)
		self.out_path = out_path
		self.loss_plot = self.out_path / loss_plot
		self.seq_plot = self.out_path / seq_plot
		self.weights_path = self.out_path / weights_path
		self.write_dot = write_dot
		self.log = self.setup_logging(self.out_path / Path("log.txt"))

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

		MASK_info = textwrap.dedent(f'''
		MASK injection cycle length: {MASK_injection.MASK_injection_cycle_length} epochs

			initial minimum MASK injection mean: {MASK_injection.initial_min_MASK_injection_mean}
			initial maximum MASK injection mean: {MASK_injection.initial_max_MASK_injection_mean}
			final minimum MASK injection mean: {MASK_injection.final_min_MASK_injection_mean}
			final maximum MASK injection mean: {MASK_injection.final_max_MASK_injection_mean}
			MASK injection stdev: {MASK_injection.MASK_injection_stdev}
		''')

		log = 	textwrap.dedent(f'''

		model hyper-parameters:
			model parameters: {training_parameters.num_params}
			d_model: {hyper_parameters.d_model}
			min_wl: {hyper_parameters.min_wl} 
			max_wl: {hyper_parameters.max_wl} 
			base_wl: {hyper_parameters.base_wl}
			d_hidden_wf: {hyper_parameters.d_hidden_wl}
			hidden_layers_wf: {hyper_parameters.hidden_layers_wl}

			d_hidden_aa: {hyper_parameters.d_hidden_aa}
			hidden_layers_aa: {hyper_parameters.hidden_layers_aa}

			number of dualcoders: {hyper_parameters.dualcoder_layers}
			number of attention heads: {hyper_parameters.num_heads}
			min_spread: {hyper_parameters.min_spread} 
			max_spread: {hyper_parameters.max_spread} 
			base_spread: {hyper_parameters.base_spread}
			min_rbf: {hyper_parameters.min_rbf} 
			max_rbf: {hyper_parameters.max_rbf} 
			d_hidden_attn: {hyper_parameters.d_hidden_attn}
			hidden_layers_attn: {hyper_parameters.hidden_layers_attn}
			temperature: {hyper_parameters.temperature}

			max sequeunce length: {data.max_size}

			dataset split ({data.num_train + data.num_val + data.num_test} clusters total): 
				train clusters: {data.num_train}
				validation clusters: {data.num_val}
				test clusters: {data.num_test}
			batch size (tokens): {data.batch_tokens}
			max batch size (samples): {data.batch_size}
			min sequence length (tokens): {data.min_seq_size}
			max sequence length (tokens): {data.max_seq_size}
			effective batch size (tokens): {data.batch_tokens * training_parameters.accumulation_steps}

			epochs: {training_parameters.epochs}
			learning rate: {training_parameters.learning_step}
			learning rate plateau scaling factor: {training_parameters.lr_scale}
			learning rate plateau patience: {training_parameters.lr_patience}
			dropout: {training_parameters.dropout}
			label-smoothing: {training_parameters.label_smoothing}
			
			{MASK_info}

			output directory: {self.out_path}
		''')

		self.log.info(log)

	def log_epoch(self, epoch, optim, model, MASK_injection):

		for param_group in optim.param_groups:
			current_lr = param_group['lr']
	
		self.log.info(textwrap.dedent(f'''
		
			{'-'*80}
			epoch {epoch.epoch}: 
			{'-'*80}
			
			current learning rate: {current_lr}
	
			training inputs contain:
				
				mean MASK injection: {round(MASK_injection.MASK_injection_mean if MASK_injection.MASK_injection_mean is not None else 0.00, 2)}
				stdev MASK injection: {round(MASK_injection.MASK_injection_stdev if MASK_injection.MASK_injection_stdev is not None else 0.00, 2)}

				''')
			)

	def log_epoch_losses(self, epoch, losses):

		output_loss, output_seq_sim = epoch.train_losses.get_avg()
		self.log.info(f"train output loss: {str(output_loss.item())}")
		self.log.info(f"train output seq_sim: {str(output_seq_sim)}\n")		
		losses.add_losses(output_loss, output_seq_sim)

	def log_val_losses(self, losses):
		
		# compute epoch loss
		loss, seq_sim = losses.get_avg()
		self.log.info(f"validation loss: {str(loss)}")
		self.log.info(f"validation seq_sim: {str(seq_sim)}\n")

	def plot_training(self, train_losses, val_losses, val_losses_context):

		# convert to numpy arrays
		for losses in [train_losses, val_losses, val_losses_context]:
			for key in losses.keys():
				losses[key].to_numpy()

		# Create the plot
		plt.plot([i + 1 for i in range(len(train_losses.losses))], train_losses.losses, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.losses))], val_losses.losses, marker='o', color='blue', label="Validation Output (no context)")
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

		plt.plot([i + 1 for i in range(len(train_losses.seq_sims))], train_losses.seq_sims, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.seq_sims))], val_losses.seq_sims, marker='o', color='blue', label="Validation Output (no context)")
		plt.plot([i + 1 for i in range(len(val_losses_context.seq_sims))], val_losses_context.seq_sims, marker='o', color='orange', label="Validation Output (w/ context)")
		
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

	def write_dot(self, model, output):
		dot = make_dot(output, params=dict(model.named_parameters()))
		dot.render(self.dot_out, format="pdf")

	def save_model(self, model):
		torch.save(model.state_dict(), self.weights_path)
		self.log.info(f"weights saved to {self.weights_path}")

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

	def plot_aa_counts(self, aa_counts, file=Path("aa_hist.png")):

		aas = aa_counts.keys()
		counts = aa_counts.values()

		plt.bar(aas, counts)

		plt.xlabel('Amino Acids')
		plt.ylabel('Count')
		plt.title('Amino Acid Counts')

		plt.savefig(self.out_path / file)

		self.log.info(f"bar graph of aa counts saved to {self.out_path / file}")

		plt.figure()

	def plot_seq_len_hist(self, seq_lens, bins=20, file=Path("seq_len_hist.png")):
		
		plt.hist(seq_lens, bins=bins, edgecolor="black", alpha=0.7)

		plt.xlabel('Sequence Length')
		plt.ylabel('Count')
		plt.title('Sequence Length Histogram')

		plt.savefig(self.out_path / file)

		self.log.info(f"histogram of sequence lengths saved to {self.out_path / file}")


		plt.figure()
# ----------------------------------------------------------------------------------------------------------------------
