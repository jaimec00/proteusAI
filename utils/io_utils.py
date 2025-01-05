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

	def log_hyperparameters(self, training_parameters, hyper_parameters, input_perturbation_parameters, data):

		noise_info = textwrap.dedent(f'''
		input perturbations:

			label-smoothing and noise cycle length: {input_perturbation_parameters.lbl_smooth_noise_cycle_length}

				initial minimum mean label-smoothing: {input_perturbation_parameters.initial_min_lbl_smooth_mean}
				final minimum mean label-smoothing: {input_perturbation_parameters.final_min_lbl_smooth_mean}
				maximum mean label-smoothing: {input_perturbation_parameters.max_lbl_smooth_mean}
				minimum stdev label-smoothing: {input_perturbation_parameters.min_lbl_smooth_stdev}
				maximum stdev label-smoothing: {input_perturbation_parameters.max_lbl_smooth_stdev}

				minimum stdev noise: {input_perturbation_parameters.min_noise_stdev}
				initial maximum stdev noise: {input_perturbation_parameters.initial_max_noise_stdev}
				final maximum stdev noise: {input_perturbation_parameters.final_max_noise_stdev}

		''')

		onehot_info = textwrap.dedent(f'''
		one-hot injection cycle length: {input_perturbation_parameters.one_hot_injection_cycle_length} epochs

			minimum one-hot injection mean: {input_perturbation_parameters.min_one_hot_injection_mean}
			initial maximum one-hot injection mean: {input_perturbation_parameters.initial_max_one_hot_injection_mean}
			final maximum one-hot injection mean: {input_perturbation_parameters.final_max_one_hot_injection_mean}
			one-hot injection stdev: {input_perturbation_parameters.one_hot_injection_stdev}
		''')


		if training_parameters.training_type in ["probs", "self-supervision"]:
			training_type = "training with probability distributions" + noise_info + onehot_info 
		elif training_parameters.training_type == "onehot":
			training_type = "training with one-hot AAs" + onehot_info
		elif training_parameters.training_type == "wf":
			training_type = "training with raw wave function features"
		else:
			raise ValueError(f"invalid training type: {training_parameters.training_type}")

		log = 	textwrap.dedent(f'''

		model hyper-parameters:
			model parameters: {training_parameters.num_params}
			d_model: {hyper_parameters.d_model}
			min_wl: {hyper_parameters.min_wl} 
			max_wl: {hyper_parameters.max_wl} 
			min_base: {hyper_parameters.min_base} 
			max_base: {hyper_parameters.max_base} 
			min_rbf: {hyper_parameters.min_rbf} 
			max_rbf: {hyper_parameters.max_rbf} 
			min_spread: {hyper_parameters.min_spread} 
			max_spread: {hyper_parameters.max_spread} 			
			number of decoders: {hyper_parameters.decoder_layers}
			number of attention heads: {hyper_parameters.num_heads}
			temperature: {hyper_parameters.temperature}
			max sequeunce length: {hyper_parameters.max_tokens}

			dataset split ({data.num_train + data.num_val + data.num_test} clusters total): 
				train clusters: {data.num_train}
				validation clusters: {data.num_val}
				test clusters: {data.num_test}
			batch size (tokens): {training_parameters.batch_size}
			possible batch sizes (samples): {training_parameters.batch_sizes}
			possible sequence lengths (tokens): {training_parameters.seq_sizes}
			effective batch size (tokens): {training_parameters.batch_size * training_parameters.accumulation_steps}

			epochs: {training_parameters.epochs}
			phase I: {round(training_parameters.epochs * training_parameters.phase_split)} epochs
			phase II: {round(training_parameters.epochs - (training_parameters.epochs * training_parameters.phase_split))} epochs

			learning rate: {training_parameters.learning_step}
			learning rate plateu scaling factor: {training_parameters.lr_scale}
			learning rate plateu patience: {training_parameters.lr_patience}
			dropout: {training_parameters.dropout}
			output label-smoothing: {training_parameters.label_smoothing}
			
			{training_type}

			output directory: {self.out_path}
		''')

		self.log.info(log)

	def log_epoch(self, epoch, optim, model, input_perturbations):

		for param_group in optim.param_groups:
			current_lr = param_group['lr']
	
		self.log.info(textwrap.dedent(f'''
		
			{'-'*80}
			epoch {epoch.epoch}: 
			{'-'*80}
			
			current learning rate: {current_lr}
	
			using {model.active_decoders} decoders

			decoder weights are {"frozen" if ((not epoch.phase) and (epoch.training_run_parent.training_parameters.training_type=="onehot")) else "not frozen"}

			training inputs contain:
				
				mean label smooth: {round(input_perturbations.lbl_smooth_mean if input_perturbations.lbl_smooth_mean is not None else 0.00, 2)}
				stdev label smooth: {round(input_perturbations.lbl_smooth_stdev if input_perturbations.lbl_smooth_stdev is not None else 0.00, 2)}

				mean noise: 0.00
				stdev noise: {round(input_perturbations.noise_stdev if input_perturbations.noise_stdev is not None else 0.00, 2)}

				mean one-hot injection: {round(input_perturbations.onehot_injection_mean if input_perturbations.onehot_injection_mean is not None else 0.00, 2)}
				stdev one-hot injection: {round(input_perturbations.onehot_injection_stdev if input_perturbations.onehot_injection_stdev is not None else 0.00, 2)}

				percent of self-supervised_batches: {round(input_perturbations.self_supervised_pct * 100, 2)}%

				''')
			)

	def log_epoch_losses(self, epoch, losses):

		input_loss, input_seq_sim = epoch.train_losses["input"].get_avg()
		self.log.info(f"train input loss: {str(input_loss.item())}")
		self.log.info(f"train input seq_sim: {str(input_seq_sim)}\n")		
		losses["input"].add_losses(input_loss, input_seq_sim)

		output_loss, output_seq_sim = epoch.train_losses["output"].get_avg()
		self.log.info(f"train output loss: {str(output_loss.item())}")
		self.log.info(f"train output seq_sim: {str(output_seq_sim)}\n")		
		losses["output"].add_losses(output_loss, output_seq_sim)

		delta_loss = output_loss - input_loss
		delta_seq_sim = output_seq_sim - input_seq_sim
		self.log.info(f"delta train loss: {str(delta_loss.item())}")
		self.log.info(f"delta train seq_sim: {str(delta_seq_sim)}\n")
		losses["delta"].add_losses(delta_loss, delta_seq_sim)

	def log_val_losses(self, losses):
		
		# compute epoch loss
		loss, seq_sim = losses.get_avg()
		self.log.info(f"validation loss: {str(loss)}")
		self.log.info(f"validation seq_sim: {str(seq_sim)}\n")


	def plot_training(self, train_losses, val_losses, use_probs=True):

		# Create the plot
		plt.plot([i + 1 for i in range(len(train_losses["output"].losses))], train_losses["output"].losses, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.losses))], val_losses.losses, marker='o', color='blue', label="Validation Output")
		if use_probs:
			plt.plot([i + 1 for i in range(len(train_losses["input"].losses))], losses["input"].losses, marker='o', color='yellow', label="Training Input")
			plt.plot([i + 1 for i in range(len(train_losses["delta"].losses))], losses["delta"].losses, marker='o', color='orange', label="Delta Training")
		
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

		plt.plot([i + 1 for i in range(len(train_losses["output"].seq_sims))], train_losses["output"].seq_sims, marker='o', color='red', label="Training Output")
		plt.plot([i + 1 for i in range(len(val_losses.seq_sims))], val_losses.seq_sims, marker='o', color='blue', label="Validation Output")

		if use_probs:
			plt.plot([i + 1 for i in range(len(train_losses["input"].seq_sims))], train_losses["input"].seq_sims, marker='o', color='yellow', label="Training Input")
			plt.plot([i + 1 for i in range(len(train_losses["delta"].seq_sims))], train_losses["delta"].seq_sims, marker='o', color='orange', label="Training Delta")


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
