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

	def __init__(self, 	out_path, 
						cel_plot="cel_plot.png", seq_plot="seq_sim_plot.png", 
						mse_plot="mse_plot.png", kldiv_plot="kldiv_plot.png", 
						vae_plot="vae_plot.png",
						test_plot="test_plot.png", 
						weights_path=Path("model_parameters.pth"), model_checkpoints=10
					):

		self.out_path = Path(out_path)
		self.out_path.mkdir(parents=True, exist_ok=True)

		self.cel_plot = self.out_path / Path(cel_plot)
		self.mse_plot = self.out_path / Path(mse_plot)
		self.kldiv_plot = self.out_path / Path(kldiv_plot)
		self.seq_plot = self.out_path / Path(seq_plot)
		self.vae_plot = self.out_path / Path(vae_plot)
		self.test_plot = self.out_path / Path(test_plot)
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

	def log_trainingrun(self, training_parameters, hyper_parameters, data):
		'''basically just prints the config file w/ a little more info'''

		log = 	textwrap.dedent(f'''

		wave function embedding parameters: {training_parameters.num_embedding_params:,}
		wave function encoding parameters: {training_parameters.num_encoding_params:,}
		wave function diffusion parameters: {training_parameters.num_diffusion_params:,}
		wave function decoding parameters: {training_parameters.num_decoding_params:,}
		wave function extraction parameters: {training_parameters.num_extraction_params:,}
		total parameters: {training_parameters.num_params:,}
		
		model hyper-parameters:
			d_model: {hyper_parameters.d_model} 
			d_latent: {hyper_parameters.d_latent} 
			N_latent: {hyper_parameters.N_latent} 
			num_aa: {hyper_parameters.num_aa}
			wave function embedding:
				none, all are learnable
			wave function encoding:
				wf preprocessing:
					d_hidden: {hyper_parameters.encoding.pre_process.d_hidden}
					hidden_layers: {hyper_parameters.encoding.pre_process.hidden_layers}
				wf post_process:
					d_hidden: {hyper_parameters.encoding.post_process.d_hidden}
					hidden_layers: {hyper_parameters.encoding.post_process.hidden_layers}
				encoders:
					self-attn:
						encoder_layers: {hyper_parameters.encoding.encoders.self_attn.layers}
						heads: {hyper_parameters.encoding.encoders.self_attn.heads}
						d_hidden_attn: {hyper_parameters.encoding.encoders.self_attn.d_hidden_attn}
						hidden_layers_attn: {hyper_parameters.encoding.encoders.self_attn.hidden_layers_attn}
					cross_attn:
						encoder_layers: {hyper_parameters.encoding.encoders.cross_attn.layers}
						heads: {hyper_parameters.encoding.encoders.cross_attn.heads}
						d_hidden_attn: {hyper_parameters.encoding.encoders.cross_attn.d_hidden_attn}
						hidden_layers_attn: {hyper_parameters.encoding.encoders.cross_attn.hidden_layers_attn}
			wave function diffusion:
				scheduler:
					alpha_bar_min: {hyper_parameters.diffusion.scheduler.alpha_bar_min}
					noise_schedule_type: {hyper_parameters.diffusion.scheduler.noise_schedule_type}
					t_max: {hyper_parameters.diffusion.scheduler.t_max}
				timestep:
					d_in: {hyper_parameters.diffusion.timestep.d_in}
					d_hidden: {hyper_parameters.diffusion.timestep.d_hidden}
					hidden_layers: {hyper_parameters.diffusion.timestep.hidden_layers} 
				wf post_process:
					d_hidden: {hyper_parameters.diffusion.post_process.d_hidden}
					hidden_layers: {hyper_parameters.diffusion.post_process.hidden_layers}
				encoders:
					encoder_layers: {hyper_parameters.diffusion.encoders.layers}
					heads: {hyper_parameters.diffusion.encoders.heads}
					d_hidden_attn: {hyper_parameters.diffusion.encoders.d_hidden_attn}
					hidden_layers_attn: {hyper_parameters.diffusion.encoders.hidden_layers_attn}
			wave function decoding:
				wf preprocessing:
					d_hidden: {hyper_parameters.decoding.pre_process.d_hidden}
					hidden_layers: {hyper_parameters.decoding.pre_process.hidden_layers}
				wf post_process:
					d_hidden: {hyper_parameters.decoding.post_process.d_hidden}
					hidden_layers: {hyper_parameters.decoding.post_process.hidden_layers}
				encoders:
					self-attn:
						encoder_layers: {hyper_parameters.decoding.encoders.self_attn.layers}
						heads: {hyper_parameters.decoding.encoders.self_attn.heads}
						d_hidden_attn: {hyper_parameters.decoding.encoders.self_attn.d_hidden_attn}
						hidden_layers_attn: {hyper_parameters.decoding.encoders.self_attn.hidden_layers_attn}
					cross-attn:
						encoder_layers: {hyper_parameters.decoding.encoders.cross_attn.layers}
						heads: {hyper_parameters.decoding.encoders.cross_attn.heads}
						d_hidden_attn: {hyper_parameters.decoding.encoders.cross_attn.d_hidden_attn}
						hidden_layers_attn: {hyper_parameters.decoding.encoders.cross_attn.hidden_layers_attn}
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
					d_hidden_attn: {hyper_parameters.extraction.encoders.d_hidden_attn}
					hidden_layers_attn: {hyper_parameters.extraction.encoders.hidden_layers_attn}

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
			effective batch size (tokens): {data.batch_tokens * training_parameters.loss.accumulation_steps:,}

		training-parameters:
			train_type: {training_parameters.train_type}
			epochs: {training_parameters.epochs}
			weights:
				use_model: {training_parameters.weights.use_model}
				use_embedding_weights: {training_parameters.weights.use_embedding_weights}
				use_encoding_weights: {training_parameters.weights.use_encoding_weights}
				use_diffusion_weights: {training_parameters.weights.use_diffusion_weights}
				use_decoding_weights: {training_parameters.weights.use_decoding_weights}
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
				noise_coords_std: {training_parameters.regularization.noise_coords_std}
				use_chain_mask: {training_parameters.regularization.use_chain_mask}
			loss:
				accumulation_steps: {training_parameters.loss.accumulation_steps} 
				label_smoothing: {training_parameters.loss.cel.label_smoothing}
				kl_div_beta: {training_parameters.loss.kl.beta}
				kl_div_kappa: {training_parameters.loss.kl.kappa}
				kl_div_midpoint: {training_parameters.loss.kl.midpoint}
				gamma: {training_parameters.loss.nll.gamma}
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

	def log_epoch_losses(self, losses, train_type):
		if train_type in ["extraction", "extraction_finetune", "old"]:
			cel, seq_sim = losses.tmp.get_avg()
			self.log.info(f"train cross entropy loss per token: {str(cel)}")
			self.log.info(f"train sequence similarity per token: {str(seq_sim)}\n")		
			losses.train.add_losses(cel, seq_sim)
		elif train_type == "vae":
			kl_div, reconstruction, loss = losses.tmp.get_avg()
			self.log.info(f"train kl divergence per token: {str(kl_div)}")
			self.log.info(f"train squared error per token: {str(reconstruction)}")
			self.log.info(f"train full loss per token: {str(loss)}\n")
			losses.train.add_losses(kl_div, reconstruction, loss)
		elif train_type == "diffusion":
			squared_error, nll, total_loss = losses.tmp.get_avg()
			self.log.info(f"train squared error per token: {str(squared_error)}")
			self.log.info(f"train negative log likelihood per token: {str(nll)}")
			self.log.info(f"train full loss per token: {str(total_loss)}\n")
			losses.train.add_losses(squared_error, nll, total_loss)

	def log_val_losses(self, losses, train_type):
		if train_type in ["extraction", "extraction_finetune", "old"]:
			cel, seq_sim = losses.tmp.get_avg()
			self.log.info(f"validation cross entropy loss per token: {str(cel)}")
			self.log.info(f"validation sequence similarity per token: {str(seq_sim)}\n")
			losses.val.add_losses(cel, seq_sim)
		elif train_type == "vae":
			kl_div, reconstruction, loss = losses.tmp.get_avg()
			self.log.info(f"validation kl divergence per token: {str(kl_div)}")
			self.log.info(f"validation squared error per token: {str(reconstruction)}")
			self.log.info(f"validation full loss per token: {str(loss)}\n")
			losses.val.add_losses(kl_div, reconstruction, loss)
		elif train_type == "diffusion":
			squared_error, nll, total_loss = losses.tmp.get_avg()
			self.log.info(f"validation squared error per token: {str(squared_error)}")
			self.log.info(f"validation negative log likelihood per token: {str(nll)}")
			self.log.info(f"validation full loss per token: {str(total_loss)}\n")
			losses.val.add_losses(squared_error, nll, total_loss)

	def log_test_losses(self, losses, train_type):
		if train_type in ["extraction", "extraction_finetune", "old"]:
			cel, seq_sim = losses.tmp.get_avg()
			self.log.info(f"test cross entropy loss per token: {str(cel)}")
			self.log.info(f"test sequence similarity per token: {str(seq_sim)}\n")
		elif train_type == "vae":
			kl_div, reconstruction, loss = losses.tmp.get_avg()
			self.log.info(f"test kl divergence per token: {str(kl_div)}")
			self.log.info(f"test squared error per token: {str(reconstruction)}")
			self.log.info(f"test full loss per token: {str(loss)}\n")
		elif train_type == "diffusion":
			seq_sim, _, _ = losses.tmp.get_avg(is_inference=True) # seq sims stored in squared errors
			self.log.info(f"test sequence similarity per token: {str(seq_sim)}\n")
		losses.test.extend_losses(losses.tmp) # include all test losses to make a histogram (per batch seq sims), not implemented

	def plot_training(self, losses, training_type):

		# convert to numpy arrays
		losses.to_numpy()

		epochs = [i + 1 for i in range(len(losses.train))]
		if training_type in ["extraction", "extraction_finetune", "old"]:
			self.plot_extraction(losses, epochs)
		elif training_type == "vae":
			self.plot_vae(losses, epochs)
		elif training_type == "diffusion":
			self.plot_diffusion(losses, epochs)

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

		plt.plot(epochs, losses.train.matches, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.matches, marker='o', color='blue', label="Validation")
		
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

	def plot_vae(self, losses, epochs):

		# Create the plot, only plotting final loss, will add functionality to plot kldiv, cel, and mse seperately later
		plt.plot(epochs, losses.train.all_losses, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.all_losses, marker='o', color='blue', label="Validation")

		# Adding title and labels
		plt.title('Loss vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.vae_plot)
		self.log.info(f"graph of loss vs. epoch saved to {self.vae_plot}")

		plt.figure()

		plt.plot(epochs, losses.train.reconstruction, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.reconstruction, marker='o', color='blue', label="Validation")
		
		# Adding title and labels
		plt.title('MSE vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('MSE')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.mse_plot)
		self.log.info(f"graph of mse vs. epoch saved to {self.mse_plot}")

		plt.figure()

		plt.plot(epochs, losses.train.kl_div, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.kl_div, marker='o', color='blue', label="Validation")
		
		# Adding title and labels
		plt.title('KLDiv vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('KLDiv')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.kldiv_plot)
		self.log.info(f"graph of kldiv vs. epoch saved to {self.kldiv_plot}")

	def plot_diffusion(self, losses, epochs):

		# Create the plot
		plt.plot(epochs, losses.train.squared_errors, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.squared_errors, marker='o', color='blue', label="Validation")

		# Adding title and labels
		plt.title('MSE vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('MSE')
		
		# Add a legend to distinguish the line plots
		plt.legend()
		
		# Display the plot
		plt.grid(True)
		plt.savefig(self.mse_plot)
		self.log.info(f"graph of mse vs. epoch saved to {self.mse_plot}")

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
		elif train_type == "vae":
			model.save_WFEncoding_weights(weights_path=weights_path("encoding"))
			model.save_WFDecoding_weights(weights_path=weights_path("decoding"))
		elif train_type == "diffusion":
			model.save_WFDiffusion_weights(weights_path=weights_path("diffusion"))

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
