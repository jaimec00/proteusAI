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

	def __init__(self, out_path, model_checkpoints=10, rank=0, world_size=1):

		self.rank = rank # deal with the loggin logic here
		self.world_size = world_size

		self.out_path = Path(out_path)
		self.out_path.mkdir(parents=True, exist_ok=True)

		self.cel_plot = self.out_path / Path("cel_plot.png")
		self.acc_plot = self.out_path / Path("acc_plot.png")
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

		total parameters: {training_parameters.num_params:,}
		
		model hyper-parameters:
			d_model: {hyper_parameters.d_model}
			topk: {hyper_parameters.topk}
			layers: {hyper_parameters.layers}
			node_embedding:
				min_wl: {hyper_parameters.node_embedding.min_wl}
				max_wl: {hyper_parameters.node_embedding.max_wl}
				base_wl: {hyper_parameters.node_embedding.base_wl}
				learn_wl: {hyper_parameters.node_embedding.learn_wl}
			edge_embedding:
				min_rbf: {hyper_parameters.edge_embedding.min_rbf}
				max_rbf: {hyper_parameters.edge_embedding.max_rbf}
				num_rbfs: {hyper_parameters.edge_embedding.num_rbfs}
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
			rng: {training_parameters.rng}
			checkpoint:
				checkpoint_path: {training_parameters.checkpoint.path}
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
				noise_coords_std: {training_parameters.regularization.noise_coords_std}
				use_chain_mask: {training_parameters.regularization.use_chain_mask}
				homo_thresh: {training_parameters.regularization.homo_thresh}
				label_smoothing: {training_parameters.regularization.label_smoothing}
			loss:
				accumulation_steps: {training_parameters.loss.accumulation_steps} {"tokens" if training_parameters.loss.token_based_step else "batches"} 
				grad_clip_norm: {training_parameters.loss.grad_clip_norm}
			lr:
				lr_type: {training_parameters.lr.lr_type}
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

		cel, seq_sim1, seq_sim3, seq_sim5, probs = losses.tmp.get_avg()
		self.log.info(f"{mode} cross entropy loss per token: {str(cel)}")
		self.log.info(f"{mode} top 1 accuracy per token: {str(seq_sim1)}")	
		self.log.info(f"{mode} top 3 accuracy per token: {str(seq_sim3)}")	
		self.log.info(f"{mode} top 5 accuracy per token: {str(seq_sim5)}")	
		self.log.info(f"{mode} true aa predicted likelihood per token: {str(probs)}\n")	
		tmp_losses.extend([cel, seq_sim1, seq_sim3, seq_sim5, probs])
		
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

	def plot_training(self, losses):

		# convert to numpy arrays
		losses.to_numpy()

		epochs = [i + 1 for i in range(len(losses.train))]

		# plot cel
		plt.plot(epochs, losses.train.cel, marker='o', color='red', label="Training")
		plt.plot(epochs, losses.val.cel, marker='o', color='blue', label="Validation")

		plt.title('Cross Entropy Loss vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Cross Entropy Loss')
		plt.legend()
		plt.grid(True)
		plt.savefig(self.cel_plot)
		self.log.info(f"plot of cross entropy loss vs. epochs saved to {self.cel_plot}")

		plt.figure()

		# plot accuracy
		plt.plot(epochs, losses.train.matches1, marker='o', color='red', label="Training (Top 1)")
		plt.plot(epochs, losses.train.matches3, marker='x', color='red', label="Training (Top 3)")
		plt.plot(epochs, losses.train.matches5, marker='^', color='red', label="Training (Top 5)")
		plt.plot(epochs, losses.train.probs, marker='v', color='red', label="Training (Predicted Likelihood of True AA)")
		plt.plot(epochs, losses.val.matches1, marker='o', color='blue', label="Validation (Top 1)")
		plt.plot(epochs, losses.val.matches3, marker='x', color='blue', label="Validation (Top 3)")
		plt.plot(epochs, losses.val.matches5, marker='^', color='blue', label="Validation (Top 5)")
		plt.plot(epochs, losses.val.probs, marker='v', color='blue', label="Validation (Predicted Likelihood of True AA)")
		
		plt.title('Accuracy vs. Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend(fontsize="x-small")
		plt.grid(True)
		plt.savefig(self.acc_plot)
		self.log.info(f"plot of accuracy vs. epochs saved to {self.acc_plot}")

	def save_checkpoint(self, model, adam=None, scheduler=None, appended_str=""):

		checkpoint = {	"model": model.module.state_dict(), 
						"adam": (None if adam is None else adam.state_dict()), 
						"scheduler": (None if scheduler is None else scheduler.state_dict())
					}
		checkpoint_path = self.out_path / Path(f"checkpoint_{appended_str}.pth")
		torch.save(checkpoint, checkpoint_path)
		self.log.info(f"checkpoint saved to {checkpoint_path}")

# ----------------------------------------------------------------------------------------------------------------------
