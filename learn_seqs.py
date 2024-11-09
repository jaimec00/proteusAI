# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		learn_seqs.py
description:	script to train, validate, and test the model
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import gc

from get_data import pt_to_data
from proteusAI import proteusAI
from utils import smooth_and_noise_labels, compute_cel_and_seq_sim, noise_uniform_distribution

# ----------------------------------------------------------------------------------------------------------------------

def main():

	print("parsing CL arguments...")
	args = init_args()

	print_log(args)

	print("initializing device...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	# setup the model
	model_items = setup_training(args, device)

	# train the model
	losses = train(model_items, args, device)

	# save weights
	args.out_path.mkdir(parents=True, exist_ok=True)
	torch.save(model_items["model"].state_dict(), args.out_path / args.weights_path)
	print(f"finished training. weights saved to {args.out_path / args.weights_path}")

	# plot the training + validation losses
	plot_training(losses, args)

	# test the model
	print(f"starting testing...")
	test(model_items, device, args.diffusion_cycles)

def print_log(args):
	log = 	f'''
model hyper-parameters:
	d_model: {args.d_model}
	number of attention heads: {args.num_heads}
	dataset split ({args.num_inputs} total): 
		train: {args.train_val} ({args.num_inputs * (args.train_val)})
		validation: {(1 - args.train_val) * args.val_test} ({args.num_inputs * ((1 - args.train_val) * args.val_test)})
		test: {(1 - args.train_val) * (1 - args.val_test)} ({args.num_inputs * ((1 - args.train_val) * (1 - args.val_test))})
	batch size: {args.batch_size}
	epochs: {args.epochs}
	learning rate: {args.learning_step}
	dropout: {args.dropout}
	output label smoothing: {args.label_smoothing}
	
	input minimum mean label smoothing: {args.min_lbl_smooth_mean}
	input maximum mean label smoothing: {args.max_lbl_smooth_mean}
	input minimum stdev label smoothing: {args.min_lbl_smooth_stdev}
	input maximum stdev label smoothing: {args.max_lbl_smooth_stdev}
	input minimum stdev noise: {args.min_noise_stdev}
	input maximum stdev noise: {args.max_noise_stdev}
	
	progressive learning phase: {int(args.epochs * args.progressive_learning_split)} epochs
	cycle length: {int(args.epochs * args.progressive_learning_split * args.cycles_split)} epochs
	real-world training phase: {int(args.epochs - (args.epochs * args.progressive_learning_split))} epochs

	output directory: {args.out_path}
'''
	print(log)

def init_args():

	parser = argparse.ArgumentParser()

	# hyper parameters
	parser.add_argument("--input_atoms", default=0, type=int, choices=[0,1,2], help="whether to train the model using only alphacarbons (0), full backbone (1), or both (2)")
	
	parser.add_argument("--d_model", default=512, type=int, help="dimensionality of input and output embeddings")
	parser.add_argument("--num_heads", default=4, type=int, help="number of attention heads to perform the training with")
	parser.add_argument("--encoder_layers", default=3, type=int, help="number of encoder layers")
	parser.add_argument("--decoder_layers", default=3, type=int, help="number of decoder layers")
	parser.add_argument("--hidden_linear_dim", default=1024, type=int, help="intermediate dimensions of feed forward layer")
	
	parser.add_argument("--train_val", default=3/4, type=float, help="how to split training and testing data")
	parser.add_argument("--val_test", default=1/2, type=float, help="how to split validation and testing data")
	parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
	parser.add_argument("--switch_batch", default=0.1, type=int, help="what fraction of all epochs to switch the batch up at")
	parser.add_argument("--batch_size", default=32, type=int, help="number of decoders")
	parser.add_argument("--learning_step", default=0.0001, type=float, help="learning step")
	parser.add_argument("--dropout", default=0.1, type=float, help="percentage of dropout")
	parser.add_argument("--label_smoothing", default=0.1, type=float, help="percentage of label smoothing to use on the labels")
	parser.add_argument("--num_inputs", default=4096, type=int, help="number of inputs to use")
	parser.add_argument("--loss_type", default="mean", type=str, choices=['sum', 'mean'], help="whether to use the 'sum' or the 'mean' for CEL")

	parser.add_argument("--min_lbl_smooth_mean", default=15/20, type=float, help="minimum input label smoothing")
	parser.add_argument("--max_lbl_smooth_mean", default=19/20, type=float, help="maximum input label smoothing")
	parser.add_argument("--min_lbl_smooth_stdev", default=0.025, type=float, help="minimum input label smoothing stdev")
	parser.add_argument("--max_lbl_smooth_stdev", default=0.25, type=float, help="maximum input label smoothing stdev")
	parser.add_argument("--min_noise_stdev", default=0.1, type=float, help="minimum standard deviation of noise to apply to inputs")
	parser.add_argument("--max_noise_stdev", default=0.2, type=float, help="maximum standard deviation of noise to apply to inputs")
	parser.add_argument("--final_stage_noise_stdev", default=0.1, type=float, help="standard deviation of noise to apply to inputs of UNIFORM distribution. "\
																					"this is for training on real world inputs. basically training the model "\
																					"to not trust the sequence probabilities when they are close to uniform, "\
																					"and instead focus on structure")

	parser.add_argument("--progressive_learning_split", default=9/10, type=float, help="ratio of progressive learning phase to real-world inputs (high label smoothing and noise)")
	parser.add_argument("--cycles_split", default=1/4, type=float, help="size of each cycle withing the progressive learning phase")
	parser.add_argument("--diffusion_cycles", default=10, type=int, help="at the end, during testing, try multiple diffusion cycles and record results")

	parser.add_argument("--out_path", default="output", type=Path, help="path to store output, such as plots and weights file.")

	parser.add_argument("--loss_plot", default="loss_vs_epoch.png", type=Path, help="path to save plot of loss vs epochs after training")
	parser.add_argument("--seq_plot", default="seq_sim_vs_epoch.png", type=Path, help="path to save plot of sequence similarity vs epochs after training")
	parser.add_argument("--weights_path", default="model_parameters.pth", type=Path, help="path to save weights after training")
	parser.add_argument("--write_dot", default=False, type=bool, help="whether to save the dot file of the computational graph")

	parser.add_argument("--data_path", default="/gpfs_backup/wangyy_data/protAI/pmpnn_data/pdb_2021aug02", type=Path, help="path to data")
	parser.add_argument("--max_load", default=8192, type=int, help="maximum number of pts to have in memory")
	parser.add_argument("--max_tokens", default=5000, type=int, help="maximum number of tokens")
	parser.add_argument("--config", default="config.yml", type=Path, help="Path to the YAML config file")

	args, _ = parser.parse_known_args()
	
	# Load YAML configuration if file exists
	if args.config.exists():
		with open(args.config, "r") as f:
			config = yaml.safe_load(f)
		parser.set_defaults(**config)

	args = parser.parse_args()

	return args


def setup_training(args, device):

	print("getting data...")

	# get pdb info, as well as validation and training clusters
	pdbs_info = pd.read_csv( args.data_path / Path("list.csv"), header=0)
	with 	open(	args.data_path / Path("test_clusters.txt"),   "r") as v, \
			open(   args.data_path / Path("valid_clusters.txt"),  "r") as t:
		val_clusters = [int(i) for i in v.read().split("\n") if i]
		test_clusters = [int(i) for i in t.read().split("\n") if i]

	# seperate training, validation, and testing
	train_pdbs = pdbs_info.loc[~pdbs_info.CLUSTER.isin(test_clusters + val_clusters), :]
	validation_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(val_clusters), :]
	test_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(test_clusters), :]
	
	del pdbs_info
	gc.collect()

	num_train = int(args.num_inputs * args.train_val)
	num_val = int(args.num_inputs * (1 - args.train_val) * args.val_test)
	num_test = int(args.num_inputs * (1 - args.train_val) * (1 - args.val_test))

	train_pdbs = train_pdbs.loc[train_pdbs.CLUSTER.isin(train_pdbs.CLUSTER.drop_duplicates().sample(n=num_train)), ["CHAINID", "CLUSTER", "SEQUENCE"]]
	val_pdbs = val_pdbs.loc[val_pdbs.CLUSTER.isin(val_pdbs.CLUSTER.drop_duplicates().sample(n=num_val)), ["CHAINID", "CLUSTER", "SEQUENCE"]]
	test_pdbs = test_pdbs.loc[test_pdbs.CLUSTER.isin(test_pdbs.CLUSTER.drop_duplicates().sample(n=num_test)), ["CHAINID", "CLUSTER", "SEQUENCE"]]

	N = args.max_tokens

	print("loading model...")
	model = proteusAI(N, args.d_model, 
					args.num_heads, 
					args.encoder_layers,
					args.decoder_layers,
					args.hidden_linear_dim, 
					args.dropout, args.use_wf)
	model.to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f"model contains {num_params} parameters")

	print("instantiating optimizer, scheduler and loss function...")
	optim = torch.optim.Adam(model.parameters(), lr=args.learning_step)

	# implementing label smoothing to downweight the effect of incorrect predictions, 
	# as the distribution will likely be wide, but the goal is for the most likely aa to be most probable, even if it is not as confident 
	# doing this because saw that even though the CEL was approaching randomness for validation, the sequence similarity continued to increase,
	# so the model was generalizing to some extent, but it was improperly "using" the loss function to penalize higher probabilities for incorrect labels
	loss_function = nn.CrossEntropyLoss(ignore_index=-1, reduction=args.loss_type, label_smoothing=args.label_smoothing)
	scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10) # no scheduling for now

	model_items = {
					"model": model, 
					"optim": optim, "loss_function": loss_function, "scheduler": scheduler, 
					"train_pdbs": train_pdbs, 
					"val_pdbs": val_pdbs,
					"test_pdbs": test_pdbs
				}

	return model_items

def train(model_items, args, device):

	# Create a DataLoader with desired batch size
	print("loading data...")
	train_clusters = model_items["train_pdbs"]
	train_data = Data(train_clusters, "cpu", args.data_path, max_size=args.max_tokens)
	train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

	val_clusters = model_items["val_pdbs"]
	val_data = Data(val_clusters, "cpu", args.data_path, max_size=args.max_tokens)
	val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

	# Now iterate over the DataLoader
	num_batches = int(len(train_data_loader))
	print(	f"\ninitializing training. "\
			f"training on {num_batches} "\
			f"batches of batch size {args.batch_size} for {args.epochs} epochs.\n"
		)
	
	# store losses
	losses = {
		"epoch_losses": [],
		"delta_epoch_losses": [],
		"epoch_seq_similarities": [],
		"delta_epoch_seq_similarities": [],
		"epoch_val_losses": [],
		"epoch_val_seq_similarities": []
	}

	scaler = GradScaler('cuda')

	# loop through epochs
	for epoch in range(args.epochs):

		if epoch % int(args.switch_batch * args.epochs) == 0: # cycle through new structures from the same clusters
			train_data = Data(train_clusters, "cpu", args.data_path, max_size=args.max_tokens)
			train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

		# make sure in training mode
		model_items["model"].train()

		# store batch losses
		batch_losses = {
			"batch_losses": [],
			"batch_seq_similarities": [],
			"delta_batch_losses": [],
			"delta_batch_seq_similarities": []
		}

		make_dot_graph = args.write_dot

		print("-"*65)
		print(f"epoch {epoch}: ")
		print("-"*65)

		for param_group in model_items["optim"].param_groups:
			current_lr = param_group['lr']
			print(f"current learning rate: {current_lr}\n")

		epoch_pbar = tqdm(total=num_batches, desc="epoch_progress", unit="step")

		learning_stage = (epoch / args.epochs) < args.progressive_learning_split

		# loop through batches
		for b_idx, (feature_batch, label_batch, key_padding_mask) in enumerate(train_data_loader):

			if learning_stage: # not final stage, where do multiple diffusion cycles
				# add noise to labels as input depending on the cycle
				cycle_stage = (b_idx % int(args.cycles_split * num_batches)) / int(args.cycles_split * num_batches)
				mean_lbl_smooth, stdev_lbl_smooth, stdev_noise = get_noise_params(cycle_stage, args)
				prediction_batch = smooth_and_noise_labels(label_batch, mean_lbl_smooth, stdev_lbl_smooth, stdev_noise, num_classes=20)
			else:
				cycle_stage = (args.epochs - epoch) / (args.epochs - args.epochs * args.progressive_learning_split)
				prediction_batch = torch.full([label_batch.size(0), label_batch.size(1), 20], 1/20)
				prediction_batch = noise_uniform_distribution(prediction_batch, key_padding_mask, stdev_noise=args.final_stage_noise_stdev)

			input_cel, input_seq_sim = compute_cel_and_seq_sim(label_batch, prediction_batch, num_classes=20)

			# forward pass on batch
			batch_loss, batch_seq_similarity = batch_step(	model_items["model"], model_items["loss_function"], 
															feature_batch, prediction_batch, label_batch, 
															key_padding_mask, device, 
															make_dot_graph=make_dot_graph, dot_out=str(args.out_path / Path("dot")), 
															use_amp=True, 
															diffusion_cycles=args.learning_stage * int(args.diffusion_cycles * cycle_stage)
														)

			make_dot_graph = False # only make the dot on the first batch, if specified

			# backward pass
			learn(model_items["optim"], batch_loss, scaler)

			# store losses
			avg_batch_loss = float(batch_loss.item()) #/ float(torch.sum(label_batch != -1))
			batch_losses["batch_losses"].append(avg_batch_loss)
			batch_losses["batch_seq_similarities"].append(batch_seq_similarity)
			delta_batch_loss =  avg_batch_loss - input_cel
			batch_losses["delta_batch_losses"].append(delta_batch_loss)
			delta_seq_sim = batch_seq_similarity - input_seq_sim
			batch_losses["delta_batch_seq_similarities"].append(delta_seq_sim)

			epoch_pbar.update(1)
		
		losses = print_batch_losses(batch_losses, losses)

		# validate learning
		losses = validation(model_items, val_data_loader, device, epoch, losses)

		model_items["scheduler"].step(losses["epoch_val_losses"][-1])

	return losses

def get_noise_params(cycle_progress, args):

	# Mean label smoothing for this cycle
	mean_lbl_smooth = args.min_lbl_smooth_mean + (args.max_lbl_smooth_mean - args.min_lbl_smooth_mean) * cycle_progress

	# Label smoothing standard deviation for this cycle
	stdev_lbl_smooth = args.max_lbl_smooth_stdev - (args.max_lbl_smooth_stdev - args.min_lbl_smooth_stdev) * cycle_progress

	# Noise standard deviation for the cycle 
	stdev_noise = args.max_noise_stdev - (args.max_noise_stdev - args.min_noise_stdev) * cycle_progress

	return mean_lbl_smooth, stdev_lbl_smooth, stdev_noise

def batch_step(model, loss_function, 
				feature_batch, prediction_batch, label_batch, key_padding_mask, 
				device, make_dot_graph=False, dot_out="dot", 
				use_amp=False, diffusion_cycles=0
				):

	feature_batch = feature_batch.to(device)
	prediction_batch = prediction_batch.to(device)
	label_batch = label_batch.to(device)
	key_padding_mask = key_padding_mask.to(device)

	if use_amp:
		with autocast('cuda'):
			# Pass protein_batch through your model
			output_prediction = model(feature_batch, prediction_batch, key_padding_mask, diffusion_cycles=diffusion_cycles)

			if make_dot_graph:
				write_dot(model, output_prediction, dot_out)

			# calculate loss
			output_prediction = output_prediction.permute(0, 2, 1)
			# label_batch = label_batch
			batch_loss = loss_function(output_prediction, label_batch)
	else:
		# Pass protein_batch through your model
		output_prediction = model(feature_batch, prediction_batch, key_padding_mask, diffusion_cycles=diffusion_cycles)

		if make_dot_graph:
			write_dot(model, output_prediction, dot_out)

		# calculate loss
		output_prediction = output_prediction.permute(0, 2, 1)
		# label_batch = label_batch
		batch_loss = loss_function(output_prediction, label_batch)

	# calculate sequence similarity via greedy selection. not used for back propogation, just for intuition
	with torch.no_grad():  # No gradients needed for this part
		norm_pred = F.softmax(output_prediction, dim=-1)
		seq_predictions = torch.argmax(norm_pred, dim=1)
		padding_mask = label_batch != -1
		valid_positions = padding_mask.sum(dim=-1)
		matches = ((seq_predictions == label_batch) & padding_mask).sum(dim=-1)
		batch_seq_similarity = ((matches / valid_positions).float()*100).mean()

	return batch_loss, batch_seq_similarity

def learn(optim, loss, scaler=None):

	optim.zero_grad()
	if scaler is not None:
		scaler.scale(loss).backward()
		scaler.step(optim)
		scaler.update()
	else:
		loss.backward()
		optim.step()

def print_batch_losses(batch_losses, losses):
	# store epoch losses
	epoch_loss = sum(batch_losses["batch_losses"]) / len(batch_losses["batch_losses"])
	print(f"train loss: {str(epoch_loss)}")
	losses["epoch_losses"].append(epoch_loss)

	# store epoch seq similarities
	epoch_seq_similarity = float(sum(batch_losses["batch_seq_similarities"]) / len(batch_losses["batch_seq_similarities"]))
	print(f"train seq_sim: {str(epoch_seq_similarity)}\n")
	losses["epoch_seq_similarities"].append(epoch_seq_similarity)

	# store delta epoch losses
	delta_epoch_loss = sum(batch_losses["delta_batch_losses"]) / len(batch_losses["delta_batch_losses"])
	print(f"delta train loss: {str(delta_epoch_loss)}")
	losses["delta_epoch_losses"].append(delta_epoch_loss)

	# store delta epoch seq similarities
	delta_epoch_seq_similarity = float(sum(batch_losses["delta_batch_seq_similarities"]) / len(batch_losses["delta_batch_seq_similarities"]))
	print(f"delta train seq_sim: {str(delta_epoch_seq_similarity)}\n")
	losses["delta_epoch_seq_similarities"].append(epoch_seq_similarity)

	return losses

def validation(model_items, data_loader, device, epoch, losses):
	
	# switch to evaluation mode to perform validation
	model_items["model"].eval()
	
	# store validation losses
	val_batch_losses = []
	val_batch_seq_similarities = []

	# turn off gradient calculation
	with torch.no_grad():

		# loop through validation batches
		for feature_batch, label_batch, key_padding_mask in data_loader:

			prediction_batch = torch.full([label_batch.size(0), label_batch.size(1), 20], 1/20)

			# step through the batch
			batch_loss, batch_seq_sim = batch_step(	model_items["model"], model_items["loss_function"], 
													feature_batch, prediction_batch, label_batch, 
													key_padding_mask, device, use_amp=False, diffusion_cycles=0
												)

			# store losses
			val_batch_losses.append(float(batch_loss.item())) # / float(torch.sum(label_batch != -1)))
			val_batch_seq_similarities.append(batch_seq_sim)

	# compute epoch loss
	val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
	print(f"validation loss: {str(val_epoch_loss)}")
	losses["epoch_val_losses"].append(val_epoch_loss)
	
	# compute epoch seq similarity
	val_epoch_seq_similarity = float(sum(val_batch_seq_similarities) / len(val_batch_seq_similarities))
	print(f"validation seq_sim: {str(val_epoch_seq_similarity)}\n")
	losses["epoch_val_seq_similarities"].append(val_epoch_seq_similarity)

	return losses

def write_dot(model, output, dot_out):
	dot = make_dot(output, params=dict(model.named_parameters()))
	dot.render(dot_out, format="pdf")

def plot_training(losses: dict, args):

	# Create the plot
	plt.plot([i + 1 for i in range(len(losses["epoch_losses"]))], losses["epoch_losses"], marker='o', color='r', label="Training")
	plt.plot([i + 1 for i in range(len(losses["epoch_val_losses"]))], losses["epoch_val_losses"], marker='o', color='b', label="Validation")

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
	plt.savefig(args.out_path / args.loss_plot)

	plt.figure()

	plt.plot([i + 1 for i in range(len(losses["epoch_seq_similarities"]))], losses["epoch_seq_similarities"], marker='o', color='red', label="Training")
	plt.plot([i + 1 for i in range(len(losses["epoch_val_seq_similarities"]))], losses["epoch_val_seq_similarities"], marker='o', color='blue', label="Validation")

	# Adding title and labels
	plt.title('Mean Sequence Similarity vs. Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Sequence Similarity')
	
	# Add a legend to distinguish the line plots
	plt.legend()
	
	# Display the plot
	plt.grid(True)
	plt.savefig(args.out_path / args.seq_plot)

	print(	f"graph of loss vs. epoch saved to {args.out_path / args.loss_plot}",
			f"graph of seq_similarity vs. epoch saved to {args.out_path / args.seq_plot}"
		)

def test(model_items, device, diffusion_cycles):

	# switch to evaluation mode
	model_items["model"].eval()

	test_data_loader = DataLoader(data.test_data, batch_size=args.batch_size, shuffle=True)

	for cycle in range(diffusion_cycles):

		# store losses
		test_losses = []
		test_seq_similarities = []

		# turn off gradient calculation
		with torch.no_grad():

			# loop through testing batches
			for feature_batch, label_batch, key_padding_mask in model_items["test_data_loader"]:

				prediction_batch = torch.full([label_batch.size(0), label_batch.size(1), 20], 1/20)

				# Pass protein_batch through your model
				# use diffusion like inference for testing, where the model "denoises" the probability matrix iteratively
				test_loss, test_seq_similarity = batch_step(model_items["model"], model_items["loss_function"], 
															feature_batch, prediction_batch, label_batch, key_padding_mask, 
															device, use_amp=False, diffusion_cycles=cycle
															)
				
				test_losses.append(float(test_loss.item()) )#/ float(torch.sum(label_batch != -1)))
				test_seq_similarities.append(test_seq_similarity)

		test_loss = sum(test_losses) / len(test_losses)
		print("-"*65)
		print(f"testing inference with {cycle} diffusion cycles")
		print(f"testing loss: {str(test_loss)}")

		test_seq_similarity = float(sum(test_seq_similarities) / len(test_seq_similarities))
		print(f"test sequence similarity: {str(test_seq_similarity)}")

		print()

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------------------------------------------------------------------------------------------------------------
