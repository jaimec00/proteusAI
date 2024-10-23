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

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from get_data import pt_to_data
from model import Model

# ----------------------------------------------------------------------------------------------------------------------

def main():

	print("parsing CL arguments...")
	args = init_args()

	print_log(args)

	print("initializing device...")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
	test(model_items, device)

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
	label smoothing: {args.label_smoothing}
	output directory: {args.out_path}
'''
	print(log)

def init_args():

	parser = argparse.ArgumentParser()

	# hyper parameters
	parser.add_argument("--input_atoms", default=0, type=int, choices=[0,1,2], help="whether to train the model using only alphacarbons (0), full backbone (1), or both (2)")
	parser.add_argument("--d_model", default=512, type=int, help="dimensionality of input and output embeddings")
	parser.add_argument("--num_heads", default=4, type=int, help="number of attention heads to perform the training with")
	parser.add_argument("--hidden_linear_dim", default=1024, type=int, help="intermediate dimensions of feed forward layer")
	parser.add_argument("--train_val", default=3/4, type=int, help="how to split training and testing data")
	parser.add_argument("--val_test", default=1/2, type=int, help="how to split validation and testing data")
	parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
	parser.add_argument("--batch_size", default=32, type=int, help="number of decoders")
	parser.add_argument("--learning_step", default=0.0001, type=int, help="learning step")
	parser.add_argument("--dropout", default=0.05, type=float, help="percentage of dropout")
	parser.add_argument("--label_smoothing", default=0.25, type=float, help="percentage of label smoothing")
	parser.add_argument("--num_inputs", default=2048, type=int, help="number of inputs to use")

	parser.add_argument("--out_path", default="output", type=Path, help="path to store output, such as plots and weights file.")

	parser.add_argument("--loss_plot", default="loss_vs_epoch.png", type=Path, help="path to save plot of loss vs epochs after training")
	parser.add_argument("--seq_plot", default="seq_sim_vs_epoch.png", type=Path, help="path to save plot of sequence similarity vs epochs after training")
	parser.add_argument("--weights_path", default="model_parameters.pth", type=Path, help="path to save weights after training")
	parser.add_argument("--write_dot", default=False, type=bool, help="whether to save the dot file of the computational graph")

	parser.add_argument("--pt_path", default="pdbs/pt", type=Path, help="path to the pt tensor files")
	parser.add_argument("--use_wf", default=True, type=bool, help="whether to use the precomputed wavefunction pt files (features) rather than computing them each time")


	args = parser.parse_args()
	
	return args


def setup_training(args, device):

	print("getting data...")
	data = pt_to_data(args.pt_path, all_bb=args.input_atoms, device=device, features=args.use_wf, num_inputs=args.num_inputs)

	# split the data into training and testing. also splits the labels
	print("splitting into training, validation, and testing...")
	data.split_train_val_test(args.train_val, args.val_test)
	N = data.train_data.bb_coords.size(1)

	# Create a DataLoader with desired batch size
	print("loading data...")
	train_data_loader = DataLoader(data.train_data, batch_size=args.batch_size, shuffle=True)
	val_data_loader = DataLoader(data.val_data, batch_size=args.batch_size, shuffle=True)
	test_data_loader = DataLoader(data.test_data, batch_size=args.batch_size, shuffle=True)


	print("loading model...")
	model = Model(N, args.d_model, 
					args.num_heads, 
					args.hidden_linear_dim, 
					args.dropout, args.use_wf)
	model.to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f"model contains {num_params} parameters")

	print("instantiating optimizer, scheduler and loss function...")
	optim = torch.optim.Adam(model.parameters(), lr=args.learning_step)

	# sum worked better for PMPNN, and it is also working better here. implementing label smoothing to downweight the effect of incorrect predictions, 
	# as the distribution will likely be wide, but the goal is for the most likely aa to be most probable, even if it is not as confident 
	# doing this because saw that even though the CEL was approaching randomness for validation, the sequence similarity continued to increase,
	# so the model was generalizing to some extent, but it was improperly "using" the loss function to penalize higher probabilities for incorrect labels
	loss_function = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum', label_smoothing=args.label_smoothing)
	scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.75, patience=4)

	model_items = {
					"model": model, 
					"optim": optim, "loss_function": loss_function, "scheduler": scheduler, 
					"train_data_loader": train_data_loader, 
					"val_data_loader": val_data_loader,
					"test_data_loader": test_data_loader
				}

	return model_items

def train(model_items, args, device):

	# Now iterate over the DataLoader
	num_batches = int(len(model_items["train_data_loader"]))
	print(	f"\ninitializing training. "\
			f"training on {num_batches} "\
			f"batches of batch size {args.batch_size} for {args.epochs} epochs.\n"
		)
	
	# store losses
	losses = {
		"epoch_losses": [],
		"epoch_seq_similarities": [],
		"epoch_val_losses": [],
		"epoch_val_seq_similarities": []
	}

	# loop through epochs
	for epoch in range(args.epochs):

		# make sure in training mode
		model_items["model"].train()

		# store batch losses
		batch_losses = []
		batch_seq_similarities = []

		make_dot_graph = args.write_dot

		print("-"*65)
		print(f"epoch {epoch}: ")
		print("-"*65)
		for param_group in model_items["optim"].param_groups:
			current_lr = param_group['lr']
			print(f"current learning rate: {current_lr}\n")

		# loo through batches
		for feature_batch, prediction_batch, label_batch, key_padding_mask in model_items["train_data_loader"]:

			# forward pass on batch
			batch_loss, batch_seq_similarity = batch_step(	model_items["model"], model_items["loss_function"], 
															feature_batch, prediction_batch, label_batch, 
															key_padding_mask, device, 
															make_dot_graph, str(args.out_path / Path("dot")) 
														)

			make_dot_graph = False # only make the dot on the first batch

			# backward pass
			learn(model_items["optim"], batch_loss)


			# store losses
			avg_batch_loss = float(batch_loss.item()) / float(torch.sum(label_batch != -1))
			batch_losses.append(avg_batch_loss)
			batch_seq_similarities.append(batch_seq_similarity)
		
		# store epoch losses
		epoch_loss = sum(batch_losses) / len(batch_losses)
		print(f"train loss: {str(epoch_loss)}")
		losses["epoch_losses"].append(epoch_loss)

		# store epoch seq similarities
		epoch_seq_similarity = float(sum(batch_seq_similarities) / len(batch_seq_similarities))
		print(f"train seq_sim: {str(epoch_seq_similarity)}\n")
		losses["epoch_seq_similarities"].append(epoch_seq_similarity)

		# validate learning
		epoch_val_loss, epoch_val_seq_similarity = validation(model_items, device, epoch)
		losses["epoch_val_losses"].append(epoch_val_loss)
		losses["epoch_val_seq_similarities"].append(epoch_val_seq_similarity)

		model_items["scheduler"].step(epoch_val_loss)

	return losses

def batch_step(model, loss_function, feature_batch, prediction_batch, label_batch, key_padding_mask, device, make_dot_graph=False, dot_out="dot"):

	feature_batch.to(device)
	prediction_batch.to(device)
	label_batch.to(device)
	key_padding_mask.to(device)

	# Pass protein_batch through your model
	output_prediction = model(feature_batch, prediction_batch, key_padding_mask)

	if make_dot_graph:
		write_dot(model, output_prediction, dot_out)

	# calculate loss
	output_prediction = output_prediction.permute(0, 2, 1)
	# label_batch = label_batch
	batch_loss = loss_function(output_prediction, label_batch)

	# calculate sequence similarity via greedy selection. not used for back propogation, just for intuition
	norm_pred = F.softmax(output_prediction, dim=-1)
	seq_predictions = torch.argmax(norm_pred, dim=1)
	padding_mask = label_batch != -1
	valid_positions = padding_mask.sum(dim=-1)
	matches = ((seq_predictions == label_batch) & padding_mask).sum(dim=-1)
	batch_seq_similarity = ((matches / valid_positions).float()*100).mean()

	return batch_loss, batch_seq_similarity

def learn(optim, loss):

	optim.zero_grad()
	loss.backward()
	optim.step()

def validation(model_items, device, epoch):
	
	# switch to evaluation mode to perform validation
	model_items["model"].eval()
	
	# store validation losses
	val_batch_losses = []
	val_batch_seq_similarities = []

	# turn off gradient calculation
	with torch.no_grad():

		# loop through validation batches
		for feature_batch, prediction_batch, label_batch, key_padding_mask in model_items["val_data_loader"]:

			# step through the batch
			batch_loss, batch_seq_sim = batch_step(	model_items["model"], model_items["loss_function"], 
													feature_batch, prediction_batch, label_batch, 
													key_padding_mask, device
												)

			# store losses
			val_batch_losses.append(float(batch_loss.item()) / float(torch.sum(label_batch != -1)))
			val_batch_seq_similarities.append(batch_seq_sim)

	# compute epoch loss
	val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
	print(f"validation loss: {str(val_epoch_loss)}")
	
	# compute epoch seq similarity
	val_epoch_seq_similarity = float(sum(val_batch_seq_similarities) / len(val_batch_seq_similarities))
	print(f"validation seq_sim: {str(val_epoch_seq_similarity)}\n")

	return val_epoch_loss, val_epoch_seq_similarity

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

def test(model_items, device):

	# switch to evaluation mode
	model_items["model"].eval()

	# store losses
	test_losses = []
	test_seq_similarities = []

	# turn off gradient calculation
	with torch.no_grad():

		# loop through testing batches
		for feature_batch, prediction_batch, label_batch, key_padding_mask in model_items["test_data_loader"]:

			# Pass protein_batch through your model
			test_loss, test_seq_similarity = batch_step(model_items["model"], model_items["loss_function"], feature_batch, prediction_batch, label_batch, key_padding_mask, device)
			
			test_losses.append(float(test_loss.item()) / float(torch.sum(label_batch != -1)))
			test_seq_similarities.append(test_seq_similarity)

	test_loss = sum(test_losses) / len(test_losses)
	print("-"*65)
	print(f"testing loss: {str(test_loss)}")

	test_seq_similarity = float(sum(test_seq_similarities) / len(test_seq_similarities))
	print(f"test sequence similarity: {str(test_seq_similarity)}")

	print()

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------------------------------------------------------------------------------------------------------------
