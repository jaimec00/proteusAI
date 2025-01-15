# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		learn_seqs.py
description:	script to train, validate, and test the model
'''
# ----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import argparse
import yaml

from utils.train_utils import TrainingRun

# ----------------------------------------------------------------------------------------------------------------------

def main():
	'''
	main script that instantiates TrainingRun object to train and test the model.
	'''

	# initialize CL args
	args = init_args()

	# initialize the training run
	training_run = TrainingRun(args)

	# setup the training
	training_run.setup_training()

	# train the model
	training_run.train()

	# test the model
	training_run.test()

def init_args():
	'''
	initializes the command-line arguments, with option to initialize from yaml file. 

	Args:
		None

	Returns:
		args (NameSpace): arguments parsed by argparse 
	'''

	parser = argparse.ArgumentParser()

	# hyper parameters
	
	# model 
	parser.add_argument("--d_model", default=512, type=int, help="dimensionality of input embeddings")
	
	# wavefunction embedding
	parser.add_argument("--min_wl", default=3.7, type=float, help="minimum wavelength to use in wavelength sampling")
	parser.add_argument("--max_wl", default=20.0, type=float, help="maximum wavelength to use in wavelength sampling")
	parser.add_argument("--base_wl", default=20.0, type=float, help="base to use in wavelength sampling")
	parser.add_argument("--d_hidden_we", default=1024, type=int, help="hidden dimensions in post wavefunction embedding MLP")
	parser.add_argument("--hidden_layers_we", default=0, type=int, help="number of hidden layers in post wavefunction embedding MLP")

	# aa embedding (just an MLP)
	parser.add_argument("--d_hidden_aa", default=1024, type=int, help="hidden dimensions in AA embedding MLP")
	parser.add_argument("--hidden_layers_aa", default=0, type=int, help="number hidden layers in AA embedding MLP")
	
	# dualcoder
	parser.add_argument("--encoder_layers", default=4, type=int, help="number of encoder layers")
	parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads")
	parser.add_argument("--min_spread", default=3.0, type=float, help="minimum spread to use for geometric attention")
	parser.add_argument("--max_spread", default=8.0, type=float, help="maximum spread to use for geometric attention")
	parser.add_argument("--base_spread", default=20.0, type=float, help="base to use for spread sampling in geometric attention")
	parser.add_argument("--d_hidden_attn", default=1024, type=int, help="hidden dimensions in geometric attention FFN")
	parser.add_argument("--hidden_layers_attn", default=0, type=int, help="number of hidden layers in geometric attention FFN")

	# training
	parser.add_argument("--num_train", default=-1, type=int, help="number of training samples to use; -1 means all available")
	parser.add_argument("--num_val", default=-1, type=int, help="number of validation samples to use; -1 means all available")
	parser.add_argument("--num_test", default=-1, type=int, help="number of test samples to use; -1 means all available")
	
	parser.add_argument("--epochs", default=50, type=int, help="number of epochs")

	# input restrictions	
	parser.add_argument("--max_batch_size", default=128, type=list, help="possible number of samples per batch, minimizes triton recompilation overhead")
	parser.add_argument("--min_seq_sizes", default=512, type=list, help="possible sequence lengths, minimizes triton recompilation overhead")
	parser.add_argument("--max_seq_sizes", default=16384, type=list, help="possible sequence lengths, minimizes triton recompilation overhead")
	parser.add_argument("--batch_tokens", default=16384, type=int, help="target number of tokens per batch")
	
	# learning parameters
	parser.add_argument("--accumulation_steps", default=2, type=int, help="grad accumulation; how many batches to process before learning step")
	parser.add_argument("--learning_step", default=0.00005, type=float, help="learning rate")
	parser.add_argument("--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer")
	parser.add_argument("--beta2", default=0.98, type=float, help="beta2 parameter for Adam optimizer")
	parser.add_argument("--epsilon", default=10e-9, type=float, help="epsilon parameter for Adam optimizer")

	parser.add_argument("--dropout", default=0.1, type=float, help="percentage of dropout")
	parser.add_argument("--label_smoothing", default=0.1, type=float, help="percentage of label smoothing to use on the output labels for loss calculation")
	parser.add_argument("--loss_type", default="mean", type=str, choices=['sum', 'mean'], help="whether to use the 'sum' or the 'mean' for CEL")
	parser.add_argument("--loss_sum_norm", default=2000, type=int, help="normalization factor for sum loss")

	parser.add_argument("--lr_type", default="custom", type=str, choices=["plateu, custom"], help="LR type")
	parser.add_argument("--lr_scale", default=0.1, type=float, help="LR scaling factor")
	parser.add_argument("--lr_patience", default=5, type=int, help="LR patience for scaling down after plateu")
	parser.add_argument("--lr_initial_min", default=5e-5, type=float,  help="initial lr rate minimum")
	parser.add_argument("--lr_initial_max", default=1e-4, type=float,  help="initial lr rate maximum")
	parser.add_argument("--lr_final_min", default=1e-5, type=float,  help="final lr rate minimum")
	parser.add_argument("--lr_final_max", default=5e-5, type=float,  help="final lr rate maximum")
	parser.add_argument("--lr_cycle_length", default=7.0, type=float,  help="epochs that make up a cycle")

	parser.add_argument("--use_amp", default=True, type=bool,  help="whether to use automatic mixed precision")
	parser.add_argument("--use_chain_mask", default=True, type=bool,  help="whether to compute loss only for chain representative of the cluster, or the whole biounit")

	# other
	parser.add_argument("--temperature", default=0.01, type=float, help="temperature for autoregressive inference (for testing)")

	# input one-hot injection
	parser.add_argument("--initial_min_MASK_injection_mean", default=0.05, type=float, help="initial minimum mean percentage of one-hot label injection in training")
	parser.add_argument("--initial_max_MASK_injection_mean", default=0.1, type=float, help="initial maximum mean percentage of one-hot label injection in training")
	parser.add_argument("--final_min_MASK_injection_mean", default=0.9, type=float, help="final minimum mean percentage of one-hot label injection in training")
	parser.add_argument("--final_max_MASK_injection_mean", default=0.95, type=float, help="final maximum mean percentage of one-hot label injection in training")
	parser.add_argument("--MASK_injection_stdev", default=0.05, type=float, help="stdev percentage of one-hot label injection in training")

	# cycle length of one-hot injection
	parser.add_argument("--MASK_injection_cycle_length", default=4.3, type=float, help="input one-hot injection cycle length. operates at different frequency than label smooth and noise cycles")

	# output
	parser.add_argument("--out_path", default="output", type=Path, help="path to store output, such as plots and weights file.")
	parser.add_argument("--loss_plot", default="loss_vs_epoch.png", type=Path, help="path to save plot of loss vs epochs after training")
	parser.add_argument("--seq_plot", default="seq_sim_vs_epoch.png", type=Path, help="path to save plot of sequence similarity vs epochs after training")
	parser.add_argument("--weights_path", default="model_parameters.pth", type=Path, help="path to save weights after training")
	parser.add_argument("--write_dot", default=False, type=bool, help="whether to save the dot file of the computational graph")

	# input
	parser.add_argument("--data_path", default="/gpfs_backup/wangyy_data/protAI/pmpnn_data/pdb_2021aug02", type=Path, help="path to data")
	parser.add_argument("--use_model", default=None, type=Path, help="use pretrained model")
	parser.add_argument("--config", default="config.yml", type=Path, help="Path to the YAML config file")

	args, _ = parser.parse_known_args()
	
	# Load YAML configuration if file exists
	if args.config.exists():
		with open(args.config, "r") as f:
			config = yaml.safe_load(f)
		parser.set_defaults(**config)

	args = parser.parse_args()

	return args

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	main()

# ----------------------------------------------------------------------------------------------------------------------
