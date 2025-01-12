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
	
	parser.add_argument("--min_wl", default=3.7, type=float, help="minimum wavelength to use in wavelength sampling")
	parser.add_argument("--max_wl", default=20.0, type=float, help="maximum wavelength to use in wavelength sampling")
	parser.add_argument("--wl_base", default=80.0, type=float, help="base to use in wavelength sampling")
	parser.add_argument("--d_hidden_wl", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")
	parser.add_argument("--hidden_layers_wl", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")

	parser.add_argument("--d_hidden_aa", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")
	parser.add_argument("--hidden_layers_aa", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")
	
	parser.add_argument("--dualcoder_layers", default=3, type=int, help="number of decoder layers")
	parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads to perform the training with")
	parser.add_argument("--min_spread", default=1.0, type=float, help="base to use in wavelength sampling")
	parser.add_argument("--max_spread", default=30.0, type=float, help="base to use in wavelength sampling")
	parser.add_argument("--spread_base", default=80.0, type=float, help="base to use in wavelength sampling")
	parser.add_argument("--min_rbf", default=0.1, type=float, help="minimum rbf scaling to apply in gaussian mha")
	parser.add_argument("--max_rbf", default=0.9, type=float, help="maximum rbf scaling to apply in gaussian mha")

	parser.add_argument("--d_hidden_attn", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")
	parser.add_argument("--hidden_layers_attn", default=1024, type=int, help="intermediate dimensions of decoder feed forward layer")
	parser.add_argument("--temperature", default=0.01, type=float, help="temperature for autoregressive inference (for testing)")
	parser.add_argument("--max_tokens", default=512, type=int, help="maximum number of tokens")
	parser.add_argument("--include_ncaa", default=False, type=bool, help="if False, masks out non-canonical AA, else X is a valid prediction")
	
	# training
	parser.add_argument("--num_train", default=-1, type=int, help="number of training samples to use; -1 means all available")
	parser.add_argument("--num_val", default=-1, type=int, help="number of validation samples to use; -1 means all available")
	parser.add_argument("--num_test", default=-1, type=int, help="number of test samples to use; -1 means all available")
	
	parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
	
	parser.add_argument("--batch_sizes", default=[1, 2, 4, 8, 16], type=list, help="possible number of samples per batch, minimizes triton recompilation overhead")
	parser.add_argument("--seq_sizes", default=[1024, 4096, 8192, 10000], type=list, help="possible sequence lengths, minimizes triton recompilation overhead")
	parser.add_argument("--batch_size", default=10000, type=int, help="tokens per batch")
	
	parser.add_argument("--accumulation_steps", default=2, type=int, help="grad accumulation; how many batches to process before learning step")
	parser.add_argument("--learning_step", default=0.00005, type=float, help="learning rate")
	parser.add_argument("--beta1", default=0.9, type=float, help="learning rate")
	parser.add_argument("--beta2", default=0.98, type=float, help="learning rate")
	parser.add_argument("--epsilon", default=10e-9, type=float, help="learning rate")
	parser.add_argument("--dropout", default=0.1, type=float, help="percentage of dropout")
	parser.add_argument("--label_smoothing", default=0.1, type=float, help="percentage of label smoothing to use on the output labels for loss calculation")
	parser.add_argument("--loss_type", default="mean", type=str, choices=['sum', 'mean'], help="whether to use the 'sum' or the 'mean' for CEL")
	parser.add_argument("--loss_sum_norm", default=2000, type=int, help="normalization factor for sum loss")

	parser.add_argument("--lr_scale", default=0.1, type=float, help="LR scaling factor")
	parser.add_argument("--lr_patience", default=5, type=int, help="LR patience for scaling down after plateu")
	parser.add_argument("--training_type", default="wf", type=str, choices=["wf", "onehot", "probs", "self-supervision"],  help="what type of training")
	parser.add_argument("--use_amp", default=True, type=bool,  help="whether to use automatic mixed precision")
	parser.add_argument("--use_chain_mask", default=True, type=bool,  help="whether to compute loss only for chain representative of the cluster, or the whole biounit")

	# input label smoothing
	parser.add_argument("--initial_min_lbl_smooth_mean", default=3/20, type=float, help="initial minimum input label smoothing")
	parser.add_argument("--final_min_lbl_smooth_mean", default=17/20, type=float, help="final minimum input label smoothing")
	parser.add_argument("--max_lbl_smooth_mean", default=21/20, type=float, help="maximum input label smoothing")

	parser.add_argument("--min_lbl_smooth_stdev", default=1/20, type=float, help="minimum input label smoothing stdev")
	parser.add_argument("--max_lbl_smooth_stdev", default=4/20, type=float, help="maximum input label smoothing stdev")

	# input noise
	parser.add_argument("--min_noise_stdev", default=0.1, type=float, help="minimum stdev of noise to apply to inputs")
	parser.add_argument("--initial_max_noise_stdev", default=0.4, type=float, help="initial maximum stdev of noise to apply to inputs")
	parser.add_argument("--final_max_noise_stdev", default=0.2, type=float, help="final maximum stdev of noise to apply to inputs")

	# label smooth and noise cycle length (oscillate with phase shift of pi; in sync)
	parser.add_argument("--lbl_smooth_noise_cycle_length", default=5.3, type=float, help="cycle length of label smoothing and noise oscillations (non-integers minimize the chance of sampling the same values every cycle)")

	# input one-hot injection
	parser.add_argument("--min_one_hot_injection_mean", default=0.15, type=float, help="minimum mean percentage of one-hot label injection in training")
	parser.add_argument("--initial_max_one_hot_injection_mean", default=0.8, type=float, help="initial maximum mean percentage of one-hot label injection in training")
	parser.add_argument("--final_max_one_hot_injection_mean", default=0.25, type=float, help="final maximum mean percentage of one-hot label injection in training")

	parser.add_argument("--one_hot_injection_stdev", default=0.25, type=float, help="stdev percentage of one-hot label injection in training")

	# cycle length of one-hot injection
	parser.add_argument("--one_hot_injection_cycle_length", default=4.3, type=float, help="input one-hot injection cycle length. operates at different frequency than label smooth and noise cycles")

	# when to start gradually introducing self-supervision, also when to complete the decoder expansion
	parser.add_argument("--phase_split", default=0.3, type=float, help="	ratio of progressive learning phase (supervised learning to real-world "\
																						"inputs (mix of supervision and self-supervision). "\
																						"also when to finish the decoder layer expansion.")

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
