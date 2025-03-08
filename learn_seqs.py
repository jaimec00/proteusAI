# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		learn_seqs.py
description:	script to train, validate, and test the model
'''
# ----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
from box import Box
import argparse
import yaml
import os

from utils.train_utils.training_run import TrainingRun

# ----------------------------------------------------------------------------------------------------------------------

def main(args):
	'''
	main script that instantiates TrainingRun object to train and test the model.
	'''

	# initialize the training run
	training_run = TrainingRun(args)

	# setup the training
	training_run.setup_training()

	# train the model
	training_run.train()

	# test the model
	training_run.test()

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--config", type=Path, help="Path to the YAML config file")

	config, _ = parser.parse_known_args()

	def merge_dicts(defaults, overrides):
		"""Recursively merges overrides into default config."""
		merged = defaults.copy()
		for key, value in overrides.items():
			if isinstance(value, dict) and key in merged:
				merged[key] = merge_dicts(merged[key], value)  # recursive merge for nested dicts
			else:
				merged[key] = value  # override value
		return merged

	default_config = Path(os.path.abspath(__file__)).parent / Path("config/train.yml")
	if not default_config.exists():
		raise FileNotFoundError(f"could not find path to default config at {default_config}.")
	with open(default_config, "r") as d:
		args = yaml.safe_load(d)

	# Load YAML configuration if file exists
	if config.config.exists():
		with open(config.config, "r") as f:
			user_args = yaml.safe_load(f)
			args = merge_dicts(args, user_args) # have user configs override defaults, if custom config specified

	# convert to box to have nested attributes that are easier to keep track of
	args = Box(args)

	main(args)

# ----------------------------------------------------------------------------------------------------------------------
