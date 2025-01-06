# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import pandas as pd
import os

from proteusAI import proteusAI
from utils.parameter_utils import (	HyperParameters, TrainingParameters, 
									InputPerturbationParameters, InputPerturbations )
from utils.io_utils import Output
from utils.data_utils import DataHolder
from utils.model_utils.featurization import protein_to_wavefunc
from utils.model_utils.gaussian_attn import attn

# ----------------------------------------------------------------------------------------------------------------------

class TrainingRun():
	'''
	the main class for training. holds all of the configuration arguments, and organnizes them into hyper-parameters, 
	training parameters, input perturbation parameters, data, and output objects
	also orchestrates the setup of training, training itself, validation, testing, outputs, etc.

	Attributes:
		hyper_parameters (HyperParameters): store model hyper_parameters
		training_parameters (TrainingParameters): stores training parameters
		input_perturbation_parameters (InputPerturbationParameters): stores the input perturbation parameters (noise, 
																		label smooth, on-hot injection), also contain
																		methods to calculate values depending on stage and
																		also to apply these perturbations 
		data (DataHolder): object to store data (contains object os Data type, inherets from Dataset), splits into train, 
							val, and test depending on arguments, also capable of loading the Data into DataLoader for 
							easy integration
		output (Output): holds information about where to write output to, also contains the logging object. comes with
						methods to print logs, plot training, and save model parameters
		train_losses (dict(Losses)):
		val_losses (Losses):
		test_losses (Losses):
		gpu (torch.device): for convenience in moving tensors to GPU
		cpu (torch.device): for loading Data before moving to GPU

	'''

	def __init__(self, args):

		self.hyper_parameters = HyperParameters(args.d_model,
												args.min_wl, args.max_wl, 
												args.min_base, args.max_base,
												args.min_rbf, args.max_rbf, 
												args.min_spread, args.max_spread, 
												args.num_heads, args.decoder_layers, 
												args.hidden_linear_dim, args.temperature, 
												args.max_tokens, args.use_model )
		
		self.training_parameters = TrainingParameters(  args.epochs, args.batch_sizes, args.seq_sizes, args.batch_size, 
														args.accumulation_steps, args.learning_step, 
														args.beta1, args.beta2, args.epsilon, 
														args.dropout, args.label_smoothing, args.include_ncaa,
														args.loss_type, args.loss_sum_norm, args.lr_scale, args.lr_patience,
														args.phase_split, args.expand_decoders, args.training_type, args.precomputed_features,
														args.use_amp, args.use_checkpoint, args.use_chain_mask,
														args.autotune_wf, args.autotune_mha )
		
		self.input_perturbation_parameters = InputPerturbationParameters(	args.initial_min_lbl_smooth_mean, args.final_min_lbl_smooth_mean, 
																			args.max_lbl_smooth_mean, args.min_lbl_smooth_stdev, args.max_lbl_smooth_stdev, 
																			args.min_noise_stdev, args.initial_max_noise_stdev, args.final_max_noise_stdev, 
																			args.lbl_smooth_noise_cycle_length, 
																			args.initial_max_one_hot_injection_mean, args.final_max_one_hot_injection_mean, 
																			args.min_one_hot_injection_mean, args.one_hot_injection_stdev, 
																			args.one_hot_injection_cycle_length,
																			self.training_parameters.use_onehot, self.training_parameters.use_probs ) 
		
		feature_path = f"3.7_20.0_20.0" # this is irrelevant now, since not precomputing features, but will update later
		self.data = DataHolder(args.data_path, args.num_train, args.num_val, args.num_test, args.max_tokens, args.batch_sizes, args.seq_sizes, args.batch_size, feature_path, args.include_ncaa)
		self.output = Output(args.out_path, args.loss_plot, args.seq_plot, args.weights_path, args.write_dot)

		self.train_losses = {
							"input": Losses(),
							"output": Losses(),
							"delta": Losses()
						}
		self.val_losses = Losses()
		self.test_losses = Losses()

		self.gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.cpu = torch.device("cpu")

	def setup_training(self):
		'''
		sets up the training by setting up the model, optimizer, scheduler, loss 
		function, scaler (if using AMP), and losses

		Args:
			None

		Returns:
			None
		'''

		self.setup_model()
		self.setup_optim()
		self.setup_scheduler()
		self.setup_loss_function()
		self.scaler = GradScaler(self.gpu) if self.gpu is not None else None

		self.output.log_hyperparameters(self.training_parameters, self.hyper_parameters, self.input_perturbation_parameters, self.data)

		# self.autotune_triton(wf=self.training_parameters.autotune_wf, mha=self.training_parameters.autotune_mha)

	def setup_model(self):
		'''
		instantiates proteusAI with given Hyper-Parameters, moves it to gpu, and 
		optionally loads model weights from pre-trained model

		Args:
			None
		
		Returns:
			None
		'''
		
		self.output.log.info("loading model...")
		
		self.model = proteusAI(	self.hyper_parameters.d_model, 
								self.hyper_parameters.num_heads, 
								self.hyper_parameters.decoder_layers,
								self.hyper_parameters.hidden_linear_dim,
								self.training_parameters.dropout,
								self.hyper_parameters.min_wl,
								self.hyper_parameters.max_wl,
								self.hyper_parameters.min_base,
								self.hyper_parameters.max_base,
								self.hyper_parameters.min_rbf,
								self.hyper_parameters.max_rbf,
								self.hyper_parameters.min_spread,
								self.hyper_parameters.max_spread,
								active_decoders=-1**(not self.training_parameters.expand_decoders), # -1 for no expansion (use all), 1 for expansion (start with one)
								use_probs=self.training_parameters.use_probs,
								include_ncaa=self.training_parameters.include_ncaa
							)
		self.model.to(self.gpu)

		if self.hyper_parameters.use_model is not None:
			pretrained_weights = torch.load(self.hyper_parameters.use_model, map_location=self.gpu)
			self.model.load_state_dict(pretrained_weights, weights_only=True)
		
		self.training_parameters.num_params = sum(p.numel() for p in self.model.parameters())

	def setup_optim(self):
		'''
		sets up the optimizer, zeros out the gradient

		Args:
			None
		
		Returns:
			None
		'''

		self.output.log.info("loading optimizer...")
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.training_parameters.learning_step, 
									betas=(self.training_parameters.beta1, self.training_parameters.beta2), 
									eps=self.training_parameters.epsilon)
		self.optim.zero_grad()

	def setup_scheduler(self):
		'''
		sets up the loss scheduler, right now only using ReduceLROnPlateu, but planning on making this configurable

		Args:
			None
		
		Returns:
			None
		'''

		self.output.log.info("loading scheduler...")
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=self.training_parameters.lr_scale, patience=self.training_parameters.lr_patience) 

	def setup_loss_function(self):
		'''
		initializes the loss function

		Args:
			None

		Returns: 
			None
		'''

		self.output.log.info("loading loss function...") 
		self.loss_function = nn.CrossEntropyLoss(ignore_index=-1, reduction="none", label_smoothing=self.training_parameters.label_smoothing)

	def autotune_triton(self, wf=False, mha=True):
		'''
		performs autotuning once with each of the possible input sizes, this allows triton to find best configuration at the beginning and 
		not recompile each time a new input size is encountered.
		'''

		# set env vars to tell functions not to autotune (i.e. test a single configuration on the fly)
		os.environ["ATTN_AUTOTUNE"] = "1" if mha else "0"
		os.environ["WF_AUTOTUNE"] = "1" if wf else "0"

		self.output.log.info(f"performing triton autotuning for: {'protein_to_wavefunc' if wf else ''}{' and ' if (wf and mha) else ''}{'attn' if mha else ''}")
		os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

		# these rely on env vars, so have to be evaluated after setting them
		from utils.model_utils.featurization import protein_to_wavefunc
		from utils.model_utils.gaussian_attn import attn

		for batch_size in self.data.batch_sizes:
			for seq_size in self.data.seq_sizes:
				
				self.output.log.info(f"testing batch_size: {batch_size}, seq_size: {seq_size}")
				coords = torch.rand((batch_size, seq_size, 3), dtype=torch.float32, device=self.gpu)
				mask = torch.zeros(batch_size, seq_size, dtype=torch.bool, device=self.gpu)
				if wf:
					protein_to_wavefunc(coords, self.hyper_parameters.d_model, self.hyper_parameters.min_wl, self.hyper_parameters.max_wl, self.hyper_parameters.base, mask)
				if mha:
					qkv = torch.rand((batch_size, self.hyper_parameters.num_heads, seq_size, self.hyper_parameters.d_model // self.hyper_parameters.num_heads), dtype=torch.float32, device=self.gpu, requires_grad=True)
					spreads = self.hyper_parameters.min_wl + (self.hyper_parameters.max_wl-self.hyper_parameters.min_wl)*(torch.logspace(0,1,self.hyper_parameters.num_heads, self.hyper_parameters.base, dtype=torch.float32, device=self.gpu)-1)/(self.hyper_parameters.base-1)
					out = attn(qkv, qkv, qkv, coords, spreads, mask, mask, self.hyper_parameters.min_rbf, self.hyper_parameters.max_rbf)
					out.sum().backward()

	def train(self):
		'''
		entry point for training the model. loads train and validation data, loops through epochs, plots training, 
		and saves the model

		Args:
			None
		
		Return:
			None
		'''

		# load the data
		self.output.log.info("loading training data...")
		self.data.load("train")
		self.output.log.info("loading validation data...")
		self.data.load("val")

		# log training info
		self.output.log.info(f"\n\ninitializing training. "\
							f"training on {len(self.data.train_data)} batches "\
							f"of batch size {self.training_parameters.batch_size} tokens "\
							f"for {self.training_parameters.epochs} epochs.\n" )
		
		# loop through epochs
		for epoch in range(self.training_parameters.epochs):

			# view current params
			# i wanna check how wavelengths and spreads are being updated
			# self.model.print_wavelengths(self.output)
			# self.model.print_spreads(self.output)
			
			epoch = Epoch(epoch, self)
			epoch.epoch_loop()


		for key in self.train_losses.keys():
			self.train_losses[key].to_numpy()
		self.val_losses.to_numpy()
		
		self.output.plot_training(self.train_losses, self.val_losses, use_probs=self.training_parameters.use_probs)
		self.output.save_model(self.model)
				
	def validation(self):
		
		self.output.log.info(f"running validation...\n")
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		val_losses = Losses()
		
		# turn off gradient calculation
		with torch.no_grad():

			input_perturbations = self.get_test_perturbations()

			# loop through validation batches
			for feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask in self.data.val_data:

				if not self.training_parameters.use_chain_mask:
					chain_mask = None

				if not self.training_parameters.precomputed_features:
					features_batch = None
					
				batch = Batch(feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask, 
							use_probs=self.training_parameters.use_probs, use_amp=False, auto_regressive=False,
							loss_type=self.training_parameters.loss_type,
							loss_sum_norm=self.training_parameters.loss_sum_norm)
				input_perturbations.apply_perturbations(batch)
				batch.batch_forward(self.model, self.loss_function, self.gpu)

				# store losses
				loss, seq_sim = batch.outputs.losses["output"].get_avg()
				val_losses.add_losses(float(loss.item()), seq_sim)

		self.val_losses.add_losses(*val_losses.get_avg())
		self.output.log_val_losses(val_losses)

	def test(self):

		# switch to evaluation mode
		self.model.eval()
		
		self.output.log.info("loading testing data...")
		self.data.load("test")

		test_losses, test_seq_sims, test_ar_seq_sims = [], [], []

		# turn off gradient calculation
		with torch.no_grad():

			input_perturbations = self.get_test_perturbations()

			# loop through testing batches
			for feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask in self.data.test_data:

				if not self.training_parameters.use_chain_mask:
					chain_mask = None
				if not self.training_parameters.precomputed_features:
					features_batch = None
					
				batch = Batch(feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask, 
								use_probs=self.training_parameters.use_probs, use_amp=False, 
								auto_regressive=self.training_parameters.auto_regressive, 
								temp=self.hyper_parameters.temperature, loss_type=self.training_parameters.loss_type,
								loss_sum_norm=self.training_parameters.loss_sum_norm)

				input_perturbations.apply_perturbations(batch)
				batch.batch_forward(self.model, self.loss_function, self.gpu)

				loss, seq_sim = batch.outputs.losses["output"].get_avg()
				_, ar_seq_sim = batch.outputs.losses["ar_output"].get_avg()
				
				test_losses.append(float(loss.item()) )
				test_seq_sims.append(seq_sim)
				test_ar_seq_sims.append(ar_seq_sim)
		
		self.output.log.info(f"testing loss: {sum(test_losses) / len(test_losses)}")
		self.output.log.info(f"test sequence similarity: {sum(test_seq_sims) / len(test_seq_sims)}")
		if None not in test_ar_seq_sims:
			self.output.log.info(f"test auto-regressive sequence similarity: {sum(test_ar_seq_sims) / len(test_ar_seq_sims)}")

	def get_test_perturbations(self):

		mean_onehot, stdev_onehot, mean_lbl_smooth, stdev_lbl_smooth, stdev_noise = [None] * 5
		self_supervised_pct = 0.0

		if self.training_parameters.use_onehot:
			mean_onehot = 0.1
			stdev_onehot = 0.0

		input_perturbations = InputPerturbations(mean_onehot, stdev_onehot, stdev_noise,
											mean_lbl_smooth, stdev_lbl_smooth,
											self_supervised_pct)

		return input_perturbations

class Epoch():	
	def __init__(self, epoch, training_run):

		self.training_run_parent = training_run

		self.epoch = epoch
		self.epochs = self.training_run_parent.training_parameters.epochs
		self.stage = epoch / self.epochs

		self.train_losses = {
								"input": Losses(),
								"output": Losses()
		}
		
		self.phase_split = self.training_run_parent.training_parameters.phase_split
		self.phase = self.stage < self.phase_split
		self.phase_i_epochs = (self.phase_split * self.epochs)
		self.phase_ii_epochs = self.epochs - self.phase_i_epochs

		self.phase_stage = self.epoch / (self.phase_ii_epochs if self.phase else self.phase_i_epochs)

		self.input_perturbations = self.training_run_parent.input_perturbation_parameters.get_input_perturbations(self)

	def epoch_loop(self):
		'''
		a single training loop through one epoch. sets up epoch input perturbation values depending on the stage (calculated in Epoch.__init__)
		then loops through batches, logs the losses, and runs validation

		Args:
			None

		Returns:
			None
		'''

		# make sure in training mode
		self.training_run_parent.model.train()

		# setup the epoch
		self.setup_epoch()

		# loop through batches
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")
		for b_idx, (feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask) in enumerate(self.training_run_parent.data.train_data):

			if not self.training_run_parent.training_parameters.use_chain_mask:
				chain_mask = None
			if not self.training_run_parent.training_parameters.precomputed_features:
				features_batch = None

					
			batch = Batch(feature_batch, label_batch, coords_batch, chain_mask, key_padding_mask, 
						b_idx=b_idx, epoch=self, use_probs=self.training_run_parent.training_parameters.use_probs, 
						use_amp=self.training_run_parent.training_parameters.use_amp, include_ncaa=self.training_run_parent.training_parameters.include_ncaa,
						use_checkpoint=self.training_run_parent.training_parameters.use_checkpoint, loss_type=self.training_run_parent.training_parameters.loss_type,
						loss_sum_norm=self.training_run_parent.training_parameters.loss_sum_norm)
			batch.batch_learn()
			# normalize by number of valid tokens if computing sum, else it is already normalized
			self.gather_batch_losses(batch, normalize=self.training_run_parent.loss_function.reduction=="sum")


			epoch_pbar.update(1)
		
		# print batch losses
		self.training_run_parent.output.log_epoch_losses(self, self.training_run_parent.train_losses)

		# run validation
		self.training_run_parent.validation()

		# lr scheduler update
		self.training_run_parent.scheduler.step(self.training_run_parent.val_losses.losses[-1])

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()

	def setup_epoch(self):
		'''
		sets up the appropriate parameters, particular the input perturbations, 
		depending on the configuration and training stage

		Args:
			None

		Returns:
			None

		Raises:
			ValueError: when self.training_parameters.training_type not valid
		
		'''
		
		if self.training_run_parent.training_parameters.training_type == "wf":
			if self.training_run_parent.training_parameters.expand_decoders:
				if not self.phase:
					num_decoders = round(((self.epoch + 1) / self.phase_i_epochs) * len(self.training_run_parent.model.decoders))
					new_decoders = num_decoders - self.training_run_parent.model.active_decoders
					self.model.add_decoder(new_decoders)

		elif self.training_run_parent.training_parameters.training_type == "onehot":
			self.training_run_parent.model.alter_decoder_weights(requires_grad=self.phase)

		self.training_run_parent.output.log_epoch(self, self.training_run_parent.optim, self.training_run_parent.model, self.input_perturbations)

	def gather_batch_losses(self, batch, normalize=False):
		self.train_losses["input"].extend_losses(batch.outputs.losses["input"], normalize)
		self.train_losses["output"].extend_losses(batch.outputs.losses["self_output" if batch.self_supervised else "output"], normalize)

class Batch():
	def __init__(self, features, labels, coords, chain_mask, key_padding_mask, 
					b_idx=None, epoch=None, use_probs=False, 
					use_amp=True, auto_regressive=False, temp=0.1, include_ncaa=False,
					use_checkpoint=True, loss_type="sum", loss_sum_norm=2000):

		self.features = features
		self.labels = labels
		self.coords = coords 
		self.chain_mask = chain_mask if chain_mask is not None else torch.zeros(labels.shape, dtype=torch.bool, device=labels.device) 
		self.use_probs = use_probs
		self.use_amp = use_amp
		self.use_checkpoint = use_checkpoint
		num_aas = 20 if not include_ncaa else 21
		if self.use_probs:
			self.predictions = torch.full(self.labels.shape, 1/num_aas).unsqueeze(-1).expand(-1,-1,num_aas)
		else:
			self.predictions = torch.zeros(self.labels.shape).unsqueeze(-1).expand(-1,-1,num_aas)

		self.key_padding_mask = key_padding_mask
		self.onehot_mask = torch.zeros(self.key_padding_mask.shape, dtype=torch.bool)

		self.b_idx = b_idx
		self.epoch_parent = epoch
		self.self_supervised = torch.rand(1) < self.epoch_parent.input_perturbations.self_supervised_pct if epoch is not None else False

		self.use_amp = use_amp
		self.auto_regressive = auto_regressive
		self.temp = temp

		self.loss_type = loss_type
		self.loss_sum_norm = loss_sum_norm

		self.outputs = None

	def move_to(self, device):

		self.features = self.features.to(device)
		self.predictions = self.predictions.to(device)
		self.labels = self.labels.to(device)
		self.coords = self.coords.to(device)
		self.chain_mask = self.chain_mask.to(device)
		self.key_padding_mask = self.key_padding_mask.to(device)
		self.onehot_mask = self.onehot_mask.to(device)

	def batch_learn(self):
		'''
		a single iteration over a batch. applies the input perturbation parameters calculated by self.setup_epoch to each batch
		performs the forward pass, as well as the backwards pass and stores losses

		Args:
			None

		Returns:
			None
		'''

		# apply proper perturbations to input (overwrites batch.predictions, if configured to do so)
		self.epoch_parent.input_perturbations.apply_perturbations(self)

		# forward pass
		self.batch_forward(self.epoch_parent.training_run_parent.model, self.epoch_parent.training_run_parent.loss_function, self.epoch_parent.training_run_parent.gpu)

		# backward pass
		self.batch_backward()

	def batch_forward(self, model, loss_function, device):
		'''
		performs the forward pass, gets the outputs and computes the losses of a batch. 
		option to use Automatic Mixed Precision (AMP)

		Args:
			batch (Batch): Batch objects, contains info about batch and useful methods
			use_amp (bool): whether to use AMP or not
			auto_regressive (bool): option for auto-regressive inference
			temp (float): temperature, only applicable if auto_regressive==True

		Returns:
			None
		  
		'''
		
		# move batch to gpu
		self.move_to(device)

		# mask one hot positions for loss. also only compute loss for the representative chain of the sequence cluster
		self.labels = torch.where(self.onehot_mask | self.chain_mask | self.key_padding_mask, -1, self.labels)

		if self.use_amp: # optional AMP
			with autocast('cuda'):
				self.outputs = self.get_outputs(model)
		else:
			self.outputs = self.get_outputs(model)

		self.outputs.get_losses(loss_function, loss_type=self.loss_type, sum_norm=self.loss_sum_norm)

	def batch_backward(self):

		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.accumulation_steps
		scaler = self.epoch_parent.training_run_parent.scaler
		optim = self.epoch_parent.training_run_parent.optim

		learn_step = (self.b_idx + 1) % accumulation_steps == 0
		loss = self.outputs.losses["self_output" if self.self_supervised else "output"].losses[-1] / accumulation_steps
		
		if scaler is not None:
			scaler.scale(loss).backward()
			if learn_step:
				scaler.step(optim)
				scaler.update()
				optim.zero_grad()
		else:
			loss.backward()
			if learn_step:
				optim.step()
				optim.zero_grad()

	def get_outputs(self, model):
		'''
		used to get output predictions for various scenarious, such as one-shot predictions from engineered inputs,
		one-shot predictions using previous outputs as inputs, or autoregressive inference
		
		Args:
			batch (Batch): Batch object, holds features, labels, and its own losses
			auto_regressive (bool): whether to do autoregressive inference
			temp (float): temperature if using autoregression

		Returns:
			None
		'''

		# compute autoregressive prediction if specified
		# cant compute cel with one hot vector, this is only for testing
		with torch.no_grad():
			if self.auto_regressive: 
				ar_output_prediction = model(self.coords, self.predictions, features=self.features, key_padding_mask=self.key_padding_mask, 
											auto_regressive=self.auto_regressive, temp=self.temp, use_checkpoint=False)
			else:
				ar_output_prediction = None

		# compute one shot also

		output_prediction = model(self.coords, self.predictions, features=self.features, key_padding_mask=self.key_padding_mask, auto_regressive=False, use_checkpoint=self.use_checkpoint)

		# use the outputs as the inputs also
		if self.self_supervised:

			# detach from computational graph
			self_prediction_batch = output_prediction.detach()

			# get percentages
			self_prediction_batch = torch.softmax(self_prediction_batch, dim=-1)

			# replace with original one hots
			self_prediction_batch = torch.where((self.onehot_mask | self.key_padding_mask).unsqueeze(-1), self.predictions, self_prediction_batch)

			# run the model
			self_output_prediction = model(self.coords, self_prediction_batch, features=self.features, key_padding_mask=self.key_padding_mask, auto_regressive=False, use_checkpoint=self.use_checkpoint)
			
		else:

			self_output_prediction = None

		return ModelOutputs(output_prediction, self_output_prediction, ar_output_prediction, 
							self.predictions, self.labels)

class ModelOutputs():

	def __init__(self, output_predictions, self_output_predictions, ar_output_predictions, input_predictions, 
				labels):
		
		self.input_predictions = input_predictions
		self.output_predictions = output_predictions
		self.self_output_predictions = self_output_predictions
		self.ar_output_predictions = ar_output_predictions

		self.labels = labels

		self.valid = self.get_valid()

		self.losses = {
			"input": Losses(),
			"output": Losses(), 
			"self_output": Losses(), 
			"ar_output": Losses()
		}


	def compute_seq_sim(self, prediction):
		
		prediction_flat = prediction.contiguous().view(-1, 20) # batch*N x 20
		labels_flat = self.labels.view(-1) # batch x N --> batch*N,

		seq_predictions = torch.argmax(prediction_flat, dim=-1) # batch*N x 20 --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		valid_positions = torch.sum(valid_mask) # 1,
		matches = ((seq_predictions == labels_flat) & (valid_mask)).sum() # 1, 
		seq_sim = ((matches / valid_positions).float()*100).mean().item()
		
		return seq_sim

	def compute_cel(self, prediction, num_classes=20):

		prediction = torch.softmax(prediction, dim=-1)

		# Flatten the tensor to simplify indexing for CEL calculation
		prediction_flat = prediction.contiguous().view(-1, num_classes)
		labels_flat = self.labels.view(-1)
		key_padding_mask_flat = (labels_flat == -1).view(-1)

		# Compute CEL: Use the log of the predicted probability for the true class
		# Gather the predicted probability for each true class in label_batch
		true_label_probs = prediction_flat[torch.arange(len(labels_flat)).long(), labels_flat.long()]
		log_probs = torch.log(true_label_probs + 1e-12)  # Avoid log(0) by adding a small epsilon
		cel = -log_probs  # Cross-entropy loss for each position

		# Mask out padded tokens (where key_padding_mask is True)
		valid_cel = cel[~key_padding_mask_flat]
		cel = valid_cel.mean().item() if len(valid_cel) > 0 else 0.0

		return cel

	def compute_cel_and_seq_sim(self, prediction, loss_function=None, loss_type="sum", sum_norm=2000):
		"""
		Compute the mean cross-entropy loss (CEL) and sequence similarity (accuracy) between
		the original label batch and the noised & smoothed label batch.

		Params:
			label_batch (torch.Tensor): A batch of correct labels with shape (batch_size, N), where each entry is the index of the correct label.
			noised_labels (torch.Tensor): A batch of smoothed and noised label predictions with shape (batch_size, N, num_classes).
			key_padding_mask (torch.Tensor): A mask with shape (batch_size, N), where True indicates padding positions.

		Returns:
			mean_cel (float): The mean cross-entropy loss over all valid positions.
			mean_seq_sim (float): The mean sequence similarity (accuracy) over all valid positions.
		"""

		cel = loss_function(prediction.view(-1, prediction.size(2)).to(torch.float32), self.labels.view(-1).long()) if loss_function else self.compute_cel(prediction)
	
		valid = self.labels.view(-1)!=-1
		
		# doing manual calculation because needs to be float32 or get overflow in sum
		cel_sum = (cel*valid).sum()
		
		if loss_type == "sum":
			cel = cel_sum / sum_norm
		elif loss_type == "mean":
			cel = cel_sum / valid.sum()


		seq_sim = self.compute_seq_sim(prediction)


		return cel, seq_sim

	def get_valid(self):
		return (self.labels != -1).sum()

	def get_losses(self, loss_function, loss_type="sum", sum_norm=2000):

		for prediction, prediction_loss in zip(	
												[self.input_predictions, self.output_predictions,
												self.self_output_predictions, self.ar_output_predictions], 
												
												[self.losses["input"], self.losses["output"], 
												self.losses["self_output"], self.losses["ar_output"]]
											):

			# for some reason gives tuple to cel and None to seq sim if do 'cel, seq_sim = self.compute...'
			cel_and_seq_sim = self.compute_cel_and_seq_sim(prediction, loss_function, loss_type, sum_norm) if prediction is not None else (None, None)
			cel, seq_sim = cel_and_seq_sim
			prediction_loss.add_losses(cel, seq_sim)

class Losses():
	'''
	class to store losses
	'''
	def __init__(self): 

		self.losses = []
		self.seq_sims = []

	def get_avg(self):


		avg_loss, avg_seq_sim = 0, 0 # will make None later, but 0 for now since operated on like a float later
		if None not in self.losses:
			avg_loss = sum(self.losses) / len(self.losses)
		if None not in self.seq_sims:
			avg_seq_sim = sum(self.seq_sims) / len(self.seq_sims)
		
		return avg_loss, avg_seq_sim

	def add_losses(self, cel, seq_sim):
		self.losses.append(cel)
		self.seq_sims.append(seq_sim)

	def extend_losses(self, other, normalize=False):
		divisor = other.valid if normalize else 1
		self.losses.extend([loss / divisor for loss in other.losses])
		self.seq_sims.extend(other.seq_sims)

	def to_numpy(self):
		self.losses = [loss.detach().to("cpu").numpy() for loss in self.losses if isinstance(loss, torch.Tensor)]


