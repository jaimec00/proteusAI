# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
from torch.nn.functional import one_hot as onehot
from torch.nn import CrossEntropyLoss as CEL

from tqdm import tqdm

from proteusAI import proteusAI
from utils.parameter_utils import HyperParameters, TrainingParameters, MASK_injection
from utils.io_utils import Output
from utils.data_utils import DataHolder

# ----------------------------------------------------------------------------------------------------------------------

class TrainingRun():
	'''
	the main class for training. holds all of the configuration arguments, and organnizes them into hyper-parameters, 
	training parameters, MASK_injection, data, and output objects
	also orchestrates the setup of training, training itself, validation, testing, outputs, etc.

	Attributes:
		hyper_parameters (HyperParameters): store model hyper_parameters
		training_parameters (TrainingParameters): stores training parameters
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
												args.min_wl, args.max_wl, args.base_wl, 
												args.d_hidden_we, args.hidden_layers_we, 
												args.d_hidden_aa, args.hidden_layers_aa,
												args.encoder_layers, args.num_heads,
												args.min_spread, args.max_spread, args.base_spread,
												args.d_hidden_attn, args.hidden_layers_attn, 
												args.temperature, args.use_model
												)
		
		self.training_parameters = TrainingParameters(  args.epochs,
														args.accumulation_steps, 
														args.lr_type, # cyclic or plataeu
														args.lr_initial_min, args.lr_initial_max, args.lr_final_min, args.lr_final_max, args.lr_cycle_length, # for cyclic
														args.lr_scale, args.lr_patience, args.lr_step, # for plataeu
														args.beta1, args.beta2, args.epsilon, # for adam optim
														args.dropout, args.label_smoothing,
														args.loss_type, args.loss_scale, 
														args.use_amp, args.use_chain_mask,
													)
		
		self.MASK_injection = MASK_injection(	args.initial_min_MASK_injection_mean, args.initial_max_MASK_injection_mean, 
												args.final_min_MASK_injection_mean, args.final_max_MASK_injection_mean, 
												args.MASK_injection_stdev, 
												args.MASK_injection_cycle_length, args.randAA_pct, args.trueAA_pct # rand and true AA pct are not implemented, want to see how the method does by itself first
											)
		
		self.data = DataHolder(	args.data_path, 
								args.num_train, args.num_val, args.num_test, 
								args.batch_tokens, args.max_batch_size, 
								args.min_seq_size, args.max_seq_size, 
								args.use_chain_mask, args.min_resolution
							)
		self.output = Output(args.out_path, args.loss_plot, args.seq_plot, args.weights_path, args.write_dot)

		self.train_losses = Losses()
		self.val_losses = Losses() # validation with all tokens being MASK
		self.val_losses_context = Losses() # validation with the same MASK injection as the training batches from current epoch
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

		self.output.log_hyperparameters(self.training_parameters, self.hyper_parameters, self.MASK_injection, self.data)

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
								self.hyper_parameters.min_wl,
								self.hyper_parameters.max_wl,
								self.hyper_parameters.base_wl,
								self.hyper_parameters.d_hidden_we,
								self.hyper_parameters.hidden_layers_we,

								self.hyper_parameters.d_hidden_aa,
								self.hyper_parameters.hidden_layers_aa,

								self.hyper_parameters.encoder_layers,
								self.hyper_parameters.num_heads, 
								self.hyper_parameters.min_spread,
								self.hyper_parameters.max_spread,
								self.hyper_parameters.base_spread,
								self.hyper_parameters.d_hidden_attn,
								self.hyper_parameters.hidden_layers_attn,

								self.training_parameters.dropout,
							)

		self.model.to(self.gpu)

		if self.hyper_parameters.use_model is not None:
			pretrained_weights = torch.load(self.hyper_parameters.use_model, map_location=self.gpu)
			self.model.load_state_dict(pretrained_weights, weights_only=True)
		
		self.training_parameters.num_params = sum(p.numel() for p in self.model.parameters())

		# compile the model
		# self.model = torch.compile(self.model)

	def setup_optim(self):
		'''
		sets up the optimizer, zeros out the gradient

		Args:
			None
		
		Returns:
			None
		'''

		self.output.log.info("loading optimizer...")
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.training_parameters.lr_step, 
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

		def cyclic_lr(epoch):

			stage = epoch / self.training_parameters.epochs
			current_min = self.training_parameters.lr_initial_min - stage*(self.training_parameters.lr_initial_min - self.training_parameters.lr_final_min)
			current_max = self.training_parameters.lr_initial_max - stage*(self.training_parameters.lr_initial_max - self.training_parameters.lr_final_max)

			midpoint = (current_max + current_min) / 2
			amplitude = (current_max - current_min) / 2
			
			lr = midpoint + amplitude*math.sin(2*math.pi*epoch/self.training_parameters.lr_cycle_length)
			
			return lr
		
		if self.training_parameters.lr_type == "plateau":
			self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=self.training_parameters.lr_scale, patience=self.training_parameters.lr_patience) 
		else:
			self.scheduler = lr_scheduler.LambdaLR(self.optim, cyclic_lr)

	def setup_loss_function(self):
		'''
		initializes the loss function

		Args:
			None

		Returns: 
			None
		'''

		self.output.log.info("loading loss function...") 
		self.loss_function = CEL(ignore_index=-1, reduction=self.training_parameters.loss_type, label_smoothing=self.training_parameters.label_smoothing)

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
							f"of batch size {self.data.batch_tokens} tokens "\
							f"for {self.training_parameters.epochs} epochs.\n" )
		
		# loop through epochs
		for epoch in range(self.training_parameters.epochs):
			epoch = Epoch(epoch, self)
			epoch.epoch_loop()

		self.output.plot_training(self.train_losses, self.val_losses, self.val_losses_context)
		self.output.save_model(self.model)
				
	def all_MASK_validation(self):
		'''run validation with no AA context'''
		self.output.log.info(f"running validation w/ no context...\n")
		self.validation()

	def batch_MASK_validation(self, epoch):
		'''run validation with the same context as the training batches from the current epoch'''
		self.output.log.info(f"running validation w/ context...\n")
		self.validation(epoch)

	def validation(self, epoch=None):		
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		val_losses = Losses()
		
		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			val_pbar = tqdm(total=len(self.data.val_data), desc="epoch_validation_progress", unit="step")
			
			# loop through validation batches
			for label_batch, coords_batch, chain_mask, key_padding_mask in self.data.val_data:
					
				batch = Batch(	label_batch, coords_batch, chain_mask, key_padding_mask, 
								epoch=epoch, use_amp=False, auto_regressive=False
							)

				if epoch is not None:
					self.MASK_injection.MASK_tokens(batch)
					# self.MASK_all(batch)
				else:
					self.MASK_all(batch)

				batch.batch_forward(self.model, self.loss_function, self.gpu)

				# store losses
				loss, seq_sim = batch.outputs.output_losses.get_avg()
				val_losses.add_losses(float(loss.item()), seq_sim)

				val_pbar.update(1)

			if epoch is None:
				self.val_losses.add_losses(*val_losses.get_avg())
			else:
				self.val_losses_context.add_losses(*val_losses.get_avg())

			self.output.log_val_losses(val_losses)

	def test(self):

		# switch to evaluation mode
		self.model.eval()
		
		self.output.log.info("loading testing data...")
		self.data.load("test")

		test_losses, test_seq_sims, test_ar_seq_sims = [], [], []

		# turn off gradient calculation
		with torch.no_grad():

			# progress bar
			test_pbar = tqdm(total=len(self.data.test_data), desc="test_progress", unit="step")

			# loop through testing batches
			for label_batch, coords_batch, chain_mask, key_padding_mask in self.data.test_data:
					
				batch = Batch(label_batch, coords_batch, chain_mask, key_padding_mask, 
								use_amp=False, 
								auto_regressive=True, 
								temp=self.hyper_parameters.temperature, 
							)

				self.MASK_all(batch)

				batch.batch_forward(self.model, self.loss_function, self.gpu)

				loss, seq_sim = batch.outputs.output_losses.get_avg()
				_, ar_seq_sim = batch.outputs.ar_output_losses.get_avg()
				
				test_losses.append(float(loss.item()) )
				test_seq_sims.append(seq_sim)
				test_ar_seq_sims.append(ar_seq_sim)

				test_pbar.update(1)
		
		self.output.log.info(f"testing loss: {sum(test_losses) / len(test_losses)}")
		self.output.log.info(f"test sequence similarity: {sum(test_seq_sims) / len(test_seq_sims)}")
		if None not in test_ar_seq_sims:
			self.output.log.info(f"test auto-regressive sequence similarity: {sum(test_ar_seq_sims) / len(test_ar_seq_sims)}")

	def MASK_all(self, batch):

		batch.predictions = onehot(torch.full((batch.size(0), batch.size(1)), 20, device=batch.predictions.device), 21).float()

class Epoch():	
	def __init__(self, epoch, training_run):

		self.training_run_parent = training_run

		self.epoch = epoch
		self.epochs = self.training_run_parent.training_parameters.epochs
		self.stage = epoch / self.epochs

		self.losses = Losses()
		
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
		self.training_run_parent.MASK_injection.calc_mean_MASK(self)
		self.training_run_parent.output.log_epoch(self, self.training_run_parent.optim, self.training_run_parent.model, self.training_run_parent.MASK_injection)

		# loop through batches
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")
		for b_idx, (label_batch, coords_batch, chain_mask, key_padding_mask) in enumerate(self.training_run_parent.data.train_data):
					
			# instantiate this batch
			batch = Batch(	label_batch, coords_batch, chain_mask, key_padding_mask, 
							b_idx=b_idx, epoch=self, 
							use_amp=self.training_run_parent.training_parameters.use_amp, 
						)

			# inject MASK tokens for prediction
			self.training_run_parent.MASK_injection.MASK_tokens(batch)
			# self.training_run_parent.MASK_all(batch)

			# learn
			batch.batch_learn()

			# compile batch losses for logging
			self.gather_batch_losses(batch)

			epoch_pbar.update(1)
		
		# print batch losses
		self.training_run_parent.output.log_epoch_losses(self, self.training_run_parent.train_losses)

		# run validation
		self.training_run_parent.batch_MASK_validation(self)
		self.training_run_parent.all_MASK_validation()

		# lr scheduler update
		self.training_run_parent.scheduler.step(self.training_run_parent.val_losses_context.losses[-1])

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()


	def gather_batch_losses(self, batch):
		self.losses.extend_losses(batch.outputs.output_losses)

class Batch():
	def __init__(self, labels, coords, chain_mask, key_padding_mask, 
					b_idx=None, epoch=None,
					use_amp=True, auto_regressive=False, temp=0.1
				):

		self.labels = labels
		self.coords = coords 
		self.chain_mask = chain_mask
		self.use_amp = use_amp

		# input is always 21 dim, last dim is mask token
		# onehot does not accept -1, make the masked labels as MASK for now, masked anyways
		self.predictions = onehot(torch.where(labels==-1, 20, labels), 21).float()

		self.key_padding_mask = key_padding_mask
		self.onehot_mask = torch.zeros(self.key_padding_mask.shape, dtype=torch.bool)

		self.b_idx = b_idx
		self.epoch_parent = epoch

		self.use_amp = use_amp
		self.auto_regressive = auto_regressive
		self.temp = temp

		self.outputs = None

	def move_to(self, device):

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

		self.outputs.get_losses(loss_function)

	def batch_backward(self):

		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.accumulation_steps
		scaler = self.epoch_parent.training_run_parent.scaler
		optim = self.epoch_parent.training_run_parent.optim

		learn_step = (self.b_idx + 1) % accumulation_steps == 0
		loss = self.outputs.output_losses.losses[-1] / accumulation_steps
		
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
				ar_output_prediction = model(self.coords, self.predictions, key_padding_mask=self.key_padding_mask, 
											auto_regressive=self.auto_regressive, temp=self.temp)
			else:
				ar_output_prediction = None

		# compute one shot also
		output_prediction = model(self.coords, self.predictions, key_padding_mask=self.key_padding_mask, auto_regressive=False)

		return ModelOutputs(output_prediction, ar_output_prediction, self.labels)

	def size(self, idx):
		return self.labels.size(idx)

class ModelOutputs():

	def __init__(self, output_predictions, ar_output_predictions, labels):
		
		self.output_predictions = output_predictions
		self.ar_output_predictions = ar_output_predictions

		self.labels = labels

		self.output_losses = Losses()
		self.ar_output_losses = Losses()

	def compute_seq_sim(self, prediction):
		
		prediction_flat = prediction.contiguous().view(-1, prediction.size(-1)) # batch*N x 20
		labels_flat = self.labels.view(-1) # batch x N --> batch*N,

		seq_predictions = torch.argmax(prediction_flat, dim=-1) # batch*N x 20 --> batch*N,
		valid_mask = labels_flat != -1 # batch*N, 
		valid_positions = torch.sum(valid_mask) # 1,
		matches = ((seq_predictions == labels_flat) & (valid_mask)).sum() # 1, 
		seq_sim = ((matches / valid_positions).float()*100).mean().item()
		
		return seq_sim

	def compute_cel(self, prediction):

		prediction = torch.softmax(prediction, dim=-1)

		# Flatten the tensor to simplify indexing for CEL calculation
		prediction_flat = prediction.contiguous().view(-1, prediciton.size(2))
		labels_flat = self.labels.view(-1)
		key_padding_mask_flat = (labels_flat == -1).view(-1)

		# Compute CEL: Use the log of the predicted probability for the true class
		# Gather the predicted probability for each true class in label_batch
		true_label_probs = prediction_flat[torch.arange(len(labels_flat)).long(), labels_flat.long()]
		log_probs = torch.log(true_label_probs + 1e-6)  # Avoid log(0) by adding a small epsilon
		cel = -log_probs  # Cross-entropy loss for each position

		# Mask out padded tokens (where key_padding_mask is True)
		valid_cel = cel[~key_padding_mask_flat]
		cel = valid_cel.mean().item() if len(valid_cel) > 0 else 0.0

		return cel

	def compute_cel_and_seq_sim(self, prediction, loss_function=None, scale=1/2000):

		cel = loss_function(prediction.view(-1, prediction.size(2)).to(torch.float32), self.labels.view(-1).long())
		if cel.isnan().any() or cel.isinf().any():
			print("num_valid: ", (self.labels!=-1).sum())

		if loss_function.reduction == "sum":
			cel = cel_sum * scale

		seq_sim = self.compute_seq_sim(prediction)

		return cel, seq_sim

	def get_losses(self, loss_function):

		for prediction, prediction_loss in zip(	[self.output_predictions, self.ar_output_predictions], 
												[self.output_losses, self.ar_output_losses]
											):

			# for some reason gives tuple to cel and None to seq sim if do 'cel, seq_sim = self.compute...'
			cel_and_seq_sim = self.compute_cel_and_seq_sim(prediction, loss_function) if prediction is not None else (None, None)
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
		'''utility when plotting losses w/ matplotlib'''
		self.losses = [loss.detach().to("cpu").numpy() for loss in self.losses if isinstance(loss, torch.Tensor)]

