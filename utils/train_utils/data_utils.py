# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		data_utils.py
description:	fetches data from proteinMPNN curated dataset and converts to DataHolder object, which can be split 
				into training/val/testing for the model. Also includes DataCleaner class for pre-processing the chains
				into biounits
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Thread
from threading import Lock
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import hashlib
import random
import psutil
import math
import os

from data.constants import alphabet, aa_2_lbl, lbl_2_aa

# ----------------------------------------------------------------------------------------------------------------------

# add safe globals
torch.serialization.add_safe_globals([defaultdict, list])

# ----------------------------------------------------------------------------------------------------------------------

class EpochBioUnits():
	def __init__(self, batch_tokens, max_batch_size, rank=0, world_size=1):
		
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size

		self.biounit_hashes = {} # store hashes of biounit samples so can order them deterministically
		self.biounits = []
		self.chains = [] # also store chain ids so can create the chain mask
		self.batches = []
	
		self.generator = torch.Generator() # on cpu, doesnt really matter
		self.rank = rank
		self.world_size = world_size

	def add_biounit(self, biounit, chain):

		# do it with a dict using the biounit hash
		biounit.sample_asmb() # sample an assembly
		self.biounit_hashes[self.hash_biounit(biounit, chain)] = [biounit.current_asmb, chain]

	def hash_biounit(self, biounit, chain):

		# hash the biounit using the coords and the chain letter so that get the same
		# data across all gpus and can split accordingly

		biounit_bytes = np.ascontiguousarray(biounit.coords.numpy()).tobytes()
		chain_bytes = chain.encode("utf-8")

		hasher = hashlib.md5()

		hasher.update(biounit_bytes)
		hasher.update(chain_bytes)

		digest_bytes = hasher.digest()

		return int.from_bytes(digest_bytes, byteorder="big")

	def clear_biounits(self):
		self.biounits = []
		self.chains = []
		self.biounit_hashes = {}
		self.batches = []

	def batch_data(self, rng=0):

		# now populate the lists after added through hashing
		# got race conditions before, this way gurantees ranks see the same order
		hashes = sorted(self.biounit_hashes.keys())
		self.biounits = [self.biounit_hashes[hash_val][0] for hash_val in hashes]
		self.chains = [self.biounit_hashes[hash_val][1] for hash_val in hashes]

		# get a list of the indexes
		idxs = list(range(len(self.biounits)))

		# also get the size of each sample for efficient batching
		idx_size = [[i, len(self.biounits[i])] for i in idxs]

		# sort first so that batches have similar sized samples
		idx_size = sorted(idx_size, key=lambda x: x[1])
		random_idx_batches = [idx_size[i:i+self.max_batch_size] for i in range(0, len(idx_size), self.max_batch_size)]

		# shuffle each mini-batch
		for i in random_idx_batches:
			random.shuffle(i) # already set the seed for random module so same for all gpus

		# send these initial batches to threads to process in parallel, and split in two recursively until the batches are the target batch size
		def batch_subset(batch_idxs):
			'''
			recursively splits batch idxs until reach target number of tokens. 
			starts at max batch dim eg 64, and splits into 2
			returns a list of lists, each inner list containing sample indexes and corresponding size
			'''
			if (sum(i[1] for i in batch_idxs) > self.batch_tokens) or (len(batch_idxs) > self.max_batch_size):
				split = len(batch_idxs) // 2
				return batch_subset(batch_idxs[:split]) + batch_subset(batch_idxs[split:])
			else: 
				return [batch_idxs]


		# parallel processing
		with ThreadPoolExecutor(max_workers=16) as executor:
			
			# submit tasks
			futures = [executor.submit(batch_subset, batch) for batch in random_idx_batches]
			
			# collect results
			for future in as_completed(futures):

				result = future.result()
				if result is not None:  # Ignore failed results
					self.batches.extend(result)

		# ok, so what i need to do is to have world size clusters that are randomly shuffled
		# this way, all gpus work on similar sized inputs, and thus are more likely to finish the fwd pass at the same time
		# first sort so that similar sized batches are next to each other
		# performs two sorts, first by the first samples index so that tiebreakers are determined this way, main sort is on batch size
		self.batches = sorted(sorted(self.batches, key=lambda x: x[0][0]), key=lambda x: len(x)) # sort the batches based on size of the batch, approximation for input size, as small batch size is typically large seq length and vice versa
		self.batches = [self.batches[gpu_rank::self.world_size] for gpu_rank in range(self.world_size)] # create mini batches for each gpu. size is (in tensor format, although not a tensor bc diff dim sizes) gpus x batches x sample x 2 (2 is idx, length)
		
		# now need to shuffle the minibatches, but shuffle in the same way for each gpu so that similarly sized batches are still grouped together
		# thinking of creating random idx list, and permuting each gpus batch this way
		batch_lengths = [len(batch) for batch in self.batches]
		shuffle_idxs = torch.randperm(min(batch_lengths), generator=self.generator.manual_seed(rng)) # use the same seed as the other gpus so that the sizes are clustered correctly
																									# use min so that all gpus have the same number of batches, at worst, drops the last n-1 batches, where n is the number of gpus, 
																									# equivilant to drop_last=True in torch.DistributedSampler
																									# introduces bias to exclude batches at tail end (long seq, small samples), but hopefully not too noticeablel, especially once expand dataset for predicted structures
		self.batches = [[batch[i] for i in shuffle_idxs] for batch in self.batches]

	def __len__(self):
		return sum(len(batch) for batch in self.batches)

	def __getitem__(self, idx):
		return self.biounits[idx]

class BioUnitCache():
	'''
	Caches the biounits that have already been loaded from disk to memory for faster retrieval
	'''
	def __init__(self):

		self.biounits = {}
		self.lock = Lock()
		self.world_size = torch.cuda.device_count()

		# check the mem allocated (using slurm env, will have to change if do something else)
		self.max_mem = int(os.environ.get("SLURM_MEM_PER_NODE")) # in MB
		self.process = psutil.Process(os.getpid()) # process object to check mem usage

	def add_biounit(self, biounit, biounit_id):

		# check current mem allocation, if > 90% of allocated, do not add to biounit cache,
		# will need to load anything that is not on cache from disk
		current_mem = (self.process.memory_info().rss * self.world_size) / (1024**2) # convert to megabytes
		if current_mem < (self.max_mem * 0.9):
			with self.lock:
				self.biounits[biounit_id] = biounit 

	def __getitem__(self, biounit_id):
		with self.lock:
			try:
				return self.biounits[biounit_id]
			except KeyError:
				return None

class BioUnit():
	def __init__(self, biounit_dict, ca_only=False, max_seq_size=float("inf")):

		self.coords = biounit_dict["coords"]
		if ca_only: 
			self.coords = self.coords[:, 1, :].unsqueeze(1) # idx 1 of dim 1 is Ca
		
		self.labels = biounit_dict["labels"] # no mask needed, masked vals have -1 for labels
		self.chain_idxs = biounit_dict["chain_idxs"] # dict of chain [start, end)
		self.seq_sims = biounit_dict["seq_sims"] # dict of dicts, indicating seq sim of each chain to all others, outer dicts contains each chain as values, inner dict is value of outer, with keys of inner dict being all other chains, and values being the seq sims
		self.asmb_xforms = [xform for xform in biounit_dict["asmb_xforms"] if xform.size(0)*self.labels.size(0) <= max_seq_size] # transforms to get biological assembly. list of tensors. first dim of each tensor is the number of copies
		self.size = self.labels.size(0)

	def sample_asmb(self):
		xform = random.sample(self.asmb_xforms, 1)[0] # N x 4 x 4
		num_copies = xform.size(0)
		R = xform[:, :3, :3] # num_copies x 3 x 3
		T = xform[:, :3, 3] # num_copies x 3

		coords = (torch.matmul(self.coords.unsqueeze(0), R.transpose(1,2).unsqueeze(1)) + T.view(num_copies, 1, 1, 3)).view(-1,self.coords.size(1),3) # 1 x N x A x 3 @ n x 1 x 3 x 3 + n x 1 x 1 x 3 --> n*N x A x 3

		# adjust sizes based on the number of copies made
		chain_idxs = {chain: [[idxs[0] + self.size*i, idxs[1] + self.size*i] for i in range(num_copies)] for chain, idxs in self.chain_idxs.items()}
		labels = self.labels.repeat(num_copies)

		self.current_asmb = BioUnit({"coords": coords, "labels": labels, "chain_idxs": chain_idxs, "seq_sims": self.seq_sims, "asmb_xforms": [xform]})

	def __len__(self):
		return self.size

class DataBatch():
	'''
	class for a batch to process, used in training_run
	'''
	def __init__(self, epoch_biounits, batch, seq_size, batch_pads, use_chain_mask, homo_thresh, device="cpu"):
		
		# initialize a list of sample for each
		labels = [epoch_biounits[idx].labels for idx, _ in batch]
		coords = [epoch_biounits[idx].coords for idx, _ in batch]
		self.chain_idxs = [[chain_copy for chain in epoch_biounits[idx].chain_idxs.values() for chain_copy in chain] for idx, _ in batch]# append the [start,stop] idxs of each chain in the sample, chain ids not necessary
		homo_masks = [self.get_homo_mask(epoch_biounits, idx, homo_thresh, device) for idx, _ in batch]
		chain_masks = [self.get_chain_mask(epoch_biounits, idx, use_chain_mask, device) for idx, _ in batch]

		# pad in batch dim
		for extra_batch in batch_pads:
			labels.append(-torch.ones(seq_size, dtype=labels[0].dtype, device=device))
			coords.append(torch.zeros(seq_size, 3, 3, dtype=coords[0].dtype, device=device))
			chain_masks.append(torch.zeros(seq_size, dtype=chain_masks[0].dtype, device=device))
			homo_masks.append(torch.zeros(seq_size, dtype=homo_masks[0].dtype, device=device))
			# no need to pad chain_idxs

		# pad in seq sim
		self.labels = self.pad_and_batch(labels, pad_val="-one", max_size=seq_size)
		self.coords = self.pad_and_batch(coords, pad_val="zero", max_size=seq_size)
		self.chain_masks = self.pad_and_batch(chain_masks, pad_val="zero", max_size=seq_size).to(torch.bool)
		self.homo_masks = self.pad_and_batch(homo_masks, pad_val="zero", max_size=seq_size).to(torch.bool)
		self.key_padding_masks = self.labels==-1

	def get_homo_mask(self, epoch_biounits, idx, homo_thresh, device):

		homo_mask = torch.zeros(epoch_biounits[idx].labels.shape, dtype=torch.bool, device=device)
		seq_sims = epoch_biounits[idx].seq_sims[epoch_biounits.chains[idx]]
		homo_chains = [chain for chain, seq_sim in seq_sims.items() if seq_sim > homo_thresh] # list of chain identifiers whose tm score is greater than the threshold compared to the representative chain
		if homo_chains: # skip if the list is empty
			# list of [start, stop] idxs 
			homo_idxs = [epoch_biounits[idx].chain_idxs[chain] for chain in homo_chains]
			for chain_copies in homo_idxs:
				for start, stop in chain_copies:
					homo_mask[start:stop] = True

		return homo_mask

	def get_chain_mask(self, epoch_biounits, idx, use_chain_mask, device):

		if use_chain_mask:
			chain_mask = torch.zeros(epoch_biounits[idx].labels.shape, dtype=torch.bool, device=device)
			for start_idx, end_idx in epoch_biounits[idx].chain_idxs[epoch_biounits.chains[idx]]:
				chain_mask[start_idx:end_idx] = True
		else:
			chain_mask = torch.ones(epoch_biounits[idx].labels.shape, dtype=torch.bool, device=device)

		return chain_mask

	def pad_and_batch(self, tensor_list, pad_val="zero", max_size=10000):

		pad_options = {
			"zero": (torch.zeros, 1, 0),
			"one": (torch.ones, 1, 0),
			"-one": (torch.ones, -1, 0),
			"inf": (torch.zeros, 1, float("inf")),
		}
		try:
			pad, weight, bias = pad_options[pad_val]
		except KeyError:
			raise ValueError(f"invalid padding option: {pad_val=}")

		pad_and_batched = torch.stack(
										[
											torch.cat(
														(	tensor, 
															weight*pad(
																		tuple([max_size - tensor.size(0)] + \
																		[tensor.size(i) for i in range(1,tensor.dim())]), 
																		dtype=tensor.dtype, device=tensor.device
																	) + bias
														), dim=0
													)
											for tensor in tensor_list
										], dim=0
									)

		return pad_and_batched

class DataHolder():

	'''
	hold Data Objects, one each for train, test and val
	using multi chain or single chain version is not handled here, as both are cleaned so that they are in the same format, 
	just make sure you run the data cleaners and specify the correct path of the CLEANED data. will add option in the data
	cleaners to automatically download and unpack later, but here are the urls for now

	single-chain: https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/{chain_set.jsonl,chain_set_splits.json}
	multi-chain: https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz 
	'''

	def __init__(self, 	data_path, num_train, num_val, num_test, 
						batch_tokens=16384, max_batch_size=128, 
						min_seq_size=64, max_seq_size=16384,
						use_chain_mask=True,
						max_resolution=3.5,
						ca_only=True, # return ca only data or full backbone
						homo_thresh=0.70,
						rank=0, world_size=1, rng=0
					):


		# define data path
		self.data_path = Path(data_path)
		self.rng = rng

		# define batch and seq sizes
		self.batch_tokens = batch_tokens # max tokens per batch
		self.max_batch_size = max_batch_size # max samples per batch
		self.max_seq_size = max_seq_size # max tokens per sample
		self.min_seq_size = min_seq_size

		# whether to mask non-cluter-representative chains in the biounit
		self.use_chain_mask = use_chain_mask
		self.ca_only = ca_only
		self.homo_thresh = homo_thresh

		# for distributed training
		self.rank = rank
		self.world_size = world_size

		# load the info about clusters
		pdb_info_path = data_path / Path("list.csv")
		val_clusters_path = data_path / Path("valid_clusters.txt")
		test_clusters_path = data_path / Path("test_clusters.txt")

		# load the df with pdb info
		pdbs_info = pd.read_csv( pdb_info_path, header=0, engine='python') # use python engine to interpret list as a list properly

		# filter based on resolution
		pdbs_info = pdbs_info.loc[pdbs_info.RESOLUTION <= max_resolution, :]

		# filter out long sequences

		# get indices of each chains biounit sizes that are less than max_seq_size
		pdbs_info["VALID_IDX"] = pdbs_info.loc[:, "BIOUNIT_SIZE"].apply(lambda x: [i for i, size in enumerate(str(x).split(";")) if int(size) <= max_seq_size])
		# remove the indices
		pdbs_info.BIOUNIT = pdbs_info.apply(lambda row: [str(row.BIOUNIT).split(";")[idx] for idx in row.VALID_IDX], axis=1)
		pdbs_info.BIOUNIT_SIZE = pdbs_info.apply(lambda row: [str(row.BIOUNIT_SIZE).split(";")[idx] for idx in row.VALID_IDX], axis=1)
		# remove any chains who do not have a biounit after the length filter, and remove the VALID IDX column
		pdbs_info = pdbs_info.loc[pdbs_info.BIOUNIT.apply(lambda x: len(x)>0), [col for col in pdbs_info.columns if col != "VALID_IDX"]].reset_index(drop=True)
		# clusters formatted differently in multi and single chain, so convert them to string for consistency
		pdbs_info.CLUSTER = pdbs_info.apply(lambda row: str(row.CLUSTER), axis=1)

		# get pdb info, as well as validation and training clusters
		with 	open(	val_clusters_path,   "r") as v, \
				open(   test_clusters_path,  "r") as t:
			val_clusters = [str(i) for i in v.read().split("\n") if i]
			test_clusters = [str(i) for i in t.read().split("\n") if i]

		# seperate training, validation, and testing
		train_pdbs = pdbs_info.loc[~pdbs_info.CLUSTER.isin(test_clusters + val_clusters), :]
		val_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(val_clusters), :]
		test_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(test_clusters), :]

		# define number of clusters to use for training, validation, and testing
		self.num_train = num_train if ((num_train < (len(train_pdbs.CLUSTER.drop_duplicates()))) and (num_train != -1)) else len(train_pdbs.CLUSTER.drop_duplicates())
		self.num_val = num_val if ((num_val < (len(val_pdbs.CLUSTER.drop_duplicates()))) and (num_val != -1)) else len(val_pdbs.CLUSTER.drop_duplicates())
		self.num_test = num_test if ((num_test < (len(test_pdbs.CLUSTER.drop_duplicates()))) and (num_test != -1)) else len(test_pdbs.CLUSTER.drop_duplicates())

		# get the chain and clusterID for each group
		self.train_pdbs = train_pdbs.loc[train_pdbs.CLUSTER.isin(train_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_train)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.val_pdbs = val_pdbs.loc[val_pdbs.CLUSTER.isin(val_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_val)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.test_pdbs = test_pdbs.loc[test_pdbs.CLUSTER.isin(test_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_test)), ["CHAINID", "CLUSTER", "BIOUNIT"]]

	def load(self, data_type):
		'''
		loads the Data Objects, allowing for the objects to be retrieved afterwards
		'''
		if data_type == "train":
			self.train_data = Data(self.data_path, self.train_pdbs, self.num_train, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask, self.ca_only, self.homo_thresh, self.rank, self.world_size, self.rng)
		elif data_type == "val":
			self.val_data = Data(self.data_path, self.val_pdbs, self.num_val, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask, self.ca_only, self.homo_thresh, self.rank, self.world_size, self.rng)
		elif data_type == "test":	
			self.test_data = Data(self.data_path, self.test_pdbs, self.num_test, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask, self.ca_only, self.homo_thresh, self.rank, self.world_size, self.rng)

class Data():
	def __init__(self, data_path, clusters_df, num_samples=None, batch_tokens=16384, max_batch_size=128, min_seq_size=512, max_seq_size=16384, use_chain_mask=True, ca_only=True, homo_thresh=0.70, rank=0, world_size=1, rng=0, device="cpu"):

		# path to pdbs
		self.pdb_path = data_path / Path("pdb")

		# define sizes
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size
		self.max_seq_size = max_seq_size
		self.min_seq_size = min_seq_size
		self.use_chain_mask = use_chain_mask
		self.ca_only = ca_only
		self.homo_thresh = homo_thresh # if tm_score is above this, then corresponding chain sequences are masked

		self.rank = rank
		self.world_size = world_size

		# should be cpu
		self.device = device

		# init the df w/ cluster info and the cache
		self.clusters_df = clusters_df
		self.biounit_cache = BioUnitCache()

		# data for current epoch
		self.epoch_biounits = EpochBioUnits(batch_tokens, max_batch_size, rank, world_size)

		# randomly sample the clusters
		self.rng = rng # start with hardcoded rng, will increment
		self.rotate_data() # will have the rng seed be updated each rotation

	def rotate_data(self):

		# clear the data from the last epoch
		self.epoch_biounits.clear_biounits()

		# get random cluster representative chains
		sampled_pdbs = self.clusters_df.groupby("CLUSTER").sample(n=1, random_state=self.rng)

		# init progress bar
		load_pbar = tqdm(total=len(sampled_pdbs), desc="data loading progress", unit="step")

		# set random module seed
		random.seed(a=self.rng)

		# define the function for loading pdbs
		def process_pdb(pdb):

			biounit_id = random.choice(pdb.at["BIOUNIT"]) # some chains have multiple biounits, choose from list of biounits
			biounit = self.biounit_cache[biounit_id]
			if biounit is None:
				biounit = self.add_data(biounit_id)
			
			chain = pdb.at["CHAINID"].split("_")[1]

			return biounit, chain

		# parallel execution
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			futures = {executor.submit(process_pdb, pdb): pdb for _, pdb in sampled_pdbs.iterrows()}
			
			# collect results
			for future in as_completed(futures):

				result = future.result()
				if None not in result:  # Ignore failed results
					biounit, chain = result
					self.epoch_biounits.add_biounit(biounit, chain)

				load_pbar.update(1)

		self.epoch_biounits.batch_data(self.rng)

		# update rng
		self.rng += 1

	def add_data(self, biounit_id):

		biounit_path = self.pdb_path / Path(f"{biounit_id.split('_')[0][1:3]}/{biounit_id}.pt") 
		biounit_raw = torch.load(biounit_path, weights_only=True)
		biounit = BioUnit(biounit_raw, self.ca_only, self.max_seq_size) # the seq size of the biounits in list.csv is the minimum of each biounit, need to filter out assemblys create by applying xforms that are too big
		self.biounit_cache.add_biounit(biounit, biounit_id)

		return biounit

	def get_seq_size(self, batch):
		'''
		compute max seq size, will add kwarg to just return the max length since i am not using triton anymore
		seq size are powers of two, unless it is between 2^n-1 and 2^(n-1 + 1/2)
		where n is log_2(max_seq_size) this is because the diff between 2^n-1 and 2^n is very big
		'''
		max_seq_size = max(i[1] for i in batch)
		# seq_pow2 = math.log(max_seq_size, 2)
		# intermediate_pow = math.log(self.max_seq_size//2 + self.max_seq_size//4, 2)
		# small_pow = math.log(self.max_seq_size//2, 2)
		
		# if (seq_pow2 > small_pow) and (seq_pow2 < intermediate_pow):
		# 	seq_pow = intermediate_pow
		# else:
		# 	seq_pow = math.ceil(seq_pow2)

		# seq_next_pow2 = int(2**seq_pow) # next power of 2

		# seq_size = max(self.min_seq_size, min(self.max_seq_size, seq_next_pow2))

		return max_seq_size # seq_size

	def get_batch_pad_size(self, batch):

		# batch_next_pow2 = 2**math.ceil(math.log(len(batch), 2)) # next power of 2
		# batch_pads = range(batch_next_pow2 - len(batch))
		return range(0) # batch_pads

	def __iter__(self):

		for batch in self.epoch_biounits.batches[self.rank]: # get the batch indices for this gpu

			batch_seq_size = self.get_seq_size(batch)
			batch_pad_size = self.get_batch_pad_size(batch)

			data_batch = DataBatch(self.epoch_biounits, batch, batch_seq_size, batch_pad_size, self.use_chain_mask, self.homo_thresh)

			yield data_batch

	def __len__(self):
		return len(self.epoch_biounits)
	
class MultiChainDataCleaner():

	'''
	class to pre-process pdbs. it is much faster to compute each chains corresponding biounit and save the complete biounits to disk before 
	training, rather than reconstructing the biounits from the individual chains on the fly like pmpnn. more disk space, but definitley worth it
	'''

	def __init__(self, 	data_path=Path("/scratch/hjc2538/projects/proteusAI/pdb_2021aug02"),
						new_data_path=Path("/scratch/hjc2538/projects/proteusAI/pdb_2021aug02_filtered"),
						test=True
				):

		# define paths
		self.data_path = data_path
		self.pdb_path = self.data_path / Path("pdb")

		# define output path
		self.new_data_path = new_data_path
		self.val_clusters_path = Path("valid_clusters.txt")
		self.test_clusters_path = Path("test_clusters.txt")
		self.all_clusters_path = Path("list.csv")

			
		# read which clusters are for validation and for testing
		with    open( self.data_path / self.val_clusters_path, 	"r") as v, \
				open( self.data_path / self.test_clusters_path,	"r") as t:

			self.val_clusters = [int(i) for i in v.read().split("\n") if i]
			self.test_clusters = [int(i) for i in t.read().split("\n") if i]

		# load the cluster dataframe, and remove high resolution and non canonical chains
		self.cluster_info = pd.read_csv(self.data_path / self.all_clusters_path, header=0)

		if test: # only include pdbs in 'l3' pdb section (e.g. 4l3q)
			self.cluster_info = self.cluster_info.loc[self.cluster_info.CHAINID.apply(lambda x: x[1:3]).eq("l3")]

		# initialize BIOUNIT and PDB columns
		self.cluster_info["BIOUNIT"] = None
		self.cluster_info["BIOUNIT_SIZE"] = None
		self.cluster_info["PDB"] = self.cluster_info.CHAINID.apply(lambda x: x.split("_")[0])

		# compute biounits
		self.get_pmpnn_pdbs()

	def get_pmpnn_pdbs(self):
		
		# no gradients
		with torch.no_grad():

			# split into chunks and send different subset of pdbs to each gpu 
			all_pdbs = self.cluster_info.PDB.drop_duplicates()

			# store each processes's results (dataframe that maps chainid to biounit) in list
			pdb_biounits = {}

			# parallel execution		
			pbar = tqdm(total=len(all_pdbs), desc="biounit computatino progress", unit="step")
			with ThreadPoolExecutor(max_workers=8) as executor:
				
				# submit tasks
				futures = {executor.submit(self.compute_biounits, pdb): pdb for _, pdb in all_pdbs.items()}
				
				# collect results
				for future in as_completed(futures):

					result = future.result()
					if result is not None:  # Ignore failed results
						pdb_biounit = result
						for chain in pdb_biounit.keys():
							pdb_biounits[chain] = pdb_biounit[chain]

					pbar.update(1)

			# now deal with the results

			# assign BIOUNIT and BIOUNIT_SIZE for the chains
			self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), "BIOUNIT"] = self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), :].apply(lambda row: ';'.join(pdb_biounits[row.CHAINID]["BIOUNIT"]), axis=1)
			self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), "BIOUNIT_SIZE"] = self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), :].apply(lambda row: ';'.join([str(i) for i in pdb_biounits[row.CHAINID]["BIOUNIT_SIZE"]]), axis=1)

			# remove unused chains, i.e. the pt file doesnt exist
			self.cluster_info = self.cluster_info.dropna(subset="BIOUNIT").reset_index(drop=True)
					
			self.write_new_clusters(self.cluster_info, self.val_clusters, self.test_clusters)
			
	def compute_biounits(self, pdbid):

		pdbs_chunk = self.cluster_info.loc[self.cluster_info.PDB.eq(pdbid), :]

		pdb_biounits = defaultdict(lambda: defaultdict(list))

		# get the biounits for this pdb
		orig_chains, chain_seq_sims, biounits = self.get_pdb_biounits(pdbid)

		# get the chains that make the biounits
		biounits_flat = [f"{pdbid}_{chain}" for biounit in biounits.keys() for chain in biounit.split(",")]
		chains = pdbs_chunk.loc[:, "CHAINID"].drop_duplicates()

		# if the chains in this pdb are not part of a biounit, treat the single chain as a biounit
		single_chains = chains.loc[~chains.isin(biounits_flat)]
		for _, chain in single_chains.items():
			biounits[chain].append([torch.eye(4).unsqueeze(0)]) # append a one element list of identity xforms

		pdb_path = self.new_data_path / Path(f"pdb/{pdbid[1:3]}")

		# loop through each biounit
		for idx, biounit in enumerate(biounits.keys()):

			# get the biounit coordinates, labels, and chain masks
			bu_coords, bu_labels, bu_chain_idxs = self.get_biounit_tensors(pdbid, biounit)
			if None in [bu_coords, bu_labels, bu_chain_idxs]: continue
			bu_size = bu_labels.size(0) * min(xform.size(0) for xform in biounits[biounit]) # dim 1 of xform is number of copies, store the minimum sze of the biounit

			seq_sims = {query_chain: 
							{target_chain: 
								chain_seq_sims[orig_chains.index(query_chain), orig_chains.index(target_chain)].item() for target_chain in bu_chain_idxs.keys()
							}
							for query_chain in bu_chain_idxs.keys()
						}
			
			
			# write the files
			path = pdb_path / Path(f"{pdbid}_{idx}.pt")
			self.write_files(bu_coords, bu_labels, bu_chain_idxs, seq_sims, biounits[biounit], path)

			# save the biounit info
			for chain in bu_chain_idxs.keys():
				pdb_biounits[f"{pdbid}_{chain}"]["BIOUNIT"].append(f"{pdbid}_{idx}")
				pdb_biounits[f"{pdbid}_{chain}"]["BIOUNIT_SIZE"].append(bu_size)

		# only return if not empty
		if pdb_biounits.keys():
			return pdb_biounits
		else:
			return None


	def write_files(self, coords, labels, chain_idxs, seq_sims, asmb_xforms, path):

		biounit_data = {
			"coords": coords, # N x 3 x 3
			"labels": labels, # N,
			"chain_idxs": chain_idxs, # {CHAINID: [start, end(exclusive)]}
			"seq_sims": seq_sims, # {query_chain1: {target_chain1: score, target_chain2: score}, query_chain2: {target_chain1: score, target_chain2: score}} etc
			"asmb_xforms": asmb_xforms
		}

		if not path.parent.exists():
			path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(biounit_data, path)

	def get_pdb_biounits(self, pdbid):
		
		pdb = self.load_pdb(pdbid)

		# first get the seq sims of the chains
		chains = pdb["chains"]
		chain_seq_sims = pdb["tm"][:, :, 1] # only want the sequence similarity between chains # num_chains x num_chains 

		# for duplicate biounits (meaning same chain compositions, but different 3D configurations of the copies), simply append the corresponding xform, so will do this with default dict
		asmb_xforms = defaultdict(list)
		for idx, biounit in enumerate(pdb["asmb_chains"]):
			asmb_xforms[biounit].append(pdb[f"asmb_xform{idx}"])

		return chains, chain_seq_sims, asmb_xforms

	def get_biounit_tensors(self, pdb, chains):

		chain_indices = {}
		chain_start_idx = 0
		biounit_coords, biounit_labels = [], []

		for chainid in chains.split(","):

			# load the chain
			chain = self.load_pdb(f"{pdb}_{chainid}")
			if chain is None: continue

			# load the mask
			mask = chain["mask"][:, :3].all(dim=1) # N, Ca, and C must be True to be valid

			# get the labels
			seq = chain["seq"]
			labels = torch.tensor([aa_2_lbl(aa) for aa in seq])[mask]

			# get the coords, removes masked pos. wf embedding for Ca only model computes virtual cb between adjacent Ca, 
			# so the approx will be more innacurate for chains with missing coords in the middle, but still a better approximation 
			# that virtual N/C lies on line connecting adjacent Ca, even if not actually adjacent. alternative is no Cb for Ca next to missing coords, which is worse  
			n = chain["xyz"][:, 0, :][mask, :]
			ca = chain["xyz"][:, 1, :][mask, :]
			c = chain["xyz"][:, 2, :][mask, :]

			# make sure same size
			assert ca.size(0) == labels.size(0) == n.size(0) == c.size(0) 

			# save chain indices, to seperate them when computing loss
			chain_indices[chainid] = [chain_start_idx, chain_start_idx + labels.size(0)]
			chain_start_idx += labels.size(0)

			biounit_coords.append(torch.stack([n, ca, c], dim=1))
			biounit_labels.append(labels)

		if biounit_coords==[] or biounit_labels==[]:
			return None, None, None

		biounit_coords = torch.cat(biounit_coords, dim=0)
		biounit_labels = torch.cat(biounit_labels, dim=0)

		return biounit_coords, biounit_labels, chain_indices

	def load_pdb(self, pdbid): 
		'''
		loads a pt file from pdb or pdb_chain id
		'''
		pdb_section = Path(pdbid[1:3])
		pdb_path = self.pdb_path / pdb_section / Path(f"{pdbid}.pt")
		if pdb_path.exists():
			pdb = torch.load(pdb_path, weights_only=True, map_location="cpu")
		else:
			return None

		return pdb

	def write_new_clusters(self, cluster_info, val_clusters, test_clusters):

		# seperate training, validation, and testing
		val_pdbs = cluster_info.loc[cluster_info.CLUSTER.isin(val_clusters), :]
		test_pdbs =cluster_info.loc[cluster_info.CLUSTER.isin(test_clusters), :]

		# get lists of unique clusters
		val_clusters = "\n".join(str(i) for i in val_pdbs.CLUSTER.unique().tolist())
		test_clusters = "\n".join(str(i) for i in test_pdbs.CLUSTER.unique().tolist())

		with    open(   self.new_data_path / self.val_clusters_path,   "w") as v, \
				open(   self.new_data_path / self.test_clusters_path,  "w") as t:
				v.write(val_clusters)
				t.write(test_clusters)

		# save training pdbs
		cluster_info.to_csv(self.new_data_path / self.all_clusters_path, index=False)


class SingleChainDataCleaner():
	'''
	clean single chain data from Ingraham et al so that it is in same format as pmpnn multi chain data, so can use the same DataHolder object 
	'''
	def __init__(self, 	data_path, new_data_path, test):

		# define paths
		self.data_path = data_path
		self.new_data_path = new_data_path
		self.pdb_path = self.new_data_path / Path("pdb")
		self.val_clusters_path = self.new_data_path / Path("valid_clusters.txt")
		self.test_clusters_path = self.new_data_path / Path("test_clusters.txt")
		self.all_clusters_path = self.new_data_path / Path("list.csv")

		self.chain_clusters = pd.read_json(self.data_path / Path("chain_set.jsonl"), lines=True)
		self.chain_clusters_split = pd.read_json(self.data_path / Path("chain_set_splits.json"), lines=True)
			
		self.val_clusters = self.chain_clusters_split.validation[0]
		self.test_clusters = self.chain_clusters_split.test[0]
		self.train_clusters = self.chain_clusters_split.train[0]

		# need these columns, 
		# 'CHAINID', 'DEPOSITION', 'RESOLUTION', 'HASH', 'CLUSTER', 'SEQUENCE', 'BIOUNIT', 'BIOUNIT_SIZE', 'PDB'
		# DEPOSITION, RESOLUTION, HASH, not used, so just fill with none, 
		# CHAINID is pdb_chain
		# CLUSTER will be the cath_id(s). to keep the same sampling logic in data, if a chain has multiple clusters, it will have seperate rows, each with one cluster
		# BIOUNIT is the same as CHAIN_ID
		# PDB also just like chainID

		self.cluster_info = {key: [] for key in ['CHAINID', 'DEPOSITION', 'RESOLUTION', 'HASH', 'CLUSTER', 'SEQUENCE', 'BIOUNIT', 'BIOUNIT_SIZE', 'PDB']}

		# compute biounits
		self.get_single_chain_pdbs(test)

	def get_single_chain_pdbs(self, test=False):
		
		current_cluster = {0: "train", 1: "validation", 2: "test"}
		for idx, cluster in enumerate([self.train_clusters, self.val_clusters, self.test_clusters]):
			
			# init progress bar
			load_pbar = tqdm(total=len(cluster), desc=f"processing {current_cluster[idx]} clusters", unit="pdbs")

			# parallel execution
			with ThreadPoolExecutor(max_workers=8) as executor:
				
				# submit tasks
				cluster = cluster[:32] if test else cluster # just process the first 32 for each cluster if testing the cleaner
				futures = {executor.submit(self.process_pdb, pdb): pdb for pdb in cluster}
				
				# collect results
				for future in as_completed(futures):

					# processed concurrently, but only one thread does this part in serial, so each row is of the same cluster
					result = future.result()
					if result is not None:  # Ignore failed results
						pdb_info = result
						for key, val in pdb_info.items():
							self.cluster_info[key].extend(val)
					load_pbar.update(1)

		# once finished, write the list.csv, valid_clusters.txt, and test_clusters.txt files
		
		# convert to df
		cluster_info_df = pd.DataFrame(self.cluster_info)

		# write the clusters
		self.write_new_clusters(cluster_info_df, self.val_clusters, self.test_clusters)

	def write_new_clusters(self, cluster_info, val_clusters, test_clusters):

		# seperate training, validation, and testing
		val_pdbs = cluster_info.loc[cluster_info.BIOUNIT.isin(["_".join(i.split(".")) for i in val_clusters]), :] # they did the split based on pdb_chainid, not cluster
		test_pdbs =cluster_info.loc[cluster_info.BIOUNIT.isin(["_".join(i.split(".")) for i in test_clusters]), :]

		# get lists of unique clusters
		val_clusters = "\n".join(str(i) for i in val_pdbs.CLUSTER.unique().tolist())
		test_clusters = "\n".join(str(i) for i in test_pdbs.CLUSTER.unique().tolist())

		with    open(   self.val_clusters_path,   "w") as v, \
				open(   self.test_clusters_path,  "w") as t:
				v.write(val_clusters)
				t.write(test_clusters)

		# save training pdbs
		cluster_info.to_csv(self.all_clusters_path, index=False)

	# define the function for loading pdbs
	def process_pdb(self, pdb):

		# extract info used by Data Class from the json
		pdb_info =  self.chain_clusters.loc[self.chain_clusters.loc[:, "name"].eq(pdb), :].iloc[0, :]
		if pdb_info.empty: return None

		CHAINID = "_".join(str(pdb_info.at["name"]).split("."))
		DEPOSITION, RESOLUTION, HASH = None, 0, None # set resolution to 0 so not filtered out, idk the resolutions, can prob write a script to fetch this data online, but assuming resolution is good enough for now
		CLUSTER = pdb_info.at["CATH"]
		SEQUENCE = pdb_info.at["seq"]
		BIOUNIT = CHAINID # only single chains
		PDB = CHAINID.split("_")[0]

		COORDS = pdb_info.at["coords"]
		N = COORDS["N"]
		Ca = COORDS["CA"]
		C = COORDS["C"]

		# missing coords are represented like this
		missing_coords = [None, None, None]

		# false is not masked, did this when started and not enough time to make the whole codebase consistent so keeping it
		mask = [(n!=missing_coords) & (ca!=missing_coords) & (c!=missing_coords) for n, ca, c in zip(N, Ca, C)]

		# remove missing coords, not a problem for full backbone model, but makes Cb approximation for Ca only model more innacurate, but better than skipping Ca whose sequence neighbor is missing
		N = [coords for coords, valid in zip(N, mask) if valid]
		Ca = [coords for coords, valid in zip(Ca, mask) if valid]
		C = [coords for coords, valid in zip(C, mask) if valid]

		# make into tensor
		coords = torch.tensor([N, Ca, C]).transpose(0,1) # 3(NCaC) x N x 3(xyz) --> N x 3(NCaC) x 3(xyz)

		# mask is baked into labels (-1)
		labels = torch.tensor([aa_2_lbl(aa) for aa in SEQUENCE])
		labels = labels[torch.tensor(mask)] # remove missing coords

		# single chain, so just [0, len(chain)]
		# did default dict for pmpnn so same here
		chain_idxs = {}
		chain_idxs[CHAINID.split("_")[1]] = [0, len(labels)]

		# create the dictionary
		chain_dict = {"coords": coords, "labels": labels, "chain_idxs": chain_idxs, "seq_sims": {chain: {chain: 1.00} for chain in chain_idxs.keys()}, "asmb_xforms": [torch.eye(4).unsqueeze(0)]}

		# define path and save
		pdb_section =  self.pdb_path / Path(PDB[1:3])
		if not pdb_section.exists():
			pdb_section.mkdir(parents=True, exist_ok=True)
		chain_path = pdb_section / Path(f"{BIOUNIT}.pt")
		torch.save(chain_dict, chain_path)

		# last part needed for the list.csv
		BIOUNIT_SIZE = labels.size(0)
		
		# define local cluster info dict
		cluster_info = {key: [] for key in ['CHAINID', 'DEPOSITION', 'RESOLUTION', 'HASH', 'CLUSTER', 'SEQUENCE', 'BIOUNIT', 'BIOUNIT_SIZE', 'PDB']}

		# now make a seperate entry for each cluster
		for cluster in CLUSTER:
			cluster_info["CHAINID"].append(CHAINID) 
			cluster_info["DEPOSITION"].append(DEPOSITION)
			cluster_info["RESOLUTION"].append(RESOLUTION)
			cluster_info["HASH"].append(HASH)
			cluster_info["CLUSTER"].append(cluster)
			cluster_info["SEQUENCE"].append(SEQUENCE)
			cluster_info["BIOUNIT"].append(BIOUNIT)
			cluster_info["BIOUNIT_SIZE"].append(BIOUNIT_SIZE)
			cluster_info["PDB"].append(PDB)		

		return cluster_info

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--clean_pdbs", default=True, type=bool, help="whether to clean the pdbs")
	parser.add_argument("--data_path", default=Path("/scratch/hjc2538/projects/proteusAI/data/single_chain/raw"), type=Path, help="path where decompressed the PMPNN dataset")
	parser.add_argument("--new_data_path", default=Path("/scratch/hjc2538/projects/proteusAI/data/single_chain/processed"), type=Path, help="path to write the filtered dataset")
	parser.add_argument("--single_chain", default=True, type=bool, help="whether to clean the single chain dataset or multi chain")
	parser.add_argument("--test", default=False, type=bool, help="test the cleaner or run")

	args = parser.parse_args()

	if args.clean_pdbs:
		if args.single_chain:
			data_cleaner = SingleChainDataCleaner(data_path=args.data_path, new_data_path=args.new_data_path, test=args.test)
		else:
			data_cleaner = MultiChainDataCleaner(data_path=args.data_path, new_data_path=args.new_data_path, test=args.test)
