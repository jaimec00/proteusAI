# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		data_utils.py
description:	fetches data from proteinMPNN curated dataset and converts to DataHolder object, which can be split into training/val/testing for the model
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Thread
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
import math
import os

# ----------------------------------------------------------------------------------------------------------------------

class DataHolder():

	'''
	hold Data Objects, one each for train, test and val
	'''

	def __init__(self, 	data_path, num_train, num_val, num_test, 
						batch_tokens=16384, max_batch_size=128, 
						max_seq_size=16384, min_seq_size=512
					):


		# define data path
		self.data_path = data_path

		# define batch and seq sizes
		self.batch_tokens = batch_tokens # max tokens per batch
		self.max_batch_size = max_batch_size # max samples per batch
		self.max_seq_size = max_seq_size # max tokens per sample
		self.min_seq_size = min_seq_size

		# load the info about clusters
		pdb_info_path = data_path / Path("list.csv")
		val_clusters_path = data_path / Path("valid_clusters.txt")
		test_clusters_path = data_path / Path("test_clusters.txt")

		pdbs_info = pd.read_csv( pdb_info_path, header=0)

		# get pdb info, as well as validation and training clusters
		with 	open(	val_clusters_path,   "r") as v, \
				open(   test_clusters_path,  "r") as t:
			val_clusters = [int(i) for i in v.read().split("\n") if i]
			test_clusters = [int(i) for i in t.read().split("\n") if i]

		# seperate training, validation, and testing
		train_pdbs = pdbs_info.loc[~pdbs_info.CLUSTER.isin(test_clusters + val_clusters), :]
		val_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(val_clusters), :]
		test_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(test_clusters), :]

		# define number of clusters to use for training, validation, and testing
		self.num_train = num_train if ((num_train < (len(train_pdbs.CLUSTER.drop_duplicates()))) and (num_train != -1)) else len(train_pdbs.CLUSTER.drop_duplicates())
		self.num_val = num_val if ((num_val < (len(val_pdbs.CLUSTER.drop_duplicates()))) and (num_val != -1)) else len(val_pdbs.CLUSTER.drop_duplicates())
		self.num_test = num_test if ((num_test < (len(test_pdbs.CLUSTER.drop_duplicates()))) and (num_test != -1)) else len(test_pdbs.CLUSTER.drop_duplicates())

		# get the chain and clusterID for each group
		self.train_pdbs = train_pdbs.loc[train_pdbs.CLUSTER.isin(train_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_train)), ["CHAINID", "CLUSTER"]]
		self.val_pdbs = val_pdbs.loc[val_pdbs.CLUSTER.isin(val_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_val)), ["CHAINID", "CLUSTER"]]
		self.test_pdbs = test_pdbs.loc[test_pdbs.CLUSTER.isin(test_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_test)), ["CHAINID", "CLUSTER"]]

	def load(self, data_type):
		'''
		loads the Data Objects, allowing for the objects to be retrieved afterwards
		'''
		if data_type == "train":
			self.train_data = Data(self.data_path, self.train_pdbs, self.num_train, self.batch_tokens, self.max_batch_size, self.max_seq_size, self.min_seq_size)
		elif data_type == "val":
			self.val_data = Data(self.data_path, self.val_pdbs, self.num_val, self.batch_tokens, self.max_batch_size, self.max_seq_size, self.min_seq_size)
		elif data_type == "test":	
			self.test_data = Data(self.data_path, self.test_pdbs, self.num_test, self.batch_tokens, self.max_batch_size, self.max_seq_size, self.min_seq_size)

class Data():
	def __init__(self, data_path, clusters_df, num_samples=None, batch_tokens=16384, max_batch_size=128, max_seq_size=16384, min_seq_size=512, device="cpu"):

		# path to pdbs
		self.pdb_path = data_path / Path("pdb")

		# define sizes
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size
		self.max_seq_size = max_seq_size
		self.min_seq_size = min_seq_size

		# should be cpu
		self.device = device

		# init the df w/ cluster info and the cache
		self.clusters_df = clusters_df
		self.biounit_cache = BioUnitCache()

		# data for current epoch
		self.epoch_biounits = EpochBioUnits(batch_tokens, max_batch_size, max_seq_size, min_seq_size)

		self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
		self.aa_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
		self.rev_aa_idx = {idx: aa for idx, aa in enumerate(self.amino_acids)}
		
		# randomly sample the clusters
		self.rotate_data()

	def rotate_data(self):

		# clear the data from the last epoch
		self.epoch_biounits.clear_data()

		# get random cluster representative chains
		sampled_pdbs = self.clusters_df.groupby("CLUSTER").sample(n=1)

		# init progress bar
		load_pbar = tqdm(total=len(sampled_pdbs), desc="data loading progress", unit="step")

		# define the function for loading pdbs
		def process_pdb(pdb):

			chain_id = pdb.at["CHAINID"]
			biounit = self.biounit_cache[chain_id]
			if biounit is None:
				biounit = self.add_data(pdb)
			
			chain = chain_id.split("_")[1]

			return biounit, chain

		# parallel execution
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			futures = {executor.submit(process_pdb, pdb): pdb for _, pdb in sampled_pdbs.iterrows()}
			
			# collect results
			for future in as_completed(futures):

				result = future.result()
				if result is not None:  # Ignore failed results
					biounit, chain = result
					self.epoch_biounits.add_biounit(biounit, chain)

				load_pbar.update(1)

		self.epoch_biounits.batch_data()

	def add_data(self, pdb):

		pdb_chain = pdb.at["CHAINID"]
		pdb_id, chain_id = chain_id.split("_")
		biounits = self.get_pdb_biounits(pdb_id)

		if biounits is None:
			return None

		biounit_chains = next(chain_id in chains for chains in biounits)

		biounit = get_biounit_tensors(pdb_id, biounit_chains)

		self.biounit_cache.add_biounit(biounit, [f"{pdb_id}_{chain}" for chain in biounit_chains])

	def get_pdb_biounits(self, pdbid):
		
		pdb = self.load_pdb(pdbid)
		if pdb is None:
			return None

		biounits = pdb["asmb_chains"]
		biounits = [biounit.split(',') for biounit in biounits]

		return biounits

	def get_biounit_tensors(self, pdb, chains):

		chain_indices = defaultdict(list)
		chain_start_idx = 0

		biounit_coords, biounit_labels = [], []

		for chainid in chains:

			# load the chain
			chain = self.load_pdb(f"{pdb}_{chainid}")
			if chain is None: continue

			# load the mask
			mask = chain["mask"][:, 1].bool()

			# get the labels
			seq = chain["seq"]
			labels = torch.tensor([self.aa_idx[aa] if aa in self.amino_acids else -1 for aa in seq])
			labels = labels[mask]

			# get the Ca coords
			ca = chain["xyz"][:, 1, :]
			ca = ca[mask]

			# make sure same size
			assert ca.size(0) == labels.size(0)

			# save chain indices, to seperate them when computing loss
			chain_indices[chainid] = [chain_start_idx, chain_start_idx + labels.size(0)]
			chain_start_idx += labels.size(0)

			biounit_coords.append(ca)
			biounit_labels.append(labels)

		if biounit_coords==[] or biounit_labels==[]:
			return None

		biounit_coords = torch.cat(biounit_coords, dim=0)
		biounit_labels = torch.cat(biounit_labels, dim=0)

		biounit = BioUnit(biounit_coords, biounit_labels, chain_indices)

		return biounit

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
																		[max_size - tensor.size(0)] + \
																		[tensor.size(i) for i in range(1,tensor.dim())], 
																		dtype=tensor.dtype, device=tensor.device
																	) + bias
														), dim=0
													)
											for tensor in tensor_list
										], dim=0
									)

		return pad_and_batched

	def __iter__(self):
		for batch in self.epoch_biounits.batches:
			
			seq_next_pow2 = 2**math.ceil(math.log(max(i[1] for i in batch), 2)) # next power of 2
			seq_size = max(self.min_seq_size, min(self.max_seq_size, seq_next_pow2))

			labels = []
			coords = []
			chain_masks = []

			for idx, _ in batch:
				labels.append(self.epoch_biounits[idx].labels)
				coords.append(self.epoch_biounits[idx].coords)
				
				start_idx, end_idx = self.epoch_biounits[idx].chain_idxs[self.epoch_biounits.chains[idx]]

				chain_mask = torch.ones(self.epoch_biounits[idx].labels, dtype=torch.bool, device=self.device)
				chain_mask[start:end] = False
				chain_masks.append(chain_mask)

			batch_next_pow2 = 2**math.ceil(math.log(len(batch), 2)) # next power of 2
			batch_pads = range(batch_next_pow2 - len(batch))
			for extra_batch in batch_pads:
				labels.append(-torch.ones(seq_size, dtype=labels[0].dtype, device=self.device))
				coords.append(torch.zeros(seq_size, 3, dtype=coords[0].dtype, device=self.device))
				chain_masks.append(torch.zeros(seq_size, dtype=chain_masks[0].dtype, device=self.device))

			labels = self.pad_and_batch(labels, pad_val="-one", max_size=seq_size)
			coords = self.pad_and_batch(coords, pad_val="zero", max_size=seq_size)
			chain_masks = self.pad_and_batch(chain_masks, pad_val="one", max_size=seq_size).to(torch.bool)
			key_padding_masks = labels==-1

			yield labels, coords, chain_masks, key_padding_masks

	def __len__(self):
		return len(self.epoch_biounits)
	
class BioUnitCache():
	'''
	Caches the biounits that have already been loaded from disk to memory for faster retrieval
	'''
	def __init__(self):

		self.biounits = {}

	def add_biounit(self, biounit, chain_ids):
		for chain_id in chain_ids:
			self.biounits[chain_id] = biounit 

	def __getitem__(self, chain_id):
		try:
			return self.biounits[chain_id]
		except KeyError:
			return None

class BioUnit():
	def __init__(self, coords, labels, chain_idxs):
		self.coords = coords
		self.labels = labels # no mask needed, masked vals have -1 for labels
		self.chain_idxs = chain_idxs # dict of chain [start, end)

	def __len__(self):
		return self.labels.size(0)

class EpochBioUnits():
	def __init__(self, batch_tokens, max_batch_size, max_seq_size):
		
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size 
		self.max_seq_size = max_seq_size

		self.biounits = []
		self.chains = [] # also store chain ids so can create the chain mask
		self.batches = []
	
	def add_biounit(self, chain):
		self.biounits.append(biounit)
		self.chains.append(chain)

	def clear_biounits(self):
		self.biounits = []
		self.chains = []
		self.batches = []

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, idx):
		return self.biounits[idx]

	def batch_data(self):

		# shuffle indexes when creating data
		idxs = list(range(len(self)))
		idx_size = [[i, len(self.biounits[i])] for i in idxs]

		# sort first and then chunk and randomize mini chunks, so that batches have similar sized samples, yet still random when do batch_subset
		idx_size = sorted(idx_size, key=lambda x: x[1])
		random_idx_batches = [idx_size[i:i+max_batch_size] for i in range(0, len(idx_size), self.max_batch_size)]
		for i in random_idx_batches:
			random.shuffle(i)

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
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			futures = [executor.submit(batch_subset, batch) for batch in random_idx_batches]
			
			# collect results
			for future in as_completed(futures):

				result = future.result()
				if result is not None:  # Ignore failed results
					self.batches.extend(result)

		# shuffle batches, as mini batches are ordered by number of samples (descending) due to previous logic
		random.shuffle(self.batches)

