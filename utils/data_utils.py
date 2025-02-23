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
import random
import math
import os
import argparse

from utils.io_utils import Output


# ----------------------------------------------------------------------------------------------------------------------

# add safe globals
torch.serialization.add_safe_globals([defaultdict, list])

# ----------------------------------------------------------------------------------------------------------------------

class EpochBioUnits():
	def __init__(self, batch_tokens, max_batch_size):
		
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size 

		self.biounits = []
		self.chains = [] # also store chain ids so can create the chain mask
		self.batches = []
	
	def add_biounit(self, biounit, chain):
		self.biounits.append(biounit)
		self.chains.append(chain)

	def clear_biounits(self):
		self.biounits = []
		self.chains = []
		self.batches = []

	def batch_data(self):

		# get a list of the indexes
		idxs = list(range(len(self.biounits)))

		# also get the size of each sample for efficient batching
		idx_size = [[i, len(self.biounits[i])] for i in idxs]

		# sort first so that batches have similar sized samples
		idx_size = sorted(idx_size, key=lambda x: x[1])
		random_idx_batches = [idx_size[i:i+self.max_batch_size] for i in range(0, len(idx_size), self.max_batch_size)]

		# shuffle each mini-batch
		for i in random_idx_batches:
			random.shuffle(i)

		# send these initial batches to threads to process in parallel, and split in two recursively until the batches are the target batch size
		def batch_subset(batch_idxs):
			'''
			recursively splits batch idxs until reach target number of tokens. 
			starts at max batch dim eg 64, and splits into 2
			returns a list of lists, each inner list containing sample indexes and corresponding size
			'''
			# print(len(batch_idxs), sum(i[1] for i in batch_idxs))
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

		# shuffle batches, as mini batches are ordered by number of samples (ascending) due to previous logic
		random.shuffle(self.batches)

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, idx):
		return self.biounits[idx]

class BioUnitCache():
	'''
	Caches the biounits that have already been loaded from disk to memory for faster retrieval
	'''
	def __init__(self):

		self.biounits = {}
		self.lock = Lock()

	def add_biounit(self, biounit, biounit_id):
		with self.lock:
			self.biounits[biounit_id] = biounit 

	def __getitem__(self, biounit_id):
		with self.lock:
			try:
				return self.biounits[biounit_id]
			except KeyError:
				return None

class BioUnit():
	def __init__(self, biounit_dict):
		self.coords = biounit_dict["coords"]
		self.labels = biounit_dict["labels"] # no mask needed, masked vals have -1 for labels
		self.labels = torch.where(self.labels==20, -1, self.labels) # don't predict NCAA
		self.chain_idxs = biounit_dict["chain_idxs"] # dict of chain [start, end)

	def __len__(self):
		return self.labels.size(0)

class DataHolder():

	'''
	hold Data Objects, one each for train, test and val
	'''

	def __init__(self, 	data_path, num_train, num_val, num_test, 
						batch_tokens=16384, max_batch_size=128, 
						min_seq_size=64, max_seq_size=16384,
						use_chain_mask=True,
						min_resolution=3.5
					):


		# define data path
		self.data_path = data_path

		# define batch and seq sizes
		self.batch_tokens = batch_tokens # max tokens per batch
		self.max_batch_size = max_batch_size # max samples per batch
		self.max_seq_size = max_seq_size # max tokens per sample
		self.min_seq_size = min_seq_size

		# whether to mask non-cluter-representative chains in the biounit
		self.use_chain_mask = use_chain_mask

		# load the info about clusters
		pdb_info_path = data_path / Path("list.csv")
		val_clusters_path = data_path / Path("valid_clusters.txt")
		test_clusters_path = data_path / Path("test_clusters.txt")

		# load the df with pdb info
		pdbs_info = pd.read_csv( pdb_info_path, header=0, engine='python') # use python engine to interpret list as a list properly

		# filter based on resolution
		pdbs_info = pdbs_info.loc[pdbs_info.RESOLUTION <= min_resolution, :]

		# filter out long sequences

		# get indices of each chains biounit sizes that are less than max_seq_size
		pdbs_info["VALID_IDX"] = pdbs_info.loc[:, "BIOUNIT_SIZE"].apply(lambda x: [i for i, size in enumerate(x.split(";")) if int(size) <= max_seq_size])
		# remove the indices
		pdbs_info.BIOUNIT = pdbs_info.apply(lambda row: [row.BIOUNIT.split(";")[idx] for idx in row.VALID_IDX], axis=1)
		pdbs_info.BIOUNIT_SIZE = pdbs_info.apply(lambda row: [row.BIOUNIT_SIZE.split(";")[idx] for idx in row.VALID_IDX], axis=1)
		# remove any chains who do not have a biounit after the length filter, and remove the VALID IDX column
		pdbs_info = pdbs_info.loc[pdbs_info.BIOUNIT.apply(lambda x: len(x)>0), [col for col in pdbs_info.columns if col != "VALID_IDX"]].reset_index(drop=True)

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
		self.train_pdbs = train_pdbs.loc[train_pdbs.CLUSTER.isin(train_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_train)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.val_pdbs = val_pdbs.loc[val_pdbs.CLUSTER.isin(val_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_val)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.test_pdbs = test_pdbs.loc[test_pdbs.CLUSTER.isin(test_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_test)), ["CHAINID", "CLUSTER", "BIOUNIT"]]

	def load(self, data_type):
		'''
		loads the Data Objects, allowing for the objects to be retrieved afterwards
		'''
		if data_type == "train":
			self.train_data = Data(self.data_path, self.train_pdbs, self.num_train, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask)
		elif data_type == "val":
			self.val_data = Data(self.data_path, self.val_pdbs, self.num_val, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask)
		elif data_type == "test":	
			self.test_data = Data(self.data_path, self.test_pdbs, self.num_test, self.batch_tokens, self.max_batch_size, self.min_seq_size, self.max_seq_size, self.use_chain_mask)

class Data():
	def __init__(self, data_path, clusters_df, num_samples=None, batch_tokens=16384, max_batch_size=128, min_seq_size=512, max_seq_size=16384, use_chain_mask=True, device="cpu"):

		# path to pdbs
		self.pdb_path = data_path / Path("pdb")

		# define sizes
		self.batch_tokens = batch_tokens
		self.max_batch_size = max_batch_size
		self.max_seq_size = max_seq_size
		self.min_seq_size = min_seq_size
		self.use_chain_mask = use_chain_mask

		# should be cpu
		self.device = device

		# init the df w/ cluster info and the cache
		self.clusters_df = clusters_df
		self.biounit_cache = BioUnitCache()

		# data for current epoch
		self.epoch_biounits = EpochBioUnits(batch_tokens, max_batch_size)

		# randomly sample the clusters
		self.rotate_data()

	def rotate_data(self):

		# clear the data from the last epoch
		self.epoch_biounits.clear_biounits()

		# get random cluster representative chains
		sampled_pdbs = self.clusters_df.groupby("CLUSTER").sample(n=1)

		# init progress bar
		load_pbar = tqdm(total=len(sampled_pdbs), desc="data loading progress", unit="step")

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

		self.epoch_biounits.batch_data()

	def add_data(self, biounit_id):

		biounit_path = self.pdb_path / Path(f"{biounit_id.split('_')[0][1:3]}/{biounit_id}.pt") 
		biounit_raw = torch.load(biounit_path, weights_only=True)
		biounit = BioUnit(biounit_raw)
		self.biounit_cache.add_biounit(biounit, biounit_id)

		return biounit


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

	def __iter__(self):
		for batch in self.epoch_biounits.batches:
			
			
			# seq size are powers of two, unless it is between 2^n-1 and 2^(n-1 + 1/2)
			# where n is log_2(max_seq_size) this is because the diff between 2^n-1 and 2^n is very big
			seq_pow2 = math.log(max(i[1] for i in batch), 2)
			intermediate_pow = math.log(self.max_seq_size//2 + self.max_seq_size//4, 2)
			small_pow = math.log(self.max_seq_size//2, 2)
			
			if (seq_pow2 > small_pow) and (seq_pow2 < intermediate_pow):
				seq_pow = intermediate_pow
			else:
				seq_pow = math.ceil(seq_pow2)

			seq_next_pow2 = int(2**seq_pow) # next power of 2

			seq_size = max(self.min_seq_size, min(self.max_seq_size, seq_next_pow2))

			labels = []
			coords = []
			chain_masks = []

			for idx, _ in batch:
				labels.append(self.epoch_biounits[idx].labels)
				coords.append(self.epoch_biounits[idx].coords)

				if self.use_chain_mask:
					
					start_idx, end_idx = self.epoch_biounits[idx].chain_idxs[self.epoch_biounits.chains[idx]]

					chain_mask = torch.ones(self.epoch_biounits[idx].labels.shape, dtype=torch.bool, device=self.device)
					chain_mask[start_idx:end_idx] = False
				else:
					# mask of zeros i.e. no mask
					chain_mask = torch.zeros(self.epoch_biounits[idx].labels.shape, dtype=torch.bool, device=self.device)

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
	
class DataCleaner():

	'''
	class to pre-process pdbs. it is much faster to compute each chains corresponding biounit and save the complete biounits to disk before 
	training, rather than reconstructing the biounits from the individual chains on the fly
	'''

	def __init__(self, 	data_path=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02"),
						new_data_path=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered"),
						pdb_path=Path("pdb"),
						all_clusters_path=Path("list.csv"),
						val_clusters_path=Path("valid_clusters.txt"),
						test_clusters_path=Path("test_clusters.txt"),
						test=True
				):

		# define paths
		self.data_path = data_path
		self.pdb_path = self.data_path / pdb_path

		# define output path
		self.output = Output(new_data_path)

			
		# read which clusters are for validation and for testing
		with    open( self.data_path / val_clusters_path, 	"r") as v, \
				open( self.data_path / test_clusters_path,	"r") as t:

			self.val_clusters = [int(i) for i in v.read().split("\n") if i]
			self.test_clusters = [int(i) for i in t.read().split("\n") if i]

		# load the cluster dataframe, and remove high resolution and non canonical chains
		self.cluster_info = pd.read_csv(self.data_path / all_clusters_path, header=0)

		if test: # only include pdbs in 'l3' pdb section (e.g. 4l3q)
			self.cluster_info = self.cluster_info.loc[self.cluster_info.CHAINID.apply(lambda x: x[1:3]).eq("l3")]

		# initialize BIOUNIT and PDB columns
		self.cluster_info["BIOUNIT"] = None
		self.cluster_info["BIOUNIT_SIZE"] = None
		self.cluster_info["PDB"] = self.cluster_info.CHAINID.apply(lambda x: x.split("_")[0])

		# useful conversions between aa and idx
		self.amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
		self.aa_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
		self.rev_aa_idx = {idx: aa for idx, aa in enumerate(self.amino_acids)}
		
		# to keep track of dataset statistics
		self.aa_distributions = {aa: 0 for aa in range(len(self.amino_acids))}
		self.seq_lengths = []
		self.max_seq_len = 0

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
			self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), "BIOUNIT_SIZE"] = self.cluster_info.loc[self.cluster_info.CHAINID.isin(pdb_biounits.keys()), :].apply(lambda row: ';'.join(pdb_biounits[row.CHAINID]["BIOUNIT_SIZE"]), axis=1)

			# remove unused chains, i.e. the pt file doesnt exist
			self.cluster_info = self.cluster_info.dropna(subset="BIOUNIT").reset_index(drop=True)
					
			self.output.write_new_clusters(self.cluster_info, self.val_clusters, self.test_clusters)
			
	def compute_biounits(self, pdbid):

		pdbs_chunk = self.cluster_info.loc[self.cluster_info.PDB.eq(pdbid), :]

		pdb_biounits = defaultdict(lambda: defaultdict(list))

		# get the biounits for this pdb
		biounits = self.get_pdb_biounits(pdbid)

		# get the chains that make the biounits
		biounits_flat = [f"{pdbid}_{chain}" for biounit in biounits for chain in biounit]
		chains = pdbs_chunk.loc[:, "CHAINID"].drop_duplicates()

		# if the chains in this pdb are not part of a biounit, treat the single chain as a biounit
		single_chains = chains.loc[~chains.isin(biounits_flat)]
		biounits.extend([chain.split("_")[1] for _, chain in single_chains.items()])

		pdb_path = self.output.out_path / Path(f"pdb/{pdbid[1:3]}")

		# loop through each biounit
		for idx, biounit in enumerate(biounits):

			# filter out ligand chains from biounit
			biounit_chains = [chain for chain in biounit if f"{pdbid}_{chain}" in chains.values]
			if not biounit_chains: continue

			# get the biounit coordinates, labels, and chain masks
			bu_coords, bu_labels, bu_chain_masks = self.get_biounit_tensors(pdbid, biounit_chains)
			if None in [bu_coords, bu_labels, bu_chain_masks]: continue
			bu_size = bu_labels.size(0)

			# write the files
			path = pdb_path / Path(f"{pdbid}_{idx}.pt")
			self.write_files(bu_coords, bu_labels, bu_chain_masks, path)

			# save the biounit info

			for chain in bu_chain_masks.keys():
				pdb_biounits[f"{pdbid}_{chain}"]["BIOUNIT"].append(f"{pdbid}_{idx}")
				pdb_biounits[f"{pdbid}_{chain}"]["BIOUNIT_SIZE"].append(bu_size)

		# only return if not empty
		if pdb_biounits.keys():
			return pdb_biounits
		else:
			return None


	def write_files(self, coords, labels, chain_masks, path):

		biounit_data = {
			"coords": coords, # N x 3
			"labels": labels, # N,
			"chain_idxs": chain_masks # {CHAINID: [start, end(exclusive)]}
		}

		if not path.parent.exists():
			path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(biounit_data, path)

	def get_pdb_biounits(self, pdbid):
		
		pdb = self.load_pdb(pdbid)
		biounits = pdb["asmb_chains"]
		
		biounits = [biounit.split(',') for biounit in biounits]

		# remove dupilcates
		biounits = list(map(list, set(map(tuple, biounits))))
		
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
			labels = torch.tensor([self.aa_idx[aa] if aa in self.amino_acids else self.aa_idx["X"] for aa in seq])
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

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--clean_pdbs", default=True, type=bool, help="whether to clean the pdbs")

	parser.add_argument("--data_path", default=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02"), type=Path, help="path where decompressed the PMPNN dataset")
	parser.add_argument("--new_data_path", default=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered_2"), type=Path, help="path to write the filtered dataset")
	parser.add_argument("--pdb_path", default=Path("pdb"), type=Path, help="path where pdbs are located, in the data_path parent directory")
	parser.add_argument("--all_clusters_path", default=Path("list.csv"), type=Path, help="path where cluster csv is located within data_path")
	parser.add_argument("--val_clusters_path", default=Path("valid_clusters.txt"), type=Path, help="path where valid clusters text file is located within data_path")
	parser.add_argument("--test_clusters_path", default=Path("test_clusters.txt"), type=Path, help="path where test clusters text file is located within data_path")
	parser.add_argument("--test", default=False, type=bool, help="test the cleaner or run")

	args = parser.parse_args()

	if args.clean_pdbs:

		data_cleaner = DataCleaner(	data_path=args.data_path, new_data_path=args.new_data_path, pdb_path=args.pdb_path, 
									all_clusters_path=args.all_clusters_path, val_clusters_path=args.val_clusters_path, test_clusters_path=args.test_clusters_path, 
									test=args.test
								)