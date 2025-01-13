# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		get_data.py
description:	converts cleaned pdbs into a Data object, which can be split into training/testing for the model
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from collections import defaultdict
import multiprocessing as mp
from threading import Thread
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import random
import math
import time
import os

from utils.io_utils import Output
from utils.model_utils.wf_embedding import wf_embedding

# ----------------------------------------------------------------------------------------------------------------------

# preprocessed data uses these
torch.serialization.add_safe_globals([defaultdict, list])

class DataHolder():

	'''
	hold Data Objects, one each for train, test and val
	'''

	def __init__(self, data_path, num_train, num_val, num_test, max_size=10000, batch_sizes=[1], seq_sizes=[10000], batch_size=10000, feature_path="3.7_20.0_20.0", include_ncaa=False):


		self.data_path = data_path
		self.batch_sizes = batch_sizes
		self.seq_sizes = seq_sizes
		self.batch_size = batch_size
		self.feature_path = feature_path
		self.include_ncaa = include_ncaa

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

		self.num_train = num_train if ((num_train < (len(train_pdbs.CLUSTER.drop_duplicates()))) and (num_train != -1)) else len(train_pdbs.CLUSTER.drop_duplicates())
		self.num_val = num_val if ((num_val < (len(val_pdbs.CLUSTER.drop_duplicates()))) and (num_val != -1)) else len(val_pdbs.CLUSTER.drop_duplicates())
		self.num_test = num_test if ((num_test < (len(test_pdbs.CLUSTER.drop_duplicates()))) and (num_test != -1)) else len(test_pdbs.CLUSTER.drop_duplicates())

		self.train_pdbs = train_pdbs.loc[train_pdbs.CLUSTER.isin(train_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_train)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.val_pdbs = val_pdbs.loc[val_pdbs.CLUSTER.isin(val_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_val)), ["CHAINID", "CLUSTER", "BIOUNIT"]]
		self.test_pdbs = test_pdbs.loc[test_pdbs.CLUSTER.isin(test_pdbs.CLUSTER.drop_duplicates().sample(n=self.num_test)), ["CHAINID", "CLUSTER", "BIOUNIT"]]

		self.max_size = max_size

	def load(self, data_type):
		if data_type == "train":
			self.train_data = Data(self.data_path, self.train_pdbs, self.num_train, self.max_size, self.feature_path, self.include_ncaa, self.batch_sizes, self.seq_sizes, self.batch_size)
		elif data_type == "val":
			self.val_data = Data(self.data_path, self.val_pdbs, self.num_val, self.max_size, self.feature_path, self.include_ncaa, self.batch_sizes, self.seq_sizes, self.batch_size)	
		elif data_type == "test":	
			self.test_data = Data(self.data_path, self.test_pdbs, self.num_test, self.max_size, self.feature_path, self.include_ncaa, self.batch_sizes, self.seq_sizes, self.batch_size)

class Data():
	def __init__(self, data_path, clusters_df, num_samples=None, max_size=10000, feature_path="3.7_20.0_20.0", include_ncaa=False, batch_sizes=[1], seq_sizes=[10000], batch_size=10000, device="cpu"):
		self.pdb_path = data_path / Path(feature_path) / Path("pdb")
		self.include_ncaa = include_ncaa
		self.max_size = max_size
		self.batch_sizes = batch_sizes

		assert all(math.log(size, 2).is_integer() for size in self.batch_sizes), "batch sizes must be powers of two"

		self.seq_sizes = seq_sizes
		self.batch_size = batch_size

		self.device = device
		self.clusters_df = clusters_df
		self.clusters = defaultdict(lambda: defaultdict(list)) # self.clusters[BioUnit][features/labels/coords/chain_masks]
		self.rotate_data()

	def rotate_data(self):
		
		def process_pdb(pdb):
			
			pdb_coords = self.clusters[pdb.at["BIOUNIT"]]["coords"]
			pdb_labels = self.clusters[pdb.at["BIOUNIT"]]["labels"]
			pdb_chain_idxs = self.clusters[pdb.at["BIOUNIT"]]["chain_idxs"]

			if not pdb_labels:
				pdb_labels, pdb_coords, pdb_chain_idxs = self.add_data(pdb)
				if None in [pdb_labels, pdb_coords, pdb_chain_idxs]: 
					return None
			else:
				pdb_labels, pdb_coords, pdb_chain_idxs = pdb_labels[0], pdb_coords[0], pdb_chain_idxs[0]

			# create a chain mask for loss compuation
			pdb_chain_idxs = pdb_chain_idxs[pdb.at["CHAINID"].split("_")[-1]]

			return pdb_labels, pdb_coords, pdb_chain_idxs

		# get random cluster representative chains
		sampled_pdbs = self.clusters_df.groupby("CLUSTER").sample(n=1)

		# initialize list for this epoch
		labels, coords, chain_idxs = [], [], []

		# init progress bar
		load_pbar = tqdm(total=len(sampled_pdbs), desc="data loading progress", unit="step")

		# parallel execution
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			futures = {executor.submit(process_pdb, pdb): pdb for _, pdb in sampled_pdbs.iterrows()}
			
			# collect results
			for future in as_completed(futures):

				result = future.result()
				if result is not None:  # Ignore failed results
					pdb_labels, pdb_coords, pdb_chain_idxs = result
					labels.append(pdb_labels)
					coords.append(pdb_coords)
					chain_idxs.append(pdb_chain_idxs)
				load_pbar.update(1)

		# store for next epoch
		self.labels = labels
		self.coords = coords
		self.chain_idxs = chain_idxs

		self.batch_data()

	def add_data(self, pdb):

		section = Path("".join(pdb.at["CHAINID"].split("_")[0][1:3]))
		pdb_path = self.pdb_path / section / Path(f"{pdb.at['BIOUNIT']}.pt")

		if pdb_path.exists():
			pdb_data = torch.load(pdb_path, weights_only=True, map_location=self.device)
			pdb_labels = pdb_data["labels"]
			if not self.include_ncaa: # mask out non-canonical amino acids
				pdb_labels = torch.where(pdb_labels==20, -1, pdb_labels)	
			pdb_coords = pdb_data["coords"]
			pdb_chain_idxs = pdb_data["chain_idxs"]

			if pdb_labels.size(0) <= self.max_size:
		
				self.clusters[pdb.at["BIOUNIT"]]["labels"].append(pdb_labels)
				self.clusters[pdb.at["BIOUNIT"]]["coords"].append(pdb_coords)
				self.clusters[pdb.at["BIOUNIT"]]["chain_idxs"].append(pdb_chain_idxs)
		
				return pdb_labels, pdb_coords, pdb_chain_idxs

		return None, None, None

	def batch_data(self):

		# shuffle indexes when creating data
		idxs = list(range(len(self.labels)))
		idx_size = [[i, self.labels[i].size(0)] for i in idxs]
		# sort first and then chunk and randomize mini chunks, so that batches have similar sized samples, yet still random when do batch_subset
		idx_size = sorted(idx_size, key=lambda x: x[1])
		# max_batch_size = len(self.labels)//4
		max_batch_size = max(self.batch_sizes)
		random_idx_batches = [idx_size[i:i+max_batch_size] for i in range(0, len(idx_size), max_batch_size)]

		for i in random_idx_batches:
			random.shuffle(i)

		# split the index list into mini batches of size max_batch_len. will send these initial batches
		# to threads to process in parallel, then sort these mini batches, and split in two recursively until the batches are
		# the target batch size

		def batch_subset(batch_idxs, max_size):
			'''
			recursively splits batch idxs until reach target number of tokens. 
			starts at max batch d eg 64, and splits into 2
			returns a list of lists, each inner list containing sample indexes and corresponding size
			'''

			if (sum(i[1] for i in batch_idxs) > self.batch_size) or (len(batch_idxs) > max_size):
				split = len(batch_idxs) // 2
				return batch_subset(batch_idxs[:split], max_size) + batch_subset(batch_idxs[split:], max_size)

			else: 
				return [batch_idxs]


		# parallel processing
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			max_batch = max(self.batch_sizes)
			futures = [executor.submit(batch_subset, batch, max_batch) for batch in random_idx_batches]
			
			# collect results
			batches = []
			for future in as_completed(futures):

				result = future.result()
				if result is not None:  # Ignore failed results
					batches.extend(result)

		# shuffle batches, as mini batches are ordered by number of samples (descending) due to previous logic
		random.shuffle(batches)
					
		self.batches = batches

	def __iter__(self):
		for batch in self.batches:
			min_size = min(self.seq_sizes)
			max_size = max(self.seq_sizes)
			seq_next_pow2 = 2**math.ceil(math.log(max(i[1] for i in batch), 2)) # next power of 2
			seq_size = max(min_size, min(max_size, seq_next_pow2))

			labels = []
			coords = []
			chain_masks = []

			for idx, _ in batch:
				labels.append(self.labels[idx])
				coords.append(self.coords[idx])
				
				start, end = self.chain_idxs[idx]
				chain_mask = torch.ones(self.labels[idx].shape, dtype=torch.bool, device=self.device)
				chain_mask[start:end] = False
				chain_masks.append(chain_mask.to(torch.bool))

			batch_next_pow2 = 2**math.ceil(math.log(len(batch), 2)) # next power of 2
			batch_pads = range(batch_next_pow2 - len(batch))
			for extra_batch in batch_pads:
				labels.append(-torch.ones(seq_size, dtype=labels[0].dtype, device=self.device))
				coords.append(torch.zeros(seq_size, 3, dtype=coords[0].dtype, device=self.device) + float("inf"))
				chain_masks.append(torch.zeros(seq_size, dtype=chain_masks[0].dtype, device=self.device))

			labels = self.pad_and_batch(labels, pad_val="-one", max_size=seq_size)
			coords = self.pad_and_batch(coords, pad_val="zero", max_size=seq_size)
			chain_masks = self.pad_and_batch(chain_masks, pad_val="one", max_size=seq_size).to(torch.bool)
			key_padding_masks = labels==-1

			yield labels, coords, chain_masks, key_padding_masks


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

	def __len__(self):
		return len(self.batches)
	

class DataCleaner():

	def __init__(self, 	data_path=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02"),
						new_data_path=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered"),
						pdb_path=Path("pdb"),
						all_clusters_path=Path("list.csv"),
						val_clusters_path=Path("valid_clusters.txt"),
						test_clusters_path=Path("test_clusters.txt"),
						include_ncaa=True,
						min_resolution=3.5,
						max_tokens=10000,
						d_model=512, min_wl=3.7, max_wl=20, base=20,
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
		self.cluster_info = self.cluster_info.loc[self.cluster_info.RESOLUTION <= min_resolution, :]
		if not include_ncaa:
			self.cluster_info = self.cluster_info.loc[~self.cluster_info.SEQUENCE.str.contains("X", na=False), :]

		# initialize BIOUNIT and PDB columns
		self.cluster_info["BIOUNIT"] = None
		self.cluster_info["PDB"] = self.cluster_info.CHAINID.apply(lambda x: x.split("_")[0])

		# maximum sequence length
		self.max_tokens = max_tokens

		# featurization params
		self.d_model = d_model
		self.min_wl = min_wl
		self.max_wl = max_wl
		self.base = base

		# useful conversions between aa and idx
		self.amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
		self.aa_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
		self.rev_aa_idx = {idx: aa for idx, aa in enumerate(self.amino_acids)}
		
		# to keep track of dataset statistics
		self.aa_distributions = {aa: 0 for aa in range(len(self.amino_acids))}
		self.seq_lengths = []
		self.max_seq_len = 0

	def get_pmpnn_pdbs(self):
		
		# no gradients
		with torch.no_grad():

			# split into chunks and send different subset of pdbs to each gpu 
			num_gpus = torch.cuda.device_count()
			devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
			all_pdbs = self.cluster_info.PDB.drop_duplicates()
			pdb_chunks = np.array_split(all_pdbs, num_gpus)

			# start a process for each gpu and have it compute the biounits for each pdb and featurize.
			# store each processes's results (dataframe that maps chainid to biounit) in list
			results = []

			# use manager to share progress between processes
			with Manager() as manager:

				# shared dict for progress
				progress = manager.dict({i: 0 for i in range(num_gpus)})

				# progress bar
				with tqdm(total=len(all_pdbs)) as pbar:

					# function to monitor progress
					def monitor_progress():
						
						# continuously update tqdm progress bar
						os.environ["WF_DONE"] = "0"
						while os.environ.get("WF_DONE") != "1":
							pbar.n = sum(progress.values())
							pbar.refresh()
							time.sleep(0.1)


					# start progress monitoring thread
					montor_thread = Thread(target=monitor_progress, daemon=True)
					montor_thread.start()

					# set processes to spawn
					mp.set_start_method('spawn', force=True)

					# run the process for each gpu
					with ProcessPoolExecutor(max_workers=num_gpus) as executor:
						futures = [
							executor.submit(self.compute_biounits, pdb_chunks[i], devices[i], progress, i)
							for i in range(num_gpus)
						]
						# gather the results
						for future in futures:
							results.append(future.result())

					os.environ["WF_DONE"] = "1"

			# now deal with the results

			# concat each process df together
			results = pd.concat(results, axis=0).drop_duplicates()

			# assign biounits for relevant chains in the total df
			# Create a mapping from results
			mapping = dict(zip(results.CHAINID, results.BIOUNIT))

			# Map values where matches are found
			self.cluster_info['BIOUNIT'] = self.cluster_info['CHAINID'].map(mapping).combine_first(self.cluster_info['BIOUNIT'])

			# remove any chains whose BIOUNIT entry is None
			self.cluster_info = self.cluster_info.dropna(subset="BIOUNIT").reset_index(drop=True)
					
			self.output.write_new_clusters(self.cluster_info, self.val_clusters, self.test_clusters)
			
	def compute_biounits(self, pdbs_chunk, device, progress, process):

		# get chain ids for each pdb
		pdbs_chunk = self.cluster_info.loc[self.cluster_info.PDB.isin(pdbs_chunk), :]

		# store biounit info locally
		chunk_biounits = {"CHAINID": [], "BIOUNIT": []}

		# loop through each pdb entry 
		coords, labels, chain_masks, paths = [], [], [], []
		all_pdbs_path = self.output.out_path / Path(f"{self.min_wl}_{self.max_wl}_{self.base}")
		batch_size = 256

		for _, pdbid in pdbs_chunk.PDB.drop_duplicates().items():

			# update process progress
			progress[process] += 1

			# get the biounits for this pdb
			biounits = self.get_pdb_biounits(pdbid)
			biounits_flat = [f"{pdbid}_{chain}" for biounit in biounits for chain in biounit]
			chains = pdbs_chunk.loc[pdbs_chunk.PDB.eq(pdbid), "CHAINID"]
			single_chains = chains.loc[chains.isin(biounits_flat)]
			biounits.extend([chain.split("_")[1] for chain in single_chains])

			pdb_path = all_pdbs_path / Path(f"pdb/{pdbid[1:3]}") 

			# loop through each biounit
			for idx, biounit in enumerate(biounits):

				# filter out ligand chains from biounit
				biounit_chains = [chain for chain in biounit if f"{pdbid}_{chain}" in chains.values]

				# get the biounit coordinates, labels, and chain masks
				bu_coords, bu_labels, bu_chain_masks = self.get_biounit_tensors(pdbid, biounit_chains)
				if None in [bu_coords, bu_labels, bu_chain_masks]:
					continue
				
				bu_coords = bu_coords.to(device)
				bu_labels = bu_labels.to(device)

				# if too big, split the biounit into its corresponding chains, returns empty list if the chains themselves are too big
				if bu_labels.size(0) > self.max_tokens:

					# list of coords/labels/masks of the biounit's chains
					bu_coords, bu_labels, bu_chain_masks = self.split_biounit(bu_coords, bu_labels, bu_chain_masks)
					# remove chains that are too long anyway
					coords.extend(bu_coords)
					labels.extend(bu_labels)
					chain_masks.extend(bu_chain_masks)
					paths.extend([pdb_path / Path(f"{pdbid}_{idx}.{i}.pt") for i in range(len(bu_labels))])

				else:
					coords.append(bu_coords)
					labels.append(bu_labels)
					chain_masks.append(bu_chain_masks)
					paths.append(pdb_path / Path(f"{pdbid}_{idx}.pt"))

				if len(coords) >= batch_size:

					self.write_files(coords, labels, chain_masks, paths, chunk_biounits)

					coords, labels, chain_masks, paths = [], [], [], []

		# incase the last few do not fit into a neat batch
		if coords:
			self.write_files(coords, labels, chain_masks, paths, chunk_biounits)

		chunk_biounits = pd.DataFrame(chunk_biounits)

		return chunk_biounits


	def write_files(self, coords, labels, chain_masks, paths, chunk_biounits):

		coords, mask = self.pad_tensors(coords)

		# get features for each biounit
		features = wf_embedding(coords, self.d_model, self.min_wl, self.max_wl, self.base, mask=mask)

		# loop through each sample along batch dim and unpad coords and features
		for i in range(features.size(0)): 


			biounit_data = {
				"coords": coords[i, ~mask[i], :], # N x 3
				"features": features[i, ~mask[i], :], # N x d_model
				"labels": labels[i], # N,
				"chain_idxs": chain_masks[i] # {CHAINID: [start, end(exclusive)]}
			}

			biounit_path = paths[i]
			if not biounit_path.parent.exists():
				biounit_path.parent.mkdir(parents=True, exist_ok=True)
			torch.save(biounit_data, biounit_path)

			pdbid = biounit_path.name.split("_")[0]
			for chain in chain_masks[i]:
				chunk_biounits["CHAINID"].append(f"{pdbid}_{chain}")
				chunk_biounits["BIOUNIT"].append(biounit_path.name.rstrip(".pt"))
			
	def pad_tensors(self, coords):

		max_len = max(sample.size(0) for sample in coords)
		# have a few fixed sizes to increase the likelihood of triton using cache instead of recompiling each time
		max_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 10000]
		max_len = next(length for length in max_lens if max_len <= length)

		masks = torch.stack([torch.cat( 
										(torch.zeros(sample.size(0), dtype=torch.bool, device=sample.device), 
										torch.ones(max_len - sample.size(0), dtype=torch.bool, device=sample.device))
										) for sample in coords
							], dim=0)
		coords = torch.stack([torch.cat(
										(sample, 
										torch.zeros(max_len - sample.size(0), sample.size(1), device=sample.device))
										) for sample in coords
							], dim=0)

		return coords, masks

	def get_pdb_biounits(self, pdbid):
		
		pdb = self.load_pdb(pdbid)
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

	def split_biounit(self, biounit_coords, biounit_labels, chain_indices):
		
		coords, labels, chain_masks = [], [], []
		for chain, (start, stop) in chain_indices.items():
			
			chain_coords = biounit_coords[start:stop, :]
			chain_labels = biounit_labels[start:stop]

			chain_size = chain_labels.size(0)
			if chain_size > self.max_tokens:
				continue

			coords.append(chain_coords)
			labels.append(chain_labels)
			chain_masks.append({chain: [0, chain_size]})

		return coords, labels, chain_masks

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

def pdb_to_torch(pdb_path: Path, parser: PDBParser, data_path: Path=None):

	amino_acids = "ACDEFGHIKLMNPQRSTVWYX"

	pdb_id = pdb_path.name.rstrip(".pdb")
	structure = parser.get_structure(pdb_id, pdb_path)
	try: 
		model = structure[0]
	except KeyError:
		return None

	sequence = ""
	ca_coords = []

	for chain_idx, chain in enumerate(model):

		for position, resi in enumerate(chain): # this assumes all residues modeled in the pdb, need to filter input pdbs from rcsb for this 

			try:
				Ca_bb = resi['CA'].coord
			except KeyError:
				continue

			pos_ca_coords = list(Ca_bb)

			ca_coords.append(pos_ca_coords)

			three_letter = resi.get_resname() 
			aa = protein_letters_3to1[three_letter[0].upper() + three_letter[1:].lower()]
			sequence += aa

	ca_coords = torch.tensor(ca_coords, dtype=torch.float32)
	ca_coords = translate_origin_to_COM(ca_coords)
	ca_coords = rotate_with_PCA(ca_coords)

	label = torch.zeros(len(sequence), 20, dtype=torch.float32)
	for pos, aa in enumerate(sequence):
		label[pos, amino_acids.index(aa)] = 1.00

	assert ca_coords.size(0) == label.size(0)
	label = torch.argmax(label, dim=-1)

	if data_path:
		pt_dir = data_path / pdb_id
		pt_dir.mkdir(parents=True)

		torch.save(ca_coords, pt_dir / f"{pdb_id}_ca.pt")
		torch.save(label, pt_dir / f"{pdb_id}_aa.pt")

	else:
		return ca_coords

def translate_origin_to_COM(bb_coords):
	com = bb_coords.mean(dim=0)
	new_bb_coords = bb_coords - com

	return new_bb_coords

def rotate_with_PCA(bb_coords):
	centered_coords = translate_origin_to_COM(bb_coords) # this should have already been done anyways
	covariance_matrix = torch.mm(centered_coords.t(), centered_coords) / (bb_coords.size(0) - 1)
	U, S, V = torch.svd(covariance_matrix)
	rotated_coords = torch.mm(centered_coords, U)

	return rotated_coords

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--clean_pdbs", default=True, type=bool, help="whether to clean the pdbs")

	parser.add_argument("--data_path", default=Path("/gpfs_backup/wangyy_data/protAI/pmpnn_data/pdb_2021aug02"), type=Path, help="path where decompressed the PMPNN dataset")
	parser.add_argument("--new_data_path", default=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered_2"), type=Path, help="path to write the filtered dataset")
	parser.add_argument("--pdb_path", default=Path("pdb"), type=Path, help="path where pdbs are located, in the data_path parent directory")
	parser.add_argument("--all_clusters_path", default=Path("list.csv"), type=Path, help="path where cluster csv is located within data_path")
	parser.add_argument("--val_clusters_path", default=Path("valid_clusters.txt"), type=Path, help="path where valid clusters text file is located within data_path")
	parser.add_argument("--test_clusters_path", default=Path("test_clusters.txt"), type=Path, help="path where test clusters text file is located within data_path")
	parser.add_argument("--include_ncaa", default=True, type=bool, help="whether to include non-canonical amino acids")
	parser.add_argument("--min_resolution", default=3.5, type=float, help="minimum pdb resolution")
	parser.add_argument("--max_tokens", default=10000, type=int, help="maximum sequence/token length")

	parser.add_argument("--d_model", default=512, type=int, help="number of feature dimensions. note that this requires d_model//2 wave functions to be computed")
	parser.add_argument("--min_wl", default=3.7, type=float, help="minimum wavelength to use for wave functions")
	parser.add_argument("--max_wl", default=20.0, type=float, help="maximum wavelength to use for wave functions")
	parser.add_argument("--base", default=20, type=int, help="base to use to samples wavelengths")
	parser.add_argument("--test", default=False, type=bool, help="test the cleaner or run")

	args = parser.parse_args()

	if args.clean_pdbs:

		data_cleaner = DataCleaner(	data_path=args.data_path, new_data_path=args.new_data_path, pdb_path=args.pdb_path, 
									all_clusters_path=args.all_clusters_path, val_clusters_path=args.val_clusters_path, test_clusters_path=args.test_clusters_path, 
									include_ncaa=args.include_ncaa, min_resolution=args.min_resolution, max_tokens=args.max_tokens,
									d_model=args.d_model, min_wl=args.min_wl, max_wl=args.max_wl, base=args.base,
									test=args.test
								)
		data_cleaner.get_pmpnn_pdbs()