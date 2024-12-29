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
from torch.utils.data import DataLoader

from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from collections import defaultdict
import multiprocessing as mp
from threading import Thread
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import time
import gc

from utils.io_utils import Output
from utils.model_utils.featurization import protein_to_wavefunc

# ----------------------------------------------------------------------------------------------------------------------

class DataHolder(Dataset):

	'''
	hold Data Objects, one each for train, test and val
	'''

	def __init__(self, data_path, num_train, num_val, num_test, max_size=10000, batch_size=32):

		torch.serialization.add_safe_globals([defaultdict, list])

		self.data_path = data_path
		self.batch_size = batch_size

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

		self.train_data_loader = None
		self.val_data_loader = None
		self.test_data_loader = None

		self.max_size = max_size

	def load(self, data_type):
		if data_type == "train":
			self.train_data = Data(self.data_path, self.train_pdbs, self.num_train, self.max_size)
			self.train_data_loader = DataLoader(self.train_data, self.batch_size, shuffle=True)
		elif data_type == "val":
			self.val_data = Data(self.data_path, self.val_pdbs, self.num_val, self.max_size)	
			self.val_data_loader = DataLoader(self.val_data, self.batch_size, shuffle=True)
		elif data_type == "test":	
			self.test_data = Data(self.data_path, self.test_pdbs, self.num_test, self.max_size)
			self.test_data_loader = DataLoader(self.test_data, self.batch_size, shuffle=True)
		
class Data(Dataset):
	def __init__(self, data_path, clusters_df, num_samples=None, max_size=10000, device="cpu"):
		self.data_path = data_path
		self.max_size = max_size
		self.device = device
		self.clusters_df = clusters_df
		self.clusters = defaultdict(lambda: defaultdict(list)) # self.clusters[BioUnit][features/labels/pw_dists/chain_masks]
		self.rotate_data()

	def rotate_data(self):
		
		sampled_pdbs = self.clusters_df.groupby("CLUSTER").sample(n=1)

		features, labels, dists, chain_masks = [], [], [], []
		for _, pdb in sampled_pdbs.iterrows():
			
			pdb_features = self.clusters[pdb.at["BIOUNIT"]]["features"]
			pdb_pw_dists = self.clusters[pdb.at["BIOUNIT"]]["pw_dists"]
			pdb_labels = self.clusters[pdb.at["BIOUNIT"]]["labels"]
			pdb_chain_idxs = self.clusters[pdb.at["BIOUNIT"]]["chain_idxs"]

			if not pdb_features:
				pdb_features, pdb_labels, pdb_pw_dists, pdb_chain_idxs = self.add_data(pdb)
				if None in [pdb_features, pdb_labels, pdb_pw_dists, pdb_chain_idxs]: 
					continue
			else:
				pdb_features, pdb_labels, pdb_pw_dists, pdb_chain_idxs = pdb_features[0], pdb_labels[0], pdb_pw_dists[0], pdb_chain_idxs[0]


			# create a chain mask
			chain_start_idx, chain_end_idx = pdb_chain_idxs[pdb.at["CHAINID"].split("_")[-1]]
			pdb_chain_masks = torch.ones(pdb_labels.shape, dtype=torch.bool)
			pdb_chain_masks[chain_start_idx:chain_end_idx] = False

			features.append(pdb_features)
			labels.append(pdb_labels)
			dists.append(pdb_pw_dists)
			chain_masks.append(pdb_chain_masks)


		# stack into batches
		self.features = torch.stack(features, dim=0).to(self.device)
		self.labels = torch.stack(labels, dim=0).to(self.device)
		self.dists = torch.stack(dists, dim=0).to(self.device)
		self.chain_masks = torch.stack(chain_masks, dim=0).to(self.device)
		self.key_padding_mask = (self.labels == -1).to(self.device)

	def add_data(self, pdb):

		section = Path("".join(pdb.at["CHAINID"].split("_")[0][1:3]))
		pdb_path = self.data_path / Path("pdb") / section / Path(f"{pdb.at['BIOUNIT']}.pt")

		if pdb_path.exists():
			pdb_data = torch.load(pdb_path, weights_only=True, map_location=self.device)
			pdb_features = pdb_data["features"]
			pdb_labels = pdb_data["labels"].long()
			pdb_dists = pdb_data["pw_dists"]
			pdb_chain_idxs = pdb_data["chain_idxs"]

			if pdb_labels.size(0) <= self.max_size:
				pdb_features, pdb_labels, pdb_dists = self.pad_tensors(pdb_features, pdb_labels, pdb_dists)
		
				self.clusters[pdb.at["BIOUNIT"]]["features"].append(pdb_features)
				self.clusters[pdb.at["BIOUNIT"]]["labels"].append(pdb_labels)
				self.clusters[pdb.at["BIOUNIT"]]["pw_dists"].append(pdb_dists)
				self.clusters[pdb.at["BIOUNIT"]]["chain_idxs"].append(pdb_chain_idxs)
		
				return pdb_features, pdb_labels, pdb_dists, pdb_chain_idxs

		return None, None, None, None

	def pad_tensors(self, pdb_features, pdb_labels, pdb_dists):

		pdb_features = torch.cat((pdb_features, torch.zeros(self.max_size - pdb_features.size(0), pdb_features.size(1))), dim=0)
		pdb_labels = torch.cat((pdb_labels, -torch.ones(self.max_size - pdb_labels.size(0))), dim=0)

		pdb_dists_tmp = torch.full((self.max_size, self.max_size), torch.inf)
		pdb_dists_tmp[:pdb_dists.size(0), :pdb_dists.size(1)] = pdb_dists

		return pdb_features, pdb_labels, pdb_dists_tmp

	def __len__(self):
		return self.features.size(0)
	
	def __getitem__(self, idx):

		item = self.features[idx]
		label = self.labels[idx]
		dists = self.dists[idx]
		chain_mask = self.chain_masks[idx]
		key_padding_mask = self.key_padding_mask[idx]
		
		return item, label, dists, chain_mask, key_padding_mask

	def unit_test(self):
		pass

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
						while sum(progress.values()) < len(all_pdbs):
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

			# now deal with the results

			# concat each process df together
			results = pd.concat(results, axis=0).drop_duplicates()

			# assign biounits for relevant chains in the total df
			# Create a mapping from results
			mapping = dict(zip(results.CHAINID, results.BIOUNIT))

			# Map values where matches are found
			self.cluster_info['BIOUNIT'] = self.cluster_info['CHAINID'].map(mapping).combine_first(self.cluster_info['BIOUNIT'])

			# remove any chains whose BIOUNIT entry is None
			self.cluster_info = self.cluster_info.dropna(subset="BIOUNIT").reset_index()
			print(self.cluster_info)
					
	def compute_biounits(self, pdbs_chunk, device, progress, process):

		# get chain ids for each pdb
		pdbs_chunk = self.cluster_info.loc[self.cluster_info.PDB.isin(pdbs_chunk), :]

		# store biounit info locally
		chunk_biounits = {"CHAINID": [], "BIOUNIT": []}

		# loop through each pdb entry 
		for _, pdbid in pdbs_chunk.PDB.items():

			# update process progress
			progress[process] += 1

			# get the biounits for this pdb
			biounits = self.get_pdb_biounits(pdbid)
			chains = pdbs_chunk.loc[pdbs_chunk.PDB.eq(pdbid), "CHAINID"]

			# # find chains that are not included in the biounits, 
			# # so they make up their own single chain biounit
			# biounits_flat = [f"{pdbid}_{chain}" for biounit in biounits for chain in biounit]
			# single_chains = chains.loc[~chains.isin(biounits_flat)].tolist()

			# print(single_chains)
			# biounits.extend([[chain.split("_")[1]] for chain in single_chains])

			# loop through each biounit
			coords, labels, chain_masks = [], [], []
			for biounit in biounits:

				# filter out ligand chains from biounit
				biounit_chains = [chain for chain in biounit if f"{pdbid}_{chain}" in chains.values]

				# get the biounit coordinates, labels, and chain masks
				bu_coords, bu_labels, bu_chain_masks = self.get_biounit_tensors(pdbid, biounit_chains)
				bu_coords = bu_coords.to(device)
				bu_labels = bu_labels.to(device)

				if None in [bu_coords, bu_labels, bu_chain_masks]:
					continue

				# if too big, split the biounit into its corresponding chains
				if bu_labels.size(0) > self.max_tokens:

					# list of coords/labels/masks of the biounit's chains
					bu_coords, bu_labels, bu_chain_masks = self.split_biounit(bu_coords, bu_labels, bu_chain_masks)
					# remove chains that are too long anyway
					coords.extend([chain_coords for chain_coords in bu_coords if chain_coords.size(0) < self.max_tokens])
					labels.extend([chain_labels for chain_labels in bu_labels if chain_labels.size(0) < self.max_tokens])
					chain_masks.extend([chain_masks for chain_masks, chain_labels in zip(bu_chain_masks, bu_labels) if chain_labels.size(0) < self.max_tokens])

				else:
					coords.append(bu_coords)
					labels.append(bu_labels)
					chain_masks.append(bu_chain_masks)
			
			# concatenate all the biounits for featurization
			if not coords: continue
			coords, mask = self.pad_tensors(coords)

			# get features for each biounit
			print(coords.shape)
			features = protein_to_wavefunc(coords, self.d_model, self.min_wl, self.max_wl, self.base, mask=mask)
			features = features.to(torch.float32)

			pdb_path = self.output.out_path / Path(f"{self.min_wl}_{self.max_wl}_{self.base}") / Path(f"pdb/{pdbid[1:3]}") 
			pdb_path.mkdir(parents=True, exist_ok=True)

			# loop through each sample along batch dim and unpad coords and features
			for i in range(features.size(0)): 

				biounit_data = {
					"coords": coords[i, ~mask[i], :], # N x 3
					"features": features[i, ~mask[i], :], # N x d_model
					"labels": labels[i], # N,
					"chain_idxs": chain_masks[i] # {CHAINID: [start, end(exclusive)]}
				}
													# min_wl_max_wl_base
				biounit_path = pdb_path / Path(f"{pdbid}_{i}.pt")
				torch.save(biounit_data, biounit_path)

				for chain in chain_masks[i]:
					chunk_biounits["CHAINID"].append(f"{pdbid}_{chain}")
					chunk_biounits["BIOUNIT"].append(f"{pdbid}_{i}")

		chunk_biounits = pd.DataFrame(chunk_biounits)

		return chunk_biounits

	def pad_tensors(self, coords):

		max_len = max(sample.size(0) for sample in coords)
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

	def split_biounit(self, pdbid, biounit_coords, biounit_labels, chain_indices):
		
		coords, labels, chain_masks = [], [], []
		for chain, (start, stop) in chain_indices.items():
			
			chain_coords = biounit_coords[start:stop, :]
			chain_labels = biounit_labels[start:stop]

			chain_size = chain_labels.size(0)
			if chain_size > self.max_tokens:
				self.output.log.info(f"skipping chain {pdbid}_{chain} of length {chain_size}.")
				continue
			elif chain_size > self.max_seq_len:
				self.max_seq_len = chain_size
				self.output.log.info(f"new max sequence length: {self.max_seq_len}")

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

	parser.add_argument("--data_path", default=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02"), type=Path, help="path where decompressed the PMPNN dataset")
	parser.add_argument("--new_data_path", default=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered_ncaa"), type=Path, help="path to write the filtered dataset")
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
	parser.add_argument("--test", default=True, type=bool, help="number of devices to parallelize the computations on")

	args = parser.parse_args()

	if args.clean_pdbs:

		data_cleaner = DataCleaner(	data_path=args.data_path, new_data_path=args.new_data_path, pdb_path=args.pdb_path, 
									all_clusters_path=args.all_clusters_path, val_clusters_path=args.val_clusters_path, test_clusters_path=args.test_clusters_path, 
									include_ncaa=args.include_ncaa, min_resolution=args.min_resolution, max_tokens=args.max_tokens,
									d_model=args.d_model, min_wl=args.min_wl, max_wl=args.max_wl, base=args.base,
									test=args.test
								)
		data_cleaner.get_pmpnn_pdbs()