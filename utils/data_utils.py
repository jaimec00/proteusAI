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

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import gc

from utils.io_utils import Output
from utils.model_utils import protein_to_wavefunc

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

	def unit_test(self):
		pass
		
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
						new_data_path=Path("/share/wangyy/hjc2538/proteusAI/pdb_2021aug02_filtered_test"),
						pdb_path=Path("pdb"),
						all_clusters_path=Path("list.csv"),
						val_clusters_path=Path("valid_clusters.txt"),
						test_clusters_path=Path("test_clusters.txt"),
						include_ncaa=True,
						min_resolution=3.5,
						max_tokens=10000,
						d_model=512, min_wl=3.7, max_wl=20, base=20, max_splits=8,
						test=True
				):

		# define paths
		self.data_path = data_path
		self.pdb_path = self.data_path / pdb_path

		self.output = Output(new_data_path)
		
		# read which clusters are for validation and for testing
		with    open( self.data_path / val_clusters_path, 	"r") as v, \
				open( self.data_path / test_clusters_path,	"r") as t:

			self.val_clusters = [int(i) for i in v.read().split("\n") if i]
			self.test_clusters = [int(i) for i in t.read().split("\n") if i]

		# load the cluster dataframe, and remove high resolution and non canonical chains
		self.cluster_info = pd.read_csv(self.data_path / all_clusters_path, header=0)

		if test: # only include pdbs in 'lw' pdb section (e.g. 4lwq)
			self.cluster_info = self.cluster_info.loc[self.cluster_info.CHAINID.apply(lambda x: x[1:3]).eq("lw")]
		self.cluster_info = self.cluster_info.loc[self.cluster_info.RESOLUTION <= min_resolution, :]
		if not include_ncaa:
			self.cluster_info = self.cluster_info.loc[~self.cluster_info.SEQUENCE.str.contains("X", na=False), :]

		# initialize BIOUNIT list. Not sure if multiple biounits per chain, but will check afterwards
		self.cluster_info["BIOUNIT"] = None
		self.cluster_info["PDB"] = self.cluster_info.CHAINID.apply(lambda x: x.split("_")[0])

		# remove chains that don't exist
		print(len(self.cluster_info))
		chain_exists = self.cluster_info.CHAINID.apply(lambda x: (self.pdb_path / Path(f"{x[1:3]}/{x}.pt")).exists())
		self.cluster_info = self.cluster_info.loc[chain_exists, :]
		print(len(self.cluster_info))

		self.cluster_info = self.cluster_info.sort_values(by="CHAINID")

		# maximum sequence length
		self.max_tokens = max_tokens
		self.d_model = d_model
		self.min_wl = min_wl
		self.max_wl = max_wl
		self.base = base
		self.max_splits = max_splits

		# useful conversions between aa and idx
		self.amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
		self.aa_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
		self.rev_aa_idx = {idx: aa for idx, aa in enumerate(self.amino_acids)}
		
		# to keep track of dataset statistics
		self.aa_distributions = {aa: 0 for aa in range(len(self.amino_acids))}
		self.seq_lengths = []
		self.max_seq_len = 0

		self.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.finished_biounits = []

	def compute_biounits(self):

		raise ValueError

		for pdbid in self.cluster_info.PDB.drop_duplicates():

			pdb_path = self.pdb_path / Path(pdbid[1:3])

			chainids = self.cluster_info.loc[self.cluster_info.PDB.eq(pdbid), "CHAINID"].apply(lambda x: x.split("_")[1]).drop_duplicates().tolist()
			chainids = [chainid for chainid in chainids if (pdb_path / Path(f"{pdbid}_{chainid}.pt")).exists()]

			pdb_pt = self.load_pdb(pdbid)
			if pdb_pt is None:
				biounits = [[chain] for chain in chainids]
			else:
				biounits = pdb_pt["asmb_chains"]
				biounits = [[chain for chain in biounit if chain in chainids] for biounit in biounits]

			# remove empty biounits
			biounits = [biounit for biounit in biounits if biounit]

			for biounit in biounits:
				biounit_name = f"{pdbid}_{'_'.join(chain for chain in biounit)}"
				self.cluster_info.loc[self.cluster_info.CHAINID.isin([f"{pdbid}_{chain}" for chain in biounit]), "BIOUNIT"] = biounit_name

			# chains without biounits that are valid are assigned their own chain as the biounit
			chain_wo_biounit = self.cluster_info.BIOUNIT.isna() & self.cluster_info.CHAINID.isin([f"{pdbid}_{chain}" for chain in chainids])
			self.cluster_info.loc[chain_wo_biounit, "BIOUNIT"] = self.cluster_info.CHAINID

			print(self.cluster_info)
			raise ValueError





	def get_pmpnn_pdbs(self):

		self.compute_biounits()
		
		# no gradients
		with torch.no_grad():

			# initialize progress bar
			pbar = tqdm(total=len(self.cluster_info.CHAINID), desc="wavefunction embedding progress", unit="wf_processed")
			
			# loop through each chain entry 
			for _, chainid in self.cluster_info.CHAINID.items():

				# get pdbid, and load the corresponding pdb (not the chain, yet)
				pdbid = chainid.split("_")[0]
				biounits, biounits_finished = self.get_pdb_biounits(pdbid)

				# first check if this chain is in any of its corresponding pdbs biounits
				chain_in_biounits = any(chainid.split("_")[1] in biounit.split("_")[1:] for biounit in biounits)
				chain_finished = (chainid in self.cluster_info.loc[self.cluster_info.CHAINID.eq(chainid), "BIOUNIT"].values[0]) or \
									(chainid.split("_")[1] in [chain for biounit in biounits_finished for chain in biounit.split("_")[1:]])

				if chain_in_biounits:
					for biounit in biounits:
						self.save_biounit(biounit)

				# if it is not, will compute the single chain as a biounit.
				elif not chain_finished:
					self.save_biounit(chainid)

				pbar.update(1)

			# plot aa count and seq length count histograms
			aa_one_distributions = {self.rev_aa_idx[aa]: count for aa, count in self.aa_distributions.items()}
			self.output.plot_aa_counts(aa_one_distributions)
			self.output.plot_seq_len_hist(self.seq_lengths)

			# remove clusters that were not used (those that have empty lists in BIOUNIT)
			self.cluster_info = self.cluster_info.loc[~self.cluster_info.BIOUNIT.isin([[]]), :]
			self.output.write_new_clusters(self.cluster_info, self.val_clusters, self.test_clusters)

	def get_pdb_biounits(self, pdbid):
		
		pdb = self.load_pdb(pdbid)
		biounits = pdb["asmb_chains"]
		biounits = [f"{pdbid}_{'_'.join(biounit.split(','))}" for biounit in biounits]

		biounits_finished = [biounit for biounit in biounits if biounit in self.finished_biounits]
		biounits = [biounit for biounit in biounits if biounit not in self.finished_biounits]

		self.finished_biounits.extend(biounits)

		return biounits, biounits_finished

	def save_biounit(self, biounit):

		pdb = biounit.split("_")[0]
		chains = biounit.split("_")[1:]

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
			return

		biounit_coords = torch.cat(biounit_coords, dim=0)
		biounit_labels = torch.cat(biounit_labels, dim=0)

		# if it is too big, split it into individual chains
		biounit_size = biounit_labels.size(0)

		if biounit_size == 0: return
		if biounit_size > self.max_tokens:
			self.split_biounit(pdb, biounit_coords, biounit_labels, chain_indices)
			return

		elif biounit_size > self.max_seq_len:
			self.max_seq_len = biounit_size
			self.output.log.info(f"new max sequence length: {self.max_seq_len}")

		biounit_features = protein_to_wavefunc(	biounit_coords.unsqueeze(0).to(self.gpu), 
												torch.zeros(biounit_coords.size(0), dtype=torch.bool, device=self.gpu).unsqueeze(0), 
												d_model=self.d_model, min_wl=self.min_wl, max_wl=self.max_wl, return_wl=False, base=self.base, 
												max_splits=self.max_splits, device=self.gpu
											).squeeze(0).to("cpu")

		biounit_data = {
			"coords": biounit_coords,
			"features": biounit_features,
			"labels": biounit_labels,
			"chain_idxs": chain_indices # [start, end(exclusive)]
		}

		# for stats on dataset
		self.seq_lengths.append(biounit_size)
		for label in biounit_labels:
			self.aa_distributions[label.item()] += 1

		# save the biounit
		biounit_path = self.output.out_path / Path("pdb") / Path(biounit[1:3]) / Path(f"{pdb}_{'_'.join(chain_indices.keys())}.pt")
		biounit_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(biounit_data, biounit_path)

		# add biounit data for each chain in the biounit. kinda dirty, modifies biounit lists inplace, not sure if it is meant to but that's what i want
		self.cluster_info.loc[self.cluster_info.CHAINID.isin([f"{pdb}_{chain_id}" for chain_id in chain_indices.keys()]), "BIOUNIT"].apply(lambda x: x.append(f"{pdb}_{'_'.join(chain_indices.keys())}"))

	def split_biounit(self, pdbid, biounit_coords, biounit_labels, chain_indices):
		
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

			chain_features = protein_to_wavefunc(	chain_coords.unsqueeze(0).to(self.gpu), 
													torch.zeros(chain_coords.size(0), dtype=torch.bool, device=self.gpu).unsqueeze(0), 
													d_model=self.d_model, min_wl=self.min_wl, max_wl=self.max_wl, 
													return_wl=False, base=self.base, max_splits=self.max_splits, device=self.gpu
												).squeeze(0).to("cpu")

			chain_data = {
				"coords": chain_coords,
				"features": chain_features,
				"labels": chain_labels,
				"chain_idxs": {chain: [0, chain_size]} # [start, end(exclusive)]
			}

			self.seq_lengths.append(chain_size)
			for label in chain_labels:
				self.aa_distributions[label.item()] += 1
			
			chain_path = self.output.out_path / Path("pdb") / Path(pdbid[1:3]) / Path(f"{pdbid}_{chain}.pt")
			chain_path.parent.mkdir(parents=True, exist_ok=True)
			torch.save(chain_data, chain_path)

			# add biounit data for this chain. 
			self.cluster_info.loc[self.cluster_info.CHAINID.eq(f"{pdbid}_{chain}"), "BIOUNIT"].apply(lambda x: x.append(f"{pdbid}_{chain}"))

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

	def unit_test(self):
		pass

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

	parser.add_argument("--d_model", default=128, type=int, help="number of feature dimensions. note that this requires d_model//2 wave functions to be computed")
	parser.add_argument("--min_wl", default=3.7, type=float, help="minimum wavelength to use for wave functions")
	parser.add_argument("--max_wl", default=20.0, type=float, help="maximum wavelength to use for wave functions")
	parser.add_argument("--base", default=40, type=int, help="base to use to samples wavelengths")
	parser.add_argument("--max_splits", default=1, type=int, help="maximum number of splits to do in case where sequence length is too long. will split along wavelength dimension")
	parser.add_argument("--num_devices", default=1, type=int, help="number of devices to parallelize the computations on")
	parser.add_argument("--test", default=False, type=bool, help="number of devices to parallelize the computations on")

	args = parser.parse_args()

	if args.clean_pdbs:

		data_cleaner = DataCleaner(	data_path=args.data_path, new_data_path=args.new_data_path, pdb_path=args.pdb_path, 
									all_clusters_path=args.all_clusters_path, val_clusters_path=args.val_clusters_path, test_clusters_path=args.test_clusters_path, 
									include_ncaa=args.include_ncaa, min_resolution=args.min_resolution, max_tokens=args.max_tokens,
									d_model=args.d_model, min_wl=args.min_wl, max_wl=args.max_wl, base=args.base, max_splits=args.max_splits,
									test=args.test
								)
		data_cleaner.get_pmpnn_pdbs()