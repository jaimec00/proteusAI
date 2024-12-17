# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		get_pdbs.py
description:	1) retrieve pdb ids that match filters using RCSB API
				2) download pdbs from RCSB server
				3) clean pdbs
					a) remove non protein atoms (should be only water if RCSB filters work correctly)
					b) cluster chains of each pdb into oligomers using HDBSCAN
					c) save each cleaned cluster as a seperate pdb
'''
# ----------------------------------------------------------------------------------------------------------------------

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa, PPBuilder
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from collections import defaultdict, Counter
from sklearn.cluster import HDBSCAN
from itertools import islice
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import requests
import gzip
import math
import time
import gc

import torch
import torch.nn.functional as F

from get_data import pdb_to_torch
from utils import protein_to_wavefunc, check_gpu_memory

# ----------------------------------------------------------------------------------------------------------------------

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--max_resolution", default=3.0, type=float, help="maximum resolution of input pdb files")
	parser.add_argument("--max_ligands", default=0.0, type=int, help="maximum number of ligands in the pdb files")
	parser.add_argument("--min_chains", default=1, type=int, help="minimum number of chains in the pdb files")
	parser.add_argument("--max_chains", default=1, type=int, help="maximum number of chains in the pdb files")
	parser.add_argument("--min_unique_chains", default=1, type=int, help="minimum number of unique chains in the pdb files")
	parser.add_argument("--max_unique_chains", default=1, type=int, help="maximum number of unique chains in the pdb files")
	parser.add_argument("--min_aa", default=500, type=float, help="minimum number of residues in the representative monomer")
	parser.add_argument("--max_aa", default=5000, type=float, help="maximum number of residues in the representative monomer")
	parser.add_argument("--include_non_canonical_aa", default=0, choices=[0,1], type=int, help="whether to include non-canonical aas") # not used, api does not support it
	parser.add_argument("--out", default="pdbs", type=Path, help="path to store raw pdbs, clean pdbs, and pt files.")
	parser.add_argument("--download_pdbs", default=0, type=int, choices=[0,1], help="whether to download the pdbs specified according to the above filters from RCSB.")
	parser.add_argument("--download_pdb_path", default="/gpfs_backup/wangyy_data/protAI/cath_nr40/canonical_chain_list.txt", type=Path, help="whether to download the pdbs specified according to the above filters from RCSB.")
	parser.add_argument("--num_download_pdbs", default=5000, type=int, help="number of pdbs to download that fulfill the specified filters.")
	parser.add_argument("--clean_pdbs", default=1, type=int, choices=[0,1], help="whether to clean the pdbs specified according to the above filters.")
	parser.add_argument("--save_pt", default=0, type=int, choices=[0,1], help="whether to save the clean pdb coordinates and labels as a .pt file")
	parser.add_argument("--save_features", default=0, type=int, choices=[0,1], help="whether to save the wave function output as a .pt file")

	args = parser.parse_args()

	if args.download_pdb_path.name:
		with open(args.download_pdb_path, "r") as f:
			pdb_and_chain_list = f.read().strip().split(" ")
			pdb_list = [[pdb[:4].upper(), pdb[4]] for pdb in pdb_and_chain_list]
	else:
		pdb_list = []


	if args.download_pdbs:
		get_pdbs(args.out / Path("raw_pdbs"), args.max_resolution,
				args.min_aa, args.max_aa,
				args.include_non_canonical_aa,
				args.min_chains, args.max_chains,
				args.min_unique_chains, args.max_unique_chains,
				args.max_ligands, args.num_download_pdbs, pdb_list=pdb_list)
	if args.clean_pdbs:
		clean_pdbs(args.out / Path("raw_pdbs"), args.out / Path("clean_pdbs"), args.max_chains, pdb_list)
	if args.save_pt:
		parser = PDBParser(QUIET=True)
		for pdb in tqdm((args.out / Path("clean_pdbs")).iterdir()):
			pdb_to_torch(pdb, parser, args.out / Path("pt"))
	if args.save_features:
		print("saving features")
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		for pt in tqdm((args.out / Path("pt")).iterdir()):

			ca = pt / Path(f"{pt.name}_ca.pt")
			coords = torch.load(ca, map_location=device, weights_only=True)[None, :, :] # for broadcasting, as 1 x N x 3 is the shape that protein_to_wavefunc expects
			key_padding_mask = torch.zeros(coords.shape[:2], dtype=torch.bool) # 1 x N
 
			# Process the batch with `protein_to_wavefunc`
			features = protein_to_wavefunc(coords, key_padding_mask)

			torch.save(features, pt / Path(f"{pt.name}_features.pt"))
# ----------------------------------------------------------------------------------------------------------------------

def batch_iterator(iterable, batch_size):
    # Generator that yields batches of a specified size
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def get_pdbs(pdb_path: Path, max_resolution: float=3.5,
			min_aa: int=20, max_aa: int=500,
			include_non_canonical_aa=0,
			min_chains: int=1, max_chains: int=1,
			min_unique_chains: int=1, max_unique_chains: int=1,
			max_ligands: int=0, num_pdbs: int=10,
			seq_identity=0.9, pdb_list=[]):

	pdb_list = list(set(pdb[0] for pdb in pdb_list))

	ids = get_pdb_ids(max_resolution,
					min_aa, max_aa,
					include_non_canonical_aa,
					min_chains, max_chains,
					min_unique_chains, max_unique_chains,
					max_ligands, num_pdbs,
					seq_identity, pdb_list=pdb_list)

	print(len(ids))

	download_pdbs(pdb_path, ids)

def get_pdb_ids(max_resolution: float=3.5,
			min_aa: int=20, max_aa: int=500,
			include_non_canonical_aa=0,
			min_chains: int=1, max_chains: int=1,
			min_unique_chains: int=1, max_unique_chains: int=1,
			max_ligands: int=0, num_pdbs: int=10,
			seq_identity: float=0.9, pdb_list=[]):

	'''
	use RCSB API to get pdb ids that match the filters
	'''


	num_pdbs = len(pdb_list) if pdb_list else num_pdbs

	# Define the API endpoint
	url = "https://search.rcsb.org/rcsbsearch/v2/query"

	def create_query(attribute, operator, value, query_type="terminal", service="text"):
		query = {	"type": query_type,
					"service": service,
					"parameters": {
						"attribute": attribute,
						"operator": operator,
						"value": value
					}
				}

		return query

	res_filter = create_query("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution)
	min_aa_filter = create_query("entity_poly.rcsb_sample_sequence_length", "greater_or_equal", min_aa)
	max_aa_filter = create_query("entity_poly.rcsb_sample_sequence_length", "less_or_equal", max_aa)
	min_unique_chains_filter = create_query("rcsb_assembly_info.polymer_entity_count", "greater_or_equal", min_unique_chains)
	max_unique_chains_filter = create_query("rcsb_assembly_info.polymer_entity_count", "less_or_equal", max_unique_chains)
	min_chains_filter = create_query("rcsb_assembly_info.polymer_entity_instance_count", "greater_or_equal", min_chains)
	max_chains_filter = create_query("rcsb_assembly_info.polymer_entity_instance_count", "less_or_equal", max_chains)
	no_nucleic_acid_filter = create_query("rcsb_assembly_info.polymer_entity_instance_count_nucleic_acid", "equals", 0)
	max_ligs_filter = create_query("rcsb_entry_info.nonpolymer_entity_count", "less_or_equal", max_ligands)

	# seq identity filter is not in the same format
	seq_identity_filter = {	"type": "terminal",
							"service": "sequence",
							"parameters": {
								"evalue_cutoff": 10,
								"identity_cutoff": seq_identity,
								"target": "pdb_protein_sequence"
							}
						}
	# include_non_canonical_aa = create_query("entity_poly.rcsb_non_std_monomer_count", "equals", 0)
	# no_missing_aa_filter = create_query("rcsb_polymer_entity.rcsb_sample_sequence_coverage", "equals", 1.0)

	max_ids = 10000
	pdb_ids = []
	num_pages = math.ceil(num_pdbs / max_ids)

	for page in range(num_pages):
		# Filter for specific PDB IDs in pdb_list
		pdb_id_filter = {	"type": "terminal",
							"service": "text",
							"parameters": {
								"attribute": "rcsb_id",
								"operator": "in",
								"value": pdb_list[page*max_ids:(page+1) * max_ids]
							}
		}

		# Define the JSON query
		query = {
					"query": {
								"type": "group",
								"logical_operator": "and",
								"nodes": [
									res_filter,
									# min_aa_filter,
									# max_aa_filter,
									# min_unique_chains_filter,
									# max_unique_chains_filter,
									# min_chains_filter,
									# max_chains_filter,
									# no_nucleic_acid_filter,
									# max_ligs_filter,
									# seq_identity_filter
									pdb_id_filter
								]
							},
							"request_options": {
												"paginate": {
													"start": 0,
													"rows":  max_ids - 1 # max return num
												}
							},
							"return_type": "entry"
		}


		# Send the POST request to the RCSB PDB API
		response = requests.post(url, json=query)

		# Check if the request was successful
		if response.status_code == 200:
			# time.sleep(3)
			data = response.json()
			pdb_ids.extend([entry['identifier'] for entry in data['result_set']])
		else:
			print(f"Failed to retrieve data: {response.status_code}")
			print(response.text)

	print(len(pdb_ids))

	# # go through the pdbs and check which ones have all canonical aas and no missing residues
	# raise ValueError

	return pdb_ids


def download_pdbs(raw_pdbs_path: Path, ids: list[str]):

	base_url = "https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/"
	raw_pdbs_path.mkdir(exist_ok=True, parents=True)

	print(f"downloading raw pdbs from rcsb.org:")
	for pdb_id in tqdm(ids):

		compressed_pdb =  raw_pdbs_path / f"{pdb_id}.ent.gz"
		decompressed_pdb = raw_pdbs_path / f"{pdb_id}.pdb"

		pdb_id = pdb_id.lower()
		url = f"{base_url}{pdb_id[1:3]}/pdb{pdb_id}.ent.gz"

		response = requests.get(url, stream=True)

		if response.status_code == 200:
			with open(compressed_pdb, 'wb') as f:
				f.write(response.content)
		else:
			print(f"Failed to download {pdb_id}. HTTP Status: {response.status_code}")
			return

		with gzip.open(compressed_pdb, 'rb') as f_in:
			with open(decompressed_pdb, 'wb') as f_out:
				f_out.write(f_in.read())

		compressed_pdb.unlink()

def clean_pdbs(raw_pdb_dir: Path, clean_pdb_dir: Path, max_chains: int, pdb_list=[]):

	# load pdb parser
	parser = PDBParser(QUIET=True)

	# class to extract chains
	class ChainAndProteinSelect(Select):
		def __init__(self, chains):
			self.chains = chains
			self.residues = [	"ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS",
								"ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
								"TYR", "VAL"
							]
			self.bb_atoms = ["CA", "C", "N", "O"]

		def accept_chain(self, chain):
			return chain.get_id() in self.chains

		def accept_residue(self, residue):
			is_aa = residue.get_resname() in self.residues
			has_bb = sum(atom.get_name() in self.bb_atoms for atom in residue) == len(self.bb_atoms)

			return is_aa and has_bb

	clean_pdb_dir.mkdir(exist_ok=True, parents=True)

	print("saving raw pdbs as clean pdbs:")
	pdb_fasta = []
	for pdb in tqdm(raw_pdb_dir.iterdir()):

		if not str(pdb.name).endswith(".pdb"): continue

		pdb_id = pdb.name.rstrip(".pdb")
		structure = parser.get_structure(pdb_id, pdb)
		model = structure[0]

		if not pdb_list:

			clusters = cluster_oligomers(model)
			if not clusters: continue
			# check if the RCSB broke the max chains filter
			if sum(len(set(chains)) for chains in clusters.values()) > max_chains:
				print(f"found pdb {pdb_id} with more than {max_chains} chains, skipping cleaning step")
				continue

			for cluster, cluster_chains in clusters.items():
				io = PDBIO()
				io.set_structure(model)
				pdb_cluster = clean_pdb_dir / f"{pdb_id}_{''.join(cluster_chains)}.pdb"
				io.save(str(pdb_cluster), select=ChainAndProteinSelect(cluster_chains))

		else:
			ppb = PPBuilder()
			chains = [pdb_and_chain[1] for pdb_and_chain in pdb_list if pdb_and_chain[0] == pdb_id]
			for chain in chains:
				if (clean_pdb_dir / Path(f"{pdb}_{chain}.pdb")).exists():
					continue
				io = PDBIO()
				# if contains_non_canonical_aa(model[chain]): 
				# 	print(f"{pdb_id} contains non-canonical AAs, skipping")
				# 	continue
				# else:
				try:
					test = model[chain]
				except KeyError:
					continue
				for pp in ppb.build_peptides(model[chain]):
					sequence = pp.get_sequence()
				chain_record = SeqRecord(Seq(sequence), id=f"{pdb_id}_{chain}", description="")
				pdb_fasta.append(chain_record)
				
				io.set_structure(model)
				pdb_cluster = clean_pdb_dir / f"{pdb_id}_{chain}.pdb"
				io.save(str(pdb_cluster), select=ChainAndProteinSelect([chain]))

	# pdb_fasta_file = clean_pdb_dir / Path("clean_chains.fasta")
	# SeqIO.write(pdb_fasta, pdb_fasta_file, "fasta")

# Function to check for any non-canonical amino acids in a chain
def contains_non_canonical_aa(chain):
    non_canonical_residues = [residue for residue in chain if not is_aa(residue, standard=True) and not residue.id[0] == "H"]
    if non_canonical_residues:
        print(f"Chain {chain.id} contains non-canonical amino acids: {[res.get_resname() for res in non_canonical_residues]}")
        return True
    return False

def cluster_oligomers(model):

	all_chain_coords = []
	all_chain_ids = []
	chain_sizes = []
	for chain in model:
		chain_coords = np.array([resi["CA"].coord for resi in chain if "CA" in resi])
		if not chain_coords.any(): continue
		chain_sizes.append(len(chain_coords[:,0]))
		all_chain_coords.append(chain_coords)
		all_chain_ids.extend(chain.id for _ in chain_coords)

	all_chain_coords = np.vstack(all_chain_coords)

	# perform clustering of chains into oligomers using DBSCAN
	if min(chain_sizes) < 2: return
	db = HDBSCAN(min_cluster_size=min(chain_sizes), allow_single_cluster=True).fit(all_chain_coords)
	labels = db.labels_

	labeled_chains = defaultdict(list)
	for label, chain in zip(labels, all_chain_ids):
		labeled_chains[chain].append(int(label))

	clusters = defaultdict(list)
	for chain, labels in labeled_chains.items():
		counter = Counter(labels)
		chain_primary_label = counter.most_common(1)[0][0]
		clusters[chain_primary_label].append(chain)

	return clusters




if __name__ == "__main__":
	main()

# ----------------------------------------------------------------------------------------------------------------------
