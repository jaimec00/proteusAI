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

from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------

def pt_to_data(pts: Path, all_bb: int=0, device="cpu", features=False, num_inputs=512):
	all_tensors = []
	all_labels = []
	for idx, pt_dir in enumerate(pts.iterdir()):

		if idx >= num_inputs: break

		if not features:
			ca_pt = pt_dir / f"{pt_dir.name}_ca.pt"
			ca_pt = torch.load(ca_pt, map_location=device, weights_only=True)
		else:
			ca_pt = pt_dir / f"{pt_dir.name}_features.pt"
			ca_pt = torch.load(ca_pt, map_location=device, weights_only=True).squeeze(0)

		aa_pt = pt_dir / f"{pt_dir.name}_aa.pt"
		aa_pt = torch.load(aa_pt, map_location=device, weights_only=True)

		all_tensors.append(ca_pt)
		all_labels.append(aa_pt)

	data = Data(all_tensors, all_labels, device)

	return data

def pdbs_to_data(pdbs: list[Path], all_bb: int=0):

	parser = PDBParser(QUIET=True)
	amino_acids = "ACDEFGHIKLMNPQRSTVWY"

	all_bb_coords, all_labels = [], []
	
	for pdb in pdbs:
	
		pdb_id = pdb.name.rstrip(".pdb")
		structure = parser.get_structure(pdb_id, pdb)
		try: 
			model = structure[0]
		except KeyError:
			continue

		sequence = ""
		bb_coords = []

		for chain_idx, chain in enumerate(model):
	
			for position, resi in enumerate(chain): # this assumes all residues modeled in the pdb, need to filter input pdbs from rcsb for this 

				Ca_bb = resi['CA'].coord
				N_bb = resi['N'].coord
				C_bb = resi['C'].coord
				O_bb = resi['O'].coord

				pos_bb_coords = [list(coords) for coords in [Ca_bb, N_bb, C_bb, O_bb]] if all_bb else list(Ca_bb)
				bb_coords.append(pos_bb_coords)

				three_letter = resi.get_resname() 
				aa = protein_letters_3to1[three_letter[0].upper() + three_letter[1:].lower()]
				sequence += aa

			break # only working with one chain for now

		bb_coords = torch.tensor(bb_coords)
		bb_coords = translate_origin_to_COM(bb_coords)
		bb_coords = rotate_with_PCA(bb_coords)

		label = torch.zeros(len(sequence), 20)
		for pos, aa in enumerate(sequence):
			label[pos, amino_acids.index(aa)] = 1

		assert bb_coords.size(0) == label.size(0)
		label = torch.argmax(label, dim=-1)

		all_bb_coords.append(bb_coords)
		all_labels.append(label)

	data = Data(all_bb_coords, all_labels)

	return data


def pdb_to_torch(pdb_path: Path, parser: PDBParser, pt_path: Path=None):

	amino_acids = "ACDEFGHIKLMNPQRSTVWY"

	pdb_id = pdb_path.name.rstrip(".pdb")
	structure = parser.get_structure(pdb_id, pdb_path)
	try: 
		model = structure[0]
	except KeyError:
		return None

	sequence = ""
	bb_coords = []
	ca_coords = []

	for chain_idx, chain in enumerate(model):

		for position, resi in enumerate(chain): # this assumes all residues modeled in the pdb, need to filter input pdbs from rcsb for this 

			Ca_bb = resi['CA'].coord
			N_bb = resi['N'].coord
			C_bb = resi['C'].coord
			O_bb = resi['O'].coord

			pos_bb_coords = [list(coords) for coords in [Ca_bb, N_bb, C_bb, O_bb]] 
			pos_ca_coords = list(Ca_bb)
			bb_coords.append(pos_bb_coords)
			ca_coords.append(pos_ca_coords)

			three_letter = resi.get_resname() 
			aa = protein_letters_3to1[three_letter[0].upper() + three_letter[1:].lower()]
			sequence += aa

		break # only working with one chain for now

	# bb_coords = torch.tensor(bb_coords)
	# bb_coords = translate_origin_to_COM(bb_coords)
	# bb_coords = rotate_with_PCA(bb_coords)

	ca_coords = torch.tensor(ca_coords, dtype=torch.float32)
	ca_coords = translate_origin_to_COM(ca_coords)
	ca_coords = rotate_with_PCA(ca_coords)

	label = torch.zeros(len(sequence), 20, dtype=torch.float32)
	for pos, aa in enumerate(sequence):
		label[pos, amino_acids.index(aa)] = 1.00

	assert ca_coords.size(0) == label.size(0)
	label = torch.argmax(label, dim=-1)

	if pt_path:
		pt_dir = pt_path / pdb_id
		pt_dir.mkdir(parents=True)

		# torch.save(bb_coords, pt_dir / f"{pdb_id}_bb.pt")
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

class Data(Dataset):

	def __init__(self, bb_coords, labels, device):

		self.device = device

		# pad the input 
		self.bb_coords = pad_sequence(bb_coords, batch_first=True, padding_value=1000).to(device)
		self.prediction = torch.full((self.bb_coords.size(0), self.bb_coords.size(1), 20), 0.05).to(device)
		self.labels = pad_sequence(labels, batch_first=True, padding_value=-1).to(device)

		self.key_padding_mask = torch.all(self.bb_coords == 1000, dim=-1).to(device)
		self.prediction[self.key_padding_mask] = 0
		self.bb_coords[self.key_padding_mask] = 0

		assert self.bb_coords.size(0) == self.labels.size(0)

	def __len__(self):
		return self.bb_coords.size(0)
	
	def __getitem__(self, idx):

		item = self.bb_coords[idx]
		prediction = self.prediction[idx]

		label = self.labels[idx]
		key_padding_mask = self.key_padding_mask[idx]
		
		return item, prediction, label, key_padding_mask
		

	def split_train_val_test(self, split_train: int=0.8, split_val_test: int=0.5):

		split_train_idx = int(self.bb_coords.size(0) * split_train)
		split_val_idx = split_train_idx + int( (self.bb_coords.size(0) - split_train_idx) * split_val_test ) 

		self.bb_coords_train = self.bb_coords[:split_train_idx]
		self.labels_train = self.labels[:split_train_idx]

		self.bb_coords_val = self.bb_coords[split_train_idx:split_val_idx]
		self.labels_val = self.labels[split_train_idx:split_val_idx]

		self.bb_coords_test = self.bb_coords[split_val_idx:]
		self.labels_test = self.labels[split_val_idx:]

		self.train_data = Data(self.bb_coords_train, self.labels_train, self.device)
		self.val_data = Data(self.bb_coords_val, self.labels_val, self.device)
		self.test_data = Data(self.bb_coords_test, self.labels_test, self.device)

# ----------------------------------------------------------------------------------------------------------------------
