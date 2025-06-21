from Bio.PDB import PDBParser
from pathlib import Path
import argparse
import torch
import numpy as np
from proteusAI import proteusAI
from data.constants import aa_2_lbl, three_2_one


def parse_pdb(pdb_path, ca_only=True):
	'''
	pdb_path (Path): path to pdb file
	ca_only (Bool): only extract ca coords (full backbone not implemented)

	parses pdb file and returns a dict containin: 
		coords (N x 3)
		labels (N)
		mask (N)
		chain_idxs ([[chain1_start, chain1_stop], [chain2_start, chain2_stop]])
	'''

	# init parser and get structure
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure(pdb_path.name, pdb_path)
	model = structure[0]

	# initialize data to store stuff
	coords = []
	labels = []
	chain_idxs = {}

	# to keep track of chain idxs
	current_resi = 0

	# loop through chains
	for chain in model:
		chain_start = current_resi # record chain start position in tensor

		# loop through residues
		for resi in chain:
		
			# only include resis with CA atom
			if resi.has_id("CA"):

				# append coordinates of CA atom
				coords.append(resi["CA"].get_coord())

				# append label
				three_letter = resi.get_resname()
				try:
					label = aa_2_lbl(three_2_one[three_letter])
				except KeyError:
					label = -1
				labels.append(label)

				# increment residue index
				current_resi+=1

		# record end of chain position
		chain_end = current_resi

		# store the chain indexes for this chain
		chain_idxs[chain.id] = [chain_start, chain_end]

	# convert to tensors
	coords = torch.tensor(np.array(coords), dtype=torch.float32)
	labels = torch.tensor(labels, dtype=torch.int64)

	# store in dictionary
	parsed_pdb = {"coords": coords, "labels": labels, "chain_idxs": chain_idxs}

	return parsed_pdb


def load_model(model_path):
	'''
	reads the model state dict to determine the arguments to initialize proteusAI, then initializes it
	'''

	# init device
	device = torch.device("cpu")

	# only loading embedding and extraction. encoding, diffusion, and decoding are kept as default since not used
	model_weights = torch.load(model_path, map_location=device, weights_only=True)
	emb_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_embedding")}
	ext_weights = {".".join(i.split(".")[1:]): model_weights[i] for i in model_weights.keys() if i.startswith("wf_extraction")}
	
	# first determine d_wf
	num_wn = emb_weights["wavenumbers"].size(0)
	d_wf = 2*num_wn

	# determine if running old model w/ no seq info, or mlm
	old = emb_weights["aa_magnitudes"].dim() == 1
	mlm = not old
	
	# now determine the extraction arguments

	# first determine d_model by checking the size of the output projection, along with number of amino acids
	num_aa, d_model = ext_weights["out_proj.weight"].shape

	# now determine if mlp_pre is included
	mlp_pre = any(param.startswith("mlp_pre") for param in ext_weights.keys())
	mlp_pre_hidden_layers = sum(param.startswith("mlp_pre.hidden_proj") and param.endswith("weight") for param in ext_weights.keys()) if mlp_pre else -1
	d_hidden_pre = ext_weights[f"mlp_pre.in_proj.weight"].size(0) if mlp_pre else 0 # check the hidden dim size

	# now determine if mlp_post is included
	mlp_post = any(param.startswith("mlp_post") for param in ext_weights.keys())
	mlp_post_hidden_layers = sum(param.startswith("mlp_post.hidden_proj") and param.endswith("weight") for param in ext_weights.keys()) if mlp_post else -1
	d_hidden_post = ext_weights[f"mlp_post.in_proj.weight"].size(0) if mlp_post else 0 

	# now determine the encoder layer arguments
	num_enc = sum(param.startswith("encoders") and param.endswith("q_proj") for param in ext_weights.keys())
	use_bias = any(param.endswith("spread_weights") for param in ext_weights.keys())
	num_heads = ext_weights[f"encoders{'.0.' if num_enc>1 else '.'}attn.q_proj"].size(0)
	enc_hidden_layers = sum(param.startswith(f"encoders{'.0.' if num_enc>0 else '.'}ffn.hidden_proj") and param.endswith("weight") for param in ext_weights.keys())
	enc_d_hidden = ext_weights[f"encoders{'.0.' if num_enc>1 else '.'}ffn.in_proj.weight"].size(0)
	min_rbf = 0.0 # min rbf not saved, but not using that anymore so set to 0


	model = proteusAI(  old=old, mlm=mlm, d_wf=d_wf, d_model=d_model, num_aas=num_aa, 
						extraction_d_hidden_pre=d_hidden_pre, extraction_hidden_layers_pre=mlp_pre_hidden_layers,
						extraction_d_hidden_post=d_hidden_post, extraction_hidden_layers_post=mlp_post_hidden_layers,
						extraction_encoder_layers=num_enc, extraction_heads=num_heads, 
						extraction_use_bias=use_bias, extraction_min_rbf=min_rbf,
						extraction_d_hidden_attn=enc_d_hidden, extraction_hidden_layers_attn=enc_hidden_layers
					)
	model = model.to(device)
	
	model.wf_embedding.load_state_dict(emb_weights, strict=True)
	model.wf_extraction.load_state_dict(ext_weights, strict=True)

	return model

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/models/3enc_geofixedmask_15A_2hiddenpre/model_parameters_e9_s2.44.pth")
	args = parser.parse_args()

	model = load_model(args.model_path)