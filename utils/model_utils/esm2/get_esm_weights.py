'''
for reference, here are the allowed models, all are based on ESM-2. 
taken from ESM github: https://github.com/facebookresearch/esm/tree/main?tab=readme-ov-file#available-models

MODEL					LAYERS		PARAMS		DATASET				Embedding_dim         
esm2_t48_15B_UR50D		48			15B			UR50/D 2021_04		5120
esm2_t36_3B_UR50D		36			3B			UR50/D 2021_04		2560
esm2_t33_650M_UR50D		33			650M		UR50/D 2021_04		1280
esm2_t30_150M_UR50D		30			150M		UR50/D 2021_04		640
esm2_t12_35M_UR50D		12			35M			UR50/D 2021_04		480
esm2_t6_8M_UR50D		6			8M			UR50/D 2021_04		320

can be downloaded dynamically based on target d_model and whether to round up or down, based on model name, or saved to disk given a specific model.
'''

from pathlib import Path
from tqdm import tqdm
import requests
import argparse
import tempfile
import torch
import io

torch.serialization.add_safe_globals([argparse.Namespace])

def get_esm_weights(d_model, round_down=True):
	'''
	gets closest match to target d_model. if round_down=True, rounds down to nearest dmodel, else rounds up
	if a valid model name is provided, simply downloads the corresponding weights
	'''
	
	model_from_dim = {
		5120: "esm2_t48_15B_UR50D",
		2560: "esm2_t36_3B_UR50D",
		1280: "esm2_t33_650M_UR50D",
		640: "esm2_t30_150M_UR50D",
		480: "esm2_t12_35M_UR50D",
		320: "esm2_t6_8M_UR50D"
	}

	if isinstance(d_model, int):
		round_func = max if round_down else min
		round_condition = (lambda dim: dim <= d_model) if round_down else (lambda dim: dim >= d_model)
		try:
			model = model_from_dim[round_func(dim for dim in model_from_dim.keys() if round_condition)]
		except KeyError as e:
			raise e(f"unable to {'round_down' if round_down else 'round up'} from {d_model}. available ESM dims are \n{model_from_dim.keys()}")
	else: # means that it is the name of the model already
		model = d_model
		if model not in model_from_dim.values():
			raise ValueError(f"{model} is not a valid ESM2 model, available models are {model_from_dim.values()}")

	state_dict = download_weights(model)
	protAI_state_dict = map_esm_to_protAI(state_dict)

	return protAI_state_dict

def download_weights(model):

	# define url
	url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model}.pt"

	# Streaming request (avoid full memory load)
	response = requests.get(url, stream=True, timeout=(10,30))
	if response.status_code != 200:
		raise Exception(f"Failed to download file: {response.status_code}")

	# Get total file size (for progress bar)
	total_size = int(response.headers.get("content-length", 0))
	chunk_size = 8192  # 8 KB per chunk

	# Create a temporary file (avoids using RAM)
	with tempfile.NamedTemporaryFile(delete=True) as temp_file:
		with tqdm(total=total_size, unit="B", unit_scale=True, desc="downloading ESM2 weights") as pbar:
			for chunk in response.iter_content(chunk_size=chunk_size):
				temp_file.write(chunk)
				pbar.update(len(chunk))

		# Seek back to the beginning of the file
		temp_file.seek(0)

		# Load PyTorch state_dict directly from the temporary file
		state_dict = torch.load(temp_file.name, map_location=torch.device("cpu"), weights_only=True)["model"]

	# model subclasses from ESM github pertaining to aa embedding (linear_no_bias and layer norm)
	aa_state_dict = {"esm2_linear_nobias.weight": state_dict['encoder.sentence_encoder.embed_tokens.weight'],
					"esm2_layernorm.weight": state_dict['encoder.sentence_encoder.emb_layer_norm_after.weight'], 
					"esm2_layernorm.bias": state_dict['encoder.sentence_encoder.emb_layer_norm_after.bias']
					}

	return aa_state_dict

def map_esm_to_protAI(state_dict):
	'''
	gets rid of tokens that proteusAI doesnt use, and reorders ESM linear_nobias tensor dims to match proteusAI alphabet

	note that all of the following links refer to the ESM github repo on 02/16/25, which corresponds to the commit hash c9c7d4f. 
	as of right now it is read-only, so there shouldn't be any changes

	this function keeps a similar syntax to what ESM uses, rather than outright defining the alphabet, so people can cross check easily
	the syntax is derived from the __init__ method of the Alphabet class in https://github.com/facebookresearch/esm/blob/main/esm/data.py#L91

	i created this function by following the logic in these methods on the ESM repo (parent method to child method, where parent method calls a child method)
		https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L355 ; 	loads esm2 model. note that each ESM2 model has its own 
																					method, this link is just one model, but all esm2 models 
																					load the same alphabet

			https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L64 ; wrapper to next child method
				https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L191 ; parses model name to check if it starts w/ esm2 and calls next method
					https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py#L175 ; initializes Alphabet object by calling the Alphabet from_architecture method with "ESM-1b" arg
						https://github.com/facebookresearch/esm/blob/main/esm/data.py#L151 ; defines standard toks, prepend_toks, and append_toks, 
						https://github.com/facebookresearch/esm/blob/main/esm/data.py#L174 ; same func as prev link, but this line initializes an Alphabet Object
							https://github.com/facebookresearch/esm/blob/main/esm/data.py#L92 ; __init__ method of Alphabet, self.all_toks is the final alphabet 

	the proteinseq_toks are from https://github.com/facebookresearch/esm/blob/main/esm/constants.py
		this is imported in https://github.com/facebookresearch/esm/blob/main/esm/data.py#L14
	'''
	
	proteinseq_toks = {
		'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
	}

	standard_toks = proteinseq_toks["toks"]
	prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
	append_toks = ("<mask>",)
	prepend_bos = True
	append_eos = True
	all_toks = list(prepend_toks)
	all_toks.extend(standard_toks)
	for i in range((8 - (len(all_toks) % 8)) % 8):
		all_toks.append(f"<null_{i  + 1}>")
	all_toks.extend(append_toks)
	# final result should be 
	# ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

	# proteus AI has these in abc order, not dealing w/ non-canonical AAs, 21st token is <mask>, i.e. model is tasked w/ predicting this
	protAI_toks = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<mask>']
	
	# reorder the 33 x ESMd_model linear_nobias layer to a 21 x ESMd_model tensor. 
	# no need to do this for the layernorm weights, as they are applied uniformly to each token
	new_linear_nobias = torch.zeros(len(protAI_toks), state_dict["esm2_linear_nobias.weight"].size(1))
	for idx, aa in enumerate(protAI_toks):
		new_linear_nobias[idx, :] = state_dict["esm2_linear_nobias.weight"][all_toks.index(aa), :]
		print(aa, idx, all_toks.index(aa))

	# modify the state dict with the update
	state_dict["esm2_linear_nobias.weight"] = new_linear_nobias

	return state_dict
	

def main(args):

	state_dict = download_weights(args.model)
	
	protAI_state_dict = map_esm_to_protAI(state_dict)

	torch.save(protAI_state_dict, args.weights_path)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="esm2_t12_35M_UR50D", 
						choices=["esm2_t48_15B_UR50D", "esm2_t36_3B_UR50D", 
								"esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D", 
								"esm2_t12_35M_UR50D", "esm2_t6_8M_UR50D"]
						)
	parser.add_argument("--weights_path", type=Path, default="esm2_weights.pt")
	args = parser.parse_args()

	main(args)