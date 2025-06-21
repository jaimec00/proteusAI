
import argparse
from pathlib import Path
import torch

from utils.inference_utils.inference_utils import parse_pdb, load_model

def main(args):

	device = torch.device("cuda")

	# load model and move to device
	model = load_model(args.model_path)
	model = model.to(device)
	
	# make sure in eval mode and not computing gradients
	model.eval()
	with torch.no_grad():

		# loop through sections    
		for section in args.pdb_path.iterdir():

			# loop through pdbs
			for pdb in section.iterdir():

				# create output dir like pmpnn does
				pdb_out_path = pdb / Path("proteusAI_output/probs")
				pdb_out_path.mkdir(exist_ok=True, parents=True)

				# loop through biounits
				for biounit in pdb.iterdir():

					# skip irrelevant files
					if biounit.name.endswith(".pdb"):

						# parse the biounit
						parsed_biounit = parse_pdb(biounit)

						# load the coords, labels, and chains, expand dim0 for batching, wrap chain idxs in list for same effect
						coords = parsed_biounit["coords"].unsqueeze(0).to(device)
						labels = parsed_biounit["labels"].unsqueeze(0).to(device)
						mask = labels == -1
						chain_idxs_dict = parsed_biounit["chain_idxs"]
						chain_idxs = [list(chain_idxs_dict.values())]

						# run the mode to obtain probabilites
						wf = model(coords_alpha=coords, chain_idxs=chain_idxs, key_padding_mask=mask, embedding=True)
						probs = model(coords_alpha=coords, wf=wf, key_padding_mask=mask, extraction=True).softmax(dim=2).squeeze(0)

						# save relevant info in pt file
						output = {"probs": probs, "labels": labels.squeeze(0), "mask": mask.squeeze(0), "chain_idxs": chain_idxs_dict}
						biounit_out_path = pdb_out_path / Path(f"{biounit.name.rstrip('.pdb')}.pt")
						torch.save(output, biounit_out_path)

if __name__ =="__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--pdb_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/data/multi_chain/evaluation/pdb")
	parser.add_argument("--model_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/models/3enc_geofixedmask_15A_2hiddenpre/model_parameters_e9_s2.44.pth")

	args = parser.parse_args()

	main(args)