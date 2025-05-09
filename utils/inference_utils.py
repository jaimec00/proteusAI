import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Thread
from Bio import PDBParser
from pathlib import Path
from tqdm import tqdm
import argparse

from proteusAI import proteusAI

torch.serialization.add_safe_globals([defaultdict, list])

# def test_cath(cath_path):



# def get_weights(weights):


def clean_cath(cath_path: Path, cath_processed_path: Path):
    '''
    saves the full backbone coords, labels (aa), and chain idxs
    '''

		# init progress bar
		load_pbar = tqdm(total=len([i for i in cath_path.iterdir()]), desc="data processing progress", unit="pdbs")

		# parallel execution
		with ThreadPoolExecutor(max_workers=8) as executor:
			
			# submit tasks
			futures = {executor.submit(process_pdb, pdb, cath_processed_path): pdb for pdb in cath_path.iterdir()}
			
			# collect results
			for future in as_completed(futures):
				load_pbar.update(1)

    def process_pdb(pdb, outpath):

        structure = parser.get_structure(pdb.name, pdb)
        try: 
            model = structure[0]
        except KeyError:
            continue

        labels = []
        bb_coords = []
        chain_idxs = []
        pos = 0

        for chain_idx, chain in enumerate(model):
            chain_start = pos

            for position, resi in enumerate(chain): # this assumes all residues modeled in the pdb, need to filter input pdbs from rcsb for this 

                Ca_bb = resi['CA'].coord
                N_bb = resi['N'].coord
                C_bb = resi['C'].coord

                pos_bb_coords = [list(coords) for coords in [N_bb, Ca_bb, C_bb, O_bb]]
                bb_coords.append(pos_bb_coords)

                three_letter = resi.get_resname() 
                aa = protein_letters_3to1[three_letter[0].upper() + three_letter[1:].lower()]
                label = amino_acids.index(aa) if aa in amino_acids else -1
                labels.append(label)
                pos += 1
            
            chain_end = pos
            chain_idxs.append([chain_start, chain_end])

        bb_coords = torch.tensor(bb_coords)
        labels = torch.tensor(labels)

        all_bb_coords.append(bb_coords)
        all_labels.append(label)


        data = {"coords": all_bb_coords, "labels": all_labels, "chain_idxs": chain_idxs}



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cath_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/dompdb")
    parser.add_argument("--cath_processed_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/dompdb_processed")
    parser.add_argument("--process_cath", type=bool, default=False)
    parser.add_argument("--model_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/models/geo_attn_old_4enc_adaptivebias/model_parameters_e29_s2.11.pth")


    test_cath()
