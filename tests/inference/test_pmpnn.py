
import argparse
from pathlib import Path
import subprocess 
import shutil

def main(args):

    parse_chains_script = f"{args.pmpnn_path}/helper_scripts/parse_multiple_chains.py"
    run_pmpnn_script = f"{args.pmpnn_path}/protein_mpnn_run.py"
    run_pmpnn_args = "--num_seq_per_target 1 --sampling_temp '0.1' --seed 37 --batch_size 1 --save_probs 1 --ca_only"
    
    # loop through sections    
    for section in args.pdb_path.iterdir():

        # loop through pdbs
        for pdb in section.iterdir():

            # parse biounits of this pdb into jsonl
            parsed_biounits_path = pdb / Path("parsed_biounits.jsonl")
            parse_biounits = f"python {parse_chains_script} --input_path={pdb} --output_path={parsed_biounits_path} --ca_only"
            subprocess.run(parse_biounits, shell=True)

            # create output dir for this pdb
            pmpnn_out_path = pdb / Path("pmpnn_output")
            pmpnn_out_path.mkdir(exist_ok=True)

            # run proteinMPNN
            run_pmpnn = f"python {run_pmpnn_script} --jsonl_path {parsed_biounits_path} --out_folder {pmpnn_out_path} {run_pmpnn_args}"
            subprocess.run(run_pmpnn, shell=True)

            # clean up

            # remove parsed chains
            parsed_biounits_path.unlink(missing_ok=True)

            # only keep the npz files with the probabilities, delete the fastas
            pmpnn_seqs = pmpnn_out_path / Path("seqs")
            shutil.rmtree(pmpnn_seqs)

if __name__ =="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb_path", type=Path, default="/scratch/hjc2538/projects/proteusAI/data/multi_chain/evaluation/pdb")
    parser.add_argument("--pmpnn_path", type=Path, default="/scratch/hjc2538/software/ProteinMPNN")

    args = parser.parse_args()

    main(args)