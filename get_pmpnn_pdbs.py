import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import gc

from utils import protein_to_wavefunc, check_gpu_memory

def main():
    
    torch.set_grad_enabled(False)

    # base dir
    path = Path("/gpfs_backup/wangyy_data/protAI/pmpnn_data/pdb_2021aug02")

    # pdb dir
    pdb_path = path / Path("pdb")


    # get pdb info, as well as validation and training clusters
    pdbs_info = pd.read_csv( path / Path("list.csv"), header=0)
    with    open(   path / Path("test_clusters.txt"),   "r") as v, \
            open(   path / Path("valid_clusters.txt"),  "r") as t:
        val_clusters = [int(i) for i in v.read().split("\n") if i]
        test_clusters = [int(i) for i in t.read().split("\n") if i]

    # remove non canonical aas and res greater than 3 A
    pdbs_info = pdbs_info.loc[~pdbs_info.SEQUENCE.str.contains("X", na=False), :]
    pdbs_info = pdbs_info.loc[pdbs_info.RESOLUTION <=3.5, :]


    # seperate training, validation, and testing
    val_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(val_clusters), :]
    test_pdbs = pdbs_info.loc[pdbs_info.CLUSTER.isin(test_clusters), :]

    # get lists of unique clusters
    val_clusters = "\n".join(str(i) for i in val_pdbs.CLUSTER.unique().tolist())
    test_clusters = "\n".join(str(i) for i in test_pdbs.CLUSTER.unique().tolist())

    with    open(   path / Path("test_clusters_filtered.txt"),   "w") as v, \
            open(   path / Path("valid_clusters_filtered.txt"),  "w") as t:
            v.write(val_clusters)
            t.write(test_clusters)

    del val_clusters, val_pdbs
    del test_clusters, test_pdbs

    # save training pdbs
    pdbs_info.to_csv(path / Path("list_filtered.csv"))

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    pbar = tqdm(total=len(pdbs_info), desc="wavefunction embedding progress", unit="wf_processed")
    device = torch.device("cuda")
    
    for _, pdb in pdbs_info.iterrows():

        # Move all computations to CPU if memory exceeds threshold
        # threshold = 0.65
        # if check_gpu_memory(threshold):
        #     print("Switching to CPU due to memory constraints.")
        #     device = torch.device("cpu")
        # else:
        #     device = torch.device("cuda")

        pdb_section = "".join(pdb.CHAINID.split("_")[0][1:3])

        pt_path = pdb_path / Path(pdb_section) / Path(f"{pdb.CHAINID}.pt")

        wf_path = pt_path.parent / Path(f"{pt_path.name.rstrip('.pt')}_wf.pt")
        if wf_path.exists() or not pt_path.exists():
            pbar.update()
            continue

        pt = torch.load(pt_path, weights_only=True)

        pt_seq = pt["seq"]

        try:
            labels = torch.tensor([aa_idx[aa] for aa in pt_seq]).to(device)
        except KeyError:
            # delete and continue
            pt_path.unlink()
            pbar.update()
            continue


        pt_ca = pt['xyz'][:, 1, :].to(device)
        mask = pt['mask'][:, 1].bool().to(device)

        pt_ca = pt_ca[mask].unsqueeze(0)
        labels = labels[mask]

        del mask
        del pt_seq
        del pt
        gc.collect()
        torch.cuda.empty_cache()

        key_padding_mask = torch.zeros(pt_ca.shape[:-1], dtype=torch.bool, device=device)
        features = protein_to_wavefunc(pt_ca, key_padding_mask)
        features = features.squeeze(0)

        features_and_labels = {
            "features": features,
            "labels": labels
        }

        torch.save(features_and_labels, str(wf_path))
                
        assert wf_path.exists()
        pt_path.unlink()

        del features
        del labels
        del features_and_labels
        del pt_ca
        del key_padding_mask
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(1)

    # remove the files that didn't pass the filter
    for section in pdb_path.iterdir():
        for file in section.iterdir():
            if not file.name.endswith("_wf.pt"):
                file.unlink()

if __name__ == "__main__":
    main()
