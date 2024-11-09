from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm

def main():

    fasta_path = Path("/gpfs_backup/wangyy_data/protAI/cath_nr40/cath-dataset-nonredundant-S40-v4_3_0.fa")
    fasta = SeqIO.parse(fasta_path, "fasta")

    canonical_aas = "ACDEFGHIKLMNPQRSTVWY"

    canonical_chains = []

    for seq_rec in tqdm(fasta):
        is_non_canonical = any(aa not in canonical_aas for aa in seq_rec.seq)
        if is_non_canonical: continue
        pdb_and_chain = seq_rec.id.strip().split("|")[-1].split("/")[0][:5]
        canonical_chains.append(pdb_and_chain)

    canonical_chains = list(set(canonical_chains))

    with open(fasta_path.parent / Path("canonical_chain_list.txt"), "w") as f:
        f.write(" ".join(chain for chain in canonical_chains))

    # print(canonical_chains, len(canonical_chains))


if __name__ == "__main__":
    main()