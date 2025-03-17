from pathlib import Path
import pandas as pd
import torch
import os

canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
alphabet = canonical_aas + ['X', '<mask>']
def aa_2_lbl(aa):
    if aa in alphabet:
        return alphabet.index(aa)
    else:
        return -1
def lbl_2_aa(label): 
    if label==-1:
        return None
    else:
        return alphabet[label]

# these volumes are from table 3 from "Volume changes on protein folding" by Yehouda Harpaz, Mark Gerstein, and Cyrus Chothia1 (1994)
# Vp is the volume of the residue when in the protein core. SCp is Vp(AA) - Vp(GLY), i.e. just the side chain volume
aa_volumes_path = Path(os.path.abspath(__file__)).parent / Path("AA_volumes.csv")
# only use the volumes in the core for now
aa_volumes = torch.tensor(pd.read_csv(aa_volumes_path, index_col=0).loc[canonical_aas, "SCp"].to_numpy(), device="cuda" if torch.cuda.is_available() else "cpu")

# approximation of the side chain lengths along arbitrary axis
aa_sizes = aa_volumes**(1/3)