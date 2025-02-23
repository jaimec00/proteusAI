
import pandas as pd
from pathlib import Path
import torch
import math

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

aa_freqs = {
    'A': 0.074,
    'C': 0.025,
    'D': 0.050,
    'E': 0.061,
    'F': 0.042,
    'G': 0.072,
    'H': 0.023,
    'I': 0.053,
    'K': 0.064,
    'L': 0.089,
    'M': 0.023,
    'N': 0.045,
    'P': 0.052,
    'Q': 0.040,
    'R': 0.052,
    'S': 0.073,
    'T': 0.056,
    'V': 0.064,
    'W': 0.013,
    'Y': 0.034
}
# blosum matrices are from https://ftp.ncbi.nlm.nih.gov/blast/matrices
blosum_scales = { # each matrix is stored in different bit units
    "blosum30": 1/5,
    "blosum45": 1/3,
    "blosum100": 1/3
}
def get_blosum_probs(name):

    # stored in 1/5 bits
    blosum_path = Path(f"{Path(__file__).parent}/{name}.csv")
    # blosum_np = pd.read_csv(blosum_path, index_col=0).loc[canonical_aas, canonical_aas].to_numpy()*blosum_scales[name] * math.log(2)
    # blosum_torch = torch.tensor(blosum_np, device=("cuda" if torch.cuda.is_available() else "cpu"))
    # blosum = torch.softmax(blosum_torch, dim=1)
    # convert to probabilities
    bits_to_probs = lambda x: pd.Series([aa_freqs[x.name]*aa_freqs[aa]*(2**(x.at[aa]*math.log(math.e,2)*blosum_scales[name])) for aa in x.index])
    blosum_np = pd.read_csv(blosum_path, index_col=0).loc[canonical_aas, canonical_aas].apply(bits_to_probs, axis=1).to_numpy()
    blosum_torch = torch.tensor(blosum_np, device=("cuda" if torch.cuda.is_available() else "cpu"))
    blosum = blosum_torch / blosum_torch.sum(dim=1, keepdim=True) # normalize the rows
    return blosum

blosum30 = get_blosum_probs("blosum30")
blosum45 = get_blosum_probs("blosum45")
blosum100 = get_blosum_probs("blosum100")

pKas = {
    "A": [2.34, 9.69],        # Alanine
    "R": [2.17, 9.04, 12.48], # Arginine (guanidinium)
    "N": [2.02, 8.80],        # Asparagine
    "D": [1.88, 9.60, 3.65],  # Aspartic acid (β-carboxyl)
    "C": [1.96, 10.28, 8.18], # Cysteine (thiol)
    "E": [2.19, 9.67, 4.25],  # Glutamic acid (γ-carboxyl)
    "Q": [2.17, 9.13],        # Glutamine
    "G": [2.34, 9.60],        # Glycine
    "H": [1.82, 9.17, 6.00],  # Histidine (imidazole)
    "I": [2.36, 9.60],        # Isoleucine
    "L": [2.36, 9.60],        # Leucine
    "K": [2.18, 8.95, 10.53], # Lysine (ε-amino)
    "M": [2.28, 9.21],        # Methionine
    "F": [1.83, 9.13],        # Phenylalanine
    "P": [1.99, 10.60],       # Proline
    "S": [2.21, 9.15],        # Serine
    "T": [2.09, 9.10],        # Threonine
    "W": [2.38, 9.39],        # Tryptophan
    "Y": [2.20, 9.11, 10.07], # Tyrosine (phenol)
    "V": [2.32, 9.62],        # Valine 
    "X": []    
}

# might change this to dist from Ca along Cb axis, but wouldnt be as impactful in the modeling of inhomogenous media since it is smaller though
aa_sizes = { # avg length along arbitrary dimension (cube root of volumes)
    'A': 4.0615481004456795, 
    'R': 5.289572472694207, 
    'N': 4.5788569702133275, 
    'D': 4.497941445275415, 
    'C': 4.414004962442103, 
    'E': 4.776856181035017, 
    'Q': 4.848807585839879, 
    'G': 0.0, 
    'H': 4.904868131524016, 
    'I': 4.986630952238645, 
    'L': 4.986630952238645, 
    'K': 5.12992784003009, 
    'M': 4.986630952238645, 
    'F': 5.12992784003009, 
    'P': 4.481404746557164, 
    'S': 4.179339196381232, 
    'T': 4.530654896083492, 
    'W': 5.462555571281397, 
    'Y': 5.2048278633942004, 
    'V': 4.7176939803165325,
    "X": 4.558434032012989 # avg size
}

