
alphabet = {
    "toks": ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '<pad>', '<mask>']
}

# need to change these dicts to single letter code keys
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

