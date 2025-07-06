
three_2_one = {
    'ALA': 'A',  # Alanine
    'CYS': 'C',  # Cysteine
    'ASP': 'D',  # Aspartic acid
    'GLU': 'E',  # Glutamic acid
    'PHE': 'F',  # Phenylalanine
    'GLY': 'G',  # Glycine
    'HIS': 'H',  # Histidine
    'ILE': 'I',  # Isoleucine
    'LYS': 'K',  # Lysine
    'LEU': 'L',  # Leucine
    'MET': 'M',  # Methionine
    'ASN': 'N',  # Asparagine
    'PRO': 'P',  # Proline
    'GLN': 'Q',  # Glutamine
    'ARG': 'R',  # Arginine
    'SER': 'S',  # Serine
    'THR': 'T',  # Threonine
    'VAL': 'V',  # Valine
    'TRP': 'W',  # Tryptophan
    'TYR': 'Y',  # Tyrosine
}

canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
alphabet = canonical_aas + ['X', '<mask>']

def aa_2_lbl(aa):
    if aa in alphabet:
        return alphabet.index(aa)
    else:
        return alphabet.index("X")

def lbl_2_aa(label): 
    if label==-1:
        return "X"
    else:
        return alphabet[label]
