import numpy as np
import pandas as pd

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_peptide(seq, max_len=12):
    """One-hot encode a peptide sequence to a (max_len, 20) matrix."""
    encoding = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_INDEX:
            encoding[i, AA_INDEX[aa]] = 1
    return encoding.flatten()

df = pd.read_csv("peptides.csv")  
df['features'] = df['sequence'].apply(one_hot_encode_peptide)

X = np.stack(df['features'].values)
