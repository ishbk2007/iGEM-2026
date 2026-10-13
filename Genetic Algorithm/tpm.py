"""
GA-style linker sequence mutation/evolution.
Uses a 20x20 transition probability matrix (TPM) loaded from CSV --
no external 'Informatics' module needed.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax


def load_tpm(csv_path: str):
    """
    Load a 20x20 TPM from CSV (first column/row = amino acid labels).
    Returns (tpm_dict, aa_order): tpm_dict maps amino acid -> its row
    of transition probabilities (aligned to aa_order); aa_order is
    taken from the file itself so it can't drift out of sync.
    """
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.index) == list(df.columns), "row/col order mismatch in TPM file"
    assert np.allclose(df.sum(axis=1), 1.0), "TPM rows must sum to 1"
    tpm_dict = {aa: df.loc[aa].values for aa in df.index}
    return tpm_dict, list(df.index)


# --- load TPM ---
tpm, olc = load_tpm("global_transition_matrix(1).csv")


# --- algorithmic parameters ---
mut = ['ins', 'del', 'sub', 'ret']
mutp_raw = [1.2/5, 0.2/5, 1/2, 1.2/5]
mutp = softmax(mutp_raw)
num_gen = 20
top_k = 5
num_daughter = 1000


def evolution(seq: str) -> str:
    """Apply one mutation pass over `seq`, producing one daughter sequence."""
    out = ''
    for i in range(len(seq)):
        choice = np.random.choice(mut, p=mutp)
        if choice == 'ret':                                    # Retention
            out += seq[i]
        elif choice == 'del':                                  # Deletion
            pass
        elif choice == 'sub':                                  # Substitution
            if i > 0:
                out += np.random.choice(olc, p=tpm[seq[i - 1]])
            else:
                out += seq[i]                                  # first residue: keep as-is
        elif choice == 'ins':                                  # Insertion
            out = out + seq[i] + np.random.choice(olc, p=tpm[seq[i]])
        else:
            raise ValueError('Mutation choice error')
    return out


def weighted_sum(arr) -> float:
    """Sum of the top_k scores in arr (assumes arr is pre-sorted descending)."""
    return sum(arr[:top_k])


if __name__ == "__main__":
    daughters = [evolution(seq) for _ in range(num_daughter)]
    print(f"\nGenerated {len(daughters)} daughters")

    for i, d in enumerate(daughters, 1):
        print(i, d)

    lengths = [len(d) for d in daughters]
    print(f"\nLength stats: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.2f} (parent length={len(seq)})")
    print("Unique daughters:", len(set(daughters)), "/", len(daughters))

    print("\nweighted_sum test:", weighted_sum([9, 7, 5, 3, 1, 0, 0]))