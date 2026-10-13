import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
fn = ['AA_count_A', 'AA_count_C', 'AA_count_D', 'AA_count_E', 'AA_count_F', 'AA_count_G', 'AA_count_H', 'AA_count_I', 'AA_count_K', 'AA_count_L', 'AA_count_M', 'AA_count_N', 'AA_count_P', 'AA_count_Q', 'AA_count_R', 'AA_count_S', 'AA_count_T', 'AA_count_V', 'AA_count_W', 'AA_count_Y', 'AA_total', 'Hydrophobic_count', 'Hydrophilic_count', 'OHE_AA_minus2_A', 'OHE_AA_minus2_C', 'OHE_AA_minus2_D', 'OHE_AA_minus2_E', 'OHE_AA_minus2_F', 'OHE_AA_minus2_G', 'OHE_AA_minus2_H', 'OHE_AA_minus2_I', 'OHE_AA_minus2_K', 'OHE_AA_minus2_L', 'OHE_AA_minus2_M', 'OHE_AA_minus2_N', 'OHE_AA_minus2_P', 'OHE_AA_minus2_Q', 'OHE_AA_minus2_R', 'OHE_AA_minus2_S', 'OHE_AA_minus2_T', 'OHE_AA_minus2_V', 'OHE_AA_minus2_W', 'OHE_AA_minus2_Y', 'OHE_AA_minus1_A', 'OHE_AA_minus1_C', 'OHE_AA_minus1_D', 'OHE_AA_minus1_E', 'OHE_AA_minus1_F', 'OHE_AA_minus1_G', 'OHE_AA_minus1_H', 'OHE_AA_minus1_I', 'OHE_AA_minus1_K', 'OHE_AA_minus1_L', 'OHE_AA_minus1_M', 'OHE_AA_minus1_N', 'OHE_AA_minus1_P', 'OHE_AA_minus1_Q', 'OHE_AA_minus1_R', 'OHE_AA_minus1_S', 'OHE_AA_minus1_T', 'OHE_AA_minus1_V', 'OHE_AA_minus1_W', 'OHE_AA_minus1_Y', 'OHE_AA_plus1_A', 'OHE_AA_plus1_C', 'OHE_AA_plus1_D', 'OHE_AA_plus1_E', 'OHE_AA_plus1_F', 'OHE_AA_plus1_G', 'OHE_AA_plus1_H', 'OHE_AA_plus1_I', 'OHE_AA_plus1_K', 'OHE_AA_plus1_L', 'OHE_AA_plus1_M', 'OHE_AA_plus1_N', 'OHE_AA_plus1_P', 'OHE_AA_plus1_Q', 'OHE_AA_plus1_R', 'OHE_AA_plus1_S', 'OHE_AA_plus1_T', 'OHE_AA_plus1_V', 'OHE_AA_plus1_W', 'OHE_AA_plus1_Y', 'OHE_AA_plus2_A', 'OHE_AA_plus2_C', 'OHE_AA_plus2_D', 'OHE_AA_plus2_E', 'OHE_AA_plus2_F', 'OHE_AA_plus2_G', 'OHE_AA_plus2_H', 'OHE_AA_plus2_I', 'OHE_AA_plus2_K', 'OHE_AA_plus2_L', 'OHE_AA_plus2_M', 'OHE_AA_plus2_N', 'OHE_AA_plus2_P', 'OHE_AA_plus2_Q', 'OHE_AA_plus2_R', 'OHE_AA_plus2_S', 'OHE_AA_plus2_T', 'OHE_AA_plus2_V', 'OHE_AA_plus2_W', 'OHE_AA_plus2_Y']

def generate_linker_features(linkers, feature_names = fn):
    """
    Convert linker sequence(s) containing '|'
    into a model-ready feature DataFrame.

    Parameters
    ----------
    linkers : str or list of str
        Linker sequence(s), e.g.
        "AAAAKLM|DEFG"
        or
        ["AAAAKLM|DEFG", "MKTLL|GGA"]

    feature_names : list
        Exact feature column names used to train the model.
        This should be the `feature_names` saved from training.

    Returns
    -------
    X : pandas.DataFrame
        Model-ready feature DataFrame with the exact
        same columns and column order as training.
    """

    # ========================================================
    # 1. Convert single sequence to list
    # ========================================================

    if isinstance(linkers, str):
        linkers = [linkers]

    # Clean sequences
    linkers = [
        str(seq).strip().upper()
        for seq in linkers
    ]

    df = pd.DataFrame({
        "Substrate": linkers
    })

    # ========================================================
    # 2. Amino acid definitions
    # ========================================================

    amino_acids = list(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    hydrophobic = set(
        "AVILMFWY"
    )

    hydrophilic = set(
        "DEKRHNQSTCPG"
    )

    # Remove "|" when calculating composition
    df["_sequence"] = (
        df["Substrate"]
        .str.replace("|", "", regex=False)
    )

    # ========================================================
    # 3. Total amino acid count
    # ========================================================

    df["AA_total"] = df["_sequence"].str.len()

    # ========================================================
    # 4. Amino acid COUNTS
    # ========================================================

    for aa in amino_acids:

        df[f"AA_count_{aa}"] = (
            df["_sequence"]
            .str.count(aa)
        )

    # ========================================================
    # 5. Amino acid PERCENTAGES
    # ========================================================

    for aa in amino_acids:

        df[f"AA_pct_{aa}"] = np.where(
            df["AA_total"] > 0,
            (df[f"AA_count_{aa}"] /
             df["AA_total"]) * 100,
            0
        )

    # ========================================================
    # 6. Hydrophobic / Hydrophilic counts
    # ========================================================

    df["Hydrophobic_count"] = df[
        "_sequence"
    ].apply(
        lambda seq:
        sum(seq.count(aa) for aa in hydrophobic)
    )

    df["Hydrophilic_count"] = df[
        "_sequence"
    ].apply(
        lambda seq:
        sum(seq.count(aa) for aa in hydrophilic)
    )

    # ========================================================
    # 7. Hydrophobic / Hydrophilic percentages
    # ========================================================

    df["Hydrophobic_pct"] = np.where(
        df["AA_total"] > 0,
        df["Hydrophobic_count"] /
        df["AA_total"] * 100,
        0
    )

    df["Hydrophilic_pct"] = np.where(
        df["AA_total"] > 0,
        df["Hydrophilic_count"] /
        df["AA_total"] * 100,
        0
    )

    # ========================================================
    # 8. Get 2 AA before and 2 AA after "|"
    # ========================================================

    def get_flanking_residues(seq):

        if "|" not in seq:
            raise ValueError(
                f"Sequence does not contain '|': {seq}"
            )

        # Make sure there is only one |
        if seq.count("|") != 1:
            raise ValueError(
                f"Sequence must contain exactly one '|': {seq}"
            )

        left, right = seq.split("|")

        # Need at least 2 residues on each side
        if len(left) < 2:
            raise ValueError(
                f"Less than 2 residues before '|': {seq}"
            )

        if len(right) < 2:
            raise ValueError(
                f"Less than 2 residues after '|': {seq}"
            )

        return (
            left[-2],    # -2
            left[-1],    # -1
            right[0],    # +1
            right[1]     # +2
        )

    flanking = df[
        "Substrate"
    ].apply(get_flanking_residues)

    df[
        [
            "AA_minus2",
            "AA_minus1",
            "AA_plus1",
            "AA_plus2"
        ]
    ] = pd.DataFrame(
        flanking.tolist(),
        index=df.index
    )

    # ========================================================
    # 9. One-hot encode the 4 positions
    # ========================================================

    positions = [
        "AA_minus2",
        "AA_minus1",
        "AA_plus1",
        "AA_plus2"
    ]

    for position in positions:

        for aa in amino_acids:

            df[
                f"OHE_{position}_{aa}"
            ] = (
                df[position] == aa
            ).astype(int)

    # ========================================================
    # 10. Remove non-feature columns
    # ========================================================

    df = df.drop(
        columns=[
            "Substrate",
            "_sequence",
            "AA_minus2",
            "AA_minus1",
            "AA_plus1",
            "AA_plus2"
        ]
    )

    # ========================================================
    # 11. Make EXACTLY the same columns as training
    # ========================================================

    missing_features = [
        col for col in feature_names
        if col not in df.columns
    ]

    extra_features = [
        col for col in df.columns
        if col not in feature_names
    ]

    if missing_features:
        raise ValueError(
            "These training features could not be generated:\n"
            + str(missing_features)
        )

    # Keep ONLY training features
    df = df.drop(
        columns=extra_features,
        errors="ignore"
    )

    # EXACT training column order
    X = df[feature_names].copy()

    # ========================================================
    # 12. Replace problematic values
    # ========================================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # For these engineered features, NaN should normally not
    # occur, but fill any unexpected missing values with 0.
    X = X.fillna(0)

    return X
#-------------------------------------------------

# ========================================================
# 1. Load the saved artifacts
# ========================================================1

with open('saved_models/feature_names.pkl', 'rb') as f:
    feature_names = joblib.load('saved_models/feature_names.pkl')

with open('saved_models/scaler_logistic_regression.pkl', 'rb') as f:
    scaler_lr = joblib.load('saved_models/scaler_logistic_regression.pkl')

with open('saved_models/logistic_regression.pkl', 'rb') as f:
    log_reg = joblib.load('saved_models/logistic_regression.pkl')

# If you also saved a label_encoder for the target classes
with open('saved_models/label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load('saved_models/label_encoder.pkl')

def check_sequence_all_aspartic(sequence):
    """
    Insert '|' at every valid position in `sequence` (need >=2 residues
    on each side), predict Aspartic / Not Aspartic at each cut site.

    Returns
    -------
    int
        0 -> every single cut site predicted "Aspartic"
        1 -> at least one cut site predicted "Not Aspartic" (stops early)
    """

    sequence = sequence.strip().upper()
    n = len(sequence)

    aspartic_idx = list(label_encoder.classes_).index("Aspartic")

    # Valid bar positions: need at least 2 residues before AND after
    # i.e. bar can go between index i-1 and i, for i in [2, n-2]
    for i in range(2, n - 1):
        linker = sequence[:i] + "|" + sequence[i:]

        X = generate_linker_features(linker, feature_names=feature_names)
        X_scaled = scaler_lr.transform(X.values)

        proba = log_reg.predict_proba(X_scaled)[0]
        prob_aspartic = proba[aspartic_idx]

        predicted_label = "Aspartic" if prob_aspartic >= 0.5 else "Not Aspartic"

        #print(f"Position {i}: {linker}  ->  {predicted_label} "
              #f"(Aspartic prob: {prob_aspartic:.4f})")

        if predicted_label == "Not Aspartic":
            #print("Found 'Not Aspartic' — stopping.")
            return 1

    #print("All positions predicted 'Aspartic'.")
    return 0


# -------------------------------------------------