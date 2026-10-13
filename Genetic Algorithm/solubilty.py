
SCM_OPTIMIZED = {
    'A': 599.425, 'C': 363.825, 'D': 507.95,  'E': 570.85,
    'F': 420.125, 'G': 378.05,  'H': 406.175, 'I': 414.75,
    'K': 445.275, 'L': 440.725, 'M': 420.875, 'N': 376.65,
    'P': 406.225, 'Q': 400.025, 'R': 370.575, 'S': 334.1,
    'T': 411.025, 'V': 424.175, 'W': 350.0,   'Y': 339.8,
}


def solubility_score(seq: str, scale: dict = SCM_OPTIMIZED) -> float:
    seq = seq.strip().upper()
    try:
        values = [scale[aa] for aa in seq]
    except KeyError as e:
        raise ValueError(f"Unknown residue '{e.args[0]}' in sequence") from e
    return sum(values) / len(values)

