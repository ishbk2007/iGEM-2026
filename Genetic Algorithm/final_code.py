
from ph import charge_isp
from hydrophobicty import gravy_score
from solubilty import solubility_score
from serine import check_sequence_all_aspartic
from tpm import evolution        # <-- only import evolution, nothing else



# ---------------------------------------------------------
# Reward vector for one sequence
# ---------------------------------------------------------
def reward_func(seq):
    return [
        *charge_isp(seq),
        gravy_score(seq),
        solubility_score(seq),
        check_sequence_all_aspartic(seq),
    ]

# ---------------------------------------------------------
# Pareto dominance (lower = better on every objective — verify this fits your metrics)
# ---------------------------------------------------------
def pareto_dominates(parent_reward, daughter_reward):
    if len(parent_reward) != len(daughter_reward):
        raise ValueError("Vectors must have the same length")
    at_least_as_good = all(p <= d for p, d in zip(parent_reward, daughter_reward))
    return at_least_as_good

def weighted_sum(reward_vec, weights=None):
    if weights is None:
        weights = [1] * len(reward_vec)
    return sum(w * r for w, r in zip(weights, reward_vec))

# ---------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------
generation = {}

def GA(seq_l, k, num_gen, num_daughter, top_k):
    if k == 0:
        return seq_l

    scored_survivors = []

    for parent in seq_l:
        parent_reward = reward_func(parent)
        for _ in range(num_daughter):
            daughter = evolution(parent)
            if len(daughter) < 4:          # hard floor — skip too-short daughters
                continue
            daughter_reward = reward_func(daughter)
            if pareto_dominates(parent_reward, daughter_reward):
                scored_survivors.append((daughter, daughter_reward))

    scored_survivors.sort(key=lambda pair: weighted_sum(pair[1]))
    top_sequences = [seq for seq, _ in scored_survivors[:top_k]]
    generation[f"Generation {num_gen - k + 1}"] = top_sequences

    return GA(top_sequences, k - 1, num_gen, num_daughter, top_k)

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    num_gen = 10
    num_daughter = 500
    top_k = 3

    initial_seq = "KDWRTLGS"   #

    print("Starting seq:", initial_seq)

    final_population = GA([initial_seq], num_gen, num_gen, num_daughter, top_k)
    print(final_population)

