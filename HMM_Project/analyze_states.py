"""
Analyze hidden states for the best-K promoter HMM.

Usage (after running preprocessing & train_hmm):
    python analyze_states.py
"""

import json
import numpy as np
from hmmlearn import hmm

from hmm_utils import DNA_TO_INT, encode_sequences, N_SYMBOLS


# ----------------------------------------------------------------------
# 1. Helper: load best K from hmm_results.json
# ----------------------------------------------------------------------
def load_best_k(results_path="hmm_results.json"):
    with open(results_path, "r") as f:
        results = json.load(f)

    best_k = None
    best_val_auprc = -np.inf

    for k_str, info in results.items():
        # skip non-integer keys like "markov_baseline" etc.
        try:
            k_int = int(k_str)
        except ValueError:
            continue

        val_auprc = info.get("val_auprc", None)
        if val_auprc is not None and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_k = k_int

    if best_k is None:
        raise RuntimeError(
            "Could not find any integer K entries with val_auprc in hmm_results.json"
        )

    print(f"Best K from hmm_results.json (by val AUPRC): {best_k} (val_auprc={best_val_auprc:.4f})")
    return best_k


# ----------------------------------------------------------------------
# 2. Helper: train a promoter-only HMM with best K
# ----------------------------------------------------------------------
def train_promoter_hmm(
    best_k,
    data_splits_path="preprocessed_data/data_splits.npz",
    max_promoters_for_fit=None,
):
    print("\nLoading data splits...")
    data = np.load(data_splits_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]

    promoters = X_train[y_train == 1]
    n_prom = len(promoters)
    print(f"Total promoter sequences in training set: {n_prom}")

    if max_promoters_for_fit is not None and max_promoters_for_fit < n_prom:
        promoters = promoters[:max_promoters_for_fit]
        print(f"Using first {len(promoters)} promoters for state analysis (subset for speed).")

    # Encode sequences using same encoding as training
    X_enc, lengths = encode_sequences(promoters)

    # Categorical HMM over 4 DNA symbols (matches HMMPromoterClassifier)
    model = hmm.CategoricalHMM(
        n_components=best_k,
        n_iter=100,
        random_state=42,
        verbose=False,
        n_features=N_SYMBOLS,
    )

    print(f"\nFitting promoter-only HMM with K={best_k} states on {len(promoters)} sequences...")
    model.fit(X_enc, lengths)
    print("HMM training for state analysis complete.")

    return model, promoters


# ----------------------------------------------------------------------
# 3. Helper: compute state-position heatmap
# ----------------------------------------------------------------------
def compute_state_position_heatmap(model, sequences):
    """
    For each promoter sequence, run Viterbi and record which state is active
    at each position. Returns a K x L matrix of frequencies.
    """
    K = model.n_components
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)

    counts = np.zeros((K, max_len), dtype=float)

    for seq in sequences:
        # Make sure we use uppercase bases to match DNA_TO_INT
        seq = seq.upper()

        # Encode sequence; fall back to 'A' (0) for any unexpected character
        seq_enc = np.array(
            [[DNA_TO_INT.get(b, 0)] for b in seq]
        )

        # Viterbi decoding
        state_path = model.predict(seq_enc)

        for pos, state in enumerate(state_path):
            counts[state, pos] += 1.0

    # Normalize per state to get frequencies across positions
    with np.errstate(divide="ignore", invalid="ignore"):
        freqs = counts / counts.sum(axis=1, keepdims=True)
        freqs[np.isnan(freqs)] = 0.0

    return freqs



# ----------------------------------------------------------------------
# 4. Helper: pretty-print emission probabilities
# ----------------------------------------------------------------------
def print_emission_profiles(model):
    """
    Print emission distributions for each hidden state in a readable way.
    """
    bases = ["A", "C", "G", "T"]
    emis = model.emissionprob_  # shape: K x 4

    print("\nEmission probabilities per state:")
    for k, row in enumerate(emis):
        joined = ", ".join(f"{b}={p:.3f}" for b, p in zip(bases, row))
        max_idx = np.argmax(row)
        print(f"  State {k}: {joined}   (peak: {bases[max_idx]})")


# ----------------------------------------------------------------------
# 5. Helper: print where each state tends to appear (position-wise)
# ----------------------------------------------------------------------
def summarize_state_positions(freqs, top_n=5):
    """
    For each state, print the top positions where it appears most frequently.
    """
    K, L = freqs.shape
    print("\nState position preferences (top positions by frequency):")
    for k in range(K):
        pos_freq = list(enumerate(freqs[k]))
        pos_freq.sort(key=lambda x: x[1], reverse=True)
        top = pos_freq[:top_n]
        pretty = ", ".join(f"pos={p} (freq={f:.3f})" for p, f in top)
        print(f"  State {k}: {pretty}")


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    best_k = load_best_k("hmm_results.json")

    # You can cap this if training is slow; otherwise leave as None
    model, promoters = train_promoter_hmm(best_k, max_promoters_for_fit=None)

    freqs = compute_state_position_heatmap(model, promoters)

    # Emission profiles → what each state "likes" (bases)
    print_emission_profiles(model)

    # Position preferences → where each state tends to fire along the sequence
    summarize_state_positions(freqs, top_n=5)

    # Optional: save freqs to disk for plotting in a notebook
    np.save("state_position_freqs.npy", freqs)
    print("\nSaved state-position frequency matrix to state_position_freqs.npy")


if __name__ == "__main__":
    main()
