"""
Train HMM models with different numbers of states
"""
import numpy as np
from hmm_utils import (
    HMMPromoterClassifier,
    evaluate_classifier,
    calculate_aic_bic,
    describe_dataset_splits,
    MarkovPromoterClassifier,
)
import json


# Load preprocessed data
print("Loading preprocessed data...")
data = np.load('preprocessed_data/data_splits.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
# NEW: print actual dataset stats (sizes, lengths, GC%)
describe_dataset_splits(X_train, y_train, X_val, y_val, X_test, y_test)


print(f"Train: {len(X_train)} sequences")
print(f"Val: {len(X_val)} sequences")
print(f"Test: {len(X_test)} sequences")

# Test different K values
k_values = [2, 3, 4, 6, 8, 10]
results = {}

for k in k_values:
    print(f"\n{'='*80}")
    print(f"Training HMM with K={k} states")
    print(f"{'='*80}")
    
    # Try 5 random initializations
    best_val_auprc = 0
    best_model = None
    
    for seed in [42, 43, 44, 45, 46]:
        print(f"\n  Seed {seed}...", end=" ")
        clf = HMMPromoterClassifier(
            n_states=k,
            n_iter=100,
            random_state=seed,
            topology='full',
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        # Quick validation check
        val_proba = clf.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(y_val, val_proba)
        print(f"Val AUPRC: {val_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model = clf
    
    # Evaluate best model on test set
    print(f"\n{'='*40}")
    print(f"Best model (val AUPRC={best_val_auprc:.4f})")
    print(f"{'='*40}")
    test_metrics = evaluate_classifier(best_model, X_test, y_test, "Test")
    
    # Calculate AIC/BIC
    model_selection = calculate_aic_bic(best_model, X_val, y_val)
    
    results[k] = {
        'test_metrics': test_metrics,
        'model_selection': model_selection,
        'val_auprc': best_val_auprc
    }

# ----------------------------------------------------------------------
# Markov-chain baseline (no hidden states)
# ----------------------------------------------------------------------
print("\n=== Training Markov baseline (1st-order) ===")
markov_clf = MarkovPromoterClassifier()
markov_clf.fit(X_train, y_train)

markov_test_metrics = evaluate_classifier(
    markov_clf, X_test, y_test, set_name="Test (Markov baseline)"
)

results["markov_baseline"] = {
    "order": 1,
    "test_metrics": {m: float(v) for m, v in markov_test_metrics.items()},
}


# ----------------------------------------------------------------------
# Training fraction experiment for the best K
# ----------------------------------------------------------------------
print("\n=== Training-fraction experiment for best K ===")

def _get_val_auprc(results_dict, k):
    """
    Helper to grab val_auprc regardless of whether keys are ints or strings.
    """
    if k in results_dict:
        return results_dict[k]["val_auprc"]
    elif str(k) in results_dict:
        return results_dict[str(k)]["val_auprc"]
    else:
        raise KeyError(f"K={k} not found in results.")

# 1) Pick best K based on validation AUPRC
best_k = max(k_values, key=lambda k: _get_val_auprc(results, k))
print(f"Best K based on validation AUPRC: {best_k}")

fractions = [0.25, 0.5, 1.0]
rng = np.random.RandomState(123)

fraction_results = {}

n_train = len(X_train)
indices = np.arange(n_train)

for frac in fractions:
    n_sub = int(n_train * frac)
    sub_idx = rng.choice(indices, size=n_sub, replace=False)

    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    print(f"\n--- Training HMM with K={best_k} on {n_sub} "
          f"({frac:.2f} of training set) sequences ---")

    clf_frac = HMMPromoterClassifier(
        n_states=best_k,
        n_iter=100,
        random_state=42,
        topology='full'
    )
    clf_frac.fit(X_sub, y_sub)

    frac_test_metrics = evaluate_classifier(
        clf_frac, X_test, y_test,
        set_name=f"Test (K={best_k}, train_frac={frac:.2f})"
    )

    fraction_results[str(frac)] = {
        "n_train": int(n_sub),
        "test_metrics": {m: float(v) for m, v in frac_test_metrics.items()},
    }

# Store in main results dict so it gets saved to hmm_results.json
results["training_fraction_experiment"] = {
    "best_k": int(best_k),
    "fractions": fraction_results,
}

print("\nSummary: training-fraction experiment (best K)")
for frac_str, info in fraction_results.items():
    tm = info["test_metrics"]
    print(f"  frac={frac_str}: n={info['n_train']}, "
          f"AUROC={tm['auroc']:.4f}, AUPRC={tm['auprc']:.4f}, F1={tm['f1']:.4f}")



# Print summary
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\nK\tAUROC\tAUPRC\tF1\tAIC\tBIC")
print("-" * 60)
for k in k_values:
    r = results[k]   # <--- use k directly, not str(k)
    print(f"{k}\t{r['test_metrics']['auroc']:.4f}\t"
          f"{r['test_metrics']['auprc']:.4f}\t"
          f"{r['test_metrics']['f1']:.4f}\t"
          f"{r['model_selection']['aic']:.1f}\t"
          f"{r['model_selection']['bic']:.1f}")

# Save results
with open('hmm_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print("\nResults saved to: hmm_results.json")