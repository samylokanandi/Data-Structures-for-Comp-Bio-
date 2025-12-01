"""
Statistical Analysis for HMM Promoter Project
==============================================
This script adds the missing statistical analyses to your existing results.

Works with your existing:
  - preprocessed_data/data_splits.npz
  - hmm_results.json
  - hmm_utils.py (HMMPromoterClassifier)

Adds:
  1. Statistical significance testing (bootstrap + p-values)
  2. Confidence intervals for all metrics
  3. Label shuffle ablation
  4. Complete training fraction (0.25, 0.5, 0.75, 1.0)
"""

import numpy as np
import json
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmm_utils import HMMPromoterClassifier, evaluate_classifier
import os

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


# ============================================================================
# LOAD YOUR EXISTING DATA
# ============================================================================

print("Loading preprocessed data...")
data = np.load('preprocessed_data/data_splits.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

print(f"✓ Loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test sequences")

# Load existing results
print("\nLoading existing results...")
with open('hmm_results.json', 'r') as f:
    existing_results = json.load(f)

print(f"✓ Loaded results for K={list(existing_results.keys())}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def retrain_model(X, y, K, seed=42):
    """Retrain an HMM model with K states"""
    clf = HMMPromoterClassifier(
        n_states=K,
        n_iter=100,
        random_state=seed,
        topology='full',
        verbose=False
    )
    clf.fit(X, y)
    return clf


# ============================================================================
# ANALYSIS 1: CONFIDENCE INTERVALS FOR ALL K VALUES
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: Confidence Intervals (Bootstrap)")
print("="*80)

def calculate_confidence_intervals(K_values, X_test, y_test, n_bootstrap=200):
    """Calculate 95% CI for each K using bootstrap"""
    
    results = []
    
    for K in K_values:
        print(f"\nProcessing K={K}...")
        
        # Retrain model (or load if you saved them)
        clf = retrain_model(X_train, y_train, K)
        
        # Get baseline scores
        y_proba = clf.predict_proba(X_test)[:, 1]
        baseline_auroc = roc_auc_score(y_test, y_proba)
        baseline_auprc = average_precision_score(y_test, y_proba)
        
        # Bootstrap
        auroc_boots = []
        auprc_boots = []
        
        for b in range(n_bootstrap):
            if b % 50 == 0:
                print(f"  Bootstrap iteration {b}/{n_bootstrap}")
            
            # Resample test set
            n_samples = len(X_test)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_test[indices]
            y_boot = y_test[indices]
            
            # Get predictions on bootstrap sample
            y_proba_boot = clf.predict_proba(X_boot)[:, 1]
            
            auroc_boots.append(roc_auc_score(y_boot, y_proba_boot))
            auprc_boots.append(average_precision_score(y_boot, y_proba_boot))
        
        auroc_boots = np.array(auroc_boots)
        auprc_boots = np.array(auprc_boots)
        
        result = {
            'K': K,
            'AUROC_mean': auroc_boots.mean(),
            'AUROC_std': auroc_boots.std(),
            'AUROC_ci_lower': np.percentile(auroc_boots, 2.5),
            'AUROC_ci_upper': np.percentile(auroc_boots, 97.5),
            'AUPRC_mean': auprc_boots.mean(),
            'AUPRC_std': auprc_boots.std(),
            'AUPRC_ci_lower': np.percentile(auprc_boots, 2.5),
            'AUPRC_ci_upper': np.percentile(auprc_boots, 97.5),
        }
        
        print(f"  AUROC: {result['AUROC_mean']:.4f} [{result['AUROC_ci_lower']:.4f}, {result['AUROC_ci_upper']:.4f}]")
        print(f"  AUPRC: {result['AUPRC_mean']:.4f} [{result['AUPRC_ci_lower']:.4f}, {result['AUPRC_ci_upper']:.4f}]")
        
        results.append(result)
    
    return pd.DataFrame(results)


# Run confidence interval analysis
K_values = [2, 3, 4, 6, 8, 10]
ci_df = calculate_confidence_intervals(K_values, X_test, y_test, n_bootstrap=200)

# Save results
os.makedirs('results', exist_ok=True)
ci_df.to_csv('results/confidence_intervals.csv', index=False)
print(f"\n✓ Saved: results/confidence_intervals.csv")


# ============================================================================
# ANALYSIS 2: STATISTICAL COMPARISON K=6 vs K=10
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: Statistical Test (K=6 vs K=10)")
print("="*80)

def compare_models_statistically(K1, K2, X_test, y_test, n_bootstrap=1000):
    """Bootstrap test comparing two models"""
    
    print(f"\nTraining models...")
    clf_k1 = retrain_model(X_train, y_train, K1)
    clf_k2 = retrain_model(X_train, y_train, K2)
    
    print(f"Running bootstrap comparison (n={n_bootstrap})...")
    
    auroc_diffs = []
    auprc_diffs = []
    
    for b in range(n_bootstrap):
        if b % 200 == 0:
            print(f"  Iteration {b}/{n_bootstrap}")
        
        # Resample
        n_samples = len(X_test)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        
        # Get predictions
        proba_k1 = clf_k1.predict_proba(X_boot)[:, 1]
        proba_k2 = clf_k2.predict_proba(X_boot)[:, 1]
        
        # Calculate metrics
        auroc_k1 = roc_auc_score(y_boot, proba_k1)
        auroc_k2 = roc_auc_score(y_boot, proba_k2)
        auprc_k1 = average_precision_score(y_boot, proba_k1)
        auprc_k2 = average_precision_score(y_boot, proba_k2)
        
        auroc_diffs.append(auroc_k2 - auroc_k1)
        auprc_diffs.append(auprc_k2 - auprc_k1)
    
    auroc_diffs = np.array(auroc_diffs)
    auprc_diffs = np.array(auprc_diffs)
    
    # Paired t-test (one-sample test on differences)
    auroc_ttest = stats.ttest_1samp(auroc_diffs, 0)
    auprc_ttest = stats.ttest_1samp(auprc_diffs, 0)
    
    results = {
        'K1': int(K1),
        'K2': int(K2),
        'AUROC': {
            'mean_diff': float(auroc_diffs.mean()),
            'std_diff': float(auroc_diffs.std()),
            'ci_lower': float(np.percentile(auroc_diffs, 2.5)),
            'ci_upper': float(np.percentile(auroc_diffs, 97.5)),
            't_statistic': float(auroc_ttest.statistic),
            'p_value': float(auroc_ttest.pvalue),
            'significant': bool(auroc_ttest.pvalue < 0.05)
        },
        'AUPRC': {
            'mean_diff': float(auprc_diffs.mean()),
            'std_diff': float(auprc_diffs.std()),
            'ci_lower': float(np.percentile(auprc_diffs, 2.5)),
            'ci_upper': float(np.percentile(auprc_diffs, 97.5)),
            't_statistic': float(auprc_ttest.statistic),
            'p_value': float(auprc_ttest.pvalue),
            'significant': bool(auprc_ttest.pvalue < 0.05)
        }
    }
    
    return results


# Run comparison
comparison_results = compare_models_statistically(6, 10, X_test, y_test, n_bootstrap=1000)

# Print results
print("\n" + "-"*60)
print("RESULTS: K=10 vs K=6")
print("-"*60)
print(f"\nAUROC Difference (K=10 - K=6):")
print(f"  Mean: {comparison_results['AUROC']['mean_diff']:.4f}")
print(f"  95% CI: [{comparison_results['AUROC']['ci_lower']:.4f}, {comparison_results['AUROC']['ci_upper']:.4f}]")
print(f"  t-statistic: {comparison_results['AUROC']['t_statistic']:.3f}")
print(f"  p-value: {comparison_results['AUROC']['p_value']:.6f}")
print(f"  Significant: {'✓ YES' if comparison_results['AUROC']['significant'] else '✗ NO'} (α=0.05)")

print(f"\nAUPRC Difference (K=10 - K=6):")
print(f"  Mean: {comparison_results['AUPRC']['mean_diff']:.4f}")
print(f"  95% CI: [{comparison_results['AUPRC']['ci_lower']:.4f}, {comparison_results['AUPRC']['ci_upper']:.4f}]")
print(f"  t-statistic: {comparison_results['AUPRC']['t_statistic']:.3f}")
print(f"  p-value: {comparison_results['AUPRC']['p_value']:.6f}")
print(f"  Significant: {'✓ YES' if comparison_results['AUPRC']['significant'] else '✗ NO'} (α=0.05)")

# Save results
with open('results/k6_vs_k10_comparison.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"\n✓ Saved: results/k6_vs_k10_comparison.json")


# ============================================================================
# ANALYSIS 3: LABEL SHUFFLE ABLATION
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: Label Shuffle Ablation")
print("="*80)

def label_shuffle_ablation(X_train, y_train, X_test, y_test, K=10, n_shuffles=5):
    """Verify model learns real signal by shuffling labels"""
    
    print(f"\nRunning {n_shuffles} shuffled-label experiments for K={K}...")
    
    shuffled_results = []
    
    for i in range(n_shuffles):
        print(f"\n  Shuffle iteration {i+1}/{n_shuffles}")
        
        # Shuffle training labels
        y_train_shuffled = y_train.copy()
        np.random.shuffle(y_train_shuffled)
        
        # Train on shuffled labels
        clf_shuffled = HMMPromoterClassifier(
            n_states=K,
            n_iter=100,
            random_state=42 + i,
            topology='full',
            verbose=False
        )
        clf_shuffled.fit(X_train, y_train_shuffled)
        
        # Evaluate on real test set
        metrics = evaluate_classifier(clf_shuffled, X_test, y_test, set_name=f"Shuffle {i+1}")
        
        shuffled_results.append({
            'iteration': i+1,
            'auroc': float(metrics['auroc']),
            'auprc': float(metrics['auprc']),
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1'])
        })
    
    # Train on real labels for comparison
    print(f"\n  Training with REAL labels for comparison...")
    clf_real = retrain_model(X_train, y_train, K)
    real_metrics = evaluate_classifier(clf_real, X_test, y_test, set_name="Real labels")
    
    results = {
        'K': K,
        'shuffled': shuffled_results,
        'shuffled_mean': {
            'auroc': float(np.mean([r['auroc'] for r in shuffled_results])),
            'auprc': float(np.mean([r['auprc'] for r in shuffled_results])),
            'accuracy': float(np.mean([r['accuracy'] for r in shuffled_results])),
            'f1': float(np.mean([r['f1'] for r in shuffled_results]))
        },
        'shuffled_std': {
            'auroc': float(np.std([r['auroc'] for r in shuffled_results])),
            'auprc': float(np.std([r['auprc'] for r in shuffled_results])),
            'accuracy': float(np.std([r['accuracy'] for r in shuffled_results])),
            'f1': float(np.std([r['f1'] for r in shuffled_results]))
        },
        'real': {
            'auroc': float(real_metrics['auroc']),
            'auprc': float(real_metrics['auprc']),
            'accuracy': float(real_metrics['accuracy']),
            'f1': float(real_metrics['f1'])
        }
    }
    
    return results


# Run label shuffle
shuffle_results = label_shuffle_ablation(X_train, y_train, X_test, y_test, K=10, n_shuffles=5)

# Print summary
print("\n" + "-"*60)
print("LABEL SHUFFLE SUMMARY")
print("-"*60)
print(f"\nReal labels (K=10):")
print(f"  AUROC: {shuffle_results['real']['auroc']:.4f}")
print(f"  AUPRC: {shuffle_results['real']['auprc']:.4f}")
print(f"  Accuracy: {shuffle_results['real']['accuracy']:.4f}")

print(f"\nShuffled labels (mean ± std over {len(shuffle_results['shuffled'])} trials):")
print(f"  AUROC: {shuffle_results['shuffled_mean']['auroc']:.4f} ± {shuffle_results['shuffled_std']['auroc']:.4f}")
print(f"  AUPRC: {shuffle_results['shuffled_mean']['auprc']:.4f} ± {shuffle_results['shuffled_std']['auprc']:.4f}")
print(f"  Accuracy: {shuffle_results['shuffled_mean']['accuracy']:.4f} ± {shuffle_results['shuffled_std']['accuracy']:.4f}")

# Save results
with open('results/label_shuffle_ablation.json', 'w') as f:
    json.dump(shuffle_results, f, indent=2)
print(f"\n✓ Saved: results/label_shuffle_ablation.json")


# ============================================================================
# ANALYSIS 4: COMPLETE TRAINING FRACTION (add 0.75)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 4: Training Fraction Experiment (Complete)")
print("="*80)

def complete_training_fraction(X_train, y_train, X_val, y_val, X_test, y_test, K=10):
    """Run training fraction with all fractions including 0.75"""
    
    fractions = [0.25, 0.5, 0.75, 1.0]
    results = []
    
    rng = np.random.RandomState(123)
    n_train = len(X_train)
    indices = np.arange(n_train)
    
    for frac in fractions:
        n_sub = int(n_train * frac)
        sub_idx = rng.choice(indices, size=n_sub, replace=False)
        
        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx]
        
        print(f"\nTraining K={K} on {frac:.2%} of data ({n_sub} sequences)...")
        
        clf = HMMPromoterClassifier(
            n_states=K,
            n_iter=100,
            random_state=42,
            topology='full',
            verbose=False
        )
        clf.fit(X_sub, y_sub)
        
        # Evaluate on both val and test
        val_metrics = evaluate_classifier(clf, X_val, y_val, set_name=f"Val (frac={frac})")
        test_metrics = evaluate_classifier(clf, X_test, y_test, set_name=f"Test (frac={frac})")
        
        results.append({
            'fraction': frac,
            'n_samples': n_sub,
            'val_auroc': float(val_metrics['auroc']),
            'val_auprc': float(val_metrics['auprc']),
            'test_auroc': float(test_metrics['auroc']),
            'test_auprc': float(test_metrics['auprc']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1': float(test_metrics['f1'])
        })
    
    return pd.DataFrame(results)


# Run complete training fraction
train_frac_df = complete_training_fraction(X_train, y_train, X_val, y_val, X_test, y_test, K=10)

# Save results
train_frac_df.to_csv('results/training_fraction_complete.csv', index=False)
print(f"\n✓ Saved: results/training_fraction_complete.csv")

# Print summary
print("\n" + "-"*60)
print("TRAINING FRACTION SUMMARY")
print("-"*60)
print(train_frac_df.to_string(index=False))


# ============================================================================
# CREATE PUBLICATION-QUALITY FIGURES
# ============================================================================

print("\n" + "="*80)
print("CREATING FIGURES")
print("="*80)

os.makedirs('figures', exist_ok=True)

# FIGURE 1: Updated with confidence intervals
print("\nCreating Figure 1 (with confidence intervals)...")

fig, ax = plt.subplots(figsize=(10, 6))

K_vals = ci_df['K'].values

# Plot AUROC with error bars
ax.errorbar(K_vals, ci_df['AUROC_mean'], 
            yerr=[ci_df['AUROC_mean'] - ci_df['AUROC_ci_lower'],
                  ci_df['AUROC_ci_upper'] - ci_df['AUROC_mean']],
            marker='o', label='AUROC', capsize=5, linewidth=2, markersize=8)

# Plot AUPRC with error bars
ax.errorbar(K_vals, ci_df['AUPRC_mean'],
            yerr=[ci_df['AUPRC_mean'] - ci_df['AUPRC_ci_lower'],
                  ci_df['AUPRC_ci_upper'] - ci_df['AUPRC_mean']],
            marker='s', label='AUPRC', capsize=5, linewidth=2, markersize=8)

ax.set_xlabel('Number of Hidden States (K)', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('HMM Performance vs. Number of Hidden States\n(with 95% Confidence Intervals)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(K_vals)

plt.tight_layout()
plt.savefig('figures/figure1_with_ci.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/figure1_with_ci.png")
plt.close()


# FIGURE 4: AIC/BIC
print("\nCreating Figure 4 (AIC/BIC)...")

# Extract AIC/BIC from existing results
aic_bic_data = []
for k in K_values:
    k_str = str(k)
    if k_str in existing_results:
        aic_bic_data.append({
            'K': k,
            'AIC': existing_results[k_str]['model_selection']['aic'],
            'BIC': existing_results[k_str]['model_selection']['bic']
        })

aic_bic_df = pd.DataFrame(aic_bic_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Raw values
ax1.plot(aic_bic_df['K'], aic_bic_df['AIC'], marker='o', label='AIC', linewidth=2, markersize=8)
ax1.plot(aic_bic_df['K'], aic_bic_df['BIC'], marker='s', label='BIC', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Hidden States (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Information Criterion Value', fontsize=12, fontweight='bold')
ax1.set_title('AIC and BIC vs. Model Complexity', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(K_values)

# Plot 2: Relative to minimum (easier to see differences)
aic_rel = aic_bic_df['AIC'] - aic_bic_df['AIC'].min()
bic_rel = aic_bic_df['BIC'] - aic_bic_df['BIC'].min()

ax2.plot(aic_bic_df['K'], aic_rel, marker='o', label='AIC (relative)', linewidth=2, markersize=8)
ax2.plot(aic_bic_df['K'], bic_rel, marker='s', label='BIC (relative)', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Hidden States (K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative IC Value (min = 0)', fontsize=12, fontweight='bold')
ax2.set_title('Relative Information Criteria', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(K_values)

plt.tight_layout()
plt.savefig('figures/figure4_aic_bic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/figure4_aic_bic.png")
plt.close()


# FIGURE 5: Training Fraction
print("\nCreating Figure 5 (Training Fraction)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUROC and AUPRC
ax1.plot(train_frac_df['fraction'], train_frac_df['test_auroc'], 
         marker='o', label='Test AUROC', linewidth=2, markersize=8)
ax1.plot(train_frac_df['fraction'], train_frac_df['test_auprc'], 
         marker='s', label='Test AUPRC', linewidth=2, markersize=8)
ax1.plot(train_frac_df['fraction'], train_frac_df['val_auprc'], 
         marker='^', label='Val AUPRC', linewidth=2, markersize=8, linestyle='--')
ax1.set_xlabel('Fraction of Training Data', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Effect of Training Set Size (K=10)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(train_frac_df['fraction'])

# Plot 2: Accuracy and F1
ax2.plot(train_frac_df['fraction'], train_frac_df['test_accuracy'], 
         marker='o', label='Test Accuracy', linewidth=2, markersize=8)
ax2.plot(train_frac_df['fraction'], train_frac_df['test_f1'], 
         marker='s', label='Test F1', linewidth=2, markersize=8)
ax2.set_xlabel('Fraction of Training Data', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy and F1 vs. Training Size', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(train_frac_df['fraction'])

plt.tight_layout()
plt.savefig('figures/figure5_training_fraction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/figure5_training_fraction.png")
plt.close()


# ============================================================================
# SAVE AIC/BIC TABLE
# ============================================================================

aic_bic_df.to_csv('results/aic_bic_table.csv', index=False)
print("\n✓ Saved: results/aic_bic_table.csv")


# ============================================================================
# GENERATE SUMMARY TABLES FOR REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING REPORT TABLES")
print("="*80)

# Table 3: AIC/BIC
print("\n### TABLE 3: AIC/BIC Comparison ###")
print("\nCopy this into your report:\n")
print(aic_bic_df.to_markdown(index=False, floatfmt='.1f'))

# Table 4: K=6 vs K=10
print("\n\n### TABLE 4: Statistical Comparison K=6 vs K=10 ###")
print("\nCopy this into your report:\n")
print(f"Metric: AUROC")
print(f"  K=6 mean: {ci_df[ci_df['K']==6]['AUROC_mean'].values[0]:.4f} ± {ci_df[ci_df['K']==6]['AUROC_std'].values[0]:.4f}")
print(f"  K=10 mean: {ci_df[ci_df['K']==10]['AUROC_mean'].values[0]:.4f} ± {ci_df[ci_df['K']==10]['AUROC_std'].values[0]:.4f}")
print(f"  Difference: {comparison_results['AUROC']['mean_diff']:.4f}")
print(f"  95% CI: [{comparison_results['AUROC']['ci_lower']:.4f}, {comparison_results['AUROC']['ci_upper']:.4f}]")
print(f"  p-value: {comparison_results['AUROC']['p_value']:.6f}")

print(f"\nMetric: AUPRC")
print(f"  K=6 mean: {ci_df[ci_df['K']==6]['AUPRC_mean'].values[0]:.4f} ± {ci_df[ci_df['K']==6]['AUPRC_std'].values[0]:.4f}")
print(f"  K=10 mean: {ci_df[ci_df['K']==10]['AUPRC_mean'].values[0]:.4f} ± {ci_df[ci_df['K']==10]['AUPRC_std'].values[0]:.4f}")
print(f"  Difference: {comparison_results['AUPRC']['mean_diff']:.4f}")
print(f"  95% CI: [{comparison_results['AUPRC']['ci_lower']:.4f}, {comparison_results['AUPRC']['ci_upper']:.4f}]")
print(f"  p-value: {comparison_results['AUPRC']['p_value']:.6f}")

# Table 5: Label Shuffle
print("\n\n### TABLE 5: Label Shuffle Ablation ###")
print("\nCopy this into your report:\n")
print(f"Real labels:     AUROC={shuffle_results['real']['auroc']:.4f}, AUPRC={shuffle_results['real']['auprc']:.4f}, Acc={shuffle_results['real']['accuracy']:.4f}")
print(f"Shuffled (mean): AUROC={shuffle_results['shuffled_mean']['auroc']:.4f}±{shuffle_results['shuffled_std']['auroc']:.4f}, AUPRC={shuffle_results['shuffled_mean']['auprc']:.4f}±{shuffle_results['shuffled_std']['auprc']:.4f}, Acc={shuffle_results['shuffled_mean']['accuracy']:.4f}±{shuffle_results['shuffled_std']['accuracy']:.4f}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ALL ANALYSES COMPLETE!")
print("="*80)

print("\nGenerated files:")
print("\nResults:")
print("  ✓ results/confidence_intervals.csv")
print("  ✓ results/k6_vs_k10_comparison.json")
print("  ✓ results/label_shuffle_ablation.json")
print("  ✓ results/training_fraction_complete.csv")
print("  ✓ results/aic_bic_table.csv")

print("\nFigures:")
print("  ✓ figures/figure1_with_ci.png (REPLACE your current Figure 1)")
print("  ✓ figures/figure4_aic_bic.png (NEW)")
print("  ✓ figures/figure5_training_fraction.png (NEW)")

print("\nNext steps:")
print("  1. Open report_text_additions.py")
print("  2. Copy the text sections into your report")
print("  3. Fill in the values shown in the tables above")
print("  4. Replace Figure 1, add Figures 4-5")
print("  5. Add Tables 3, 4, 5")
print("  6. Update Discussion with overfitting paragraph")
print("  7. Submit!")

print("\n" + "="*80)