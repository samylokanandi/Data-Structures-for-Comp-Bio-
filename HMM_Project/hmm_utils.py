"""
HMM Utilities for Promoter Detection
Contains HMMPromoterClassifier and evaluation functions
"""

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             accuracy_score, f1_score, precision_recall_curve,
                             roc_curve, confusion_matrix)
import math


# DNA to integer encoding
DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
N_SYMBOLS = 4

def encode_sequence(sequence):
    """
    Convert DNA sequence to integer array.
    
    Args:
        sequence: String of DNA bases (A, C, G, T)
    
    Returns:
        numpy array of integers (0-3)
    """
    return np.array([DNA_TO_INT[base.upper()] for base in sequence])

def encode_sequences(sequences):
    """
    Encode multiple sequences and create lengths array.
    
    Args:
        sequences: List or array of DNA sequences
    
    Returns:
        X: 2D array of encoded sequences (concatenated)
        lengths: Array of sequence lengths
    """
    encoded = [encode_sequence(seq) for seq in sequences]
    lengths = np.array([len(seq) for seq in encoded])
    X = np.concatenate(encoded).reshape(-1, 1)
    return X, lengths


class HMMPromoterClassifier:
    """
    Hidden Markov Model for promoter vs non-promoter classification.
    """
    
    def __init__(self, n_states=4, n_iter=100, random_state=42, 
                 topology='full', verbose=False):
        """
        Initialize HMM classifier.
        
        Args:
            n_states: Number of hidden states
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
            topology: 'full' for fully connected, 'left-right' for left-to-right
            verbose: Whether to print training progress
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.topology = topology
        self.verbose = verbose
        
        # Separate models for promoters and non-promoters
        self.promoter_model = None
        self.nonpromoter_model = None
        
        self._create_models()
    
    def _create_models(self):
        """Create HMM models with specified topology."""
        params = 'ste'  # Train start probs, transition, and emission probs
        init_params = 'ste'
        
        if self.topology == 'left-right':
            # Left-to-right topology: can only move forward or stay
            self.promoter_model = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=self.verbose,
                params=params,
                init_params=init_params,
                n_features=N_SYMBOLS
            )
            self.nonpromoter_model = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state + 1,
                verbose=self.verbose,
                params=params,
                init_params=init_params,
                n_features=N_SYMBOLS
            )
            
            # Set left-to-right transition matrix
            for model in [self.promoter_model, self.nonpromoter_model]:
                transmat = np.zeros((self.n_states, self.n_states))
                for i in range(self.n_states):
                    for j in range(i, self.n_states):
                        transmat[i, j] = 1.0
                transmat = transmat / transmat.sum(axis=1, keepdims=True)
                model.transmat_ = transmat
                
        else:  # full topology
            self.promoter_model = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=self.verbose,
                params=params,
                init_params=init_params,
                n_features=N_SYMBOLS
            )
            self.nonpromoter_model = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state + 1,
                verbose=self.verbose,
                params=params,
                init_params=init_params,
                n_features=N_SYMBOLS
            )
    
    def fit(self, X_train, y_train):
        """
        Train separate HMMs for promoters and non-promoters.
        
        Args:
            X_train: Array of DNA sequences
            y_train: Array of labels (1=promoter, 0=non-promoter)
        
        Returns:
            self
        """
        # Separate sequences by class
        promoter_seqs = X_train[y_train == 1]
        nonpromoter_seqs = X_train[y_train == 0]
        
        if self.verbose:
            print(f"\nTraining HMM with {self.n_states} states ({self.topology} topology)")
            print(f"Promoter sequences: {len(promoter_seqs)}")
            print(f"Non-promoter sequences: {len(nonpromoter_seqs)}")
        
        # Encode sequences
        X_promoter, lengths_promoter = encode_sequences(promoter_seqs)
        X_nonpromoter, lengths_nonpromoter = encode_sequences(nonpromoter_seqs)
        
        # Train models
        if self.verbose:
            print("Training promoter model...")
        self.promoter_model.fit(X_promoter, lengths_promoter)
        
        if self.verbose:
            print("Training non-promoter model...")
        self.nonpromoter_model.fit(X_nonpromoter, lengths_nonpromoter)
        
        if self.verbose:
            print("Training complete!")
        
        return self
    
    def predict_log_proba(self, X):
        """
        Predict log probability for each class.
        
        Args:
            X: Array of DNA sequences
        
        Returns:
            Array of shape (n_samples, 2) with log probabilities
        """
        X_encoded, lengths = encode_sequences(X)
        
        # Get log likelihoods from both models
        promoter_scores = []
        nonpromoter_scores = []
        
        start_idx = 0
        for length in lengths:
            seq = X_encoded[start_idx:start_idx+length]
            promoter_scores.append(self.promoter_model.score(seq))
            nonpromoter_scores.append(self.nonpromoter_model.score(seq))
            start_idx += length
        
        return np.column_stack([nonpromoter_scores, promoter_scores])
    
    def predict_proba(self, X):
        """
        Predict probability for each class (normalized).
        
        Args:
            X: Array of DNA sequences
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        log_proba = self.predict_log_proba(X)
        
        # Convert to probabilities using softmax
        max_score = np.max(log_proba, axis=1, keepdims=True)
        exp_scores = np.exp(log_proba - max_score)
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Array of DNA sequences
        
        Returns:
            Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def get_emission_matrix(self, model_type='promoter'):
        """
        Get emission probability matrix for visualization.
        
        Args:
            model_type: 'promoter' or 'nonpromoter'
        
        Returns:
            Emission matrix of shape (n_states, 4)
        """
        if model_type == 'promoter':
            return self.promoter_model.emissionprob_
        else:
            return self.nonpromoter_model.emissionprob_


def evaluate_classifier(clf, X, y, set_name="Test"):
    """
    Comprehensive evaluation of classifier.
    
    Args:
        clf: Trained classifier with predict_proba method
        X: Sequences
        y: True labels
        set_name: Name of dataset (for printing)
    
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]  # Probability of being promoter
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'auroc': roc_auc_score(y, y_proba),
        'auprc': average_precision_score(y, y_proba),
        'f1': f1_score(y, y_pred),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    j_scores = tpr - fpr
    best_threshold_idx = np.argmax(j_scores)
    metrics['youden_j'] = j_scores[best_threshold_idx]
    metrics['best_threshold'] = thresholds[best_threshold_idx]
    
    # Print results
    print(f"\n{set_name} Set Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Youden's J:  {metrics['youden_j']:.4f}")
    
    return metrics


def calculate_aic_bic(clf, X, y):
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        clf: Trained HMM classifier
        X: Sequences
        y: Labels
    
    Returns:
        Dictionary with AIC and BIC values
    """
    # Calculate log likelihood
    log_likelihood = 0
    promoter_seqs = X[y == 1]
    nonpromoter_seqs = X[y == 0]
    
    if len(promoter_seqs) > 0:
        X_prom, lens_prom = encode_sequences(promoter_seqs)
        log_likelihood += clf.promoter_model.score(X_prom, lens_prom) * len(promoter_seqs)
    
    if len(nonpromoter_seqs) > 0:
        X_nonprom, lens_nonprom = encode_sequences(nonpromoter_seqs)
        log_likelihood += clf.nonpromoter_model.score(X_nonprom, lens_nonprom) * len(nonpromoter_seqs)
    
    # Number of parameters
    n_states = clf.n_states
    n_params = 2 * (n_states * (n_states - 1) + n_states * (N_SYMBOLS - 1) + (n_states - 1))
    
    n_samples = len(X)
    
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_samples) * n_params - 2 * log_likelihood
    
    return {
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'n_params': n_params
    }

def _gc_content(seq: str) -> float:
    """Compute GC fraction for a single sequence."""
    seq = seq.upper()
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def describe_dataset_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Print a concise summary of train/val/test splits:
    - # sequences and class balance
    - length distribution
    - GC content distribution
    """

    def _summarize(name, X, y):
        lengths = np.array([len(s) for s in X])
        gc_vals = np.array([_gc_content(s) for s in X])
        n = len(X)
        pos = int(np.sum(y))
        neg = n - pos

        print(f"\n{name} set:")
        print(f"  # sequences: {n} (promoters={pos}, non-promoters={neg})")
        print(f"  length: min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}, median={np.median(lengths):.1f}")
        print(f"  GC%:    mean={gc_vals.mean():.3f}, std={gc_vals.std():.3f}")

    print("=" * 80)
    print("DATASET SUMMARY")
    _summarize("Train", X_train, y_train)
    _summarize("Val", X_val, y_val)
    _summarize("Test", X_test, y_test)
    print("=" * 80)

class MarkovPromoterClassifier:
    """
    Simple 1st-order Markov baseline:
    - Train separate 1st-order Markov chains for promoters and non-promoters
    - Classify sequences by log-likelihood ratio
    Compatible with evaluate_classifier().
    """

    def __init__(self, pseudocount: float = 1e-3):
        self.pseudocount = pseudocount
        self.promoter_log_probs = None
        self.nonpromoter_log_probs = None

    def _train_markov(self, sequences):
        # Bigram counts with pseudocount smoothing
        bases = ["A", "C", "G", "T"]
        counts = {a: {b: self.pseudocount for b in bases} for a in bases}

        for seq in sequences:
            seq = seq.upper()
            for a, b in zip(seq, seq[1:]):
                if a in counts and b in counts[a]:
                    counts[a][b] += 1

        # Convert to log-probabilities
        log_probs = {}
        for a in bases:
            total = sum(counts[a].values())
            log_probs[a] = {b: math.log(counts[a][b] / total) for b in bases}
        return log_probs

    def fit(self, X_train, y_train):
        promoters = [s for s, y in zip(X_train, y_train) if y == 1]
        nonpromoters = [s for s, y in zip(X_train, y_train) if y == 0]

        self.promoter_log_probs = self._train_markov(promoters)
        self.nonpromoter_log_probs = self._train_markov(nonpromoters)
        return self

    def _loglik(self, seq, log_probs):
        seq = seq.upper()
        if len(seq) < 2:
            return 0.0

        ll = 0.0
        for a, b in zip(seq, seq[1:]):
            if a in log_probs and b in log_probs[a]:
                ll += log_probs[a][b]
            else:
                # Very low probability for unseen transition
                ll += math.log(self.pseudocount / 4.0)
        return ll

    def predict_log_proba(self, X):
        """Return log-probabilities for [non-promoter, promoter]."""
        rows = []
        for seq in X:
            lp = self._loglik(seq, self.promoter_log_probs)
            ln = self._loglik(seq, self.nonpromoter_log_probs)
            rows.append([ln, lp])
        return np.array(rows)

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        max_log = np.max(log_proba, axis=1, keepdims=True)
        exp_scores = np.exp(log_proba - max_log)
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


if __name__ == "__main__":
    print("HMM utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - encode_sequence()")
    print("  - encode_sequences()")
    print("  - HMMPromoterClassifier")
    print("  - evaluate_classifier()")
    print("  - calculate_aic_bic()")