"""
Complete Preprocessing Pipeline for HMM Promoter Detection
Run this script locally to reproduce all preprocessing steps.

Requirements:
    pip install pandas numpy scipy scikit-learn matplotlib seaborn

Usage:
    python preprocessing_pipeline.py <path_to_tableData.tsv>

Example:
    python preprocessing_pipeline.py ./tableData.tsv
"""

import sys
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

def main(tsv_file_path):
    """Main preprocessing pipeline"""
    
    print("="*80)
    print("HMM PROMOTER DETECTION - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Create output directory
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: DATA LOADING AND CLEANING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("="*80)
    
    print(f"\nLoading data from: {tsv_file_path}")
    df = pd.read_csv(tsv_file_path, sep='\t')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    seq_col = '6)pmSequence'
    
    # Check for missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Remove rows with missing sequences
    df_valid = df.dropna(subset=[seq_col]).copy()
    print(f"\nSequences with valid data: {len(df_valid)} / {len(df)} "
          f"({len(df_valid)/len(df)*100:.1f}%)")
    
    # Validate sequence content
    def has_only_acgt(seq):
        """Check if sequence contains only A, C, G, T"""
        return set(seq.upper()).issubset({'A', 'C', 'G', 'T'})
    
    clean_seqs = df_valid[seq_col].apply(has_only_acgt)
    df_clean = df_valid[clean_seqs].copy()
    
    print(f"Sequences with only A/C/G/T: {len(df_clean)} / {len(df_valid)}")
    print(f"Removed {len(df_valid) - len(df_clean)} sequences with ambiguous bases")
    
    # Verify sequence length
    df_clean['seq_length'] = df_clean[seq_col].str.len()
    print(f"\nSequence length statistics:")
    print(df_clean['seq_length'].describe())
    
    # Calculate GC content
    def calculate_gc_content(seq):
        """Calculate GC content of a sequence"""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0
    
    df_clean['gc_content'] = df_clean[seq_col].apply(calculate_gc_content)
    
    print(f"\nGC content statistics:")
    print(f"  Mean: {df_clean['gc_content'].mean():.3f}")
    print(f"  Std:  {df_clean['gc_content'].std():.3f}")
    print(f"  Min:  {df_clean['gc_content'].min():.3f}")
    print(f"  Max:  {df_clean['gc_content'].max():.3f}")
    
    # Save cleaned promoter data
    clean_file = os.path.join(output_dir, 'promoters_clean.csv')
    df_clean.to_csv(clean_file, index=False)
    print(f"\nSaved cleaned promoters to: {clean_file}")
    
    # ========================================================================
    # STEP 2: NEGATIVE SAMPLE GENERATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: NEGATIVE SAMPLE GENERATION (GC-MATCHED)")
    print("="*80)
    
    promoter_seqs = df_clean[seq_col].values
    seq_length = int(df_clean['seq_length'].mode()[0])
    
    print(f"\nPromoters: {len(promoter_seqs)}")
    print(f"Sequence length: {seq_length}bp")
    
    # Calculate GC content for all promoters
    promoter_gc_contents = [calculate_gc_content(seq) for seq in promoter_seqs]
    
    print(f"\nPromoter GC content: mean={np.mean(promoter_gc_contents):.3f}, "
          f"std={np.std(promoter_gc_contents):.3f}")
    
    def generate_random_sequence(length, gc_content):
        """
        Generate a random DNA sequence with target GC content.
        """
        num_gc = int(length * gc_content)
        num_at = length - num_gc
        
        # Split GC equally between G and C, AT equally between A and T
        num_g = num_gc // 2
        num_c = num_gc - num_g
        num_a = num_at // 2
        num_t = num_at - num_a
        
        # Create sequence
        bases = ['G'] * num_g + ['C'] * num_c + ['A'] * num_a + ['T'] * num_t
        np.random.shuffle(bases)
        
        return ''.join(bases)
    
    # Generate GC-matched negative samples
    print(f"\nGenerating {len(promoter_seqs)} GC-matched negative sequences...")
    negative_seqs = []
    
    for gc in promoter_gc_contents:
        # Add some noise to avoid exact matching
        noisy_gc = gc + np.random.normal(0, 0.02)
        noisy_gc = np.clip(noisy_gc, 0.2, 0.8)
        
        neg_seq = generate_random_sequence(seq_length, noisy_gc)
        negative_seqs.append(neg_seq)
    
    negative_seqs = np.array(negative_seqs)
    
    # Verify GC matching
    negative_gc_contents = [calculate_gc_content(seq) for seq in negative_seqs]
    print(f"Negative GC content: mean={np.mean(negative_gc_contents):.3f}, "
          f"std={np.std(negative_gc_contents):.3f}")
    
    # Statistical test for GC matching
    t_stat, p_value = stats.ttest_ind(promoter_gc_contents, negative_gc_contents)
    print(f"\nT-test for GC content difference:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.3f}")
    print(f"  Difference: {np.mean(promoter_gc_contents) - np.mean(negative_gc_contents):.3f}")
    
    if p_value > 0.05:
        print("  ✓ GC contents are not significantly different")
    else:
        print(f"  ⚠ GC contents differ (p={p_value:.3f}), but difference is small")
    
    # ========================================================================
    # STEP 3: CREATE COMBINED DATASET
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CREATE COMBINED DATASET")
    print("="*80)
    
    # Combine promoters and negatives
    all_sequences = np.concatenate([promoter_seqs, negative_seqs])
    all_labels = np.concatenate([np.ones(len(promoter_seqs)), 
                                 np.zeros(len(negative_seqs))])
    
    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"Positives (promoters): {(all_labels == 1).sum()}")
    print(f"Negatives (non-promoters): {(all_labels == 0).sum()}")
    print(f"Class balance: {(all_labels == 1).sum() / len(all_labels) * 100:.1f}%")
    
    # ========================================================================
    # STEP 4: TRAIN/VAL/TEST SPLITS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAIN/VAL/TEST SPLITS (80/10/10)")
    print("="*80)
    
    # First split: 80% train, 20% temp (for val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_sequences, all_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_labels
    )
    
    # Second split: split temp into 50/50 for val and test (10% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"\nTrain set: {len(X_train)} sequences")
    print(f"  Positives: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"  Negatives: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    
    print(f"\nValidation set: {len(X_val)} sequences")
    print(f"  Positives: {(y_val == 1).sum()} ({(y_val == 1).sum()/len(y_val)*100:.1f}%)")
    print(f"  Negatives: {(y_val == 0).sum()} ({(y_val == 0).sum()/len(y_val)*100:.1f}%)")
    
    print(f"\nTest set: {len(X_test)} sequences")
    print(f"  Positives: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Negatives: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    
    # ========================================================================
    # STEP 5: SAVE DATA SPLITS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: SAVING DATA SPLITS")
    print("="*80)
    
    # Save as numpy arrays
    npz_file = os.path.join(output_dir, 'data_splits.npz')
    np.savez(npz_file,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"\nSaved NumPy arrays to: {npz_file}")
    
    # Save as CSV files for easy inspection
    def save_split_to_csv(sequences, labels, filename):
        df = pd.DataFrame({
            'sequence': sequences,
            'label': labels.astype(int),
            'class': ['promoter' if l == 1 else 'non-promoter' for l in labels]
        })
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath}")
    
    save_split_to_csv(X_train, y_train, 'train_data.csv')
    save_split_to_csv(X_val, y_val, 'val_data.csv')
    save_split_to_csv(X_test, y_test, 'test_data.csv')
    
    # Save summary statistics
    summary = {
        'total_sequences': len(all_sequences),
        'promoters': int((all_labels == 1).sum()),
        'non_promoters': int((all_labels == 0).sum()),
        'sequence_length': seq_length,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'promoter_gc_mean': float(np.mean(promoter_gc_contents)),
        'promoter_gc_std': float(np.std(promoter_gc_contents)),
        'negative_gc_mean': float(np.mean(negative_gc_contents)),
        'negative_gc_std': float(np.std(negative_gc_contents)),
        'gc_ttest_pvalue': float(p_value)
    }
    
    summary_file = os.path.join(output_dir, 'data_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")
    
    # ========================================================================
    # STEP 6: CREATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sequence length distribution
    axes[0, 0].hist(df_clean['seq_length'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Sequence Length (bp)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Sequence Lengths', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(df_clean['seq_length'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df_clean["seq_length"].mean():.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. GC content distribution (promoters vs negatives)
    axes[0, 1].hist(promoter_gc_contents, bins=30, alpha=0.6, label='Promoters', 
                    edgecolor='black', color='blue')
    axes[0, 1].hist(negative_gc_contents, bins=30, alpha=0.6, label='Non-promoters', 
                    edgecolor='black', color='red')
    axes[0, 1].set_xlabel('GC Content', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('GC Content Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Confidence levels
    if '15)confidenceLevel' in df_clean.columns:
        conf_counts = df_clean['15)confidenceLevel'].value_counts()
        axes[1, 0].bar(range(len(conf_counts)), conf_counts.values, 
                       color=['#2ecc71', '#f39c12', '#e74c3c'][:len(conf_counts)],
                       edgecolor='black')
        axes[1, 0].set_xticks(range(len(conf_counts)))
        axes[1, 0].set_xticklabels(conf_counts.index, fontsize=10)
        axes[1, 0].set_xlabel('Confidence Level', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Promoters by Confidence Level', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 4. Class balance in splits
    split_data = {
        'Train': [int((y_train == 1).sum()), int((y_train == 0).sum())],
        'Val': [int((y_val == 1).sum()), int((y_val == 0).sum())],
        'Test': [int((y_test == 1).sum()), int((y_test == 0).sum())]
    }
    
    x = np.arange(len(split_data))
    width = 0.35
    
    promoter_counts = [split_data[split][0] for split in split_data]
    nonpromoter_counts = [split_data[split][1] for split in split_data]
    
    axes[1, 1].bar(x - width/2, promoter_counts, width, label='Promoters', 
                   color='#3498db', edgecolor='black')
    axes[1, 1].bar(x + width/2, nonpromoter_counts, width, label='Non-promoters', 
                   color='#e74c3c', edgecolor='black')
    axes[1, 1].set_xlabel('Split', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Class Balance Across Splits', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(split_data.keys())
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    viz_file = os.path.join(output_dir, 'data_exploration.png')
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {viz_file}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    
    print(f"\nOutput directory: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  1. promoters_clean.csv - Cleaned promoter data ({len(df_clean)} sequences)")
    print(f"  2. train_data.csv - Training set ({len(X_train)} sequences)")
    print(f"  3. val_data.csv - Validation set ({len(X_val)} sequences)")
    print(f"  4. test_data.csv - Test set ({len(X_test)} sequences)")
    print(f"  5. data_splits.npz - NumPy arrays for efficient loading")
    print(f"  6. data_summary.json - Summary statistics")
    print(f"  7. data_exploration.png - Visualizations")
    
    print(f"\nDataset statistics:")
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Promoters: {(all_labels == 1).sum()}")
    print(f"  Non-promoters: {(all_labels == 0).sum()}")
    print(f"  Sequence length: {seq_length}bp")
    print(f"  GC content (promoters): {np.mean(promoter_gc_contents):.3f} ± {np.std(promoter_gc_contents):.3f}")
    print(f"  GC content (negatives): {np.mean(negative_gc_contents):.3f} ± {np.std(negative_gc_contents):.3f}")
    
    print("\n" + "="*80)
    print("Ready for HMM training!")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocessing_pipeline.py <tableData.tsv>")
        print("\nExample:")
        print("  python preprocessing_pipeline.py ./tableData.tsv")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    
    if not os.path.exists(tsv_file):
        print(f"Error: File not found: {tsv_file}")
        sys.exit(1)
    
    main(tsv_file)