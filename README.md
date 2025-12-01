HMM Data Structures Final Project: Promoter Detection in E. coli

This project explores whether a discrete Hidden Markov Model (HMM) can accurately distinguish E. coli σ⁷⁰ promoters from non-promoter sequences, and how the number of hidden states (K) changes the model’s performance. The overall goal is to understand how HMM complexity affects what the model actually learns about promoter structure.

The repository includes the full preprocessing pipeline, all HMM training scripts, hyperparameter sweeps, ablations, and analysis/visualization code.

Project Overview: 

Bacterial promoters follow a loose structure (−35 motif → spacer → −10 motif → TSS), but there’s enough biological variability that simple consensus matching doesn’t work well. Hidden Markov Models are a good match because they can learn ordered but flexible subregions of the promoter without needing position-specific labels.

In this project, I:

  Built a 7,928-sequence labeled dataset (3,964 σ70 promoters + 3,964 GC-matched negatives)
  Preprocessed, cleaned, and encoded the data
  Implemented and trained discrete HMMs with varying hidden-state counts:
  K = 2, 3, 4, 6, 8, 10
  Compared against a first-order Markov baseline
  Added label-shuffle ablations and training-fraction experiments
  Analyzed hidden state behavior using Viterbi decoding
  Evaluated performance with AUROC, AUPRC, Accuracy, F1, Sensitivity, and Specificity
The best model used K = 10 hidden states, achieving ~0.83 AUROC and ~0.84 AUPRC on the held-out test set.

Dataset Details

Source: RegulonDB v11 – dataset RDBECOLIDLF00011
Organism: E. coli K-12 MG1655
Promoters: 3,964 experimentally supported σ⁷⁰ promoters
Length: Every promoter is already provided as an 81 bp window centered on the TSS.

Negatives:
Generated 3,964 synthetic non-promoter sequences with GC content matched per promoter to avoid trivial classification shortcuts.

Final dataset:
      7,928 sequences
      81 bp each
      50% promoters / 50% negatives
      Splits: 80% train / 10% validation / 10% test
      Encoded into integers: A=0, C=1, G=2, T=3
