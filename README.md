# DCT-Enhanced Lithography Hotspot Detection

Controlled experiments evaluating DCT-based frequency-domain representations for lithography hotspot detection on the ICCAD-12 benchmark.

## Overview

This repository contains the controlled experimental campaign conducted to investigate whether DCT-based input representations improve lithography hotspot detection compared with the spatial-domain baseline.

The experiments were designed to address reviewer feedback concerning:

- consistent validation protocols,
- DCT coefficient-count ablation,
- zig-zag ordering ablation,
- block-size screening,
- multi-seed evaluation and statistical testing,
- comparison with prior frequency-domain work,
- limitations and future work.

## Experimental Protocol

All experiments use the same data protocol:

```text
Official training partition
        |
        +---- 80% training
        |
        +---- 20% validation
               
Official test partition
        |
        +---- untouched

The train/validation split is stratified and fixed using split seed 42.

Training seeds used for the multi-seed experiments are:
- 42
- 123
- 2024
The official ICCAD-12 test partition is kept untouched throughout this experimental campaign.
Models
The spatial-domain baseline uses the custom CNN architecture derived from the reference lithography hotspot detection work.

The network contains 12,873 parameters and uses:
- three convolutional layers per basic block,
- ELU activations,
- batch normalization,
- max pooling,
- dropout,
- a 10-unit dense layer,
- a binary sigmoid output.
DCT Representations
The repository evaluates several frequency-domain configurations.
Full-image DCT
A 2-D DCT is applied to the complete 224 × 224 image.

The resulting coefficients are transformed using magnitude, log1p, and per-image min-max normalization before being replicated across three channels.
Truncated DCT
A subset of DCT coefficients is retained using a standard diagonal zig-zag ordering.

Evaluated coefficient counts:
- 64
- 256
- 1,024
- 4,096
- 16,384
The retained coefficients remain at their original 2-D frequency-domain positions; unselected coefficients are zeroed.
Zig-zag Ordering
The standard diagonal zig-zag ordering is compared against row-major coefficient selection at a target of 4,096 coefficients.
Block DCT
Initial block-size screening was performed using:
- 16 × 16 blocks
- 56 × 56 blocks
These experiments were limited to the B1 benchmark because of computational cost.

For block configurations, the requested coefficient target and actual retained coefficient count can differ when the number of blocks does not divide the target evenly.
Benchmarks
Experiments are conducted on ICCAD-12 benchmarks:
- B1
- B2
- B3
- B4
- B5
The dataset is highly imbalanced, so balanced accuracy is used as the primary evaluation metric.
Repository Structure
dct_hotspot_experiments/
├── configs/
├── evaluation/
├── models/
├── preprocessing/
├── results/
├── splits/
├── training/
├── README.md
└── .gitignore

configs/
Experimental configuration files.
evaluation/
Evaluation and metric computation scripts.
models/
CNN model definitions.
preprocessing/
Spatial, full-image DCT, block-DCT, zig-zag, and row-major preprocessing implementations.
results/
Experiment histories and the consolidated metrics.csv file.
splits/
Fixed stratified train/validation splits generated with split seed 42.
training/
Training and experiment execution scripts.
Reproducibility
The controlled campaign fixes the train/validation partition across experiments and records the training seed for multi-seed runs.

The repository does not contain the original ICCAD-12 dataset. The dataset is expected to be available separately, with its root path specified in the configuration.

Generated model checkpoints and virtual environments are excluded from version control.
Scope
This repository contains the controlled experimental campaign used for the revised study. Earlier exploratory experiments are maintained separately and are not duplicated here.
License
This repository is intended for academic and research use.