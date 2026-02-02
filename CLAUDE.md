# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **nnU-Net V2** (v2.6.3), a self-configuring deep learning framework for medical image segmentation. The branch `CHD_diseaseVector` is used for congenital heart disease (CHD) segmentation work. nnU-Net automatically analyzes datasets, configures U-Net architectures, and handles preprocessing/training/inference without manual tuning.

## Environment Setup

```bash
# Required environment variables (paths to data directories)
export nnUNet_raw=/path/to/raw_data
export nnUNet_preprocessed=/path/to/preprocessed_data
export nnUNet_results=/path/to/trained_models

# Optional: data augmentation process count (default: 12)
export nnUNet_n_proc_DA=12
```

## Install & Build

```bash
pip install -e .  # editable install for development
# Requires Python 3.10+, PyTorch >=2.1.2,<2.9.0 (2.9.0 has 3D conv + AMP regression)
```

## Standard Workflow Commands

```bash
# 1. Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# 2. Train (repeat for folds 0-4, configs: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
nnUNetv2_train DATASET_ID CONFIG FOLD [--npz] [-device cuda]

# 3. Find best configuration
nnUNetv2_find_best_configuration DATASET_ID

# 4. Predict
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID
```

## Testing

```bash
# Integration test (downloads example data, validates Dice > 0.99)
python nnunetv2/tests/integration_tests/run_nnunet_inference.py
```

CI runs on Ubuntu and macOS with PyTorch 2.4.0 via `.github/workflows/run_tests_nnunet.yml`.

## Architecture

The pipeline has three stages: **Planning → Training → Inference**.

### Key Modules (`nnunetv2/`)

- **`experiment_planning/`** — Dataset fingerprint extraction and automatic architecture design. Produces `nnUNetPlans.json` which drives all downstream configuration (network topology, batch/patch size, spacing, augmentation params).
- **`preprocessing/`** — Intensity normalization (`CTNormalization`, `ZScoreNormalization`) and voxel spacing resampling. Outputs cached `.npz`/`.npy` files.
- **`training/nnUNetTrainer/`** — Main trainer class (`nnUNetTrainer.py`, ~72KB). Orchestrates data loading, augmentation, loss (Dice+CE with deep supervision), optimization (SGD + PolyLR), and 5-fold cross-validation. Subclass variants in `variants/`.
- **`training/dataloading/`** — `nnUNetDataset` (lazy-loading with caching) and `nnUNetDataLoader` (oversamples foreground).
- **`training/data_augmentation/`** — Uses `batchgeneratorsv2` transforms. Custom spatial, intensity, and noise augmentations.
- **`inference/`** — `nnUNetPredictor` handles sliding window prediction, test-time augmentation (mirroring), and multi-model ensembling.
- **`imageio/`** — Abstraction layer for NIfTI (nibabel), SimpleITK, TIFF, and natural image formats.
- **`utilities/get_network_from_plans.py`** — Instantiates networks from plans using `dynamic-network-architectures` library.

### Design Patterns

- **Plans-driven**: All configuration flows from `nnUNetPlans.json`. Modifying plans changes the entire pipeline behavior.
- **Extensible trainer**: Subclass `nnUNetTrainer` to customize loss, augmentation, architecture, or training loop. Trainer class is selected by string name at runtime.
- **Three data paths**: `nnUNet_raw` (input), `nnUNet_preprocessed` (cached), `nnUNet_results` (outputs). Configured in `nnunetv2/paths.py`.
- **DDP support**: Multi-GPU via `torch.distributed` with automatic rank detection.

### Entry Points

All CLI commands are defined in `pyproject.toml` under `[project.scripts]`. Key mapping:
- `nnUNetv2_train` → `nnunetv2/run/run_training.py:run_training_entry`
- `nnUNetv2_predict` → `nnunetv2/inference/predict_from_raw_data.py:predict_entry_point`
- `nnUNetv2_plan_and_preprocess` → `nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py`

## Key Dependencies

- `dynamic-network-architectures` — Builds U-Net architectures from plans
- `batchgeneratorsv2` — Data augmentation framework
- `acvl-utils` — Shared utilities
- `SimpleITK`, `nibabel` — Medical image I/O
