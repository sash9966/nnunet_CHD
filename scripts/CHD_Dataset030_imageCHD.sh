#!/bin/bash
#SBATCH --job-name=CHD_D030_imageCHD
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
# NOTE: This script runs 3 experiments (fullres DA5 + 2 cascade pipelines) across 5 folds.
# Total training runs: 5 (fullres) + 10 (lowres) + 10 (cascade) = 25 x 200 epochs.
# Wall time may exceed 96h on a single GPU. If so, split into two jobs:
#   Job A: Phase 0 + Phase 1 (fullres) + Phase 2 (lowres)
#   Job B: Phase 3 (symlinks) + Phase 4 (cascade fullres) + Phase 5 (inference)

set -euo pipefail

# -------------------------
# 1. Environment
# -------------------------
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONUNBUFFERED=1

# -------------------------
# 2. Config
# -------------------------
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"

# Experiments:
#   Experiment 1 — DA5 fullres baseline (standalone, no cascade)
#   Experiment 2 — Cascade baseline:  DA5_200epochs (lowres) → CascadeFullresBaseline_200epochs
#   Experiment 3 — Cascade topology:  DA5CascadeTopo_200epochs (lowres) → CascadeFullresTopo_200epochs
#
# Topology loss (soft-clDice) is applied to AO/PA classes.
# Label names must match "AO" and "PA" in Dataset030's dataset.json.

# Parallel arrays: lowres trainer -> cascade fullres trainer
LOWRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"
    "nnUNetTrainerDA5CascadeTopo_200epochs"
)
CASCADE_TRAINERS=(
    "nnUNetTrainerDA5CascadeFullresBaseline_200epochs"
    "nnUNetTrainerDA5CascadeFullresTopo_200epochs"
)

# -------------------------
# Phase 0: Plan and preprocess all three configs
# -------------------------
echo "================================================================"
echo "Phase 0: Plan and Preprocess (fullres + lowres + cascade)"
echo "================================================================"
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    -pl ${PLANNER} \
    -c ${FULLRES} ${LOWRES} ${CASCADE} \
    --verify_dataset_integrity

# -------------------------
# Phase 1: DA5 fullres baseline (Experiment 1) — all 5 folds
# -------------------------
echo "================================================================"
echo "Phase 1: Fullres DA5 Baseline — 5 folds"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    echo "--- Fullres DA5: fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${FULLRES} ${FOLD} \
        -tr nnUNetTrainerDA5_200epochs \
        -p ${PLANS} \
        --npz
done

# -------------------------
# Phase 2: Lowres training for both cascade experiments — all folds
# -------------------------
echo "================================================================"
echo "Phase 2: Lowres Training (Experiments 2 and 3)"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        echo "--- LowRes: ${LR_TRAINER} | fold ${FOLD} ---"
        nnUNetv2_train ${DATASET_ID} ${LOWRES} ${FOLD} \
            -tr ${LR_TRAINER} \
            -p ${PLANS} \
            --npz
    done
done

# -------------------------
# Phase 3: Symlink lowres predicted_next_stage into cascade trainer directories
# -------------------------
echo "================================================================"
echo "Phase 3: Symlink Lowres Predictions"
echo "================================================================"
for i in "${!LOWRES_TRAINERS[@]}"; do
    echo "--- Linking: ${LOWRES_TRAINERS[$i]} -> ${CASCADE_TRAINERS[$i]} ---"
    python "${REPO}/scripts/setup_cascade_predictions.py" \
        --lowres_trainer  "${LOWRES_TRAINERS[$i]}" \
        --cascade_trainer "${CASCADE_TRAINERS[$i]}" \
        --dataset         "${DATASET_NAME}" \
        --plans           "${PLANS}" \
        --folds           0 1 2 3 4
done

# -------------------------
# Phase 4: Cascade fullres training — all folds
# -------------------------
echo "================================================================"
echo "Phase 4: Cascade Fullres Training (Experiments 2 and 3)"
echo "================================================================"
for CASCADE_TRAINER in "${CASCADE_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        echo "--- Cascade: ${CASCADE_TRAINER} | fold ${FOLD} ---"
        nnUNetv2_train ${DATASET_ID} ${CASCADE} ${FOLD} \
            -tr ${CASCADE_TRAINER} \
            -p ${PLANS} \
            --npz
    done
done

# -------------------------
# Phase 5: Test-set inference (5-fold ensemble for all experiments)
# -------------------------
echo "================================================================"
echo "Phase 5: Inference on Test Set (ensemble all folds)"
echo "================================================================"
mkdir -p "${PRED_BASE}"

# --- Experiment 1: DA5 fullres baseline ---
FULLRES_PRED="${PRED_BASE}/DA5_fullres"
mkdir -p "${FULLRES_PRED}"
echo "--- Inference: DA5 fullres baseline ---"
nnUNetv2_predict \
    -i "${IN_DIR}" \
    -o "${FULLRES_PRED}" \
    -d ${DATASET_ID} \
    -c ${FULLRES} \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainerDA5_200epochs \
    -p ${PLANS}

# --- Experiments 2 and 3: Cascade (lowres predict then cascade predict) ---
for i in "${!LOWRES_TRAINERS[@]}"; do
    LR_TRAINER="${LOWRES_TRAINERS[$i]}"
    CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"

    LR_PRED="${PRED_BASE}/${LR_TRAINER}/ensemble"
    CASCADE_PRED="${PRED_BASE}/${CASCADE_TRAINER}/ensemble"
    mkdir -p "${LR_PRED}" "${CASCADE_PRED}"

    echo "--- Inference lowres: ${LR_TRAINER} ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" \
        -o "${LR_PRED}" \
        -d ${DATASET_ID} \
        -c ${LOWRES} \
        -f 0 1 2 3 4 \
        -tr ${LR_TRAINER} \
        -p ${PLANS}

    echo "--- Inference cascade: ${CASCADE_TRAINER} ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" \
        -o "${CASCADE_PRED}" \
        -d ${DATASET_ID} \
        -c ${CASCADE} \
        -f 0 1 2 3 4 \
        -tr ${CASCADE_TRAINER} \
        -p ${PLANS} \
        -prev_stage_predictions "${LR_PRED}"
done

echo "================================================================"
echo "Dataset030 pipeline complete."
echo "Results in: ${PRED_BASE}"
echo "================================================================"
