#!/bin/bash
#SBATCH --job-name=CHD_D001_cascade
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#
# Dataset001_all_imageCHD — Cascade ablation at 200 epochs
# Updated from CHD_Cascade_allFolds.sh (was 100 epochs) to 200 epochs throughout.
#
# 4 cascade experiments (lowres -> fullres):
#   DA5                -> CascadeFullresBaseline   (no conditioning, no topology)
#   DA5CascadeFiLM     -> CascadeFullresFiLM        (disease conditioning only)
#   DA5CascadeTopo     -> CascadeFullresTopo         (topology loss only)
#   DA5CascadeFiLMTopo -> CascadeFullresFiLMTopo     (conditioning + topology)
#
# FiLM trainers require disease_map.json in $nnUNet_preprocessed/Dataset001_all_imageCHD/
# Topology loss applies soft-clDice on AO (label 6) and PA (label 7).
#
# NOTE: 4 trainers x 5 folds x 2 stages = 40 training runs x 200 epochs.
# This will very likely exceed 96h on a single GPU. Recommended split:
#   Job A: Phase 0 (preprocess) + Phase 1 (all lowres training)
#   Job B: Phase 2 (symlinks) + Phase 3 (all cascade fullres) + Phase 4 (inference)

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
DATASET_ID=1
DATASET_NAME="Dataset001_all_imageCHD"
PLANS="nnUNetResEncUNetMPlans"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"

# Parallel arrays: index i pairs lowres trainer with its cascade fullres counterpart
LOWRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"
    "nnUNetTrainerDA5CascadeFiLM_200epochs"
    "nnUNetTrainerDA5CascadeTopo_200epochs"
    "nnUNetTrainerDA5CascadeFiLMTopo_200epochs"
)
CASCADE_TRAINERS=(
    "nnUNetTrainerDA5CascadeFullresBaseline_200epochs"
    "nnUNetTrainerDA5CascadeFullresFiLM_200epochs"
    "nnUNetTrainerDA5CascadeFullresTopo_200epochs"
    "nnUNetTrainerDA5CascadeFullresFiLMTopo_200epochs"
)

# -------------------------
# Phase 0: Preprocess lowres and cascade configs
# (Plans assumed to exist from previous runs; only preprocesses if not already done)
# -------------------------
echo "================================================================"
echo "Phase 0: Preprocess 3d_lowres + 3d_cascade_fullres"
echo "================================================================"
nnUNetv2_preprocess \
    -d ${DATASET_ID} \
    -plans_name ${PLANS} \
    -c ${LOWRES} ${CASCADE} \
    -n 4 2

# -------------------------
# Phase 1: Train all lowres trainers x all folds
# -------------------------
echo "================================================================"
echo "Phase 1: Lowres Training (all 4 trainers x 5 folds)"
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
# Phase 2: Symlink lowres predicted_next_stage into cascade trainer directories
# -------------------------
echo "================================================================"
echo "Phase 2: Symlink Lowres Predictions"
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
# Phase 3: Train all cascade fullres trainers x all folds
# -------------------------
echo "================================================================"
echo "Phase 3: Cascade Fullres Training (all 4 trainers x 5 folds)"
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
# Phase 4: Test-set inference (fold 0 only; use -f 0 1 2 3 4 for full ensemble)
# -------------------------
echo "================================================================"
echo "Phase 4: Inference on Test Set (fold 0)"
echo "================================================================"
mkdir -p "${PRED_BASE}"

LR_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}"

for i in "${!LOWRES_TRAINERS[@]}"; do
    LR_TRAINER="${LOWRES_TRAINERS[$i]}"
    CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"
    LR_PRED="${PRED_BASE}/${LR_TRAINER}/fold_0"
    CASCADE_PRED="${PRED_BASE}/${CASCADE_TRAINER}/fold_0"
    LR_MODEL="${LR_MODEL_BASE}/${LR_TRAINER}__${PLANS}__${LOWRES}"
    CASCADE_MODEL="${LR_MODEL_BASE}/${CASCADE_TRAINER}__${PLANS}__${CASCADE}"
    mkdir -p "${LR_PRED}" "${CASCADE_PRED}"

    echo "--- Inference lowres: ${LR_TRAINER} ---"
    if [[ "${LR_TRAINER}" == *"FiLM"* ]]; then
        python -m nnunetv2.inference.predict_disease_conditioned \
            -i "${IN_DIR}" \
            -o "${LR_PRED}" \
            -m "${LR_MODEL}" \
            -f 0
    else
        nnUNetv2_predict \
            -i "${IN_DIR}" \
            -o "${LR_PRED}" \
            -d ${DATASET_ID} \
            -c ${LOWRES} \
            -f 0 \
            -tr ${LR_TRAINER} \
            -p ${PLANS}
    fi

    echo "--- Inference cascade: ${CASCADE_TRAINER} ---"
    if [[ "${CASCADE_TRAINER}" == *"FiLM"* ]]; then
        python -m nnunetv2.inference.predict_disease_conditioned \
            -i "${IN_DIR}" \
            -o "${CASCADE_PRED}" \
            -m "${CASCADE_MODEL}" \
            -f 0 \
            --prev_stage_predictions "${LR_PRED}"
    else
        nnUNetv2_predict \
            -i "${IN_DIR}" \
            -o "${CASCADE_PRED}" \
            -d ${DATASET_ID} \
            -c ${CASCADE} \
            -f 0 \
            -tr ${CASCADE_TRAINER} \
            -p ${PLANS} \
            -prev_stage_predictions "${LR_PRED}"
    fi
done

echo "================================================================"
echo "Dataset001 cascade pipeline complete."
echo "Results in: ${PRED_BASE}"
echo "================================================================"
