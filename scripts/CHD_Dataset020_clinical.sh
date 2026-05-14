#!/bin/bash
#SBATCH --job-name=CHD_D020_clinical
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#
# CLINICAL DEPLOYMENT MODEL — Dataset020 (Fanwei data + ImageCHD combined)
#
# Approach: 5-fold cross-validation ensemble (preferred over fold "all" for deployment).
#
# Why 5-fold ensemble over fold "all":
#   - Each model sees 80% of data; ensemble average of 5 softmax outputs gains
#     +1-3% Dice compared to any single model, including fold "all".
#   - True out-of-sample validation per fold — training metrics are trustworthy.
#   - Deployment is identical: nnUNetv2_predict -f 0 1 2 3 4 handles ensembling
#     automatically; no extra code needed.
#   - If you later want a fold-all single model, just add:
#       nnUNetv2_train 20 3d_fullres all -tr nnUNetTrainerDA5_200epochs -p ${PLANS} --npz
#
# No disease conditioning (Dataset020 has no diagnosis labels).
# No topology loss — keeping this clean for the baseline clinical model.
# Two experiments:
#   Experiment 1: DA5 fullres baseline
#   Experiment 2: Cascade (DA5 lowres → CascadeFullresBaseline)
#
# NOTE: 5 folds x 200 epochs x (fullres + lowres + cascade) may approach or
# exceed 48h on a single GPU. If needed, split:
#   Job A: Phase 0 + Phase 1 (fullres) + Phase 2 (lowres)
#   Job B: Phase 3 (symlinks) + Phase 4 (cascade) + Phase 5 (inference)

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
DATASET_ID=20
DATASET_NAME="Dataset020FanweiDataandImageCHD_HU"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"

LR_TRAINER="nnUNetTrainerDA5_200epochs"
CASCADE_TRAINER="nnUNetTrainerDA5CascadeFullresBaseline_200epochs"

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
# Phase 1: DA5 fullres baseline — all 5 folds
# -------------------------
echo "================================================================"
echo "Phase 1: Fullres DA5 Baseline — 5 folds"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    echo "--- Fullres DA5: fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${FULLRES} ${FOLD} \
        -tr ${LR_TRAINER} \
        -p ${PLANS} \
        --npz
done

# -------------------------
# Phase 2: Lowres training for cascade — all 5 folds
# -------------------------
echo "================================================================"
echo "Phase 2: Lowres Training (for cascade)"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    echo "--- LowRes DA5: fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${LOWRES} ${FOLD} \
        -tr ${LR_TRAINER} \
        -p ${PLANS} \
        --npz
done

# -------------------------
# Phase 3: Symlink lowres predicted_next_stage into cascade trainer directory
# -------------------------
echo "================================================================"
echo "Phase 3: Symlink Lowres Predictions"
echo "================================================================"
python "${REPO}/scripts/setup_cascade_predictions.py" \
    --lowres_trainer  "${LR_TRAINER}" \
    --cascade_trainer "${CASCADE_TRAINER}" \
    --dataset         "${DATASET_NAME}" \
    --plans           "${PLANS}" \
    --folds           0 1 2 3 4

# -------------------------
# Phase 4: Cascade fullres training — all 5 folds
# -------------------------
echo "================================================================"
echo "Phase 4: Cascade Fullres Training"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    echo "--- Cascade fullres: fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${CASCADE} ${FOLD} \
        -tr ${CASCADE_TRAINER} \
        -p ${PLANS} \
        --npz
done

# -------------------------
# Phase 5: Inference on clinical test cases (5-fold ensemble)
# -------------------------
echo "================================================================"
echo "Phase 5: Inference on Clinical Test Set (5-fold ensemble)"
echo "================================================================"
mkdir -p "${PRED_BASE}"

# --- Experiment 1: Fullres baseline ---
FULLRES_PRED="${PRED_BASE}/DA5_fullres_ensemble"
mkdir -p "${FULLRES_PRED}"
echo "--- Inference: DA5 fullres ensemble ---"
nnUNetv2_predict \
    -i "${IN_DIR}" \
    -o "${FULLRES_PRED}" \
    -d ${DATASET_ID} \
    -c ${FULLRES} \
    -f 0 1 2 3 4 \
    -tr ${LR_TRAINER} \
    -p ${PLANS}

# --- Experiment 2: Cascade (lowres predict → cascade predict) ---
LR_PRED="${PRED_BASE}/${LR_TRAINER}_lowres_ensemble"
CASCADE_PRED="${PRED_BASE}/${CASCADE_TRAINER}_ensemble"
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

echo "================================================================"
echo "Dataset020 clinical pipeline complete."
echo "Fullres predictions : ${FULLRES_PRED}"
echo "Cascade predictions : ${CASCADE_PRED}"
echo "================================================================"
echo ""
echo "Next step for Slicer deployment:"
echo "  nnUNetv2_find_best_configuration ${DATASET_ID} -c ${FULLRES} ${CASCADE}"
echo "  Then export the winning model folder as the Slicer extension weights."
