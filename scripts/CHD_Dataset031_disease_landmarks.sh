#!/bin/bash
# =============================================================================
#  CHD_Dataset031_disease_landmarks.sh
#  Dataset031_imageCHD_DiseaseLandmarks — 7 anatomy labels + unified
#  septal-defect label (id 8), derived from Dataset030 GT + diagnosis flags.
#
#  Regular nnU-Net pipeline, 3d_fullres, fold 0:
#     Phase 0   build Dataset031 from Dataset030 (chd_landmarks; source read-only)
#     Phase 1   plan_and_preprocess (3d_fullres only)
#     Phase 2   train DA5 baseline           → predict test set
#     Phase 3   train DA5 DiseaseLandmark     → predict test set
#
#  RESUME: each phase writes a .done marker; resubmission skips completed phases.
#          nnU-Net auto-resumes interrupted training from checkpoint_latest.pth.
#
#  Prereqs: git pull (this file + the chd_landmarks package), env below set.
#           The diagnosis xlsx ships in the repo ($REPO/imageCHD_dataset_info.xlsx).
# =============================================================================
#SBATCH --job-name=D031-landmarks
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D031-landmarks_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D031-landmarks_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
REPO="/scratch/users/sastocke/nnunet_CHD"
SOURCE_NAME="Dataset030_imageCHD_HU"
DATASET_ID=31
DATASET_NAME="Dataset031_imageCHD_DiseaseLandmarks"
PLANNER="nnUNetPlannerResEncM"
FULLRES="3d_fullres"
FOLD=0
METADATA="${REPO}/imageCHD_dataset_info.xlsx"      # shipped in repo

TRAINER_BASE="nnUNetTrainerDA5_200epochs"
TRAINER_DLM="nnUNetTrainerDA5DiseaseLandmark_200epochs"

# Inference: images are identical to the source test set (labels differ only by the added class)
IN_DIR="${nnUNet_raw}/${SOURCE_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset031_disease_landmarks"
mkdir -p "${CKPT_DIR}" "${PRED_BASE}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ─────────────────────────────────────────────
# Phase 0 — build Dataset031 (source Dataset030 READ-ONLY)
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} from ${SOURCE_NAME}"
  python -m chd_landmarks.cli build-dataset \
      --source-dataset "${nnUNet_raw}/${SOURCE_NAME}" \
      --target-dataset-id "${DATASET_ID}" \
      --target-dataset-name imageCHD_DiseaseLandmarks \
      --metadata "${METADATA}" \
      --out-root "${nnUNet_raw}" --overwrite
  touch "${CKPT_DIR}/00_build.done"
else
  echo "[Phase 0] build already done — skipping"
fi

# ─────────────────────────────────────────────
# Phase 1 — plan & preprocess (3d_fullres only; Dataset031 ONLY)
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  echo "[Phase 1] plan_and_preprocess -d ${DATASET_ID}"
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" \
      -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
else
  echo "[Phase 1] preprocess already done — skipping"
fi

# ─────────────────────────────────────────────
# Phase 2 — train DA5 baseline → predict
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/02_train_base.done" ]; then
  echo "[Phase 2] train ${TRAINER_BASE}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER_BASE}"
  touch "${CKPT_DIR}/02_train_base.done"
fi
if [ ! -f "${CKPT_DIR}/02_pred_base.done" ]; then
  echo "[Phase 2] predict ${TRAINER_BASE}"
  nnUNetv2_predict -i "${IN_DIR}" -o "${PRED_BASE}/DA5_baseline" \
      -d "${DATASET_ID}" -c "${FULLRES}" -f "${FOLD}" -tr "${TRAINER_BASE}"
  touch "${CKPT_DIR}/02_pred_base.done"
fi

# ─────────────────────────────────────────────
# Phase 3 — train DA5 DiseaseLandmark → predict
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/03_train_dlm.done" ]; then
  echo "[Phase 3] train ${TRAINER_DLM}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER_DLM}"
  touch "${CKPT_DIR}/03_train_dlm.done"
fi
if [ ! -f "${CKPT_DIR}/03_pred_dlm.done" ]; then
  echo "[Phase 3] predict ${TRAINER_DLM}"
  nnUNetv2_predict -i "${IN_DIR}" -o "${PRED_BASE}/DA5DiseaseLandmark" \
      -d "${DATASET_ID}" -c "${FULLRES}" -f "${FOLD}" -tr "${TRAINER_DLM}"
  touch "${CKPT_DIR}/03_pred_dlm.done"
fi

echo "=============================================================="
echo "DONE. Predictions:"
echo "  ${PRED_BASE}/DA5_baseline"
echo "  ${PRED_BASE}/DA5DiseaseLandmark"
echo "Copy these into the Alison SegmentationDetailStandard predictions"
echo "folder as new methods to evaluate them (septal label = 8)."
echo "=============================================================="
