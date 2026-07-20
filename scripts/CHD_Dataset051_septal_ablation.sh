#!/bin/bash
# =============================================================================
#  CHD_Dataset051_septal_ablation.sh
#  Rebuild Dataset051 (v2 septal label, ASD-merge FIX, missing-myo excluded from
#  TRAINING) and run the septal-focus ABLATION on fold 0, 3d_fullres:
#     Arm 0  DA5 baseline                       (label 8 present, no focus) — reference
#     Arm 1  DA5 + septal oversampling          (sampling lever)
#     Arm 2  DA5 + septal FN-weighted Tversky   (loss lever)
#     Arm 3  DA5 + oversampling + Tversky        (combined)
#  Same fold-0 split as Dataset030 (filtered to the included, myo-present cases).
#  Predictions -> compare with Step 10 of the Alison dice_analysis.ipynb.
#
#  RESUME: .done markers per phase; resubmit to continue (5 arms won't all fit 48h).
# =============================================================================
#SBATCH --job-name=D051-septal-abl
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D051-septal-abl_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D051-septal-abl_%j.err

set -euo pipefail
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

REPO="/scratch/users/sastocke/nnunet_CHD"; cd "$REPO"
SOURCE_NAME="Dataset030_imageCHD_HU"
DATASET_ID=51
DATASET_NAME="Dataset051_imageCHD_DiseaseLandmarksV2"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; FOLD=0
METADATA="${REPO}/imageCHD_dataset_info.xlsx"
IN_DIR="${nnUNet_raw}/${SOURCE_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/septal_ablation"
mkdir -p "${CKPT_DIR}" "${PRED_BASE}" /scratch/users/sastocke/nnunet_CHD/logs

ARMS=( "nnUNetTrainerDA5_200epochs"
       "nnUNetTrainerDA5SeptalOversample_200epochs"
       "nnUNetTrainerDA5SeptalTversky_200epochs"
       "nnUNetTrainerDA5SeptalTverskyV2_200epochs"
       "nnUNetTrainerDA5SeptalOversampleTverskyV2_200epochs"
       "nnUNetTrainerDA5SeptalOversampleTversky_200epochs" )

# ---- Phase 0: rebuild Dataset051 (ASD-fixed v2 label, exclude missing-myo from training)
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  python -m chd_landmarks.cli build-dataset \
      --source-dataset "${nnUNet_raw}/${SOURCE_NAME}" \
      --target-dataset-id "${DATASET_ID}" --target-dataset-name imageCHD_DiseaseLandmarksV2 \
      --metadata "${METADATA}" --out-root "${nnUNet_raw}" --overwrite --require-myo
  touch "${CKPT_DIR}/00_build.done"
fi

# ---- Phase 1: plan & preprocess (Dataset051 only)
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
fi

# ---- Phase 1b: split = Dataset030 split filtered to Dataset051's (myo-present) cases
if [ ! -f "${CKPT_DIR}/01b_splits.done" ]; then
  python tools/filter_splits_to_dataset.py \
      --source-splits "${nnUNet_preprocessed}/${SOURCE_NAME}/splits_final.json" \
      --target-dataset "${nnUNet_raw}/${DATASET_NAME}" \
      --out "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
  touch "${CKPT_DIR}/01b_splits.done"
fi

# ---- Phases 2..: each ablation arm -> train + predict
for TR in "${ARMS[@]}"; do
  if [ ! -f "${CKPT_DIR}/train_${TR}.done" ]; then
    echo "[train] ${TR}"
    nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TR}" -p "${PLANS}"
    touch "${CKPT_DIR}/train_${TR}.done"
  fi
  if [ ! -f "${CKPT_DIR}/pred_${TR}.done" ]; then
    echo "[predict] ${TR}"
    nnUNetv2_predict -i "${IN_DIR}" -o "${PRED_BASE}/${TR}" \
        -d "${DATASET_ID}" -c "${FULLRES}" -f "${FOLD}" -tr "${TR}" -p "${PLANS}"
    touch "${CKPT_DIR}/pred_${TR}.done"
  fi
done

echo "DONE. Ablation predictions in ${PRED_BASE}/<arm>/. Copy into the Alison"
echo "predictions folder and compare with Step 10 (septal-defect metrics)."
