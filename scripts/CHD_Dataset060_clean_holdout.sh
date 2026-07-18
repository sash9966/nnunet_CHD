#!/bin/bash
# =============================================================================
#  CHD_Dataset060_clean_holdout.sh
#  Dataset060: CLEAN-TRAIN / DIRTY-HOLDOUT partition of ImageCHD.
#    TEST  = all missing-myocardium cases (+ diagnosis-stratified topup to ~10%)
#    TRAIN = all clean (myo-present) cases, with septal_defect label (v2, ASD-fixed)
#  New partition (NOT the Dataset030/050/051 split) — for a high-quality clinical
#  model + a reported ImageCHD holdout clinicians can re-segment.
#
#  Runs the septal-focus ablation on fold 0, 3d_fullres:
#    DA5 baseline / +oversample / +Tversky / +both  -> predict Dataset060 imagesTs.
#  RESUME: .done markers; resubmit to continue.
# =============================================================================
#SBATCH --job-name=D060-clean-abl
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D060-clean-abl_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D060-clean-abl_%j.err

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
DATASET_ID=60
DATASET_NAME="Dataset060_imageCHD_CleanHoldout"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; FOLD=0
METADATA="${REPO}/imageCHD_dataset_info.xlsx"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"     # Dataset060's OWN held-out set
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/clean_holdout"
mkdir -p "${CKPT_DIR}" "${PRED_BASE}" /scratch/users/sastocke/nnunet_CHD/logs

ARMS=( "nnUNetTrainerDA5_200epochs"
       "nnUNetTrainerDA5SeptalOversample_200epochs"
       "nnUNetTrainerDA5SeptalTversky_200epochs"
       "nnUNetTrainerDA5SeptalOversampleTversky_200epochs" )

# ---- Phase 0: build Dataset060 (clean train, missing-myo -> holdout test) -----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  python tools/build_dataset060_clean_holdout.py \
      --source-dataset "${nnUNet_raw}/${SOURCE_NAME}" \
      --target-id "${DATASET_ID}" --target-name imageCHD_CleanHoldout \
      --metadata "${METADATA}" --out-root "${nnUNet_raw}" --test-frac 0.10 --overwrite
  touch "${CKPT_DIR}/00_build.done"
fi

# ---- Phase 1: plan & preprocess (fold split auto-generated from clean labelsTr)
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
fi

# ---- Phases 2..: ablation arms -> train + predict on the clean holdout --------
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

echo "DONE. Dataset060 clean-holdout ablation. Predictions in ${PRED_BASE}/<arm>/."
echo "Holdout (incl. missing-myo) is Dataset060/labelsTs — report + clinician re-seg."
