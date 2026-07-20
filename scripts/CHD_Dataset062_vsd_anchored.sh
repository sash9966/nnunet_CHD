#!/bin/bash
# =============================================================================
#  CHD_Dataset062_vsd_anchored.sh
#  Dataset062 — septal_defect via the v3 VSD-ANCHORED derivation (the "purest").
#    TRAIN = all clean (myo-present) cases; septal_defect (id 8) derived reliably.
#    TEST  = missing-myo cases (+stratified topup ~10%); septal ALSO derived
#            (degraded, for visual inspection).
#  No myo hole-filling; ImageCHD as-is. NEW partition (not 030/050/051).
#
#  Phase 0 builds the dataset — you can stop after it and inspect labelsTr/labelsTs
#  before training (the .done markers let you resubmit to continue).
# =============================================================================
#SBATCH --job-name=D062-vsd-anch
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D062-vsd-anch_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D062-vsd-anch_%j.err

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
DATASET_ID=62
DATASET_NAME="Dataset062_imageCHD_VSDanchored"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; FOLD=0
METADATA="${REPO}/imageCHD_dataset_info.xlsx"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/vsd_anchored"
mkdir -p "${CKPT_DIR}" "${PRED_BASE}" /scratch/users/sastocke/nnunet_CHD/logs

ARMS=( "nnUNetTrainerDA5_200epochs"
       "nnUNetTrainerDA5SeptalOversample_200epochs"
       "nnUNetTrainerDA5SeptalTversky_200epochs"
       "nnUNetTrainerDA5SeptalTverskyV2_200epochs"
       "nnUNetTrainerDA5SeptalOversampleTverskyV2_200epochs"
       "nnUNetTrainerDA5SeptalOversampleTversky_200epochs" )

# ---- Phase 0: build Dataset062 (v3 VSD-anchored septal derivation) ------------
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  python tools/build_dataset062_vsd_anchored.py \
      --source-dataset "${nnUNet_raw}/${SOURCE_NAME}" \
      --target-id "${DATASET_ID}" --target-name imageCHD_VSDanchored \
      --metadata "${METADATA}" --out-root "${nnUNet_raw}" --test-frac 0.10 --overwrite
  touch "${CKPT_DIR}/00_build.done"
  echo ">>> Dataset062 built. Inspect labelsTr/ + labelsTs/ before training if you want."
fi

# ---- Phase 1: plan & preprocess ----------------------------------------------
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
fi

# ---- Phases 2..: septal ablation arms -> train + predict ---------------------
for TR in "${ARMS[@]}"; do
  if [ ! -f "${CKPT_DIR}/train_${TR}.done" ]; then
    echo "[train] ${TR}"; nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TR}" -p "${PLANS}"
    touch "${CKPT_DIR}/train_${TR}.done"
  fi
  if [ ! -f "${CKPT_DIR}/pred_${TR}.done" ]; then
    echo "[predict] ${TR}"
    nnUNetv2_predict -i "${IN_DIR}" -o "${PRED_BASE}/${TR}" \
        -d "${DATASET_ID}" -c "${FULLRES}" -f "${FOLD}" -tr "${TR}" -p "${PLANS}"
    touch "${CKPT_DIR}/pred_${TR}.done"
  fi
done
echo "DONE. Dataset062 (v3 VSD-anchored). Predictions in ${PRED_BASE}/<arm>/."
