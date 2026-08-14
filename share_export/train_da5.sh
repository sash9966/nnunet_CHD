#!/bin/bash
# =============================================================================
#  train_da5.sh — train a DA5 model on your own dataset, the same way the shared
#  weights were made. Plain nnU-Net workflow: plan_and_preprocess -> train.
#
#  Prereq: `pip install -e .` (this fork) done, and your data laid out as:
#     $nnUNet_raw/DatasetXXX_YourName/
#         imagesTr/<case>_0000.nii.gz      (CT; _0000 = channel 0)
#         labelsTr/<case>.nii.gz           (integer labels 0..N)
#         dataset.json                     (nnU-Net v2 format; channel_names + labels + numTraining)
#  See https://github.com/MIC-DKFZ/nnUNet for dataset.json details.
# =============================================================================
set -euo pipefail

# ============================ ### EDIT THESE ### =============================
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

DATASET_ID=200                                   # your Dataset<ID>_<name> folder id
TRAINER="nnUNetTrainerDA5_500epochs"             # or _100epochs / _200epochs
FOLDS="0"                                        # "0" for one fold, or "0 1 2 3 4", or "all"
# ============================================================================

PLANNER="nnUNetPlannerResEncM"                   # ResEnc "M" plans (what the shared models use)
PLANS="nnUNetResEncUNetMPlans"
CONFIG="3d_fullres"

echo "[1/2] plan & preprocess (Dataset ${DATASET_ID})"
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${CONFIG}" --verify_dataset_integrity

echo "[2/2] train ${TRAINER}  folds: ${FOLDS}"
for f in ${FOLDS}; do
  echo "  --- fold ${f} ---"
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${f}" -tr "${TRAINER}" -p "${PLANS}"
done

echo "DONE. Model: ${nnUNet_results}/Dataset$(printf '%03d' ${DATASET_ID})_*/${TRAINER}__${PLANS}__${CONFIG}/"
echo "Predict with it: see INFERENCE.md"
