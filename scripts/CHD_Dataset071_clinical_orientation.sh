#!/bin/bash
# =============================================================================
#  CHD_Dataset071_clinical_orientation.sh
#  Dataset071_ImageCHDClinicalOrientation — clinical (LPS) 7-class anatomy model.
#
#     Phase 0   BUILD Dataset071 (RAS->LPS reorient, clean 7-class labels) on the
#               compute node -- much faster than the throttled login node
#     Phase 1   plan_and_preprocess (3d_fullres, ResEncM)
#     Phase 1b  reuse Dataset060 splits_final.json (verified case-set match) -> 5-fold
#               (falls back to nnU-Net auto 5-fold if that split is absent)
#     Phase 2   train DA5 @200 and @500 epochs, all 5 folds
#
#  RESUME: a (trainer,fold) whose checkpoint_final.pth exists is SKIPPED; one with a
#          checkpoint_latest.pth is RESUMED (--c). 10 runs (2 schedules x 5 folds),
#          the 500-epoch folds are long -> just resubmit this script until done.
#  Prereq: git pull. The build sources images+labels from Dataset030 (the original
#          HU dataset), so it does not depend on Dataset060 existing.
# =============================================================================
#SBATCH --job-name=D071-clinical
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D071-clinical_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D071-clinical_%j.err

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
DATASET_ID=71
DATASET_NAME="Dataset071_ImageCHDClinicalOrientation"
BUILD_IMAGES_DS="Dataset030_imageCHD_HU"             # image source (same content as D060; D030 always present)
BUILD_LABELS_DS="Dataset030_imageCHD_HU"             # clean 7-class labels (no septal id 8)
SPLITS_SRC_NAME="Dataset060_imageCHD_CleanHoldout"   # reuse this dataset's 5-fold split if present
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"          # plans id the ResEncM planner writes (pass via -p)
FULLRES="3d_fullres"
FOLDS=(0 1 2 3 4)
TRAINERS=("nnUNetTrainerDA5_200epochs" "nnUNetTrainerDA5_500epochs")

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/clinical_orientation"
mkdir -p "${CKPT_DIR}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ─────────────────────────────────────────────
# Phase 0 — BUILD Dataset071 (reorient RAS->LPS; runs here on the compute node)
#   Images + clean 7-class labels both from Dataset030 (no dependency on Dataset060).
#   The builder self-verifies LPS + geometry + lossless labels/HU and exits non-zero
#   on any failure, so a bad build aborts the job before preprocessing.
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} from ${BUILD_IMAGES_DS} (RAS->LPS)"
  python tools/build_dataset071_clinical_orientation.py \
      --nnunet-raw "${nnUNet_raw}" \
      --images-dataset "${BUILD_IMAGES_DS}" \
      --labels-dataset "${BUILD_LABELS_DS}" \
      --overwrite
  touch "${CKPT_DIR}/00_build.done"
else
  echo "[Phase 0] build already done — skipping"
fi

# ─────────────────────────────────────────────
# Phase 1 — plan & preprocess (3d_fullres only; Dataset071 ONLY)
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
# Phase 1b — reuse Dataset060 5-fold split (ABORT if the training case sets differ)
#   Dataset071's training set = all myo-present cases; Dataset060's = clean-train.
#   These SHOULD be identical, but Dataset060 may have moved a few clean cases into
#   its test set (stratified topup). We verify the split's case-set == Dataset071's
#   training case-set EXACTLY before copying — never train on a mismatched split.
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/01b_splits.done" ]; then
  SRC_SPLITS="${nnUNet_preprocessed}/${SPLITS_SRC_NAME}/splits_final.json"
  if [ ! -f "${SRC_SPLITS}" ]; then
    echo "[Phase 1b] WARNING: ${SRC_SPLITS} not found (Dataset060 absent or never trained)."
    echo "           Falling back to nnU-Net's auto-generated 5-fold split over"
    echo "           Dataset071's training cases (folds will NOT match Dataset060)."
    touch "${CKPT_DIR}/01b_splits.done"
  else
  echo "[Phase 1b] verifying + copying ${SPLITS_SRC_NAME} splits_final.json"
  python3 - "${DATASET_NAME}" "${SPLITS_SRC_NAME}" <<'PY'
import json, os, sys
from pathlib import Path
raw = os.environ['nnUNet_raw']; pre = os.environ['nnUNet_preprocessed']
ds, src = sys.argv[1], sys.argv[2]
suf = "_0000.nii.gz"
tr = {p.name[:-len(suf)] for p in Path(raw, ds, 'imagesTr').glob('*' + suf)}
sp = Path(pre, src, 'splits_final.json')
if not sp.is_file():
    sys.exit(f"ERROR: {sp} not found — train (or split) {src} first so its splits exist")
splits = json.load(open(sp))
covered = set()
for fold in splits:
    covered |= set(fold['train']) | set(fold['val'])
extra   = covered - tr     # in split but not a Dataset071 training case -> would CRASH training
missing = tr - covered     # a Dataset071 training case not in any fold -> silently untrained
print(f"[splits] Dataset071 train={len(tr)}  {src} splits cover={len(covered)}  folds={len(splits)}")
if extra:
    sys.exit(f"ABORT: split references {len(extra)} case(s) NOT in Dataset071 train "
             f"(would crash): {sorted(extra)[:8]}")
if missing:
    sys.exit(f"ABORT: {len(missing)} Dataset071 train case(s) NOT in the {src} split "
             f"(training sets differ — do NOT reuse this split): {sorted(missing)[:8]}")
dst = Path(pre, ds, 'splits_final.json')
json.dump(splits, open(dst, 'w'), indent=1)
print(f"[splits] MATCH — copied {src} split -> {dst} ({len(splits)} folds)")
PY
  touch "${CKPT_DIR}/01b_splits.done"
  fi
else
  echo "[Phase 1b] splits already set — skipping"
fi

# ─────────────────────────────────────────────
# Phase 2 — train DA5 @200 and @500 epochs, all 5 folds
#   200-epoch CV finishes first (trainers loop is outer), then 500-epoch.
# ─────────────────────────────────────────────
for TR in "${TRAINERS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    OUT="${nnUNet_results}/${DATASET_NAME}/${TR}__${PLANS}__${FULLRES}/fold_${FOLD}"
    if [ -f "${OUT}/checkpoint_final.pth" ]; then
      echo "[skip] ${TR} fold ${FOLD} — already complete"
      continue
    fi
    CONT=""
    if [ -f "${OUT}/checkpoint_latest.pth" ]; then
      CONT="--c"
      echo "[resume] ${TR} fold ${FOLD} from checkpoint_latest.pth"
    else
      echo "[train]  ${TR} fold ${FOLD} (fresh)"
    fi
    nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TR}" -p "${PLANS}" ${CONT}
  done
done

echo "=============================================================="
echo "DONE. Dataset071 clinical (LPS) — DA5 @200 & @500, folds 0-4."
echo "Models under: ${nnUNet_results}/${DATASET_NAME}/nnUNetTrainerDA5_{200,500}epochs__${PLANS}__${FULLRES}/"
echo "(If SLURM walltime cut it short, just resubmit — completed folds skip, partial folds resume.)"
echo "=============================================================="
