#!/bin/bash
# =============================================================================
#  CHD_Dataset090_pseudolabel.sh
#  First pseudo-label run: ImageCHD (Dataset071, myo-intact) + usable Fanwei +
#  usable clinical LCC pseudo-labels. Held-out (unusable + Dataset080) -> imagesTs.
#
#     Phase 0   BUILD Dataset090 (buckets + sanity checks + split_config)
#     Phase 1   plan_and_preprocess (only imagesTr/labelsTr are processed)
#     Phase 1b  write splits_final.json: ImageCHD does the 5-fold VAL (reusing
#               Dataset071's folds if present); pseudo-label cases are in EVERY
#               fold's TRAIN, NEVER val (noisy labels -> keep val clean, no leak)
#     Phase 2   train DA5 200e, fold 0 (a first "how do we do" run)
#
#  RESUME: build/preprocess/split guarded; training skips/resumes on checkpoint.
#  Prereq: run CHD_backproject_ds071.sh first so the LCC native pseudo-labels exist.
# =============================================================================
#SBATCH --job-name=D090-pseudo
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D090-pseudo_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D090-pseudo_%j.err

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

REPO="/scratch/users/sastocke/nnunet_CHD"
DATASET_ID=90
DATASET_NAME="Dataset090_ImageCHDPseudoCombined"
IMAGECHD_SRC="Dataset071_ImageCHDClinicalOrientation"
NUM_FOLDS=5; SPLIT_SEED=42
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLDS=(0)   # set (0 1 2 3 4) for full CV

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/pseudo"
mkdir -p "${CKPT_DIR}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ---- Phase 0: build ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME}"
  python tools/build_dataset090_pseudolabel.py --nnunet-raw "${nnUNet_raw}" \
      --imagechd-dataset "${IMAGECHD_SRC}" --target-id "${DATASET_ID}" \
      --target-name "ImageCHDPseudoCombined" \
      --clinical-root "${REPO}/ClinicalImagesPHICleared" \
      --overwrite
  touch "${CKPT_DIR}/00_build.done"
else
  echo "[Phase 0] build already done — skipping"
fi

# ---- Phase 1: plan & preprocess ----
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
else
  echo "[Phase 1] preprocess already done — skipping"
fi

# ---- Phase 1b: splits — ImageCHD 5-fold val (reuse 071 folds if present); pseudo always train ----
if [ ! -f "${CKPT_DIR}/01b_splits.done" ]; then
  echo "[Phase 1b] writing splits_final.json (ImageCHD val; pseudo train-only)"
  python3 - "${DATASET_NAME}" "${IMAGECHD_SRC}" "${NUM_FOLDS}" "${SPLIT_SEED}" <<'PY'
import json, os, sys, random
from pathlib import Path
raw = os.environ['nnUNet_raw']; pre = os.environ['nnUNet_preprocessed']
ds, chd_src, K, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
meta = json.loads(Path(raw, ds, 'split_meta.json').read_text())
chd = set(meta['imagechd']); pseudo = sorted(meta['pseudo_train'])
src_splits = Path(pre, chd_src, 'splits_final.json')
if src_splits.is_file():
    base = json.loads(src_splits.read_text())
    # keep only ImageCHD cases that actually exist in Dataset090, restricted to val/train
    splits = []
    for fold in base:
        val = sorted(set(fold['val']) & chd)
        train_chd = sorted(set(fold['train']) & chd)
        # any ImageCHD case not covered by this fold's train/val -> put in train (safety)
        covered = set(val) | set(train_chd)
        train_chd = sorted(set(train_chd) | (chd - covered - set(val)))
        splits.append({"train": sorted(train_chd + pseudo), "val": val})
    print(f"[splits] reused {chd_src} folds ({len(base)}) for ImageCHD val")
else:
    print(f"[splits] WARNING: {src_splits} not found -> fresh {K}-fold over ImageCHD")
    chd_sorted = sorted(chd); random.Random(seed).shuffle(chd_sorted)
    folds = [chd_sorted[i::K] for i in range(K)]
    splits = [{"train": sorted([c for c in chd_sorted if c not in set(folds[k])] + pseudo),
               "val": sorted(folds[k])} for k in range(K)]
# assertions: no pseudo in val, no leak, ImageCHD covered
for k, s in enumerate(splits):
    assert not (set(s['val']) & set(pseudo)), f"fold {k}: pseudo case in val!"
    assert not (set(s['val']) & set(s['train'])), f"fold {k}: train/val overlap!"
    assert set(s['val']) <= chd, f"fold {k}: non-ImageCHD case in val!"
allval = set().union(*[set(s['val']) for s in splits])
assert allval == chd, f"ImageCHD not fully validated once ({len(allval)} vs {len(chd)})"
Path(pre, ds, 'splits_final.json').write_text(json.dumps(splits, indent=1))
print(f"[splits] ImageCHD={len(chd)} pseudo(train-only)={len(pseudo)} folds={len(splits)}")
print(f"[splits] per-fold (train,val): {[(len(s['train']), len(s['val'])) for s in splits]}")
PY
  touch "${CKPT_DIR}/01b_splits.done"
else
  echo "[Phase 1b] split already written — skipping"
fi

# ---- Phase 2: train ----
for FOLD in "${FOLDS[@]}"; do
  OUT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FOLD}"
  if [ -f "${OUT}/checkpoint_final.pth" ]; then echo "[skip] fold ${FOLD} complete"; continue; fi
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[train] ${TRAINER} fold ${FOLD} ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
done

# ---- Phase 3: held-out inference with the Dataset090 model (grid512 route) ----
#   imagesTs = unusable + quick_check + Dataset080 (all held out from training).
#   resize -> predict (Dataset090) -> resample back to native + LCC.  All under nnunet_CHD.
HOLDOUT="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
HRESIZED="${nnUNet_raw}/${DATASET_NAME}/imagesTs_imagechd_grid"
PREDROOT="${nnUNet_raw}/${DATASET_NAME}/predictions"
GRID="${PREDROOT}/ds090__grid512"
FINAL="${PREDROOT}/ds090__grid2native_lcc"
FF="${FOLDS[0]}"; FOLDSTR="${FOLDS[*]}"
MODELCKPT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FF}/checkpoint_final.pth"
if [ -f "${MODELCKPT}" ] && ls "${HOLDOUT}"/*.nii.gz >/dev/null 2>&1; then
  echo "[Phase 3] held-out inference (resize -> predict ds090 -> backproject+LCC)"
  mkdir -p "${PREDROOT}"
  ls "${HRESIZED}"/*.nii.gz >/dev/null 2>&1 || \
    python tools/resize_to_imagechd_grid.py --input "${HOLDOUT}" --output "${HRESIZED}" --overwrite
  if ! ls "${GRID}"/*.nii.gz >/dev/null 2>&1; then
    mkdir -p "${GRID}"
    nnUNetv2_predict -i "${HRESIZED}" -o "${GRID}" -d "${DATASET_ID}" -c "${FULLRES}" \
        -tr "${TRAINER}" -p "${PLANS}" -f ${FOLDSTR} -chk checkpoint_final.pth
  fi
  ls "${FINAL}"/*.nii.gz >/dev/null 2>&1 || \
    python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" --native-dir "${HOLDOUT}" \
        --output-dir "${FINAL}" --overwrite
  echo "[Phase 3] held-out native+LCC predictions -> ${FINAL}"
else
  echo "[Phase 3] skipped (model checkpoint or imagesTs missing)"
fi

echo "=============================================================="
echo "DONE. Dataset090 pseudo-label run (everything under nnunet_CHD)."
echo "  model:            ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "  held-out images:  ${HOLDOUT}/  (unusable + quick_check + Dataset080)"
echo "  held-out labels:  ${FINAL}/  (native + LCC; overlay on the original CT)"
echo "=============================================================="
