#!/bin/bash
# =============================================================================
#  CHD_Dataset090_train5fold.sh
#  Full 5-fold training of Dataset090 (pseudo-label run 1).
#  Same build/preprocess/split as the original run (guarded -> skipped if done);
#  trains folds 0,1,2,3,4 with the SAME trainer/plans/config.
#
#  NOTE: does NOT run held-out inference (Phase 3) — that is a separate script
#  (CHD_predict_dataset080_compare.sh). We deliberately DO NOT touch the existing
#  predictions/ds090__grid2native_lcc/ so the QC-approved labels used to build
#  Dataset091 stay exactly as you inspected them.
# =============================================================================
#SBATCH --job-name=D090-5fold
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D090-5fold_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D090-5fold_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# --- env guard: fail LOUDLY if the env's python isn't active (a /scratch purge can delete the
#     interpreter, leaving dangling symlinks -> `python` falls back to system 2.7 -> f-string SyntaxError) ---
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python '+sys.version.split()[0]+' active, expected 3.10 from conda env nnunet310. The env did not activate (likely a scratch purge deleted the interpreter). Recreate the env before running; do not edit the code.')"

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
TRAINER="nnUNetTrainerDA5_200epochs"; FOLDS=(0 1 2 3 4)

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/pseudo"
mkdir -p "${CKPT_DIR}" "${REPO}/logs"
cd "${REPO}"

# ---- Phase 0: build (guarded; already done in the first run) ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME}"
  python tools/build_dataset090_pseudolabel.py --nnunet-raw "${nnUNet_raw}" \
      --imagechd-dataset "${IMAGECHD_SRC}" --target-id "${DATASET_ID}" \
      --target-name "ImageCHDPseudoCombined" \
      --clinical-root "${REPO}/ClinicalImagesPHICleared" --overwrite
  touch "${CKPT_DIR}/00_build.done"
else echo "[Phase 0] build already done — skipping"; fi

# ---- Phase 1: plan & preprocess (guarded) ----
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
else echo "[Phase 1] preprocess already done — skipping"; fi

# ---- Phase 1b: splits (ImageCHD 5-fold val, reuse 071 folds; pseudo train-only) ----
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
    base = json.loads(src_splits.read_text()); splits = []
    for fold in base:
        val = sorted(set(fold['val']) & chd)
        train_chd = sorted(set(fold['train']) & chd)
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
else echo "[Phase 1b] split already written — skipping"; fi

# ---- Phase 2: train folds 0-4 ----
for FOLD in "${FOLDS[@]}"; do
  OUT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FOLD}"
  if [ -f "${OUT}/checkpoint_final.pth" ]; then echo "[skip] fold ${FOLD} complete"; continue; fi
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[train] ${TRAINER} fold ${FOLD} ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
done

echo "=============================================================="
echo "DONE. Dataset090 trained on folds ${FOLDS[*]}."
echo "  model: ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "  next:  sbatch scripts/CHD_predict_dataset080_compare.sh (after D091 also trains)"
echo "=============================================================="
