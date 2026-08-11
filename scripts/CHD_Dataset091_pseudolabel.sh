#!/bin/bash
# =============================================================================
#  CHD_Dataset091_pseudolabel.sh
#  Dataset091 = Dataset090 + QC-approved ds090 pseudo-label cases promoted to train.
#  Identical setup to Dataset090 except for the added cases; full 5-fold training.
#
#     Phase 0   BUILD Dataset091 from Dataset090 (symlink 090 verbatim + add promoted)
#     Phase 1   plan_and_preprocess (--verify_dataset_integrity)
#     Phase 1b  splits: ImageCHD 5-fold val (SAME 071 folds as 090); pseudo (incl the
#               promoted cases) train-only  ->  090 and 091 share identical val folds
#     Phase 2   train folds 0,1,2,3,4
#
#  Promoted labels come from the ds090 QC'd predictions (predictions/ds090__grid2native_lcc).
#  BAF004 is EXCLUDED (Dataset080 held-out test) — the builder hard-fails if it ever appears.
# =============================================================================
#SBATCH --job-name=D091-5fold
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D091-5fold_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D091-5fold_%j.err

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
DATASET_ID=91
DATASET_NAME="Dataset091_ImageCHDPseudoCombinedV2"
SRC_DATASET="Dataset090_ImageCHDPseudoCombined"
IMAGECHD_SRC="Dataset071_ImageCHDClinicalOrientation"
NUM_FOLDS=5; SPLIT_SEED=42
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLDS=(0 1 2 3 4)

# QC-approved ds090 pseudo-label cases to promote (8 Fanwei + BAF007). NOT BAF004 (Dataset080).
PROMOTED="CT_052_7910,CT_528_0579,CT_584_09_no,CT_731_6,CT_747_68,CT_754_49,CT_860_8,CT_914_49,BAF007"
# hard "never promote" list (safety): BAF004 + the do-not-include cases you listed.
EXCLUDE="BAF004,CHIPS002,CHIPS016,BAF005,CT_704_49,CT_853_56_no,CT_881_8,CT_110_69,CT_335_058,CT_790_069,CT_793_0569_no"

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/pseudo"
mkdir -p "${CKPT_DIR}" "${REPO}/logs"
cd "${REPO}"

# ---- Phase 0: build Dataset091 from Dataset090 ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} from ${SRC_DATASET} (+${PROMOTED})"
  python tools/build_dataset091_from_090.py --nnunet-raw "${nnUNet_raw}" \
      --src-dataset "${SRC_DATASET}" --target-id "${DATASET_ID}" \
      --target-name "ImageCHDPseudoCombinedV2" \
      --promoted "${PROMOTED}" --exclude "${EXCLUDE}" --overwrite
  touch "${CKPT_DIR}/00_build.done"
else echo "[Phase 0] build already done — skipping"; fi

# ---- Phase 1: plan & preprocess ----
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
echo "DONE. Dataset091 trained on folds ${FOLDS[*]}."
echo "  model: ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "  next:  sbatch scripts/CHD_predict_dataset080_compare.sh"
echo "=============================================================="
