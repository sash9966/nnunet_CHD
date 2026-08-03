#!/bin/bash
# =============================================================================
#  CHD_Dataset081_mix.sh
#  Train on Dataset081 = ImageCHD (071) + clinical (080, oversampled 8x).
#
#     Phase 0   BUILD Dataset081 (symlink 071 + oversampled clinical from 080)
#     Phase 1   plan_and_preprocess (3d_fullres, ResEncM)
#     Phase 1b  write splits_final.json: ImageCHD does the 5-fold VAL; clinical
#               (all copies) is in EVERY fold's TRAIN, never val -> no duplicate
#               leakage, all 3 clinical cases fully used. (Real clinical eval =
#               external inference, not an internal fold.)
#     Phase 2   train DA5 200e (from scratch), fold 0 by default
#
#  RESUME: build/preprocess/split guarded by .done; training skips/resumes on ckpt.
# =============================================================================
#SBATCH --job-name=D081-mix
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D081-mix_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D081-mix_%j.err

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
DATASET_ID=81
DATASET_NAME="Dataset081_ImageCHDplusClinical"
IMAGECHD_DS="Dataset071_ImageCHDClinicalOrientation"
CLINICAL_DS="Dataset080_ClincalCaseSanjibDetailed"
DUP_FACTOR=8
NUM_FOLDS=5
SPLIT_SEED=42
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"
FOLDS=(0)                     # set to (0 1 2 3 4) for the full CV

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/mix"
mkdir -p "${CKPT_DIR}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ---- Phase 0: build Dataset081 ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} (${IMAGECHD_DS} + ${CLINICAL_DS} x${DUP_FACTOR})"
  python tools/build_dataset081_mix.py --nnunet-raw "${nnUNet_raw}" \
      --imagechd-dataset "${IMAGECHD_DS}" --clinical-dataset "${CLINICAL_DS}" \
      --target-id "${DATASET_ID}" --target-name "ImageCHDplusClinical" \
      --dup-factor "${DUP_FACTOR}" --overwrite
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

# ---- Phase 1b: clinical-always-train split (no clinical in val -> no dup leakage) ----
if [ ! -f "${CKPT_DIR}/01b_splits.done" ]; then
  echo "[Phase 1b] writing splits_final.json (ImageCHD 5-fold val; clinical always train)"
  python3 - "${DATASET_NAME}" "${NUM_FOLDS}" "${SPLIT_SEED}" <<'PY'
import json, os, sys, random
from pathlib import Path
raw = os.environ['nnUNet_raw']; pre = os.environ['nnUNet_preprocessed']
ds, K, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
meta = json.loads(Path(raw, ds, 'split_meta.json').read_text())
chd = sorted(meta['imagechd']); clin = sorted(meta['clinical_instances'])
random.Random(seed).shuffle(chd)
folds = [chd[i::K] for i in range(K)]            # deterministic 5-fold over ImageCHD
splits = []
for k in range(K):
    val = sorted(folds[k])
    train = sorted([c for c in chd if c not in set(folds[k])] + clin)  # clinical ALWAYS in train
    assert not (set(val) & set(train)), "leakage: id in val and train"
    assert not (set(clin) & set(val)), "clinical case ended up in val!"
    splits.append({"train": train, "val": val})
Path(pre, ds, 'splits_final.json').write_text(json.dumps(splits, indent=1))
print(f"[splits] ImageCHD={len(chd)} clinical(train-only)={len(clin)} folds={K}")
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

echo "=============================================================="
echo "DONE. Dataset081 mix model: ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "Predict with: -d ${DATASET_ID} -tr ${TRAINER} -p ${PLANS} -c ${FULLRES} -f ${FOLDS[*]}"
echo "=============================================================="
