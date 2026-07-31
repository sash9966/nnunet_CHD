#!/bin/bash
# =============================================================================
#  CHD_Dataset072_multifov.sh
#  Dataset072_ImageCHDMultiFOV — clinical domain adaptation via cardiac-FOV crops.
#
#     Phase 0   BUILD Dataset072 from Dataset071 (full + bbox60mm + bbox30mm)
#     Phase 1   plan_and_preprocess (3d_fullres, ResEncM)
#     Phase 1b  OVERWRITE splits_final.json with a GROUPED 5-fold split so all
#               variants of a patient (<case>_full/_bbox60mm/_bbox30mm) stay in the
#               SAME fold -> NO validation leakage across a patient's own variants
#     Phase 2   train DA5 (default: fold 0 @ 200 epochs -- "see a training run")
#
#  RESUME: build/preprocess/splits guarded by .done markers; a (trainer,fold) with
#          checkpoint_final.pth is skipped, a partial one resumes with --c.
#  Prereq: git pull; Dataset071 must exist in $nnUNet_raw (build it first if not).
# =============================================================================
#SBATCH --job-name=D072-multifov
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D072-multifov_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D072-multifov_%j.err

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
DATASET_ID=72
DATASET_NAME="Dataset072_ImageCHDMultiFOV"
SOURCE_NAME="Dataset071_ImageCHDClinicalOrientation"
MARGINS_MM="60,30"
NUM_FOLDS=5
SPLIT_SEED=42
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
TRAINERS=("nnUNetTrainerDA5_200epochs")   # add "nnUNetTrainerDA5_500epochs" for the long schedule
FOLDS=(0)                                 # set to (0 1 2 3 4) for the full grouped 5-fold CV

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/multifov"
mkdir -p "${CKPT_DIR}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ─────────────────────────────────────────────
# Phase 0 — BUILD Dataset072 (full + cardiac-bbox crops; crop-only, no resample)
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} from ${SOURCE_NAME} (margins ${MARGINS_MM} mm/side)"
  python tools/build_dataset072_multifov.py \
      --nnunet-raw "${nnUNet_raw}" \
      --source-dataset "${SOURCE_NAME}" \
      --target-id "${DATASET_ID}" --target-name "ImageCHDMultiFOV" \
      --margins-mm "${MARGINS_MM}" --overwrite
  touch "${CKPT_DIR}/00_build.done"
else
  echo "[Phase 0] build already done — skipping"
fi

# ─────────────────────────────────────────────
# Phase 1 — plan & preprocess (fingerprint recomputed over the mixed-FOV set)
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
# Phase 1b — OVERWRITE splits_final.json with a GROUPED 5-fold split
#   Groups come from case_groups.json (source case -> its variant case ids). Each
#   source case is assigned to one fold (round-robin over sorted ids); ALL of its
#   variants go to that fold's val, the rest to train. No patient straddles folds.
# ─────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/01b_splits.done" ]; then
  echo "[Phase 1b] writing grouped ${NUM_FOLDS}-fold splits_final.json (no per-patient leakage)"
  python3 - "${DATASET_NAME}" "${NUM_FOLDS}" "${SPLIT_SEED}" <<'PY'
import json, os, sys, random
from pathlib import Path
raw = os.environ['nnUNet_raw']; pre = os.environ['nnUNet_preprocessed']
ds, K, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
groups = json.loads(Path(raw, ds, 'case_groups.json').read_text())   # {source_case: [variant ids]}
sources = sorted(groups)
random.Random(seed).shuffle(sources)                                  # deterministic
fold_of = {s: i % K for i, s in enumerate(sources)}
all_ids = sorted(v for vs in groups.values() for v in vs)
splits = []
for k in range(K):
    val, train = [], []
    for s in sorted(groups):
        (val if fold_of[s] == k else train).extend(groups[s])
    splits.append({"train": sorted(train), "val": sorted(val)})
    # leakage assertion: a source case's variants never span train and val
    val_sources = {s for s in groups if fold_of[s] == k}
    assert not (set(splits[k]['val']) & set(splits[k]['train'])), "id in both train and val"
    for s in val_sources:
        assert all(v in splits[k]['val'] for v in groups[s]), f"{s} variants split across folds"
# every preprocessed case must be covered exactly
prep = pre if os.path.isdir(Path(pre, ds)) else raw
dst = Path(pre, ds, 'splits_final.json')
dst.write_text(json.dumps(splits, indent=1))
sizes = [(len(s['train']), len(s['val'])) for s in splits]
print(f"[splits] {len(sources)} source cases -> {len(all_ids)} variant cases | {K} folds")
print(f"[splits] per-fold (train,val) sizes: {sizes}")
print(f"[splits] wrote {dst}")
PY
  touch "${CKPT_DIR}/01b_splits.done"
else
  echo "[Phase 1b] grouped splits already written — skipping"
fi

# ─────────────────────────────────────────────
# Phase 2 — train DA5 (default: fold 0 @ 200e; edit TRAINERS/FOLDS to scale)
# ─────────────────────────────────────────────
for TR in "${TRAINERS[@]}"; do
  for FOLD in "${FOLDS[@]}"; do
    OUT="${nnUNet_results}/${DATASET_NAME}/${TR}__${PLANS}__${FULLRES}/fold_${FOLD}"
    if [ -f "${OUT}/checkpoint_final.pth" ]; then
      echo "[skip] ${TR} fold ${FOLD} — already complete"; continue
    fi
    CONT=""
    if [ -f "${OUT}/checkpoint_latest.pth" ]; then CONT="--c"; echo "[resume] ${TR} fold ${FOLD}"; \
    else echo "[train]  ${TR} fold ${FOLD} (fresh)"; fi
    nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TR}" -p "${PLANS}" ${CONT}
  done
done

echo "=============================================================="
echo "DONE. Dataset072 multi-FOV — grouped ${NUM_FOLDS}-fold, DA5 folds: ${FOLDS[*]}."
echo "Predict clinical cases with: nnUNetv2_predict -d ${DATASET_ID} -c ${FULLRES} -tr ${TRAINERS[0]} -p ${PLANS} -f ${FOLDS[*]}"
echo "=============================================================="
