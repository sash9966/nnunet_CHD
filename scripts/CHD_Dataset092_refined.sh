#!/bin/bash
# =============================================================================
#  CHD_Dataset092_refined.sh
#  Dataset092 = Dataset091 with the QC-approved Fanwei labels REPLACED by the promptable-refinement
#  output (nnInteractive + SeqSeg). *** IDENTICAL CASE LIST TO D091 *** — only the LABELS differ, so
#  D091 vs D092 isolates ONE variable: label quality / native-resolution refinement.
#  Extra data is deliberately NOT mixed in here; that is Dataset093 (D092 + newly-usable cases), so
#  D092 vs D093 isolates the additional-data effect. One variable per increment.
#  IDENTICAL trainer/plans/folds/seed to Dataset090 and Dataset091 so the A/B on Dataset080 is valid.
#
#     Phase 0   BUILD Dataset092 from Dataset091 (relabel QC-approved ONLY — same case list)
#     Phase 1   plan_and_preprocess (--verify_dataset_integrity)
#     Phase 1b  splits: ImageCHD 5-fold val (SAME 071 folds as 090/091); pseudo train-only
#     Phase 2   train folds 0,1,2,3,4
#
#  Dataset080 (BAF004/CHIPS*) is NEVER in training — the builder hard-fails if it appears.
# =============================================================================
#SBATCH --job-name=D092-5fold
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D092-5fold_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D092-5fold_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python '+sys.version.split()[0]+' active, expected 3.10 from conda env nnunet310. The env did not activate (likely a scratch purge deleted the interpreter). Recreate the env before running; do not edit the code.')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=f          # torch.compile/Triton JIT needs Python.h from $CONDA_PREFIX/include;
                                 # the scratch-purge repair restored lib/ but not include/, so compile
                                 # dies at Epoch 0 ("Python.h: No such file"). Disabling it is safe
                                 # (slightly slower). Remove once the headers are restored.

REPO="/scratch/users/sastocke/nnunet_CHD"
DATASET_ID=92
DATASET_NAME="Dataset092_ImageCHDRefined"
SRC_DATASET="Dataset091_ImageCHDPseudoCombinedV2"
IMAGECHD_SRC="Dataset071_ImageCHDClinicalOrientation"
NUM_FOLDS=5; SPLIT_SEED=42
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLDS=(0 1 2 3 4)          # identical to D090/D091

# refined labels = nnInteractive + SeqSeg, stitched (step 5 output)
REFINED_DIR="${REFINED_DIR:-/scratch/users/sastocke/chd_refinement/out/refined_fanwei_merged}"

# QC-approved refined cases ("Can Use After Check"). Only cases ALREADY in D091 are relabelled here;
# CT_704_49 / CT_853_56_no were held out by the earlier QC and are added in D093, NOT here.
QC_APPROVED="CT_853_56_no,CT_462_49,CT_710_579,CT_325_459_no,CT_309_069_no,CT_584_09_no,CT_860_8,CT_201_09,CT_704_49,CT_260_5810,CT_502_5910,CT_234_69,CT_273,CT_747_68,CT_010_49,CT_535_8,CT_041_089,CT_342_156,CT_697_57,CT_728_459,CT_190_49,CT_754_49,CT_298_49,CT_395_459_no"
# never-train safety list (Dataset080 test set + the clinical holdouts)
EXCLUDE="BAF004,BAF005,CHIPS001,CHIPS002,CHIPS005,CHIPS006,CHIPS007,CHIPS010,CHIPS016"

NEW_FLAG="--no-new-cases"   # D092 = label-swap ONLY; identical case list to D091 (single variable)

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/refined"
mkdir -p "${CKPT_DIR}" "${REPO}/logs"
cd "${REPO}"
source scripts/_provenance.sh
stamp_provenance "D092-build+train" "${nnUNet_results}/${DATASET_NAME}" \
    "src=${SRC_DATASET}" "refined=${REFINED_DIR}" "trainer=${TRAINER}" "design=label_swap_only(case_list==D091)"

# ---- Phase 0: build Dataset092 from Dataset091 ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} from ${SRC_DATASET}"
  [ -d "${REFINED_DIR}" ] || { echo "FATAL: refined labels dir not found: ${REFINED_DIR}"; exit 1; }
  python tools/build_dataset092_refined.py --nnunet-raw "${nnUNet_raw}" \
      --src-dataset "${SRC_DATASET}" --target-id "${DATASET_ID}" --target-name "ImageCHDRefined" \
      --refined-labels-dir "${REFINED_DIR}" \
      --qc-approved "${QC_APPROVED}" --exclude "${EXCLUDE}" ${NEW_FLAG} --overwrite
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


# ---- Phase 3: chain the frozen-Dataset080 evaluation ----
# Only reached if every fold trained (set -e aborts earlier otherwise). Submitted as its own job so it
# gets its own log/allocation instead of eating this job's wall clock. AUTO_EVAL=0 to disable.
if [ "${AUTO_EVAL:-1}" = "1" ]; then
  EVAL_JOB="$(sbatch --parsable scripts/CHD_predict_dataset080_d092.sh 2>/dev/null || true)"
  if [ -n "$EVAL_JOB" ]; then echo "[chain] queued frozen-D080 evaluation as job ${EVAL_JOB}"
  else echo "[chain] WARNING: could not auto-submit the eval job — run it manually"; fi
fi
echo "=============================================================="
echo "DONE. Dataset092 trained on folds ${FOLDS[*]}."
echo "  model: ${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "  eval:  auto-submitted (see [chain] above); AUTO_EVAL=0 to skip"
echo "=============================================================="
