#!/bin/bash
# =============================================================================
#  CHD_TCV_arm1_donoronly.sh   (workstream C — TCV transplant matching)
#  ARM 1: train on DONORS ONLY -> test on held-out RECIPIENTS.
#  This is the DEFENSIBLE donor->recipient generalization arm: the 'x' cohort has undefined
#  provenance, so if any x case is actually a recipient it would leak the test domain into training.
#  Arm 2 (CHD_TCV_arm2_donorx.sh) adds x and is the better-powered model; compare the two.
#  Fold 0 only — this is a pipeline test, not a final number (arm 1 trains on ~12 donors).
#
#     Phase 0  build Dataset511 (absolute symlinks + SimpleITK read-test on every image)
#     Phase 1  plan_and_preprocess
#     Phase 2  train fold 0
#     Phase 3  predict held-out recipients + TCV volume-error eval (MAPE/bias/Bland-Altman/Dice)
# =============================================================================
#SBATCH --job-name=TCV-arm1
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/TCV-arm1_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/TCV-arm1_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: python <3.9 active — env did not activate (scratch purge?). Recreate the env.')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=f          # torch.compile/Triton JIT needs Python.h from $CONDA_PREFIX/include;
                                 # the scratch-purge repair restored lib/ but not include/, so compile
                                 # dies at Epoch 0 ("Python.h: No such file"). Disabling it is safe
                                 # (slightly slower). Remove once the headers are restored.
REPO="/scratch/users/sastocke/nnunet_CHD"; cd "$REPO"

# ===== arm config =====
export DS_ID=511
export DS_NAME="Dataset511_TCV_DonorOnly_ToRecipient"
export TRAIN_PREFIXES="d"        # <-- ARM 1: donors only
export INFER_PREFIXES="r"        # held-out recipients (masks -> labelsTs, scoring only)
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLD=0
# ======================

CKPT="${nnUNet_results}/${DS_NAME}/.checkpoints"; mkdir -p "$CKPT" "$REPO/logs"
source scripts/_provenance.sh
stamp_provenance "TCV-arm1-donoronly" "${nnUNet_results}/${DS_NAME}" \
    "train=${TRAIN_PREFIXES}" "test=${INFER_PREFIXES}" "trainer=${TRAINER}" "fold=${FOLD}"

# ---- Phase 0: build ----
if [ ! -f "$CKPT/00_build.done" ]; then
  echo "[Phase 0] building ${DS_NAME}  (train='${TRAIN_PREFIXES}'  test='${INFER_PREFIXES}')"
  bash scripts/build_tcv_ds.sh
  touch "$CKPT/00_build.done"
else echo "[Phase 0] already built — skipping"; fi

# ---- Phase 1: preprocess ----
if [ ! -f "$CKPT/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "$DS_ID" -pl "$PLANNER" -c "$FULLRES" --verify_dataset_integrity
  touch "$CKPT/01_preprocess.done"
else echo "[Phase 1] already preprocessed — skipping"; fi

# ---- Phase 2: train fold 0 ----
OUT="${nnUNet_results}/${DS_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FOLD}"
if [ ! -f "${OUT}/checkpoint_final.pth" ]; then
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[Phase 2] training ${TRAINER} fold ${FOLD} ${CONT}"
  nnUNetv2_train "$DS_ID" "$FULLRES" "$FOLD" -tr "$TRAINER" -p "$PLANS" ${CONT}
else echo "[Phase 2] fold ${FOLD} already complete — skipping"; fi

# ---- Phase 3: predict held-out recipients + volume-error eval ----
TS="${nnUNet_raw}/${DS_NAME}/imagesTs"
GT="${nnUNet_raw}/${DS_NAME}/labelsTs"
PRED="/scratch/users/sastocke/chd_refinement/tcv_pred_${DS_ID}_f${FOLD}"
EVAL="/scratch/users/sastocke/chd_refinement/tcv_eval_${DS_ID}"
if [ -d "$TS" ] && [ -n "$(ls -A "$TS" 2>/dev/null)" ]; then
  mkdir -p "$PRED" "$EVAL"
  if [ -z "$(ls -A "$PRED" 2>/dev/null)" ]; then
    echo "[Phase 3] predicting held-out recipients (fold ${FOLD}, native spacing)"
    nnUNetv2_predict -i "$TS" -o "$PRED" -d "$DS_ID" -c "$FULLRES" -tr "$TRAINER" -p "$PLANS" \
        -f "$FOLD" --disable_tta || echo "  [warn] prediction failed"
  fi
  echo "[Phase 3] TCV volume-error eval (MAPE / bias / Bland-Altman / Dice)"
  python tools/eval_tcv_volume.py --gt-dir "$GT" --pred-dir "$PRED" --out "$EVAL" || echo "  [warn] eval failed"
else echo "[Phase 3] no imagesTs — skipping prediction/eval"; fi

echo "=============================================================="
echo "DONE. ${DS_NAME} (train=${TRAIN_PREFIXES} -> test=${INFER_PREFIXES}), fold ${FOLD}."
echo "  model : ${nnUNet_results}/${DS_NAME}/${TRAINER}__${PLANS}__${FULLRES}/"
echo "  preds : ${PRED}"
echo "  eval  : ${EVAL}/tcv_volume_eval.csv"
echo "=============================================================="
