#!/bin/bash
set -euo pipefail   # abort immediately on any error; treat unset vars as errors
#SBATCH --job-name=CHD_Cascade
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu

module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

# Initialize conda for non-interactive shell:
source /scratch/users/sastocke/conda_envs/nnunet310/etc/profile.d/conda.sh 2>/dev/null || true
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh

conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# -------------------------
# nnU-Net paths
# -------------------------
export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export nnUNet_n_proc_DA=0   # fix for dataloader issue

REPO="/scratch/users/sastocke/nnunet_CHD"

echo "=== ENV SANITY ==="
echo "python: $(which python)"
python -c "import sys; print('sys.executable=', sys.executable); print('sys.prefix=', sys.prefix)"

# -------------------------
# Dataset & Architecture Config
# -------------------------
DATASET_NAME="Dataset001_all_imageCHD"
DATASET_ID=1

LOWRES_CONFIG="3d_lowres"
CASCADE_CONFIG="3d_cascade_fullres"
PLANS="nnUNetResEncUNetMPlans"

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"

# All 5 folds for training; fold 0 only for test-set prediction
TRAIN_FOLDS=(0 1 2 3 4)
PRED_FOLD=0   # fold used for single-fold test-set predictions

# -------------------------
# Preprocess both configs (safe to re-run; skips if already done)
# -------------------------
echo "=== PREPROCESSING ==="
nnUNetv2_preprocess \
    -d ${DATASET_ID} \
    -plans_name ${PLANS} \
    -c ${LOWRES_CONFIG} ${CASCADE_CONFIG} \
    -n 4 2

# -------------------------
# Trainer pairing
#
#   Baseline  : plain DA5 lowres         → CascadeFullresBaseline
#   FiLM      : DA5 + FiLM lowres        → CascadeFullresFiLM       (FiLM retained at high-res)
#   Topo      : DA5 + topology loss       → CascadeFullresTopo        (no FiLM at high-res)
#   FiLM+Topo : DA5 + FiLM + topo loss   → CascadeFullresFiLMTopo   (FiLM retained at high-res)
# -------------------------
declare -A CASCADE_TRAINER_MAP
CASCADE_TRAINER_MAP["nnUNetTrainerDA5_100epochs"]="nnUNetTrainerDA5CascadeFullresBaseline_100epochs"
CASCADE_TRAINER_MAP["nnUNetTrainerDA5CascadeFiLM_100epochs"]="nnUNetTrainerDA5CascadeFullresFiLM_100epochs"
CASCADE_TRAINER_MAP["nnUNetTrainerDA5CascadeTopo_100epochs"]="nnUNetTrainerDA5CascadeFullresTopo_100epochs"
CASCADE_TRAINER_MAP["nnUNetTrainerDA5CascadeFiLMTopo_100epochs"]="nnUNetTrainerDA5CascadeFullresFiLMTopo_100epochs"

LOWRES_TRAINERS=(
  "nnUNetTrainerDA5_100epochs"
  "nnUNetTrainerDA5CascadeFiLM_100epochs"
  "nnUNetTrainerDA5CascadeTopo_100epochs"
  "nnUNetTrainerDA5CascadeFiLMTopo_100epochs"
)

echo "=== CONFIG ==="
echo "DATASET=${DATASET_NAME} (id=${DATASET_ID})"
echo "PLANS=${PLANS}"
echo "TRAIN_FOLDS=${TRAIN_FOLDS[*]}"
echo "PRED_FOLD=${PRED_FOLD}"
echo "LOWRES_TRAINERS=${LOWRES_TRAINERS[*]}"

PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
mkdir -p "${PRED_BASE}"

# ================================================================
# PHASE 1 — Train ALL low-res trainers on ALL folds
#
# WHY: The cascade-fullres trainer for fold N uses lowres soft
# predictions as a spatial prior for its *training* cases, which
# are the validation cases of all OTHER folds.  Every fold must be
# trained before any cascade-fullres fold can start.
# The --npz flag saves soft predictions to
#   {trainer}__{plans}__3d_lowres/predicted_next_stage/3d_cascade_fullres/
# which the cascade trainer reads at the start of each training run.
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 1 — Low-Res Training: ALL trainers x ALL folds"
echo "================================================================"

for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  for FOLD in "${TRAIN_FOLDS[@]}"; do
    echo ""
    echo "--- LowRes: ${LR_TRAINER} | fold ${FOLD} ---"
    nnUNetv2_train \
      ${DATASET_ID} \
      ${LOWRES_CONFIG} \
      ${FOLD} \
      -tr ${LR_TRAINER} \
      -p ${PLANS} \
      --npz
  done
done

# ================================================================
# PHASE 2 — Symlink lowres priors into cascade trainer directories
#
# When lowres and cascade trainers have different class names nnU-Net
# cannot auto-locate the predicted_next_stage folder.  This script
# creates a symlink from the cascade trainer's expected location to
# where the lowres trainer actually wrote its predictions.
# Must run after ALL lowres folds complete so the folder exists.
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 2 — Symlink lowres priors for cascade trainers"
echo "================================================================"

for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"
  echo ""
  echo "--- Symlinking: ${LR_TRAINER} → ${CASCADE_TRAINER} ---"
  python "${REPO}/scripts/setup_cascade_predictions.py" \
    --lowres_trainer  "${LR_TRAINER}" \
    --cascade_trainer "${CASCADE_TRAINER}" \
    --dataset         "${DATASET_NAME}" \
    --plans           "${PLANS}" \
    --folds           "${TRAIN_FOLDS[@]}"
done

# ================================================================
# PHASE 3 — Train ALL cascade-fullres trainers on ALL folds
#
# Each cascade trainer reads the symlinked lowres predictions for
# the training cases of each fold.  With all 5 lowres folds done,
# every training case now has a lowres prior available.
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 3 — Cascade High-Res Training: ALL trainers x ALL folds"
echo "================================================================"

for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"
  for FOLD in "${TRAIN_FOLDS[@]}"; do
    echo ""
    echo "--- CascadeFullres: ${CASCADE_TRAINER} | fold ${FOLD} ---"
    nnUNetv2_train \
      ${DATASET_ID} \
      ${CASCADE_CONFIG} \
      ${FOLD} \
      -tr ${CASCADE_TRAINER} \
      -p ${PLANS} \
      --npz
  done
done

# ================================================================
# PHASE 4 — Test-set inference (lowres then cascade, per trainer)
#
# Uses fold PRED_FOLD only.  The cascade prediction step requires
# the lowres predictions as --prev_stage_predictions.
# FiLM trainers use predict_disease_conditioned (reads disease_map.json
# from the model folder and injects the disease vector at inference).
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 4 — Test-set predictions (fold ${PRED_FOLD})"
echo "================================================================"

for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"

  LR_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}/${LR_TRAINER}__${PLANS}__${LOWRES_CONFIG}"
  CASC_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}/${CASCADE_TRAINER}__${PLANS}__${CASCADE_CONFIG}"

  LR_PRED_DIR="${PRED_BASE}/${LR_TRAINER}/imagesTs/fold_${PRED_FOLD}"
  CASC_PRED_DIR="${PRED_BASE}/${CASCADE_TRAINER}/imagesTs/fold_${PRED_FOLD}"

  mkdir -p "${LR_PRED_DIR}" "${CASC_PRED_DIR}"

  # -------------------------------------------------------
  echo ""
  echo "--- Lowres predict: ${LR_TRAINER} | fold ${PRED_FOLD} ---"
  # -------------------------------------------------------
  if [[ "${LR_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" \
      -o "${LR_PRED_DIR}" \
      -m "${LR_MODEL_BASE}" \
      -f ${PRED_FOLD}
  else
    nnUNetv2_predict \
      -i "${IN_DIR}" \
      -o "${LR_PRED_DIR}" \
      -d ${DATASET_ID} \
      -c ${LOWRES_CONFIG} \
      -f ${PRED_FOLD} \
      -tr ${LR_TRAINER} \
      -p ${PLANS}
  fi

  # -------------------------------------------------------
  echo ""
  echo "--- Cascade predict: ${CASCADE_TRAINER} | fold ${PRED_FOLD} ---"
  # -------------------------------------------------------
  if [[ "${CASCADE_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" \
      -o "${CASC_PRED_DIR}" \
      -m "${CASC_MODEL_BASE}" \
      -f ${PRED_FOLD} \
      --prev_stage_predictions "${LR_PRED_DIR}"
  else
    nnUNetv2_predict \
      -i "${IN_DIR}" \
      -o "${CASC_PRED_DIR}" \
      -d ${DATASET_ID} \
      -c ${CASCADE_CONFIG} \
      -f ${PRED_FOLD} \
      -tr ${CASCADE_TRAINER} \
      -p ${PLANS} \
      -prev_stage_predictions "${LR_PRED_DIR}"
  fi

done

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE."
echo "================================================================"
echo ""
echo "Prediction folders (fold ${PRED_FOLD}):"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"
  echo "  lowres  : ${PRED_BASE}/${LR_TRAINER}/imagesTs/fold_${PRED_FOLD}"
  echo "  cascade : ${PRED_BASE}/${CASCADE_TRAINER}/imagesTs/fold_${PRED_FOLD}"
done
