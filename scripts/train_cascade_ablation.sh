#!/bin/bash
set -euo pipefail   # abort immediately on any error; treat unset vars as errors
#SBATCH --job-name=CHD_Cascade
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

# Fold 0 only for the initial comparison run
FOLD=0

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
# Lowres trainer → Cascade fullres trainer mapping
#
# Each pair writes to a unique results folder so all four ablations
# can be compared side-by-side without overwriting each other.
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
echo "FOLD=${FOLD}"
echo "LOWRES_TRAINERS=${LOWRES_TRAINERS[*]}"

# Output layout:
#   predictions/
#     {LR_TRAINER}/imagesTs/fold_0/        ← lowres test predictions
#     {CASCADE_TRAINER}/imagesTs/fold_0/   ← cascade test predictions
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
mkdir -p "${PRED_BASE}"

# -------------------------
# Main Training & Inference Loop
# -------------------------
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"

  echo "================================================================"
  echo "PIPELINE: LowRes (${LR_TRAINER}) -> Cascade (${CASCADE_TRAINER})"
  echo "================================================================"

  LR_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}/${LR_TRAINER}__${PLANS}__${LOWRES_CONFIG}"
  CASC_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}/${CASCADE_TRAINER}__${PLANS}__${CASCADE_CONFIG}"

  # Prediction output folders — trainer name first for easy comparison
  LR_PRED_DIR="${PRED_BASE}/${LR_TRAINER}/imagesTs/fold_${FOLD}"
  CASC_PRED_DIR="${PRED_BASE}/${CASCADE_TRAINER}/imagesTs/fold_${FOLD}"

  # -------------------------------------------------------
  echo "FOLD ${FOLD}: 1. Low-Res Training (${LR_TRAINER})"
  # -------------------------------------------------------
  nnUNetv2_train \
    ${DATASET_ID} \
    ${LOWRES_CONFIG} \
    ${FOLD} \
    -tr ${LR_TRAINER} \
    -p ${PLANS} \
    --npz

  # -------------------------------------------------------
  echo "FOLD ${FOLD}: 2. Low-Res Prediction on Test Set"
  # -------------------------------------------------------
  mkdir -p "${LR_PRED_DIR}"

  if [[ "${LR_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" \
      -o "${LR_PRED_DIR}" \
      -m "${LR_MODEL_BASE}" \
      -f ${FOLD}
  else
    nnUNetv2_predict \
      -i "${IN_DIR}" \
      -o "${LR_PRED_DIR}" \
      -d ${DATASET_ID} \
      -c ${LOWRES_CONFIG} \
      -f ${FOLD} \
      -tr ${LR_TRAINER} \
      -p ${PLANS}
  fi

  # -------------------------------------------------------
  echo "FOLD ${FOLD}: 3. Symlinking Lowres Priors for Cascade Training"
  # -------------------------------------------------------
  # Symlinks fold_0/validation/ (written by --npz during step 1) into the
  # cascade trainer folder so it can use them as spatial priors during training.
  python "${REPO}/scripts/setup_cascade_predictions.py" \
    --lowres_trainer "${LR_TRAINER}" \
    --cascade_trainer "${CASCADE_TRAINER}" \
    --dataset "${DATASET_NAME}" \
    --plans "${PLANS}" \
    --folds ${FOLD}

  # -------------------------------------------------------
  echo "FOLD ${FOLD}: 4. Cascade High-Res Training (${CASCADE_TRAINER})"
  # -------------------------------------------------------
  nnUNetv2_train \
    ${DATASET_ID} \
    ${CASCADE_CONFIG} \
    ${FOLD} \
    -tr ${CASCADE_TRAINER} \
    -p ${PLANS} \
    --npz

  # -------------------------------------------------------
  echo "FOLD ${FOLD}: 5. Cascade High-Res Prediction on Test Set"
  # -------------------------------------------------------
  mkdir -p "${CASC_PRED_DIR}"

  if [[ "${CASCADE_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" \
      -o "${CASC_PRED_DIR}" \
      -m "${CASC_MODEL_BASE}" \
      -f ${FOLD} \
      --prev_stage_predictions "${LR_PRED_DIR}"
  else
    nnUNetv2_predict \
      -i "${IN_DIR}" \
      -o "${CASC_PRED_DIR}" \
      -d ${DATASET_ID} \
      -c ${CASCADE_CONFIG} \
      -f ${FOLD} \
      -tr ${CASCADE_TRAINER} \
      -p ${PLANS} \
      -prev_stage_predictions "${LR_PRED_DIR}"
  fi

done

echo ""
echo "PIPELINE COMPLETE."
echo ""
echo "Prediction folders:"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  CASCADE_TRAINER="${CASCADE_TRAINER_MAP[$LR_TRAINER]}"
  echo "  lowres  : ${PRED_BASE}/${LR_TRAINER}/imagesTs/fold_${FOLD}"
  echo "  cascade : ${PRED_BASE}/${CASCADE_TRAINER}/imagesTs/fold_${FOLD}"
done
