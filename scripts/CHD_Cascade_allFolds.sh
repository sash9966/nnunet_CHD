#!/bin/bash
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

set -euo pipefail

# -------------------------
# 1. Environment
# -------------------------
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONUNBUFFERED=1

# -------------------------
# 2. Config
# -------------------------
DATASET_ID=1
DATASET_NAME="Dataset001_all_imageCHD"
PLANS="nnUNetResEncUNetMPlans"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"

LOWRES_TRAINERS=(
  "nnUNetTrainerDA5_100epochs"
  "nnUNetTrainerDA5CascadeFiLM_100epochs"
  "nnUNetTrainerDA5CascadeTopo_100epochs"
  "nnUNetTrainerDA5CascadeFiLMTopo_100epochs"
)

CASCADE_TRAINERS=(
  "nnUNetTrainerDA5CascadeFullresBaseline_100epochs"
  "nnUNetTrainerDA5CascadeFullresFiLM_100epochs"
  "nnUNetTrainerDA5CascadeFullresTopo_100epochs"
  "nnUNetTrainerDA5CascadeFullresFiLMTopo_100epochs"
)

# -------------------------
# Phase 1: Train all lowres trainers x all folds
# -------------------------
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "--- LowRes: ${LR_TRAINER} | fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${LOWRES} ${FOLD} -tr ${LR_TRAINER} -p ${PLANS} --npz
  done
done

# -------------------------
# Phase 2: Symlink lowres priors into cascade trainer directories
# -------------------------
for i in "${!LOWRES_TRAINERS[@]}"; do
  python "${REPO}/scripts/setup_cascade_predictions.py" \
    --lowres_trainer  "${LOWRES_TRAINERS[$i]}" \
    --cascade_trainer "${CASCADE_TRAINERS[$i]}" \
    --dataset         "${DATASET_NAME}" \
    --plans           "${PLANS}" \
    --folds           0 1 2 3 4
done

# -------------------------
# Phase 3: Train all cascade trainers x all folds
# -------------------------
for i in "${!CASCADE_TRAINERS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    echo "--- Cascade: ${CASCADE_TRAINERS[$i]} | fold ${FOLD} ---"
    nnUNetv2_train ${DATASET_ID} ${CASCADE} ${FOLD} -tr ${CASCADE_TRAINERS[$i]} -p ${PLANS} --npz
  done
done

# -------------------------
# Phase 4: Test-set predictions (fold 0 only)
# -------------------------
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
mkdir -p "${PRED_BASE}"

for i in "${!LOWRES_TRAINERS[@]}"; do
  LR_TRAINER="${LOWRES_TRAINERS[$i]}"
  CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"
  LR_MODEL="${nnUNet_results}/${DATASET_NAME}/${LR_TRAINER}__${PLANS}__${LOWRES}"
  CASC_MODEL="${nnUNet_results}/${DATASET_NAME}/${CASCADE_TRAINER}__${PLANS}__${CASCADE}"
  LR_PRED="${PRED_BASE}/${LR_TRAINER}/fold_0"
  CASC_PRED="${PRED_BASE}/${CASCADE_TRAINER}/fold_0"
  mkdir -p "${LR_PRED}" "${CASC_PRED}"

  if [[ "${LR_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" -o "${LR_PRED}" -m "${LR_MODEL}" -f 0
  else
    nnUNetv2_predict -i "${IN_DIR}" -o "${LR_PRED}" \
      -d ${DATASET_ID} -c ${LOWRES} -f 0 -tr ${LR_TRAINER} -p ${PLANS}
  fi

  if [[ "${CASCADE_TRAINER}" == *"FiLM"* ]]; then
    python -m nnunetv2.inference.predict_disease_conditioned \
      -i "${IN_DIR}" -o "${CASC_PRED}" -m "${CASC_MODEL}" -f 0 \
      --prev_stage_predictions "${LR_PRED}"
  else
    nnUNetv2_predict -i "${IN_DIR}" -o "${CASC_PRED}" \
      -d ${DATASET_ID} -c ${CASCADE} -f 0 -tr ${CASCADE_TRAINER} -p ${PLANS} \
      -prev_stage_predictions "${LR_PRED}"
  fi
done

echo "Done."
