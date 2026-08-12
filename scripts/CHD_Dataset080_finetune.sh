#!/bin/bash
# =============================================================================
#  CHD_Dataset080_finetune.sh
#  Fine-tune the Dataset071 model on the 3 clinical cases (Dataset080).
#
#     Phase 0  verify Dataset080 exists + a Dataset071 checkpoint to load
#     Phase 1  extract fingerprint -> MOVE Dataset071's plans onto 080 (so the
#              architecture matches the pretrained weights) -> preprocess 080
#     Phase 2  train nnUNetTrainerDA5_finetune (100ep, lr 1e-3) on -f all,
#              initialised from Dataset071 fold-0 weights (-pretrained_weights)
#
#  Only 3 cases -> -f all (nnU-Net's fixed 250 iters/epoch already oversamples
#  them); DA5 augmentation + short low-LR schedule guard against overfitting.
#  RESUME: preprocess guarded by .done; training skips if checkpoint_final.pth
#          exists, resumes with --c otherwise.
# =============================================================================
#SBATCH --job-name=D080-finetune
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D080-finetune_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D080-finetune_%j.err

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
CLIN_ID=80
CLIN_NAME="Dataset080_ClinicalCaseSanjibDetailed"
SRC_NAME="Dataset071_ImageCHDClinicalOrientation"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
SRC_TRAINER="nnUNetTrainerDA5_200epochs"    # the trained Dataset071 model
FT_TRAINER="nnUNetTrainerDA5_finetune"
PRETRAINED="${nnUNet_results}/${SRC_NAME}/${SRC_TRAINER}__${PLANS}__${FULLRES}/fold_0/checkpoint_final.pth"

CKPT_DIR="${nnUNet_results}/${CLIN_NAME}/.checkpoints/finetune"
mkdir -p "${CKPT_DIR}" /scratch/users/sastocke/nnunet_CHD/logs
cd "${REPO}"

# ---- Phase 0: prerequisites ----
if [ ! -d "${nnUNet_raw}/${CLIN_NAME}/imagesTr" ]; then
  echo "ERROR: ${nnUNet_raw}/${CLIN_NAME} not found (build/place Dataset080 first)"; exit 1
fi
if [ ! -f "${PRETRAINED}" ]; then
  echo "ERROR: pretrained weights not found: ${PRETRAINED}"
  echo "       (train Dataset071 ${SRC_TRAINER} fold 0 first, or point PRETRAINED at checkpoint_best.pth)"; exit 1
fi

# ---- Phase 1: fingerprint -> move 071 plans -> preprocess (matches architecture) ----
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  echo "[Phase 1] extract_fingerprint + move ${SRC_NAME} plans -> ${CLIN_NAME} + preprocess"
  nnUNetv2_extract_fingerprint -d "${CLIN_ID}"
  nnUNetv2_move_plans_between_datasets -s "${SRC_NAME}" -t "${CLIN_ID}" -sp "${PLANS}" -tp "${PLANS}"
  nnUNetv2_preprocess -d "${CLIN_ID}" -plans_name "${PLANS}" -c "${FULLRES}"
  touch "${CKPT_DIR}/01_preprocess.done"
else
  echo "[Phase 1] preprocess already done — skipping"
fi

# ---- Phase 2: fine-tune on all 3 clinical cases ----
OUT="${nnUNet_results}/${CLIN_NAME}/${FT_TRAINER}__${PLANS}__${FULLRES}/fold_all"
if [ -f "${OUT}/checkpoint_final.pth" ]; then
  echo "[Phase 2] fine-tune already complete — skipping"
else
  CONT=""
  if [ -f "${OUT}/checkpoint_latest.pth" ]; then CONT="--c"; echo "[Phase 2] resuming fine-tune"; \
  else echo "[Phase 2] fine-tuning ${FT_TRAINER} from ${SRC_NAME} fold-0 weights"; fi
  if [ -n "${CONT}" ]; then
    nnUNetv2_train "${CLIN_ID}" "${FULLRES}" all -tr "${FT_TRAINER}" -p "${PLANS}" ${CONT}
  else
    nnUNetv2_train "${CLIN_ID}" "${FULLRES}" all -tr "${FT_TRAINER}" -p "${PLANS}" \
        -pretrained_weights "${PRETRAINED}"
  fi
fi

echo "=============================================================="
echo "DONE. Fine-tuned model: ${OUT}"
echo "Predict with: -d ${CLIN_ID} -tr ${FT_TRAINER} -p ${PLANS} -c ${FULLRES} -f all"
echo "=============================================================="
