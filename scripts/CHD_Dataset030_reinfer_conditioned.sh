#!/bin/bash
# =============================================================================
#  CHD_Dataset030_reinfer_conditioned.sh
#  Re-run fold-0 inference for all AuxDiag and CrossAttn trainers using
#  predict_disease_conditioned.py (previous runs used nnUNetv2_predict which
#  ignored disease vectors for FiLMAuxDiag and all CrossAttn trainers).
#
#  Overwrites existing predictions in-place. Safe to resubmit — each trainer
#  writes a .done marker and is skipped if already completed.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-reinfer
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-reinfer_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-reinfer_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# Environment
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
# Configuration
# ─────────────────────────────────────────────
DATASET_NAME="Dataset030_imageCHD_HU"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/reinfer_conditioned"

mkdir -p "${CKPT_DIR}" "${PRED_BASE}"

TRAINERS=(
    # AuxDiag group
    "nnUNetTrainerDA5AuxDiag_200epochs"        # no conditioning → falls back to standard predict
    "nnUNetTrainerDA5AuxDiagTopo_200epochs"    # no conditioning → falls back to standard predict
    "nnUNetTrainerDA5FiLMAuxDiag_200epochs"    # FiLM → needs disease vec
    # CrossAttn group
    "nnUNetTrainerDA5CrossAttn_200epochs"      # cross-attn → needs disease vec
    "nnUNetTrainerDA5CrossAttnTopo_200epochs"  # cross-attn → needs disease vec
    "nnUNetTrainerDA5AuxDiagCrossAttn_200epochs" # cross-attn → needs disease vec
)

sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

echo ""
echo "Re-inference with disease conditioning — $(date '+%Y-%m-%d %H:%M:%S')"
echo "SLURM job: ${SLURM_JOB_ID:-manual}"
echo ""

for TRAINER in "${TRAINERS[@]}"; do
    KEY="reinfer_$(sk ${TRAINER})"
    OUT_DIR="${PRED_BASE}/$(sk ${TRAINER})"
    MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
    mkdir -p "${OUT_DIR}"

    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
        continue
    fi

    echo "================================================================"
    echo "${KEY}"
    echo "  model : ${MODEL_DIR}"
    echo "  output: ${OUT_DIR}"
    echo "================================================================"

    python -m nnunetv2.inference.predict_disease_conditioned \
        -i "${IN_DIR}" \
        -o "${OUT_DIR}" \
        -m "${MODEL_DIR}" \
        -f 0

    mark_done "${KEY}"
    echo "[DONE] ${KEY}"
done

echo ""
echo "All done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in ${PRED_BASE}/"
