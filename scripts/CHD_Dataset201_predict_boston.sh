#!/bin/bash
# =============================================================================
#  CHD_Dataset201_predict_boston.sh
#  Dataset201_ClinicalBostonMRI: 27 seven-label MRI cases; ResEnc-M / DA5 / 100 epochs.
#  Predict remaining 50 Boston cases, score six labels, prepare myocardium review.
#  Same environment, phase markers and provenance as the existing CHD scripts.
#  Added 2026-09-18. Submit this script directly with sbatch.
# =============================================================================
#SBATCH --job-name=D201-Boston-predict
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D201-Boston-predict_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D201-Boston-predict_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: wrong Python; activate nnunet310')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Sherlock nnunet310 lacks Python.h needed by torch.compile/Triton.
# Use ordinary CUDA training; compilation is an optional optimization.
export nnUNet_compile=false
echo "[env] nnUNet_compile=${nnUNet_compile} (CUDA training without JIT compilation)"
REPO="/scratch/users/sastocke/nnunet_CHD"
cd "${REPO}"

DATASET_NAME="Dataset201_ClinicalBostonMRI"
TRAINER="nnUNetTrainerDA5_100epochs"
PLANS="nnUNetResEncUNetMPlans"
MODEL="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__3d_fullres"
CHECKPOINT="${MODEL}/fold_all/checkpoint_final.pth"
OUTPUT="${nnUNet_raw}/${DATASET_NAME}/predictions/ds201__native"
REVIEW="${nnUNet_results}/${DATASET_NAME}/Boston50_review"
[ -f "${CHECKPOINT}" ] || { echo 'Train Dataset201 fold all first' >&2; exit 1; }
[ ! -e "${REVIEW}" ] || { echo 'Review folder already exists; preserving edits' >&2; exit 1; }
mkdir -p "${OUTPUT}"
source scripts/_provenance.sh
stamp_provenance "D201-Boston-predict" "${OUTPUT}" "CHECKPOINT=${CHECKPOINT}" "TRAINER=${TRAINER}"
sha256sum "${CHECKPOINT}" > "${OUTPUT}/teacher_SHA256.txt"
nnUNetv2_predict -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" -o "${OUTPUT}" \
  -d 201 -c 3d_fullres -tr "${TRAINER}" -p "${PLANS}" -f all -chk checkpoint_final.pth
python tools/mri_boston.py pseudo --dataset "${nnUNet_raw}/${DATASET_NAME}" \
  --predictions "${OUTPUT}" --checkpoint "${CHECKPOINT}" --output "${REVIEW}"
echo "Review full volumes before promotion: ${REVIEW}"
