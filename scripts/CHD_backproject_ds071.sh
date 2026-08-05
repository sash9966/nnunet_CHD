#!/bin/bash
# =============================================================================
#  CHD_backproject_ds071.sh
#  Turn the Dataset071 grid512 predictions into NATIVE-spacing, LCC-cleaned labels
#  aligned to the original CTs -- the seed labels for the correct-and-retrain loop.
#
#  For each image set: grid512 prediction (predictions/ds071) --> resample back to
#  the native grid (guided by the original image's geometry) --> largest-connected-
#  component cleanup per label --> predictions/ds071__native_labels/<case>.nii.gz
#
#  NOTE: this step is CPU-bound (scipy zoom + connected components); a GPU is
#  allocated only because it's requested -- remove --gpus to schedule faster.
# =============================================================================
#SBATCH --job-name=CHD-backproj-071
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/CHD-backproj-071_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/CHD-backproj-071_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
REPO="/scratch/users/sastocke/nnunet_CHD"
cd "${REPO}"
mkdir -p /scratch/users/sastocke/nnunet_CHD/logs

# backproject <pred_dir> <native_dir> <out_dir>  (skips if pred_dir missing)
backproject () {
  local pred="$1" native="$2" out="$3"
  if [ ! -d "${pred}" ]; then echo "[skip] ${pred} not found"; return; fi
  echo "[backproject] ${pred} -> ${out}  (native geom from ${native}, +LCC)"
  python tools/backproject_predictions_to_native.py \
      --pred-dir "${pred}" --native-dir "${native}" --output-dir "${out}" --overwrite
}

FANWEI="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata"
CLIN="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared"

# Fanwei (the 50-case bootstrap set)
backproject "${FANWEI}/predictions/ds071" "${FANWEI}/imagesTr" "${FANWEI}/predictions/ds071__native_labels"
# Clinical set
backproject "${CLIN}/predictions/ds071"   "${CLIN}/imagesTs"   "${CLIN}/predictions/ds071__native_labels"

echo "=============================================================="
echo "DONE. Native-spacing, LCC'd seed labels:"
echo "  ${FANWEI}/predictions/ds071__native_labels/"
echo "  ${CLIN}/predictions/ds071__native_labels/"
echo "Overlay each on its ORIGINAL image (imagesTr/imagesTs *_0000.nii.gz), then send the keep/drop list."
echo "=============================================================="
