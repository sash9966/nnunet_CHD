#!/bin/bash
# =============================================================================
#  CHD_predict_fanwei_ds072.sh
#  Resize Dataset012_Fanweidata images to the ImageCHD grid (512x512x221), then
#  run inference with the Dataset072 (multi-FOV) model.
#
#     Phase A  resize imagesTr -> imagesTr_imagechd_grid  (tools/resize_to_imagechd_grid.py)
#     Phase B  nnUNetv2_predict with Dataset072 -> predictions/ds072__grid512/ (+ manifest)
#
#  Standalone: own job name + own output folder, so it runs alongside other jobs.
#  Idempotent: skips resize / prediction if already present.
# =============================================================================
#SBATCH --job-name=CHD-pred-fanwei-072
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/CHD-pred-fanwei-072_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/CHD-pred-fanwei-072_%j.err

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
cd "${REPO}"

# ---- Fanwei image set ----
FANWEI_SRC="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/imagesTr"
FANWEI_RESIZED="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/imagesTr_imagechd_grid"
FANWEI_PRED="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/predictions"

# ---- Dataset072 model ----
DS_ID=72
DS_NAME="Dataset072_ImageCHDMultiFOV"
TR="nnUNetTrainerDA5_200epochs"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
FOLDS="0"
CHK="checkpoint_final.pth"        # switch to checkpoint_best.pth if you prefer
GRID_SIZE="512x512x221"; GRID_SPACING="1x1x1.9059633016586304"

OUT="${FANWEI_PRED}/ds072__grid512"
mkdir -p "${FANWEI_PRED}" /scratch/users/sastocke/nnunet_CHD/logs

# ---- prerequisites ----
if [ ! -d "${FANWEI_SRC}" ]; then
  echo "ERROR: ${FANWEI_SRC} not found"; exit 1
fi
FIRST_FOLD="${FOLDS%% *}"
CKPT="${nnUNet_results}/${DS_NAME}/${TR}__${PLANS}__${FULLRES}/fold_${FIRST_FOLD}/${CHK}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: Dataset072 checkpoint not found: ${CKPT}"
  echo "       (train Dataset072 first, or fix TR/FOLDS/CHK; try checkpoint_best.pth)"; exit 1
fi

# ---- Phase A: resize Fanwei -> ImageCHD grid ----
if ls "${FANWEI_RESIZED}"/*.nii.gz >/dev/null 2>&1; then
  echo "[resize] ${FANWEI_RESIZED} already present — skipping"
else
  echo "[resize] ${FANWEI_SRC} -> ${FANWEI_RESIZED}  (${GRID_SIZE}, spacing ${GRID_SPACING})"
  python tools/resize_to_imagechd_grid.py --input "${FANWEI_SRC}" --output "${FANWEI_RESIZED}" --overwrite
fi

# ---- Phase B: predict with Dataset072 ----
if ls "${OUT}"/*.nii.gz >/dev/null 2>&1; then
  echo "[predict] ${OUT} already has predictions — skipping"
else
  mkdir -p "${OUT}"
  echo "[predict] Dataset072 -d ${DS_ID} -tr ${TR} -f ${FOLDS} -chk ${CHK}  ->  ${OUT}"
  nnUNetv2_predict -i "${FANWEI_RESIZED}" -o "${OUT}" -d "${DS_ID}" -c "${FULLRES}" \
      -tr "${TR}" -p "${PLANS}" -f ${FOLDS} -chk "${CHK}"
  {
    echo "model_tag:    ds072"
    echo "dataset:      ${DS_ID}  (${DS_NAME})"
    echo "trainer:      ${TR}"
    echo "plans:        ${PLANS}"
    echo "config:       ${FULLRES}"
    echo "folds:        ${FOLDS}"
    echo "checkpoint:   ${CHK}"
    echo "image_set:    Dataset012_Fanweidata"
    echo "input_space:  grid512 (resized to ${GRID_SIZE}, spacing ${GRID_SPACING}; labels on resized grid)"
    echo "input_dir:    ${FANWEI_RESIZED}"
    echo "generated:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  } > "${OUT}/PREDICTION_INFO.txt"
fi

echo "=============================================================="
echo "DONE. Fanwei x Dataset072 predictions (resized grid): ${OUT}"
echo "To map back to native Fanwei spacing:"
echo "  python tools/backproject_predictions_to_native.py \\"
echo "    --pred-dir ${OUT} --native-dir ${FANWEI_SRC} \\"
echo "    --output-dir ${FANWEI_PRED}/ds072__native_labels"
echo "=============================================================="
