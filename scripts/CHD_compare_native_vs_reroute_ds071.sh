#!/bin/bash
# =============================================================================
#  CHD_compare_native_vs_reroute_ds071.sh
#  For BOTH Fanwei and the clinical set, produce two NATIVE-geometry label sets
#  from the Dataset071 model so you can overlay each on the ORIGINAL CT and judge:
#
#     A) ds071__native            direct inference on the NATIVE image (expected to break)
#     B) ds071__grid2native_lcc   resize->512x512x221 -> predict -> resample back
#                                 to native (guided by the original image) -> LCC
#
#  Per set the steps are: (A) native predict | (B1) resize | (B2) grid512 predict |
#  (B3) backproject+LCC. Idempotent: a step is skipped if its output already exists
#  (delete a folder to force a fresh run). Needs a GPU for the two predict steps.
#
#  FOLDS="0" = fast single-fold VERIFICATION (default). Set "0 1 2 3 4" for the full
#  5-fold ensemble (much slower, esp. the native-direct pass on big volumes).
# =============================================================================
#SBATCH --job-name=CHD-cmp-nvr-071
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/CHD-cmp-nvr-071_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/CHD-cmp-nvr-071_%j.err

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
REPO="/scratch/users/sastocke/nnunet_CHD"; cd "${REPO}"
mkdir -p /scratch/users/sastocke/nnunet_CHD/logs

# ---- Dataset071 model ----
DS_ID=71
DS_NAME="Dataset071_ImageCHDClinicalOrientation"
TR="nnUNetTrainerDA5_200epochs"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
FOLDS="0"                         # single-fold verification; "0 1 2 3 4" = full ensemble
CHK="checkpoint_best.pth"
GRID_SIZE="512x512x221"

FIRST_FOLD="${FOLDS%% *}"
CKPT="${nnUNet_results}/${DS_NAME}/${TR}__${PLANS}__${FULLRES}/fold_${FIRST_FOLD}/${CHK}"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: Dataset071 checkpoint not found: ${CKPT} (fix FOLDS/CHK, try checkpoint_final.pth)"; exit 1
fi

manifest () {  # $1=out_dir  $2=space  $3=imgset  $4=input_dir
  {
    echo "model:       ${DS_ID} (${DS_NAME}) | ${TR} | ${PLANS} | ${FULLRES} | folds ${FOLDS} | ${CHK}"
    echo "image_set:   $3"
    echo "input_space: $2"
    echo "input_dir:   $4"
    echo "generated:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  } > "$1/PREDICTION_INFO.txt"
}

# run_set <label> <native_src> <resized_dir> <pred_root>
run_set () {
  local label="$1" src="$2" resized="$3" proot="$4"
  echo "=================  ${label}  ================="
  if [ ! -d "${src}" ]; then echo "[skip ${label}] ${src} not found"; return; fi
  local NAT="${proot}/ds071__native"
  local GRID="${proot}/ds071__grid512"
  local REROUTE="${proot}/ds071__grid2native_lcc"
  mkdir -p "${proot}"

  # A) direct inference on NATIVE images
  if ls "${NAT}"/*.nii.gz >/dev/null 2>&1; then
    echo "[A native] ${NAT} exists — skipping"
  else
    echo "[A native] predict on native ${src}"
    mkdir -p "${NAT}"
    nnUNetv2_predict -i "${src}" -o "${NAT}" -d "${DS_ID}" -c "${FULLRES}" -tr "${TR}" -p "${PLANS}" -f ${FOLDS} -chk "${CHK}"
    manifest "${NAT}" "NATIVE (direct, no resize)" "${label}" "${src}"
  fi

  # B1) resize native -> ImageCHD grid
  if ls "${resized}"/*.nii.gz >/dev/null 2>&1; then
    echo "[B1 resize] ${resized} exists — skipping"
  else
    echo "[B1 resize] ${src} -> ${resized} (${GRID_SIZE})"
    python tools/resize_to_imagechd_grid.py --input "${src}" --output "${resized}" --overwrite
  fi

  # B2) predict on resized (grid512)
  if ls "${GRID}"/*.nii.gz >/dev/null 2>&1; then
    echo "[B2 grid512] ${GRID} exists — skipping"
  else
    echo "[B2 grid512] predict on resized ${resized}"
    mkdir -p "${GRID}"
    nnUNetv2_predict -i "${resized}" -o "${GRID}" -d "${DS_ID}" -c "${FULLRES}" -tr "${TR}" -p "${PLANS}" -f ${FOLDS} -chk "${CHK}"
    manifest "${GRID}" "grid512 (resized ${GRID_SIZE})" "${label}" "${resized}"
  fi

  # B3) backproject grid512 -> native + LCC
  if ls "${REROUTE}"/*.nii.gz >/dev/null 2>&1; then
    echo "[B3 reroute] ${REROUTE} exists — skipping"
  else
    echo "[B3 reroute] backproject ${GRID} -> ${REROUTE} (+LCC)"
    python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" --native-dir "${src}" \
        --output-dir "${REROUTE}" --overwrite
  fi
}

FANWEI="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata"
CLIN="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared"

run_set "Fanwei"   "${FANWEI}/imagesTr" "${FANWEI}/imagesTr_imagechd_grid" "${FANWEI}/predictions"
run_set "Clinical" "${CLIN}/imagesTs"   "${CLIN}/imagesTs_imagechd_grid"   "${CLIN}/predictions"

echo "=============================================================="
echo "DONE. Compare (both in NATIVE geometry — overlay on the ORIGINAL *_0000.nii.gz):"
echo "  A) direct-native : predictions/ds071__native/"
echo "  B) reroute+LCC   : predictions/ds071__grid2native_lcc/"
echo "(intermediate resized-grid preds: predictions/ds071__grid512/)"
echo "FOLDS=${FOLDS} — set to '0 1 2 3 4' for the full ensemble."
echo "=============================================================="
