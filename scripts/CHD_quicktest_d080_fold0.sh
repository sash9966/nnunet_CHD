#!/bin/bash
# =============================================================================
#  CHD_quicktest_d080_fold0.sh
#  QUICK fold-0 sanity check on the held-out Dataset080, two questions at once:
#    (1) does Dataset091 (pseudo-labels) beat Dataset090?
#    (2) NATIVE inference vs the resize->512^3 route — now that D090/D091 train on
#        native-FOV clinical/Fanwei data, does direct native inference already work?
#  So we predict FOUR ways (D090/D091 x native/grid), all in D080's native geometry, and
#  Dice them vs labelsTr (tools/dice_analysis_d080.py, matches dice_analysis.ipynb).
#
#    native: nnUNetv2_predict directly on D080/imagesTr  (nnU-Net resamples internally + back)
#    grid  : resize -> 512^3 -> predict -> backproject to native (--no-lcc)
#  Full 5-fold + ensemble comes later via CHD_predict_dataset080_compare.sh.
# =============================================================================
#SBATCH --job-name=D080-f0-quick
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D080-f0-quick_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D080-f0-quick_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: wrong python (env not active); recreate the env first')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
REPO="/scratch/users/sastocke/nnunet_CHD"; cd "${REPO}"; mkdir -p "${REPO}/logs"

D080="Dataset080_ClinicalCaseSanjibDetailed"
D080_IMG="${nnUNet_raw}/${D080}/imagesTr"
D080_GT="${nnUNet_raw}/${D080}/labelsTr"
RESIZED="${nnUNet_raw}/${D080}/imagesTr_imagechd_grid"
PREDROOT="${nnUNet_raw}/${D080}/predictions"
GRIDROOT="${PREDROOT}/_grid512"
PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; TRAINER="nnUNetTrainerDA5_200epochs"; CHK="checkpoint_final.pth"
mkdir -p "${PREDROOT}" "${GRIDROOT}"

ls "${D080_IMG}"/*_0000.nii.gz >/dev/null 2>&1 || { echo "ERROR: no images in ${D080_IMG}"; exit 1; }
ls "${D080_GT}"/*.nii.gz       >/dev/null 2>&1 || { echo "ERROR: no GT labels in ${D080_GT}"; exit 1; }

ckpt_of () { echo "${nnUNet_results}/$1/${TRAINER}__${PLANS}__${FULLRES}/fold_0/${CHK}"; }

# predict_native <dataset_id> <full_dataset_name> <tag>  — direct native inference (no resize)
predict_native () {
  local DSID="$1" DSNAME="$2" TAG="$3"
  local CKPT; CKPT="$(ckpt_of "${DSNAME}")"
  if [ ! -f "${CKPT}" ]; then echo "  [skip ${TAG}] fold 0 checkpoint missing"; return 0; fi
  local FINAL="${PREDROOT}/${TAG}"
  if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then echo "  [done ${TAG}] exists"; return 0; fi
  echo "  [predict ${TAG}] NATIVE  d=${DSID} fold 0"
  mkdir -p "${FINAL}"
  nnUNetv2_predict -i "${D080_IMG}" -o "${FINAL}" -d "${DSID}" -c "${FULLRES}" \
      -tr "${TRAINER}" -p "${PLANS}" -f 0 -chk "${CHK}"
}

# predict_grid <dataset_id> <full_dataset_name> <tag>  — resize -> predict -> backproject
predict_grid () {
  local DSID="$1" DSNAME="$2" TAG="$3"
  local CKPT; CKPT="$(ckpt_of "${DSNAME}")"
  if [ ! -f "${CKPT}" ]; then echo "  [skip ${TAG}] fold 0 checkpoint missing"; return 0; fi
  ls "${RESIZED}"/*_0000.nii.gz >/dev/null 2>&1 || \
    python tools/resize_to_imagechd_grid.py --input "${D080_IMG}" --output "${RESIZED}" --overwrite
  local GRID="${GRIDROOT}/${TAG}" FINAL="${PREDROOT}/${TAG}"
  if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then echo "  [done ${TAG}] exists"; return 0; fi
  echo "  [predict ${TAG}] GRID  d=${DSID} fold 0"
  mkdir -p "${GRID}"
  nnUNetv2_predict -i "${RESIZED}" -o "${GRID}" -d "${DSID}" -c "${FULLRES}" \
      -tr "${TRAINER}" -p "${PLANS}" -f 0 -chk "${CHK}"
  python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" \
      --native-dir "${D080_IMG}" --output-dir "${FINAL}" --no-lcc --overwrite
}

predict_native 90 "Dataset090_ImageCHDPseudoCombined"   "ds090_native_f0"
predict_native 91 "Dataset091_ImageCHDPseudoCombinedV2" "ds091_native_f0"
predict_grid   90 "Dataset090_ImageCHDPseudoCombined"   "ds090_grid_f0"
predict_grid   91 "Dataset091_ImageCHDPseudoCombinedV2" "ds091_grid_f0"

# ---- Dice + plots (matches dice_analysis.ipynb); baseline = D090 native ----
python tools/dice_analysis_d080.py --gt-dir "${D080_GT}" \
    --pred "D090 native=${PREDROOT}/ds090_native_f0" \
    --pred "D091 native=${PREDROOT}/ds091_native_f0" \
    --pred "D090 grid=${PREDROOT}/ds090_grid_f0" \
    --pred "D091 grid=${PREDROOT}/ds091_grid_f0" \
    --baseline "D090 native" \
    --out-dir "${PREDROOT}/dice_fold0"

echo "=============================================================="
echo "DONE (fold-0 quick test: native vs grid, D090 vs D091)."
echo "  predictions: ${PREDROOT}/ds09{0,1}_{native,grid}_f0/"
echo "  Dice + plots: ${PREDROOT}/dice_fold0/  (summary_*.csv, delta_median.csv,"
echo "                dice_violin*.png, delta_heatmap.png)"
echo "  READ: if 'D09x native' ~= or > 'D09x grid', native inference works -> drop the resize."
echo "        if 'D091' > 'D090', the pseudo-labels are already helping."
echo "  NOTE: fold-0 only — full 5-fold ensemble later via CHD_predict_dataset080_compare.sh."
echo "=============================================================="
