#!/bin/bash
# =============================================================================
#  CHD_quicktest_d080_fold0.sh
#  QUICK fold-0 sanity check: does Dataset091 (pseudo-labels) already beat Dataset090
#  on the held-out Dataset080, using only FOLD 0 of each? (Full 5-fold + ensemble comes
#  later via CHD_predict_dataset080_compare.sh.)
#
#  resize D080 -> 512^3 -> predict (D090 f0, D091 f0) -> backproject native (--no-lcc)
#  -> Dice + violin + Δ-heatmap (tools/dice_analysis_d080.py, matches dice_analysis.ipynb).
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

# ---- resize D080 once ----
ls "${RESIZED}"/*_0000.nii.gz >/dev/null 2>&1 || \
  python tools/resize_to_imagechd_grid.py --input "${D080_IMG}" --output "${RESIZED}" --overwrite

# predict_f0 <dataset_id> <full_dataset_name> <tag>
predict_f0 () {
  local DSID="$1" DSNAME="$2" TAG="$3"
  local CKPT="${nnUNet_results}/${DSNAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_0/${CHK}"
  if [ ! -f "${CKPT}" ]; then echo "  [skip ${TAG}] fold 0 checkpoint missing: ${CKPT}"; return 0; fi
  local GRID="${GRIDROOT}/${TAG}" FINAL="${PREDROOT}/${TAG}"
  if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then echo "  [done ${TAG}] exists"; return 0; fi
  echo "  [predict ${TAG}] d=${DSID} fold 0"
  mkdir -p "${GRID}"
  nnUNetv2_predict -i "${RESIZED}" -o "${GRID}" -d "${DSID}" -c "${FULLRES}" \
      -tr "${TRAINER}" -p "${PLANS}" -f 0 -chk "${CHK}"
  python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" \
      --native-dir "${D080_IMG}" --output-dir "${FINAL}" --no-lcc --overwrite
}

predict_f0 90 "Dataset090_ImageCHDPseudoCombined"   "ds090_fold0"
predict_f0 91 "Dataset091_ImageCHDPseudoCombinedV2" "ds091_fold0"

# ---- Dice + plots (matches dice_analysis.ipynb) ----
python tools/dice_analysis_d080.py --gt-dir "${D080_GT}" \
    --pred "Dataset090 (f0)=${PREDROOT}/ds090_fold0" \
    --pred "Dataset091 (f0)=${PREDROOT}/ds091_fold0" \
    --baseline "Dataset090 (f0)" \
    --out-dir "${PREDROOT}/dice_fold0"

echo "=============================================================="
echo "DONE (fold-0 quick test)."
echo "  predictions: ${PREDROOT}/ds090_fold0/ , ds091_fold0/"
echo "  Dice + plots: ${PREDROOT}/dice_fold0/  (summary_*.csv, delta_median.csv,"
echo "                dice_violin_090_vs_091.png, delta_heatmap.png)"
echo "  NOTE: fold-0 only — rerun the full 5-fold ensemble comparison once all folds are in."
echo "=============================================================="
