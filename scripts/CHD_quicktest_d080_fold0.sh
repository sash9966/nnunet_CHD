#!/bin/bash
# =============================================================================
#  CHD_quicktest_d080_fold0.sh
#  QUICK fold-0 sanity check on the held-out Dataset080, NATIVE inference only:
#    predict D080 in its native spacing/FOV with D090 fold0 and D091 fold0
#    (nnUNetv2_predict resamples input->plan spacing->back internally), then Dice
#    vs labelsTr (tools/dice_analysis_d080.py, matches dice_analysis.ipynb).
#  Tells us: does native inference already work, and is D091 (pseudo-labels) > D090?
#  Full 5-fold + ensemble comes later via CHD_predict_dataset080_compare.sh.
# =============================================================================
#SBATCH --job-name=D080-f0-quick
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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
PREDROOT="${nnUNet_raw}/${D080}/predictions"
PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; TRAINER="nnUNetTrainerDA5_200epochs"; CHK="checkpoint_final.pth"
mkdir -p "${PREDROOT}"

ls "${D080_IMG}"/*_0000.nii.gz >/dev/null 2>&1 || { echo "ERROR: no images in ${D080_IMG}"; exit 1; }
ls "${D080_GT}"/*.nii.gz       >/dev/null 2>&1 || { echo "ERROR: no GT labels in ${D080_GT}"; exit 1; }

# predict_native <dataset_id> <full_dataset_name> <tag>  — direct native inference (no resize)
predict_native () {
  local DSID="$1" DSNAME="$2" TAG="$3"
  local CKPT="${nnUNet_results}/${DSNAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_0/${CHK}"
  if [ ! -f "${CKPT}" ]; then echo "  [skip ${TAG}] fold 0 checkpoint missing: ${CKPT}"; return 0; fi
  local FINAL="${PREDROOT}/${TAG}"
  if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then echo "  [done ${TAG}] exists"; return 0; fi
  echo "  [predict ${TAG}] NATIVE  d=${DSID} fold 0"
  mkdir -p "${FINAL}"
  nnUNetv2_predict -i "${D080_IMG}" -o "${FINAL}" -d "${DSID}" -c "${FULLRES}" \
      -tr "${TRAINER}" -p "${PLANS}" -f 0 -chk "${CHK}"
}

predict_native 90 "Dataset090_ImageCHDPseudoCombined"   "ds090_native_f0"
predict_native 91 "Dataset091_ImageCHDPseudoCombinedV2" "ds091_native_f0"

# ---- Dice + plots (matches dice_analysis.ipynb); baseline = D090 native ----
python tools/dice_analysis_d080.py --gt-dir "${D080_GT}" \
    --pred "D090 native=${PREDROOT}/ds090_native_f0" \
    --pred "D091 native=${PREDROOT}/ds091_native_f0" \
    --baseline "D090 native" \
    --out-dir "${PREDROOT}/dice_fold0"

echo "=============================================================="
echo "DONE (fold-0 quick test, NATIVE inference)."
echo "  predictions: ${PREDROOT}/ds090_native_f0/ , ds091_native_f0/"
echo "  Dice + plots: ${PREDROOT}/dice_fold0/  (summary_*.csv, delta_median.csv,"
echo "                dice_violin*.png, delta_heatmap.png)"
echo "  READ: if D091 > D090, the pseudo-labels are already helping (native inference)."
echo "  NOTE: fold-0 only — full 5-fold ensemble later via CHD_predict_dataset080_compare.sh."
echo "=============================================================="
