#!/bin/bash
# =============================================================================
#  CHD_predict_dataset080_compare.sh
#  Predict the Dataset080 held-out expert cases with BOTH Dataset090 and Dataset091,
#  for each fold (0-4) AND the 5-fold ensemble, via the clinical resize route:
#      resize D080 -> 512x512x221 -> nnUNetv2_predict -> backproject to native (--no-lcc)
#
#  Output (native geometry, aligns with Dataset080/labelsTr for offline scoring):
#      Dataset080.../predictions/ds090_fold{0..4}/ , ds090_ensemble/
#      Dataset080.../predictions/ds091_fold{0..4}/ , ds091_ensemble/
#
#  --no-lcc on purpose: raw model output, so LCC post-processing can't mask the
#  090-vs-091 difference. Re-run backproject without --no-lcc offline if you want LCC.
#  Dice / per-label / violin / heatmap: done offline in the local notebook vs labelsTr.
# =============================================================================
#SBATCH --job-name=D080-cmp
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D080-cmp_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D080-cmp_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# --- env guard: fail LOUDLY if the env's python isn't active (a /scratch purge can delete the
#     interpreter, leaving dangling symlinks -> `python` falls back to system 2.7 -> f-string SyntaxError) ---
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python '+sys.version.split()[0]+' active, expected 3.10 from conda env nnunet310. The env did not activate (likely a scratch purge deleted the interpreter). Recreate the env before running; do not edit the code.')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

REPO="/scratch/users/sastocke/nnunet_CHD"
cd "${REPO}"

D080="Dataset080_ClinicalCaseSanjibDetailed"
D080_IMG="${nnUNet_raw}/${D080}/imagesTr"          # expert-labelled cases (GT in labelsTr)
D080_RESIZED="${nnUNet_raw}/${D080}/imagesTr_imagechd_grid"
PREDROOT="${nnUNet_raw}/${D080}/predictions"
GRIDROOT="${PREDROOT}/_grid512"

PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; TRAINER="nnUNetTrainerDA5_200epochs"
mkdir -p "${PREDROOT}" "${GRIDROOT}" "${REPO}/logs"

if ! ls "${D080_IMG}"/*_0000.nii.gz >/dev/null 2>&1; then
  echo "ERROR: no images in ${D080_IMG} (expected <case>_0000.nii.gz)"; exit 1
fi

# ---- resize D080 to the ImageCHD grid once (shared by all models/folds) ----
if ! ls "${D080_RESIZED}"/*_0000.nii.gz >/dev/null 2>&1; then
  echo "[resize] ${D080_IMG} -> ${D080_RESIZED} (512x512x221)"
  python tools/resize_to_imagechd_grid.py --input "${D080_IMG}" --output "${D080_RESIZED}" --overwrite
else echo "[resize] already done — skipping"; fi

# predict_one <dataset_id> <mtag> "<folds>" <out_tag>
predict_one () {
  local DSID="$1" MTAG="$2" FOLDSTR="$3" TAG="$4"
  local MODELROOT="${nnUNet_results}/Dataset$(printf '%03d' "${DSID}")_*/${TRAINER}__${PLANS}__${FULLRES}"
  # verify every requested fold has a final checkpoint
  local missing=0
  for f in ${FOLDSTR}; do
    ls ${MODELROOT}/fold_${f}/checkpoint_final.pth >/dev/null 2>&1 || { echo "  [skip ${TAG}] fold ${f} checkpoint missing"; missing=1; }
  done
  [ "${missing}" -eq 1 ] && return 0

  local GRID="${GRIDROOT}/${TAG}"
  local FINAL="${PREDROOT}/${TAG}"
  if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then echo "  [done ${TAG}] exists — skipping"; return 0; fi
  echo "  [predict ${TAG}] d=${DSID} folds='${FOLDSTR}'"
  mkdir -p "${GRID}"
  nnUNetv2_predict -i "${D080_RESIZED}" -o "${GRID}" -d "${DSID}" -c "${FULLRES}" \
      -tr "${TRAINER}" -p "${PLANS}" -f ${FOLDSTR} -chk checkpoint_final.pth
  python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" \
      --native-dir "${D080_IMG}" --output-dir "${FINAL}" --no-lcc --overwrite
  echo "  [predict ${TAG}] -> ${FINAL}"
}

for M in "90:ds090" "91:ds091"; do
  DSID="${M%%:*}"; MTAG="${M##*:}"
  echo "==== model ${MTAG} (Dataset${DSID}) ===="
  for f in 0 1 2 3 4; do predict_one "${DSID}" "${MTAG}" "${f}" "${MTAG}_fold${f}"; done
  predict_one "${DSID}" "${MTAG}" "0 1 2 3 4" "${MTAG}_ensemble"
done

echo "=============================================================="
echo "DONE. Dataset080 predictions (native, raw / no-LCC):"
echo "  ${PREDROOT}/ds090_fold{0..4}, ds090_ensemble"
echo "  ${PREDROOT}/ds091_fold{0..4}, ds091_ensemble"
echo "  GT for scoring: ${nnUNet_raw}/${D080}/labelsTr"
echo "  -> pull these to the local notebook for Dice / per-label / violin / heatmap."
echo "=============================================================="
