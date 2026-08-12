#!/bin/bash
# =============================================================================
#  CHD_regen_ds071_fanwei_labels.sh
#  Regenerate the purged ds071 LCC pseudo-labels for Fanwei (+ clinical) by re-running
#  the Dataset071 model:  resize -> 512^3 -> nnUNetv2_predict -> backproject + LCC.
#  Writes back to the ORIGINAL paths so Dataset090's labelsTr symlinks resolve again:
#      <root>/predictions/ds071             (grid512 preds)
#      <root>/predictions/ds071__grid2native_lcc   (native + LCC = the labels D090 uses)
# =============================================================================
#SBATCH --job-name=regen-ds071
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/regen-ds071_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/regen-ds071_%j.err

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

DS_ID=71; DS_NAME="Dataset071_ImageCHDClinicalOrientation"
TR="nnUNetTrainerDA5_200epochs"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; CHK="checkpoint_final.pth"
MODELDIR="${nnUNet_results}/${DS_NAME}/${TR}__${PLANS}__${FULLRES}"

# --- detect which folds of the ds071 model survive ---
FOLDS=""
for f in 0 1 2 3 4; do [ -f "${MODELDIR}/fold_${f}/${CHK}" ] && FOLDS="${FOLDS} ${f}"; done
FOLDS="$(echo ${FOLDS} | xargs)"
if [ -z "${FOLDS}" ]; then
  echo "FATAL: no Dataset071 checkpoints under ${MODELDIR}"
  echo "       The ds071 model was purged too. Restore it from OAK"
  echo "       (/oak/stanford/groups/amarsden/sastocke/nnUNet_results/${DS_NAME}) or retrain ds071 first."
  exit 1
fi
echo "[ds071] regenerating labels with folds: ${FOLDS}"

# process <root_dir> <images_subdir>
process () {
  local ROOT="$1" IMG="$2"
  if [ ! -d "${IMG}" ]; then echo "[skip] ${IMG} not found"; return; fi
  local RESIZED="${IMG}_imagechd_grid"
  local GRID="${ROOT}/predictions/ds071"
  local FINAL="${ROOT}/predictions/ds071__grid2native_lcc"
  mkdir -p "${ROOT}/predictions"
  echo "[resize]  ${IMG} -> ${RESIZED}"
  python tools/resize_to_imagechd_grid.py --input "${IMG}" --output "${RESIZED}" --overwrite
  echo "[predict] ds071 folds '${FOLDS}' -> ${GRID}"
  nnUNetv2_predict -i "${RESIZED}" -o "${GRID}" -d "${DS_ID}" -c "${FULLRES}" \
      -tr "${TR}" -p "${PLANS}" -f ${FOLDS} -chk "${CHK}"
  echo "[backproject+LCC] -> ${FINAL}"
  python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" \
      --native-dir "${IMG}" --output-dir "${FINAL}" --overwrite
  echo "[done] ${FINAL}"
}

process "${nnUNet_raw}/Dataset012_Fanweidata"        "${nnUNet_raw}/Dataset012_Fanweidata/imagesTr"
process "${REPO}/ClinicalImagesPHICleared"           "${REPO}/ClinicalImagesPHICleared/imagesTs"

echo "=============================================================="
echo "DONE. Regenerated ds071 LCC labels at the original paths:"
echo "  ${nnUNet_raw}/Dataset012_Fanweidata/predictions/ds071__grid2native_lcc/"
echo "  ${REPO}/ClinicalImagesPHICleared/predictions/ds071__grid2native_lcc/"
echo "  -> now re-run: bash scripts/fix_d090_symlinks.sh   (the labelsTr should resolve)"
echo "=============================================================="
