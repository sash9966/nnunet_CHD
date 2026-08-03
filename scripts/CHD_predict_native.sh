#!/bin/bash
# =============================================================================
#  CHD_predict_native.sh
#  Predict on the NON-RESIZED (native-spacing) images. Standalone + different
#  job name + different output folders than CHD_predict_clinical.sh, so it runs
#  ALONGSIDE that (resized) job without touching it.
#
#  For each trained model, predicts each image set at its NATIVE spacing/FOV and
#  writes a PREDICTION_INFO.txt manifest recording exactly what it ran on:
#
#     predictions/<model>__native/     native input; labels align to the ORIGINAL CT
#
#  (No resizing here. The resized 512x512x221 predictions come from the other
#  script, which writes to predictions/<model>/.)
#
#  Idempotent: skips a folder that already has predictions; re-run to fill in
#  models that finish training later.
# =============================================================================
#SBATCH --job-name=CHD-predict-native
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/CHD-predict-native_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/CHD-predict-native_%j.err

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
PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
cd "${REPO}"

# ---- image sets: label, NATIVE source dir, prediction root ----
CLIN_LABEL="ClinicalImagesPHICleared"
CLIN_SRC="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/imagesTs"
CLIN_PRED="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/predictions"
FANWEI_LABEL="Dataset012_Fanweidata"
FANWEI_SRC="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/imagesTr"
FANWEI_PRED="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/predictions"
mkdir -p "${CLIN_PRED}" "${FANWEI_PRED}" /scratch/users/sastocke/nnunet_CHD/logs

# ---- model table (parallel arrays) ----
TAGS=(       "ds071"                                  "finetune080"                          "ds081" )
DSIDS=(      71                                       80                                     81 )
DSNAMES=(    "Dataset071_ImageCHDClinicalOrientation" "Dataset080_ClincalCaseSanjibDetailed" "Dataset081_ImageCHDplusClinical" )
TRS=(        "nnUNetTrainerDA5_200epochs"             "nnUNetTrainerDA5_finetune"            "nnUNetTrainerDA5_200epochs" )
FOLDSTRS=(   "0 1 2 3 4"                              "all"                                  "0" )
CHKS=(       "checkpoint_best.pth"                    "checkpoint_final.pth"                 "checkpoint_final.pth" )

# predict_native: run one model on one NATIVE input dir + write a manifest
# args: IN_DIR OUT_DIR IMGSET_LABEL
predict_native () {
  local in_dir="$1" out_dir="$2" imgset="$3"
  if ls "${out_dir}"/*.nii.gz >/dev/null 2>&1; then echo "    [skip] ${out_dir} already has predictions"; return; fi
  mkdir -p "${out_dir}"
  echo "    [predict] model=${TAG} set=${imgset} space=NATIVE  ->  ${out_dir}"
  nnUNetv2_predict -i "${in_dir}" -o "${out_dir}" -d "${DSID}" -c "${FULLRES}" -tr "${TR}" -p "${PLANS}" \
      -f ${FOLDSTR} -chk "${CHK}"
  {
    echo "model_tag:    ${TAG}"
    echo "dataset:      ${DSID}  (${DSN})"
    echo "trainer:      ${TR}"
    echo "plans:        ${PLANS}"
    echo "config:       ${FULLRES}"
    echo "folds:        ${FOLDSTR}"
    echo "checkpoint:   ${CHK}"
    echo "image_set:    ${imgset}"
    echo "input_space:  NATIVE (no resize; native spacing/FOV; labels align to the ORIGINAL CT)"
    echo "input_dir:    ${in_dir}"
    echo "generated:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  } > "${out_dir}/PREDICTION_INFO.txt"
}

# ---- for each trained model, predict both image sets at native spacing ----
for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"; DSID="${DSIDS[$i]}"; DSN="${DSNAMES[$i]}"
  TR="${TRS[$i]}"; FOLDSTR="${FOLDSTRS[$i]}"; CHK="${CHKS[$i]}"
  FIRST_FOLD="${FOLDSTR%% *}"
  CKPT="${nnUNet_results}/${DSN}/${TR}__${PLANS}__${FULLRES}/fold_${FIRST_FOLD}/${CHK}"
  if [ ! -f "${CKPT}" ]; then
    echo "[${TAG}] not trained yet (${CKPT} missing) — skipping"; continue
  fi
  echo "[${TAG}] predicting NATIVE on both sets"
  predict_native "${CLIN_SRC}"   "${CLIN_PRED}/${TAG}__native"   "${CLIN_LABEL}"
  predict_native "${FANWEI_SRC}" "${FANWEI_PRED}/${TAG}__native" "${FANWEI_LABEL}"
done

echo "=============================================================="
echo "DONE. Native (non-resized) predictions — each folder has PREDICTION_INFO.txt:"
echo "  ${CLIN_PRED}/<model>__native/"
echo "  ${FANWEI_PRED}/<model>__native/"
echo "(Resized 512x512x221 predictions come from CHD_predict_clinical.sh -> predictions/<model>/.)"
echo "=============================================================="
