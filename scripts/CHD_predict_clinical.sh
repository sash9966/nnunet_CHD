#!/bin/bash
# =============================================================================
#  CHD_predict_clinical.sh
#  Resize the two clinical image sets to the ImageCHD grid, then run inference
#  with every model that is already trained. Predictions go into per-model
#  subfolders so you can tell which model produced which labels.
#
#     Phase A  resize -> ImageCHD grid (512x512x221) via tools/resize_to_imagechd_grid.py
#     Phase B  for each AVAILABLE model (checkpoint exists) predict BOTH sets
#
#  Models attempted (skipped automatically if not trained yet):
#     ds071        Dataset071 DA5_200e, 5-fold ENSEMBLE, checkpoint_best   (quick win, no training)
#     finetune080  Dataset080 fine-tuned, -f all, checkpoint_final
#     ds081        Dataset081 mix DA5_200e, fold 0, checkpoint_final
#  Re-run after each model finishes training; done sets are skipped.
# =============================================================================
#SBATCH --job-name=CHD-predict-clinical
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/CHD-predict-clinical_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/CHD-predict-clinical_%j.err

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

# ---- image sets: source, resized dir, prediction root ----
CLIN_SRC="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/imagesTs"
CLIN_RESIZED="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/imagesTs_imagechd_grid"
CLIN_PRED="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/predictions"
FANWEI_SRC="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/imagesTr"
FANWEI_RESIZED="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/imagesTr_imagechd_grid"
FANWEI_PRED="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw/Dataset012_Fanweidata/predictions"
mkdir -p "${CLIN_PRED}" "${FANWEI_PRED}" /scratch/users/sastocke/nnunet_CHD/logs

# ---- Phase A: resize both sets to the ImageCHD grid (once) ----
resize_set () {  # $1=src  $2=out
  if ls "$2"/*.nii.gz >/dev/null 2>&1; then echo "[resize] $2 already present — skipping"; return; fi
  echo "[resize] $1 -> $2"
  python tools/resize_to_imagechd_grid.py --input "$1" --output "$2" --overwrite
}
resize_set "${CLIN_SRC}" "${CLIN_RESIZED}"
resize_set "${FANWEI_SRC}" "${FANWEI_RESIZED}"

# ---- model table (parallel arrays) ----
TAGS=(       "ds071"                              "finetune080"                        "ds081" )
DSIDS=(      71                                   80                                   81 )
DSNAMES=(    "Dataset071_ImageCHDClinicalOrientation" "Dataset080_ClinicalCaseSanjibDetailed" "Dataset081_ImageCHDplusClinical" )
TRS=(        "nnUNetTrainerDA5_200epochs"         "nnUNetTrainerDA5_finetune"          "nnUNetTrainerDA5_200epochs" )
FOLDSTRS=(   "0 1 2 3 4"                          "all"                                "0" )
CHKS=(       "checkpoint_best.pth"                "checkpoint_final.pth"               "checkpoint_final.pth" )

predict_set () {  # $1=in  $2=out  + model env: DSID TR FOLDSTR CHK
  if ls "$2"/*.nii.gz >/dev/null 2>&1; then echo "  [skip] $2 already has predictions"; return; fi
  mkdir -p "$2"
  echo "  [predict] -d ${DSID} -tr ${TR} -f ${FOLDSTR} -chk ${CHK}  ->  $2"
  nnUNetv2_predict -i "$1" -o "$2" -d "${DSID}" -c "${FULLRES}" -tr "${TR}" -p "${PLANS}" \
      -f ${FOLDSTR} -chk "${CHK}"
}

# ---- Phase B: for each trained model, predict both resized sets ----
for i in "${!TAGS[@]}"; do
  TAG="${TAGS[$i]}"; DSID="${DSIDS[$i]}"; DSN="${DSNAMES[$i]}"
  TR="${TRS[$i]}"; FOLDSTR="${FOLDSTRS[$i]}"; CHK="${CHKS[$i]}"
  FIRST_FOLD="${FOLDSTR%% *}"                       # first fold in the list
  CKPT="${nnUNet_results}/${DSN}/${TR}__${PLANS}__${FULLRES}/fold_${FIRST_FOLD}/${CHK}"
  if [ ! -f "${CKPT}" ]; then
    echo "[${TAG}] not trained yet (${CKPT} missing) — skipping"; continue
  fi
  echo "[${TAG}] predicting both sets"
  predict_set "${CLIN_RESIZED}"   "${CLIN_PRED}/${TAG}"
  predict_set "${FANWEI_RESIZED}" "${FANWEI_PRED}/${TAG}"
done

echo "=============================================================="
echo "DONE. Predictions:"
echo "  clinical: ${CLIN_PRED}/<model>/"
echo "  fanwei  : ${FANWEI_PRED}/<model>/"
echo "Re-run after finetune080 / ds081 finish training to fill in those subfolders."
echo "=============================================================="
