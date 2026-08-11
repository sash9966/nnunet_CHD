#!/bin/bash
# =============================================================================
#  CHD_Dataset100_finalclinic.sh
#  CLINIC-FACING ALL-DATA model.  Dataset100 = Dataset091 + ALL Dataset080 cases
#  (Dataset080 intentionally in TRAINING).  Trains fold 'all' (no held-out).
#
#  *** NOT an evaluation model. Dataset080 is in training -> its Dice from this
#      model is NOT unbiased. For qualitative clinic feedback / deployment only. ***
#
#     Phase 0  BUILD Dataset100 (Dataset091 verbatim + all Dataset080 -> train)
#     Phase 1  plan_and_preprocess (--verify_dataset_integrity)
#     Phase 2  train fold 'all'  (same trainer/plans/config as Dataset091)
#     Phase 3  EXPORT clinic weights to a clearly-named folder
#     Phase 4  predict the clinic review cases (resize route, +LCC)
#     Phase 5  QC overlay sheets for visual inspection
# =============================================================================
#SBATCH --job-name=D100-clinic
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D100-clinic_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D100-clinic_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# --- env guard: fail LOUDLY if the conda python didn't activate (old python -> f-string SyntaxError) ---
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python '+sys.version.split()[0]+' active, expected 3.10 from conda env nnunet310. The env did NOT activate (this is why the build SyntaxErrors at the first f-string). Fix the environment, do not touch the code.')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

REPO="/scratch/users/sastocke/nnunet_CHD"
DATASET_ID=100
DATASET_NAME="Dataset100_FinalClinic"
SRC_DATASET="Dataset091_ImageCHDPseudoCombinedV2"
D080_NAME="Dataset080_ClincalCaseSanjibDetailed"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLD="all"

# clinic review input (what to segment for qualitative feedback). Default: the Dataset080 clinic
# cases. Point this at a fresh unlabeled clinic folder if you have one.
REVIEW_INPUT="${nnUNet_raw}/${D080_NAME}/imagesTr"

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/clinic"
mkdir -p "${CKPT_DIR}" "${REPO}/logs"
cd "${REPO}"

# ---- Phase 0: build ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0] building ${DATASET_NAME} (Dataset091 + all ${D080_NAME})"
  python tools/build_dataset100_finalclinic.py --nnunet-raw "${nnUNet_raw}" \
      --src-dataset "${SRC_DATASET}" --d080-name "${D080_NAME}" \
      --target-id "${DATASET_ID}" --target-name "FinalClinic" --overwrite
  touch "${CKPT_DIR}/00_build.done"
else echo "[Phase 0] build already done — skipping"; fi

# ---- Phase 1: plan & preprocess (authoritative integrity check) ----
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
else echo "[Phase 1] preprocess already done — skipping"; fi

# ---- Phase 2: train fold 'all' (ALL-DATA, no held-out) ----
OUT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FOLD}"
if [ -f "${OUT}/checkpoint_final.pth" ]; then
  echo "[Phase 2] fold '${FOLD}' already trained — skipping"
else
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[Phase 2] train ${TRAINER} fold '${FOLD}' ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
fi

# ---- Phase 3: export clinic weights (self-contained model folder) ----
EXPORT="${nnUNet_results}/${DATASET_NAME}/CLINIC_MODEL_allData"
if [ -f "${OUT}/checkpoint_final.pth" ] && [ ! -f "${EXPORT}/fold_all/checkpoint_final.pth" ]; then
  echo "[Phase 3] exporting clinic weights -> ${EXPORT}"
  mkdir -p "${EXPORT}/fold_all"
  cp "${OUT}/checkpoint_final.pth" "${EXPORT}/fold_all/"
  MODELDIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
  for f in plans.json dataset.json dataset_fingerprint.json; do
    [ -f "${MODELDIR}/${f}" ] && cp "${MODELDIR}/${f}" "${EXPORT}/" || true
  done
  cat > "${EXPORT}/HOW_TO_USE.txt" <<EOF
Clinic-facing ALL-DATA CHD model (Dataset100_FinalClinic).
Trained on ALL trusted data (Dataset091 + Dataset080). Qualitative review only —
Dataset080 was in training, so its Dice from this model is NOT an unbiased estimate.

Predict (input CT must be resized to the 512x512x221 ImageCHD grid first — see the repo
tools/resize_to_imagechd_grid.py; direct native-spacing inference is unreliable):
  nnUNetv2_predict -i <resized_ct_dir> -o <out_dir> -d ${DATASET_ID} -c ${FULLRES} \\
      -tr ${TRAINER} -p ${PLANS} -f all -chk checkpoint_final.pth
then resample predictions back to native with tools/backproject_predictions_to_native.py.
Labels: 0=bg 1=LV-BP 2=RV-BP 3=LA 4=RA 5=Myo 6=Ao 7=PA.
EOF
else echo "[Phase 3] export exists or model missing — skipping"; fi

# ---- Phase 4: predict clinic review cases (resize -> predict -> backproject +LCC) ----
RRESIZED="${nnUNet_raw}/${DATASET_NAME}/review_input_imagechd_grid"
PREDROOT="${nnUNet_raw}/${DATASET_NAME}/predictions"
GRID="${PREDROOT}/clinic_review__grid512"
FINAL="${PREDROOT}/clinic_review__grid2native_lcc"
if [ -f "${OUT}/checkpoint_final.pth" ] && ls "${REVIEW_INPUT}"/*_0000.nii.gz >/dev/null 2>&1; then
  echo "[Phase 4] predicting clinic review cases from ${REVIEW_INPUT}"
  mkdir -p "${PREDROOT}"
  ls "${RRESIZED}"/*_0000.nii.gz >/dev/null 2>&1 || \
    python tools/resize_to_imagechd_grid.py --input "${REVIEW_INPUT}" --output "${RRESIZED}" --overwrite
  if ! ls "${GRID}"/*.nii.gz >/dev/null 2>&1; then
    mkdir -p "${GRID}"
    nnUNetv2_predict -i "${RRESIZED}" -o "${GRID}" -d "${DATASET_ID}" -c "${FULLRES}" \
        -tr "${TRAINER}" -p "${PLANS}" -f all -chk checkpoint_final.pth
  fi
  ls "${FINAL}"/*.nii.gz >/dev/null 2>&1 || \
    python tools/backproject_predictions_to_native.py --pred-dir "${GRID}" \
        --native-dir "${REVIEW_INPUT}" --output-dir "${FINAL}" --overwrite
  echo "[Phase 4] clinic predictions (native +LCC) -> ${FINAL}"
else echo "[Phase 4] skipped (model or review input missing)"; fi

# ---- Phase 5: QC overlay sheets ----
QC="${PREDROOT}/clinic_review__qc"
if ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then
  echo "[Phase 5] QC overlays -> ${QC}"
  python tools/make_qc_overlays.py --image-dir "${REVIEW_INPUT}" --label-dir "${FINAL}" --output-dir "${QC}"
else echo "[Phase 5] skipped (no predictions)"; fi

echo "=============================================================="
echo "DONE. Clinic-facing all-data model (Dataset100_FinalClinic)."
echo "  weights (send to clinic): ${EXPORT}/"
echo "  clinic predictions:       ${FINAL}/"
echo "  QC sheets:                ${QC}/index.html"
echo "  REMINDER: Dataset080 is in TRAINING here — do NOT report its Dice as unbiased."
echo "=============================================================="
