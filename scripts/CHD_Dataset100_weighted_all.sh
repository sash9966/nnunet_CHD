#!/bin/bash
# =============================================================================
#  CHD_Dataset100_weighted_all.sh
#  Clinic-facing ALL-DATA model, CASE-WEIGHTED, fold 'all', 500 epochs.
#  Same data as Dataset100 (Dataset091 + all Dataset080) but the TRAIN dataloader
#  samples cases by trust: Dataset080 expert 3x, ImageCHD GT 1x, QC'd promoted 1x,
#  usable clinical pseudo 0.5x, Fanwei pseudo 0.5x (FOV/volume diversity, not trust).
#  Distinct trainer name -> separate results folder from the unweighted D100 run.
#
#  build (emits case_weights.json) -> preprocess -> copy weights to preprocessed ->
#  train fold 'all' (nnUNetTrainerDA5CaseWeighted_500epochs) -> export -> predict
#  clinic (NATIVE, no resize) -> QC overlays.
# =============================================================================
#SBATCH --job-name=D100-wt-all
#SBATCH --partition=bioe        # long 500e job; if bioe is backlogged switch to: gpu (check walltime)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D100-wt-all_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D100-wt-all_%j.err

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

DATASET_ID=100
DATASET_NAME="Dataset100_FinalClinic"
SRC_DATASET="Dataset091_ImageCHDPseudoCombinedV2"
D080_NAME="Dataset080_ClinicalCaseSanjibDetailed"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5CaseWeighted_500epochs"; FOLD="all"
REVIEW_INPUT="${nnUNet_raw}/${D080_NAME}/imagesTr"     # clinic cases to segment for review (native)

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/weighted"
mkdir -p "${CKPT_DIR}"

# ---- Phase 0a: ensure Dataset091 exists (build in-job if missing) ----
PROMOTED="CT_052_7910,CT_528_0579,CT_584_09_no,CT_731_6,CT_747_68,CT_754_49,CT_860_8,CT_914_49,BAF007"
EXCLUDE="BAF004,CHIPS002,CHIPS016,BAF005,CT_704_49,CT_853_56_no,CT_881_8,CT_110_69,CT_335_058,CT_790_069,CT_793_0569_no"
if [ ! -f "${nnUNet_raw}/${SRC_DATASET}/dataset.json" ]; then
  echo "[Phase 0a] building ${SRC_DATASET}"
  python tools/build_dataset091_from_090.py --nnunet-raw "${nnUNet_raw}" \
      --src-dataset "Dataset090_ImageCHDPseudoCombined" --target-id 91 \
      --target-name "ImageCHDPseudoCombinedV2" --promoted "${PROMOTED}" --exclude "${EXCLUDE}" --overwrite
else echo "[Phase 0a] ${SRC_DATASET} already built"; fi

# ---- Phase 0b: build Dataset100 + case_weights.json (D080 3x, Fanwei 0.5x, rest per defaults) ----
if [ ! -f "${CKPT_DIR}/00_build.done" ]; then
  echo "[Phase 0b] building ${DATASET_NAME} (+ case_weights.json)"
  python tools/build_dataset100_finalclinic.py --nnunet-raw "${nnUNet_raw}" \
      --src-dataset "${SRC_DATASET}" --d080-name "${D080_NAME}" \
      --target-id "${DATASET_ID}" --target-name "FinalClinic" \
      --w-imagechd 1.0 --w-d080 3.0 --w-promoted 1.0 --w-fanwei 0.5 --w-clinical 0.5 --overwrite
  touch "${CKPT_DIR}/00_build.done"
else echo "[Phase 0b] build already done"; fi

# ---- Phase 1: plan & preprocess ----
if [ ! -f "${CKPT_DIR}/01_preprocess.done" ]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  touch "${CKPT_DIR}/01_preprocess.done"
else echo "[Phase 1] preprocess already done"; fi

# ---- Phase 1c: make case_weights.json findable by the mixin (raw + preprocessed base) ----
cp -f "${nnUNet_raw}/${DATASET_NAME}/case_weights.json" \
      "${nnUNet_preprocessed}/${DATASET_NAME}/case_weights.json" 2>/dev/null || true
echo "[Phase 1c] case_weights.json present: raw=$( [ -f "${nnUNet_raw}/${DATASET_NAME}/case_weights.json" ] && echo yes || echo NO )  preprocessed=$( [ -f "${nnUNet_preprocessed}/${DATASET_NAME}/case_weights.json" ] && echo yes || echo NO )"

# ---- Phase 2: train fold 'all' with the case-weighted trainer ----
OUT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}/fold_${FOLD}"
if [ -f "${OUT}/checkpoint_final.pth" ]; then
  echo "[Phase 2] fold '${FOLD}' already trained — skipping"
else
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[Phase 2] train ${TRAINER} fold '${FOLD}' ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
fi

# ---- Phase 3: export clinic weights ----
EXPORT="${nnUNet_results}/${DATASET_NAME}/CLINIC_MODEL_weighted_all"
MODELDIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
if [ -f "${OUT}/checkpoint_final.pth" ] && [ ! -f "${EXPORT}/fold_all/checkpoint_final.pth" ]; then
  echo "[Phase 3] exporting -> ${EXPORT}"
  mkdir -p "${EXPORT}/fold_all"; cp "${OUT}/checkpoint_final.pth" "${EXPORT}/fold_all/"
  for j in plans.json dataset.json dataset_fingerprint.json; do
    [ -f "${MODELDIR}/${j}" ] && cp "${MODELDIR}/${j}" "${EXPORT}/" || true
  done
  cat > "${EXPORT}/HOW_TO_USE.txt" <<EOF
Clinic-facing ALL-DATA CHD model (Dataset100), CASE-WEIGHTED, fold 'all', 500 epochs.
Trained on ALL trusted data with per-case sampling weights (D080 expert 3x, ImageCHD 1x,
QC'd promoted 1x, clinical/Fanwei pseudo 0.5x). Qualitative review only — Dataset080 was in
training, so its Dice from this model is NOT unbiased.
Predict NATIVE (no resize needed; nnU-Net resamples internally):
  nnUNetv2_predict -i <ct_dir> -o <out> -d ${DATASET_ID} -c ${FULLRES} -tr ${TRAINER} -p ${PLANS} -f all -chk checkpoint_final.pth
Labels: 0=bg 1=LV-BP 2=RV-BP 3=LA 4=RA 5=Myo 6=Ao 7=PA.
EOF
else echo "[Phase 3] export exists or model missing — skipping"; fi

# ---- Phase 4: predict clinic review cases (NATIVE) ----
FINAL="${nnUNet_raw}/${DATASET_NAME}/predictions/clinic_review_weighted_native"
if [ -f "${OUT}/checkpoint_final.pth" ] && ls "${REVIEW_INPUT}"/*_0000.nii.gz >/dev/null 2>&1; then
  if ! ls "${FINAL}"/*.nii.gz >/dev/null 2>&1; then
    echo "[Phase 4] predicting clinic review (native) -> ${FINAL}"
    mkdir -p "${FINAL}"
    nnUNetv2_predict -i "${REVIEW_INPUT}" -o "${FINAL}" -d "${DATASET_ID}" -c "${FULLRES}" \
        -tr "${TRAINER}" -p "${PLANS}" -f all -chk checkpoint_final.pth
  fi
  # ---- Phase 5: QC overlays ----
  python tools/make_qc_overlays.py --image-dir "${REVIEW_INPUT}" --label-dir "${FINAL}" \
      --output-dir "${nnUNet_raw}/${DATASET_NAME}/predictions/clinic_review_weighted_qc"
else echo "[Phase 4/5] skipped (model or review input missing)"; fi

echo "=============================================================="
echo "DONE. Case-weighted all-data clinic model (Dataset100)."
echo "  model:   ${MODELDIR}/fold_all/"
echo "  weights: ${EXPORT}/  (predict -f all)"
echo "  clinic:  ${FINAL}/  + QC sheets alongside"
echo "  REMINDER: Dataset080 is in TRAINING (weighted 3x) — its Dice here is NOT unbiased."
echo "=============================================================="
