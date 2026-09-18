#!/bin/bash
# =============================================================================
#  CHD_Dataset200_mri.sh
#  Dataset200_MRI: 17 seven-label MRI cases; ResEnc-M / DA5 / 200 epochs.
#  Train fold all on every MRI, then export the model for review.
#  Same environment, phase markers and provenance as the existing CHD scripts.
#  Added 2026-09-18. Submit this script directly with sbatch.
# =============================================================================
#SBATCH --job-name=D200-MRI-all
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D200-MRI-all_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D200-MRI-all_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: wrong Python; activate nnunet310')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Sherlock nnunet310 lacks Python.h needed by torch.compile/Triton.
# Use ordinary CUDA training; compilation is an optional optimization.
export nnUNet_compile=false
echo "[env] nnUNet_compile=${nnUNet_compile} (CUDA training without JIT compilation)"
REPO="/scratch/users/sastocke/nnunet_CHD"
cd "${REPO}"

DATASET_ID=200
DATASET_NAME="Dataset200_MRI"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"; FOLDS=(all)
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/mri"
MODELDIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
mkdir -p "${CKPT_DIR}" "${REPO}/logs"
# Do not run the two D200 scripts simultaneously (shared preprocessing/unpacking).
exec 9>"${CKPT_DIR}/run.lock"
flock -n 9 || { echo 'Another Dataset200 job is running. Resubmit after it finishes.' >&2; exit 1; }
source scripts/_provenance.sh
RUN_RECORD="${CKPT_DIR}/runs/all_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_RECORD}"
cp "scripts/CHD_Dataset200_mri.sh" "${RUN_RECORD}/"
stamp_provenance "D200-all" "${RUN_RECORD}" \
    "DATASET=${DATASET_NAME}" "TRAINER=${TRAINER}" "PLANS=${PLANS}" "FOLDS=${FOLDS[*]}" "nnUNet_compile=${nnUNet_compile}"
python -m pip freeze > "${RUN_RECORD}/packages.txt"

# ---- Phase 0: verify the uploaded seven-label MRI dataset ----
[ -f "${nnUNet_raw}/${DATASET_NAME}/dataset.json" ] || { echo 'Dataset200_MRI is missing from nnUNet_raw' >&2; exit 1; }
(cd "${nnUNet_raw}/${DATASET_NAME}" && sha256sum -c SHA256SUMS)
cp "${nnUNet_raw}/${DATASET_NAME}/SHA256SUMS" "${RUN_RECORD}/"
cp "${nnUNet_raw}/${DATASET_NAME}/dataset.json" "${RUN_RECORD}/"
python - <<'CHECK'
import json, os
from pathlib import Path
p = Path(os.environ['nnUNet_raw']) / 'Dataset200_MRI'
m = json.loads((p / 'dataset.json').read_text())
assert m['numTraining'] == 17 and m['channel_names'] == {'0': 'MRI'}, m
assert set(m['labels'].values()) == set(range(8)), m['labels']
images = {f.name[:-12] for f in (p / 'imagesTr').glob('*_0000.nii.gz')}
labels = {f.name[:-7] for f in (p / 'labelsTr').glob('*.nii.gz')}
assert images == labels and len(images) == 17, 'Expected 17 matching image/label pairs'
CHECK

# ---- Phase 1: plan and preprocess once; resume through the existing marker ----
if [ -f "${CKPT_DIR}/01_preprocess.done" ]; then
  cmp "${RUN_RECORD}/SHA256SUMS" "${CKPT_DIR}/SHA256SUMS" || { echo 'Dataset changed since preprocessing; inspect before restarting' >&2; exit 1; }
  [ -f "${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}.json" ] || { echo 'Preprocessed plans missing; inspect phase marker' >&2; exit 1; }
  echo '[Phase 1] preprocessing already done — skipping'
else
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl "${PLANNER}" -c "${FULLRES}" --verify_dataset_integrity
  cp "${RUN_RECORD}/SHA256SUMS" "${CKPT_DIR}/SHA256SUMS"
  touch "${CKPT_DIR}/01_preprocess.done"
fi
cp "${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}.json" "${RUN_RECORD}/"

# ---- Phase 2: train; completed folds skip, interrupted folds resume ----
for FOLD in "${FOLDS[@]}"; do
  OUT="${MODELDIR}/fold_${FOLD}"
  if [ -f "${OUT}/checkpoint_final.pth" ]; then
    echo "[skip] fold ${FOLD} complete"
    if [ "${FOLD}" != all ] && [ ! -f "${OUT}/validation/summary.json" ]; then
      nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" --val
    fi
    continue
  fi
  CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  stamp_provenance "D200-all-fold${FOLD}" "${OUT}" "TRAINER=${TRAINER}" "PLANS=${PLANS}" "RUN_RECORD=${RUN_RECORD}" "nnUNet_compile=${nnUNet_compile}"
  echo "[Phase 2] train ${TRAINER} fold ${FOLD} ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
done
if [ -f "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json" ]; then
  cp "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json" "${RUN_RECORD}/"
fi

# ---- Phase 3: export the all-case model for review/handoff ----
EXPORT="${nnUNet_results}/${DATASET_NAME}/CLINIC_MODEL_all"
mkdir -p "${EXPORT}/fold_all"
cp "${MODELDIR}/fold_all/checkpoint_final.pth" "${EXPORT}/fold_all/"
for j in plans.json dataset.json dataset_fingerprint.json; do
  [ ! -f "${MODELDIR}/${j}" ] || cp "${MODELDIR}/${j}" "${EXPORT}/"
done
stamp_provenance "D200-MRI-export-all" "${EXPORT}" "TRAINER=${TRAINER}" "PLANS=${PLANS}" "RUN_RECORD=${RUN_RECORD}" "nnUNet_compile=${nnUNet_compile}"
cat > "${EXPORT}/HOW_TO_USE.txt" <<USAGE
Dataset200_MRI: 17 MRI cases, seven foreground classes, DA5 200 epochs, fold all.
All cases were used for training. Any fold_all validation is IN-SAMPLE.
Use the separate five-fold run for held-out performance; clinical review is required.
Native MRI prediction (CASE_0000.nii.gz; no CT-grid resize):
  nnUNetv2_predict -i <mri_dir> -o <out_dir> -d 200 -c ${FULLRES} -tr ${TRAINER} -p ${PLANS} -f all -chk checkpoint_final.pth
The target environment must contain this repository's trainer implementation.
Labels: 0=bg 1=LV-BP 2=RV-BP 3=LA 4=RA 5=Myo 6=Aorta 7=Pulmonary.
Run record: ${RUN_RECORD}
USAGE
echo "DONE. All-case MRI weights: ${EXPORT}"
echo 'Next: sbatch scripts/CHD_Dataset200_train5fold.sh'
