#!/bin/bash
# =============================================================================
#  CHD_Dataset201_train5fold.sh
#  Dataset201_ClinicalBostonMRI: 27 seven-label MRI cases; ResEnc-M / DA5 / 100 epochs.
#  Train folds 0–4 sequentially; held-out validation uses native MRI labels.
#  Same environment, phase markers and provenance as the existing CHD scripts.
#  Added 2026-09-18. Submit this script directly with sbatch.
# =============================================================================
#SBATCH --job-name=D201-MRI-5fold
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D201-MRI-5fold_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D201-MRI-5fold_%j.err

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

DATASET_ID=201
DATASET_NAME="Dataset201_ClinicalBostonMRI"
PLANNER="nnUNetPlannerResEncM"; PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_100epochs"; FOLDS=(0 1 2 3 4)
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/mri"
MODELDIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
mkdir -p "${CKPT_DIR}"
source scripts/_provenance.sh
RUN_RECORD="${CKPT_DIR}/runs/5fold_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_RECORD}"
cp "scripts/CHD_Dataset201_train5fold.sh" "${RUN_RECORD}/"
stamp_provenance "D201-5fold" "${RUN_RECORD}" \
    "DATASET=${DATASET_NAME}" "TRAINER=${TRAINER}" "PLANS=${PLANS}" "FOLDS=${FOLDS[*]}" "nnUNet_compile=${nnUNet_compile}"
python -m pip freeze > "${RUN_RECORD}/packages.txt"

# ---- Phase 0: verify the uploaded seven-label MRI dataset ----
[ -f "${nnUNet_raw}/${DATASET_NAME}/dataset.json" ] || { echo 'Dataset201_ClinicalBostonMRI is missing from nnUNet_raw' >&2; exit 1; }
(cd "${nnUNet_raw}/${DATASET_NAME}" && sha256sum -c SHA256SUMS)
cp "${nnUNet_raw}/${DATASET_NAME}/SHA256SUMS" "${RUN_RECORD}/"
cp "${nnUNet_raw}/${DATASET_NAME}/dataset.json" "${RUN_RECORD}/"
python - <<'CHECK'
import json, os
from pathlib import Path
p = Path(os.environ['nnUNet_raw']) / 'Dataset201_ClinicalBostonMRI'
m = json.loads((p / 'dataset.json').read_text())
assert m['numTraining'] == 27 and m['channel_names'] == {'0': 'MRI'}, m
assert set(m['labels'].values()) == set(range(8)), m['labels']
images = {f.name[:-12] for f in (p / 'imagesTr').glob('*_0000.nii.gz')}
labels = {f.name[:-7] for f in (p / 'labelsTr').glob('*.nii.gz')}
assert images == labels and len(images) == 27, 'Expected 27 matching image/label pairs'
CHECK

# ---- Phase 1: plan and preprocess once; resume through the existing marker ----
# Only shared preparation is serialized; training releases this lock.
exec 9>"${CKPT_DIR}/run.lock"
echo '[Phase 1] waiting for shared preparation, if another job is preparing data'
flock 9
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

# Finish first-time unpacking before either job starts training. Use the same
# dataset-class API as this repository's trainer (NumPy or Blosc2).
python -c '
import os, sys
from pathlib import Path
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
base = Path(os.environ["nnUNet_preprocessed"]) / sys.argv[1]
configuration = PlansManager(str(base / (sys.argv[2] + ".json"))).get_configuration(sys.argv[3])
folder = str(base / configuration.data_identifier)
infer_dataset_class(folder).unpack_dataset(folder, overwrite_existing=False, num_processes=4, verify=True)
' "${DATASET_NAME}" "${PLANS}" "${FULLRES}"
python tools/mri_boston.py splits \
  --dataset "${nnUNet_raw}/${DATASET_NAME}" \
  --baseline "${nnUNet_preprocessed}/Dataset200_MRI/splits_final.json" \
  --output "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
cp "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json" "${RUN_RECORD}/"
flock -u 9
exec 9>&-
echo '[Phase 1] shared preparation complete; training can run concurrently'

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
  stamp_provenance "D201-5fold-fold${FOLD}" "${OUT}" "TRAINER=${TRAINER}" "PLANS=${PLANS}" "RUN_RECORD=${RUN_RECORD}" "nnUNet_compile=${nnUNet_compile}"
  echo "[Phase 2] train ${TRAINER} fold ${FOLD} ${CONT}"
  nnUNetv2_train "${DATASET_ID}" "${FULLRES}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" ${CONT}
done
if [ -f "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json" ]; then
  cp "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json" "${RUN_RECORD}/"
fi

echo "DONE. Five-fold MRI results: ${MODELDIR}/fold_0 through fold_4"
echo 'Held-out scores are in each fold/validation/summary.json.'
echo 'The fold_all scores are in-sample and must not be used as a held-out comparison.'
