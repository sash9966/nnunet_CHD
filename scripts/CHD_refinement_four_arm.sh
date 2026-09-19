#!/bin/bash
# Explicit stages: init | seqseg | nni | assemble | build | preprocess | train | evaluate
# Examples and environment settings: docs/refinement_four_arm.md
#SBATCH --job-name=CHD-refine4
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/refine4_%A_%a.out
#SBATCH --error=logs/refine4_%A_%a.err
set -euo pipefail
REPO="${REPO:-/scratch/users/sastocke/nnunet_CHD}"
RUN="${RUN:-$REPO/refinement_runs/accepted50_conservative_v3}"
export RUN REPO
export NNI_MODEL="${NNI_MODEL:-/scratch/users/sastocke/chd_refinement/models/nninteractive/models/nnInteractive_v1.0}"
export SEQSEG_MODEL="${SEQSEG_MODEL:-/scratch/users/sastocke/chd_refinement/seqseg_weights/aorta_ct_mr/Dataset006_SEQAORTANDFEMOCT/nnUNetTrainer__nnUNetPlans__3d_fullres}"
STAGE="${1:?Specify a stage}"; shift
cd "$REPO"
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
case "$STAGE" in
  nni) conda activate /scratch/users/sastocke/conda_envs/chd_nninteractive310 ;;
  seqseg) conda activate /scratch/users/sastocke/conda_envs/chd_seqseg310 ;;
  *) conda activate /scratch/users/sastocke/conda_envs/nnunet310 ;;
esac
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 nnUNet_compile=f
export nnUNet_raw="${nnUNet_raw:-$REPO/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$REPO/nnUNet_results}"
python -c 'import sys; assert sys.version_info >= (3, 10), "Expected Python >=3.10 in the selected conda environment"'
ARMS=(baseline chambers seqseg combined)
case "$STAGE" in
  prepare)
    python tools/verify_refinement_baseline.py --run "$RUN" --folds "${FOLDS:-0,1,2,3,4}"
    if [ ! -f "$RUN/training.json" ]; then
      python - <<'CHECK'
import os
from pathlib import Path
for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
    for dataset in range(95, 98):
        matches = list(Path(os.environ[key]).glob('Dataset%03d_*' % dataset))
        if matches:
            raise SystemExit('Dataset ID occupied; refusing overwrite: '+str(matches))
CHECK
    fi
    if [ ! -f "$RUN/run.json" ]; then bash scripts/CHD_refinement_four_arm.sh init; fi
    bash scripts/CHD_refinement_four_arm.sh seqseg
    bash scripts/CHD_refinement_four_arm.sh nni
    bash scripts/CHD_refinement_four_arm.sh assemble
    if [ ! -f "$RUN/training.json" ]; then bash scripts/CHD_refinement_four_arm.sh build; fi
    ;;
  predict)
    TASK="${SLURM_ARRAY_TASK_ID:?Submit prediction as --array=0-3}"
    (( TASK >= 0 && TASK < 4 )) || exit 2
    python tools/evaluate_refinement_models.py --run "$RUN" --arm "${ARMS[$TASK]}" --folds "${FOLDS:-0,1,2,3,4}" ;;
  init)
    python tools/run_refinement_ablation.py --run "$RUN" init \
      --images-dir "${IMAGES_DIR:-$nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTr}" \
      --seeds-dir "${SEEDS_DIR:-$nnUNet_raw/Dataset090_ImageCHDPseudoCombined/labelsTr}" \
      --accepted-csv "${ACCEPTED_CSV:-$nnUNet_raw/Dataset090_ImageCHDPseudoCombined/split_config.csv}" \
      --expected-cases 50 "$@" ;;
  nni)
    python tools/run_refinement_ablation.py --run "$RUN" nni --model "${NNI_MODEL:?Set explicit nnInteractive trained-model folder}" \
      --fold "${NNI_FOLD:-0}" --checkpoint "${NNI_CHECKPOINT:-checkpoint_final.pth}" "$@" ;;
  seqseg)
    python tools/run_refinement_ablation.py --run "$RUN" seqseg --model "${SEQSEG_MODEL:?Set explicit SeqSeg trained-model folder}" "$@" ;;
  assemble)
    python tools/run_refinement_ablation.py --run "$RUN" assemble "$@" ;;
  evaluate)
    python tools/run_refinement_ablation.py --run "$RUN" evaluate \
      --gt-dir "${GT_DIR:?Set ground-truth folder for the complete run case list}" "$@" ;;
  build)
    python tools/build_refinement_ablation.py --run "$RUN" build --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" --results "$nnUNet_results" \
      --reference-plans "${REFERENCE_PLANS:-$nnUNet_preprocessed/Dataset090_ImageCHDPseudoCombined/nnUNetResEncUNetMPlans.json}" "$@" ;;
  preprocess)
    TASK="${SLURM_ARRAY_TASK_ID:?Submit preprocessing as --array=0-3}"
    (( TASK >= 0 && TASK < 4 )) || exit 2
    python tools/build_refinement_ablation.py --run "$RUN" preprocess --arm "${ARMS[$TASK]}" --workers 4 "$@" ;;
  train)
    TASK="${SLURM_ARRAY_TASK_ID:?Submit training as --array=0-19 (or select fold tasks)}"
    (( TASK >= 0 && TASK < 20 )) || exit 2
    python tools/build_refinement_ablation.py --run "$RUN" train --arm "${ARMS[$((TASK / 5))]}" \
      --fold "$((TASK % 5))" --results "$nnUNet_results" "$@" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
