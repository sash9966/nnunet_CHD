#!/bin/bash
# ImageCHD-only baseline on the eight Dataset080 cases; no training.
# Environment follows CHD_Dataset100_weighted_all.sh. Inference follows the
# completed D090/D091 evaluation in CHD_predict_dataset080_d092.sh.
# Submit from the Sherlock checkout after making its logs directory:
#   mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
#   sbatch scripts/CHD_Dataset071_eval_dataset080.sh
# Uses the audited September D090_ens/D091_ens comparator folders below.
# Each submission creates a fresh output directory; partial runs are not reused.
#SBATCH --job-name=D071-on-D080
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D071-on-D080_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D071-on-D080_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python >=3.9 required; check conda activation')"

REPO="/scratch/users/sastocke/nnunet_CHD"
cd "$REPO"
export nnUNet_raw="$REPO/nnUNet_raw"
export nnUNet_preprocessed="$REPO/nnUNet_preprocessed"
export nnUNet_results="$REPO/nnUNet_results"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=f

TRAINER="nnUNetTrainerDA5_200epochs"
PLANS="nnUNetResEncUNetMPlans"
CONFIG="3d_fullres"
MODEL="$nnUNet_results/Dataset071_ImageCHDClinicalOrientation/${TRAINER}__${PLANS}__${CONFIG}"
D080="$nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed"
D090_PRED="/scratch/users/sastocke/chd_refinement/d080_incremental/D090_ens"
D091_PRED="/scratch/users/sastocke/chd_refinement/d080_incremental/D091_ens"
mkdir -p "$REPO/evaluations/d080_imagechd_baseline"
RUN=$(mktemp -d "$REPO/evaluations/d080_imagechd_baseline/run_${SLURM_JOB_ID:-local}_XXXXXX")
export MPLCONFIGDIR="$RUN/.mplconfig"
echo "[output] $RUN"
trap 'echo "Run stopped before completion. Inspect this directory and the job log: $RUN" >&2' ERR

# Fail before GPU inference if dependencies, cases, geometry, comparator metadata
# or any of the five final checkpoints are missing. Ground truth is used only by
# this evaluator; nnUNetv2_predict receives imagesTr alone.
python -c "import nibabel, numpy, matplotlib, scipy"
CHECK_ARGS=(--images "$D080/imagesTr" --gt "$D080/labelsTr" --model "$MODEL"
            --d090 "$D090_PRED" --d091 "$D091_PRED" --out "$RUN")
python tools/evaluate_d071_on_d080.py prepare "${CHECK_ARGS[@]}"

# Native input/output: no external resize, backprojection or LCC. Each model
# retains its own trained plans/normalization; D071 and D090 have different patch
# dimensions. Explicit final checkpoints and no TTA match the September ensemble.
PREDICT=(nnUNetv2_predict -i "$D080/imagesTr" -o "$RUN/D071_ens"
         -d Dataset071_ImageCHDClinicalOrientation -c "$CONFIG" -tr "$TRAINER" -p "$PLANS"
         -f 0 1 2 3 4 -chk checkpoint_final.pth --disable_tta
         -step_size 0.5 -npp 3 -nps 3)
printf '%q ' "${PREDICT[@]}" > "$RUN/inference_command.txt"
printf '\n' >> "$RUN/inference_command.txt"
"${PREDICT[@]}"

# Rescore all three models against exactly the same GT, leaving old masks/CSVs
# unchanged. Every patient and score, including zeros, is retained in the figure.
python tools/evaluate_d071_on_d080.py evaluate "${CHECK_ARGS[@]}"
python tools/plot_d080_three_model_violins.py --analysis-dir "$RUN/analysis"
touch "$RUN/COMPLETE"
trap - ERR
echo "[done] D071 predictions: $RUN/D071_ens"
echo "[done] Comparison CSVs and figure2_d080_three_model_violins.{png,svg,pdf}: $RUN/analysis"
