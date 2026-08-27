#!/bin/bash
# =============================================================================
#  CHD_refine_step1_prompts.sh  (workstream D — promptable refinement)
#  Turn LCC pseudo-labels into prompts (centerline / points / bbox / adaptive lasso)
#  + a QC overlay NIfTI, saved so you can inspect them in Slicer before running any model.
#  Runs under the KNOWN-GOOD nnunet310 env (has numpy/scipy/skimage/networkx/nibabel).
# =============================================================================
#SBATCH --job-name=refine-prompts
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/refine-prompts_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/refine-prompts_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import numpy,scipy,skimage,networkx,nibabel" || { echo "FATAL: prompt deps missing in env"; exit 1; }

REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
# ===== EDIT: where the LCC pseudo-labels live, and where prompts go =====
LCC_DIR="${1:-$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc}"
OUT_DIR="${2:-/scratch/users/sastocke/chd_refinement/prompts/ds090}"
# =======================================================================
mkdir -p "$OUT_DIR" "$REPO/logs"
echo "[prompts] LCC labels: $LCC_DIR"
echo "[prompts] output:     $OUT_DIR"

python tools/label_to_prompts.py --labels-dir "$LCC_DIR" --out-dir "$OUT_DIR" --write-qc \
    --structures LV-BP,RV-BP,LA,RA,Aorta,Pulmonary

echo "DONE. Per-case <case>_prompts.json + <case>_prompts_qc.nii.gz in $OUT_DIR"
echo "  QC legend: 10=fg points, 11=neg points, 12=vessel centerline, 13=chamber lasso"
