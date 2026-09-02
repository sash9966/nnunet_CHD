#!/bin/bash
# FANWEI production run — JOB 2: nnInteractive refine ALL 7 structures. Run after JOB 1.
#   sbatch scripts/CHD_refine_fanwei_step2.sh
#SBATCH --job-name=fw-nnI
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/fw-nnI_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/fw-nnI_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
LCC="${FANWEI_LCC:-$REPO/nnUNet_raw/Dataset012_Fanweidata/predictions/ds071__grid2native_lcc}"
OUT=/scratch/users/sastocke/chd_refinement/out/nninteractive_fanwei
rm -rf "$OUT"
exec bash scripts/CHD_refine_step2_nninteractive.sh "$LCC" "$OUT"
