#!/bin/bash
# D080 validation — JOB 1: prompts (+QC) for the expert-GT Dataset080 cases. Bare-submittable.
#   sbatch scripts/CHD_refine_d080_step1.sh
#SBATCH --job-name=d080-prompts
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/d080-prompts_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/d080-prompts_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
LCC="$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc"
OUT=/scratch/users/sastocke/chd_refinement/prompts/d080
D080="BAF004,CHIPS001,CHIPS002,CHIPS005,CHIPS006,CHIPS007,CHIPS010,CHIPS016"
exec bash scripts/CHD_refine_step1_prompts.sh "$LCC" "$OUT" "$D080"
