#!/bin/bash
# D080 validation — JOB 2: nnInteractive refine ALL 7 structures. Bare-submittable (run after JOB 1).
#   sbatch scripts/CHD_refine_d080_step2.sh
#SBATCH --job-name=d080-nnI
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/d080-nnI_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/d080-nnI_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
LCC="$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc"
OUT=/scratch/users/sastocke/chd_refinement/out/nninteractive_d080
D080="BAF004,CHIPS001,CHIPS002,CHIPS005,CHIPS006,CHIPS007,CHIPS010,CHIPS016"
rm -rf "$OUT"   # clean rerun: fresh refined masks + prompts every time
exec bash scripts/CHD_refine_step2_nninteractive.sh "$LCC" "$OUT" "$D080"
