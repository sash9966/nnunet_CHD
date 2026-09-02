#!/bin/bash
# FANWEI production run — JOB 1: prompts (+QC) for all Fanwei cases. Bare-submittable.
#   sbatch scripts/CHD_refine_fanwei_step1.sh
#SBATCH --job-name=fw-prompts
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/fw-prompts_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/fw-prompts_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
LCC="${FANWEI_LCC:-$REPO/nnUNet_raw/Dataset012_Fanweidata/predictions/ds071__grid2native_lcc}"
OUT=/scratch/users/sastocke/chd_refinement/prompts/fanwei
echo "[fanwei] available Fanwei prediction dirs (pick with FANWEI_LCC=... if the default is wrong):"
ls -d "$REPO"/nnUNet_raw/Dataset012_Fanweidata/predictions/*/ 2>/dev/null || echo "  (none found)"
echo "[fanwei] using LCC: $LCC"
[ -d "$LCC" ] || { echo "FATAL: LCC dir not found: $LCC"; exit 1; }
rm -rf "$OUT"
exec bash scripts/CHD_refine_step1_prompts.sh "$LCC" "$OUT"
