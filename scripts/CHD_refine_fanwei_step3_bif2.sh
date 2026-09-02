#!/bin/bash
# FANWEI — JOB 3b: SeqSeg vessels, tree to ~2 bifurcations. Run after JOB 1.
#   sbatch scripts/CHD_refine_fanwei_step3_bif2.sh
#SBATCH --job-name=fw-seqseg-bif2
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/fw-seqseg-bif2_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/fw-seqseg-bif2_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
PROMPTS=/scratch/users/sastocke/chd_refinement/prompts/fanwei
OUT=/scratch/users/sastocke/chd_refinement/out/seqseg_fanwei_bif2
# trunk + up to 6 daughters ~= stop after 2 bifurcations (verify from the log's "Connections of branches are:")
export SEQSEG_MAX_BRANCHES=7 SEQSEG_MAX_STEPS=300 SEQSEG_MAX_STEPS_PER_BRANCH=60
rm -rf "$OUT"
exec bash scripts/CHD_refine_step3_seqseg.sh "$PROMPTS" "$OUT"
