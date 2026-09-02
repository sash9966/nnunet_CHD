#!/bin/bash
# D080 validation — JOB 3: SeqSeg vessels, then JOB 4 (eval vs expert GT) auto-runs at the end.
# Bare-submittable (run after JOB 2).  sbatch scripts/CHD_refine_d080_step3.sh
#SBATCH --job-name=d080-seqseg
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/d080-seqseg_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/d080-seqseg_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
LCC="$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc"
PROMPTS=/scratch/users/sastocke/chd_refinement/prompts/d080
SEQOUT=/scratch/users/sastocke/chd_refinement/out/seqseg_d080
GT="$REPO/nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/labelsTr"
D080="BAF004,CHIPS001,CHIPS002,CHIPS005,CHIPS006,CHIPS007,CHIPS010,CHIPS016"
rm -rf "$SEQOUT" "$(dirname "$SEQOUT")/eval_$(basename "$SEQOUT")"   # clean rerun: fresh traces + eval
# args: PROMPTS_DIR  SEQSEG_OUT  CASES  GT_DIR(->runs job4 eval)  EVAL_LCC_DIR
exec bash scripts/CHD_refine_step3_seqseg.sh "$PROMPTS" "$SEQOUT" "$D080" "$GT" "$LCC"
