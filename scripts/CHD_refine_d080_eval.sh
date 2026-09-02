#!/bin/bash
# D080 validation — EVAL as a job (bare-submittable): LCC vs nnInteractive vs nnInteractive+SeqSeg
# vs expert GT, on the 3 D080 cases that have an LCC seed. Writes a Dice table + CSV + provenance.
#   sbatch scripts/CHD_refine_d080_eval.sh
#SBATCH --job-name=d080-eval
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/d080-eval_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/d080-eval_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
source scripts/_provenance.sh
PY=/scratch/users/sastocke/conda_envs/nnunet310/bin/python   # known-good env for eval deps
B=/scratch/users/sastocke/chd_refinement
OUT="$B/eval_d080_v2"
CASES="BAF004,CHIPS002,CHIPS016"
LCC="nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc"
GT="nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/labelsTr"

stamp_provenance "d080-eval" "$OUT" "lcc=$LCC" "nnI=out/nninteractive_d080" "seqseg=out/seqseg_d080" "cases=$CASES"

"$PY" tools/eval_vs_gt.py \
  --gt-dir "$GT" --lcc-dir "$LCC" \
  --nninteractive-dir "$B/out/nninteractive_d080/refined" \
  --seqseg-dir "$B/out/seqseg_d080" \
  --out "$OUT" --cases "$CASES"
