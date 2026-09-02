#!/bin/bash
# Submit the whole D080 pipeline as a SLURM dependency chain so each step only starts after the
# previous one SUCCEEDS (no more races between nnInteractive and the eval). Step 3 runs the eval itself.
# Run on a LOGIN node (not sbatch):  bash scripts/CHD_refine_d080_submit_all.sh
set -euo pipefail
cd /scratch/users/sastocke/nnunet_CHD
j1=$(sbatch --parsable scripts/CHD_refine_d080_step1.sh)
echo "step1 prompts            : $j1"
j2=$(sbatch --parsable --dependency=afterok:"$j1" scripts/CHD_refine_d080_step2.sh)
echo "step2 nnInteractive      : $j2   (after $j1)"
j3=$(sbatch --parsable --dependency=afterok:"$j2" scripts/CHD_refine_d080_step3.sh)
echo "step3 SeqSeg + eval(job4): $j3   (after $j2)"
echo ""
echo "watch:   squeue -u sastocke"
echo "results: logs/d080-seqseg_${j3}.out   (Dice table at the end)"
