#!/bin/bash
# Submit the Fanwei pipeline as a dependency chain. Run on a LOGIN node:
#   bash scripts/CHD_refine_fanwei_submit_all.sh
set -euo pipefail
cd /scratch/users/sastocke/nnunet_CHD
j1=$(sbatch --parsable scripts/CHD_refine_fanwei_step1.sh)
echo "step1 prompts        : $j1"
j2=$(sbatch --parsable --dependency=afterok:"$j1" scripts/CHD_refine_fanwei_step2.sh)
echo "step2 nnInteractive  : $j2   (after $j1)"
j3=$(sbatch --parsable --dependency=afterok:"$j1" scripts/CHD_refine_fanwei_step3_bif1.sh)
echo "step3a SeqSeg bif1   : $j3   (after $j1, runs parallel to step2)"
j4=$(sbatch --parsable --dependency=afterok:"$j2" scripts/CHD_refine_fanwei_step4_postlcc.sh)
echo "step4 post-LCC       : $j4   (after $j2)"
echo ""
echo "optional deeper tree: sbatch --dependency=afterok:$j1 scripts/CHD_refine_fanwei_step3_bif2.sh"
echo "watch: squeue -u sastocke"
