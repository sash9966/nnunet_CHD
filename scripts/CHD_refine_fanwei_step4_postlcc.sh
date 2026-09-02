#!/bin/bash
# FANWEI — JOB 4: post-refinement LCC variant, so we can compare refined vs refined+LCC.
#   sbatch scripts/CHD_refine_fanwei_step4_postlcc.sh
#SBATCH --job-name=fw-postlcc
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/fw-postlcc_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/fw-postlcc_%j.err
set -euo pipefail
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
source scripts/_provenance.sh
PY=/scratch/users/sastocke/conda_envs/nnunet310/bin/python
IN=/scratch/users/sastocke/chd_refinement/out/nninteractive_fanwei/refined
OUT=/scratch/users/sastocke/chd_refinement/out/nninteractive_fanwei_lcc
rm -rf "$OUT"
stamp_provenance "fanwei-postlcc" "$OUT" "in=$IN" "skip_labels=none"
echo "===== variant A: LCC on ALL labels (incl. vessels) ====="
"$PY" tools/lcc_postprocess.py --in-dir "$IN" --out-dir "$OUT"
echo ""
echo "===== variant B: LCC on chambers/myo only, vessels (6,7) left intact ====="
"$PY" tools/lcc_postprocess.py --in-dir "$IN" --out-dir "${OUT}_keepvessels" --skip-labels 6,7
