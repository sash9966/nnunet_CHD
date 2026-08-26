#!/bin/bash
# =============================================================================
#  CHD_refine_step3_seqseg.sh  (workstream D — vessels: Aorta / Pulmonary)
#  SeqSeg (numisveinsson/SeqSeg) traces vessels from a SINGLE SEED. We already produce seeds:
#  label_to_prompts.py writes per-case endpoints (voxel + world) for Aorta/Pulmonary in
#  <case>_prompts.json (run step 1 first). This script extracts those seeds and calls `seqseg`.
#
#  !!! NEEDS 3 THINGS I DON'T HAVE FROM YOUR ENV SUMMARY — fill them in / confirm:
#    1. SEQSEG_NNUNET_WEIGHTS : SeqSeg's own trained nnU-Net VESSEL model (-nnunet_results_path).
#                               (SeqSeg segments local patches with a vessel nnU-Net; not our CHD model.)
#    2. SEQSEG_CONFIG         : a SeqSeg config name (e.g. aorta_tutorial) selecting units/params.
#    3. SEED INGESTION        : how this SeqSeg build takes the seed (CLI flag vs a seed file).
#                               Run `seqseg --help` and paste it to me; I'll wire the exact flags.
#  Documented CLI (from the repo): seqseg -data_dir DIR -nnunet_results_path W -config_name CFG [-unit mm -scale 0.1]
# =============================================================================
#SBATCH --job-name=refine-seqseg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/refine-seqseg_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/refine-seqseg_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/chd_seqseg310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import seqseg" 2>/dev/null && echo "[env] seqseg import OK" || echo "[env] (seqseg imports as CLI? check 'seqseg --help')"

REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
# ===== EDIT =====
IMG_DIR="${1:-$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTr}"
PROMPTS_DIR="${2:-/scratch/users/sastocke/chd_refinement/prompts/ds090}"     # from step 1
OUT_DIR="${3:-/scratch/users/sastocke/chd_refinement/out/seqseg_ds090}"
SEQSEG_NNUNET_WEIGHTS="/scratch/users/sastocke/chd_refinement/models/seqseg_nnunet"   # <-- FILL / CONFIRM
SEQSEG_CONFIG="aorta_tutorial"                                                        # <-- CONFIRM
# ================
mkdir -p "$OUT_DIR" "$REPO/logs"

echo "seqseg --help (so we can wire the exact seed flag):"
seqseg --help 2>&1 | head -40 || true

echo ""
echo "Per-case seeds we generated (Aorta/Pulmonary endpoints, world coords):"
for j in "$PROMPTS_DIR"/*_prompts.json; do
  c=$(basename "$j" _prompts.json)
  python - "$j" "$c" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); c=sys.argv[2]
for name in ("Aorta","Pulmonary"):
    s=j["structures"].get(name)
    if s and s.get("endpoints_world"):
        print(f"  {c} {name} seeds(world): {s['endpoints_world']}")
PY
done

echo ""
echo "STOP: fill SEQSEG_NNUNET_WEIGHTS + SEQSEG_CONFIG + confirm the seed flag from --help,"
echo "then I'll finalize the per-case 'seqseg -data_dir ... -nnunet_results_path ... -config_name ... <seed>' loop."
