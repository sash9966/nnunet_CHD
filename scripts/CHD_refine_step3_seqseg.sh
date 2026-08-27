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
#SBATCH --partition=bioe
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
SEQSEG_NNUNET_RESULTS="/scratch/users/sastocke/chd_refinement/seqseg_weights/aorta_ct_mr/nnUNet_results"  # Zenodo 15020477
SEQSEG_CONFIG="aorta_tutorial"          # <-- CONFIRM the config for Dataset006_SEQAORTANDFEMOCT
SEQSEG_MODEL="Dataset006_SEQAORTANDFEMOCT"   # CT aorta/femoral model (if SeqSeg asks for a dataset)
SEQSEG_SCALE="0.1"                       # aortic model trained on cm; mm CT -> scale 0.1
# ================
echo "[seqseg] weights: $SEQSEG_NNUNET_RESULTS  (model $SEQSEG_MODEL, scale $SEQSEG_SCALE)"
[ -d "$SEQSEG_NNUNET_RESULTS" ] && echo "  weights dir present" || echo "  !! weights dir MISSING — unzip Zenodo nnUNet_results.zip there"
mkdir -p "$OUT_DIR" "$REPO/logs"

# SeqSeg reads its nnU-Net weights via the nnUNet_results env var (doctor showed it unset)
export nnUNet_results="$SEQSEG_NNUNET_RESULTS"
echo "[seqseg] nnUNet_results set -> $nnUNet_results"

echo "===== SeqSeg 2.1.0 CLI probe (run single = one image + seeds.json) ====="
echo "----- seqseg run single --help -----";  seqseg run single --help 2>&1 | head -80 || true
echo "----- seqseg run batch  --help -----";  seqseg run batch  --help 2>&1 | head -40 || true
echo "----- seqseg init dataset --help -----"; seqseg init dataset --help 2>&1 | head -40 || true
echo "----- seeds.json TEMPLATE (scaffold a throwaway dataset and cat it) -----"
SCAF="$OUT_DIR/_seed_schema_probe"; rm -rf "$SCAF"
seqseg init dataset "$SCAF" 2>&1 | head -20 || seqseg init dataset --out "$SCAF" 2>&1 | head -20 || true
find "$SCAF" -maxdepth 2 -name "seeds*.json" -exec sh -c 'echo "== {} =="; cat "{}"' \; 2>/dev/null || echo "  (adjust init syntax from its --help above)"
echo "----- seqseg doctor (should now see the weights) -----"; seqseg doctor 2>&1 | tail -12 || true

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
echo "Documented call shape (finalize the SEED flag from --help above):"
echo "  seqseg -data_dir <DIR> -nnunet_results_path $SEQSEG_NNUNET_RESULTS -config_name $SEQSEG_CONFIG -unit mm -scale $SEQSEG_SCALE <SEED_FLAG ...>"
echo "STOP: paste 'seqseg --help' so I wire the exact seed flag; then I enable the per-case loop."
