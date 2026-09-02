#!/bin/bash
# =============================================================================
#  CHD_refine_step3_seqseg.sh  (workstream D — vessels: Aorta / Pulmonary)
#  AUTONOMOUS SeqSeg tracing: self-discovers the nnU-Net model folder under the weights root
#  (auto-downloads the Zenodo weights if absent), then runs `seqseg run single` per case per vessel,
#  seeding from the endpoints (world coords + radius) produced by step 1 (label_to_prompts.py).
#  SeqSeg 2.1.0 API:
#    seqseg run single --image IMG --outdir OUT --model-folder <..__..__3d_fullres> \
#        --nnunet-type 3d_fullres --train-dataset Dataset006_SEQAORTANDFEMOCT --scale 0.1 --unit mm \
#        --seed X Y Z R [--seed X Y Z R ...]
# =============================================================================
#SBATCH --job-name=refine-seqseg
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/refine-seqseg_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/refine-seqseg_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/chd_seqseg310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import seqseg" 2>/dev/null && echo "[env] seqseg import OK" || { echo "FATAL: seqseg not importable"; exit 1; }

REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
# ===== EDIT =====
PROMPTS_DIR="${1:-/scratch/users/sastocke/chd_refinement/prompts/ds090}"             # step 1 output (<case>_prompts.json)
OUT_DIR="${2:-/scratch/users/sastocke/chd_refinement/out/seqseg_ds090}"
CASES="${3:-}"                                       # optional comma-separated case filter
GT_DIR="${4:-}"                                       # if set -> run job-4 eval_vs_gt at the end (e.g. Dataset080/labelsTr)
EVAL_LCC_DIR="${5:-$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc}"
WSEARCH=/scratch/users/sastocke/chd_refinement/seqseg_weights          # search anywhere under here
WROOT="$WSEARCH/aorta_ct_mr"                                            # download target if missing
ZENODO_URL="https://zenodo.org/records/15020477/files/nnUNet_results.zip?download=1"
TRAIN_DATASET=Dataset006_SEQAORTANDFEMOCT
SEQSEG_CONFIG="${SEQSEG_CONFIG:-aorta_tutorial}"   # the 'global' default lacks keys (ADD_RADIUS); use a real config
# tracing caps (env-overridable; empty = SeqSeg default). Cap bifurcation depth so the vessel tree
# stays shallow enough to be useful clinically (the uncapped run produced a 731-step, 65-branch tree).
SEQSEG_MAX_BRANCHES="${SEQSEG_MAX_BRANCHES:-}"
SEQSEG_MAX_STEPS="${SEQSEG_MAX_STEPS:-}"
SEQSEG_MAX_STEPS_PER_BRANCH="${SEQSEG_MAX_STEPS_PER_BRANCH:-}"
SCALE=0.1; UNIT=mm; VESSELS="Aorta Pulmonary"
# native CT images live in the SOURCE datasets — search all per case:
IMG_DIRS=(
  "$REPO/nnUNet_raw/Dataset012_Fanweidata/imagesTr"
  "$REPO/ClinicalImagesPHICleared/imagesTs"
  "$REPO/nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/imagesTr"
  "$REPO/nnUNet_raw/Dataset071_ImageCHDClinicalOrientation/imagesTr"
  "$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTr"
  "$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTs"
)
find_image () { local c="$1" d; for d in "${IMG_DIRS[@]}"; do [ -f "$d/${c}_0000.nii.gz" ] && { echo "$d/${c}_0000.nii.gz"; return 0; }; done; return 1; }
want_case () { [ -z "$CASES" ] && return 0; case ",$CASES," in *",$1,"*) return 0;; *) return 1;; esac; }
CAPS=""
[ -n "$SEQSEG_MAX_BRANCHES" ]        && CAPS="$CAPS --max-n-branches $SEQSEG_MAX_BRANCHES"
[ -n "$SEQSEG_MAX_STEPS" ]           && CAPS="$CAPS --max-n-steps $SEQSEG_MAX_STEPS"
[ -n "$SEQSEG_MAX_STEPS_PER_BRANCH" ] && CAPS="$CAPS --max-n-steps-per-branch $SEQSEG_MAX_STEPS_PER_BRANCH"
# ================
mkdir -p "$OUT_DIR" "$WROOT" "$REPO/logs"
source scripts/_provenance.sh; stamp_provenance "refine-step3-seqseg" "$OUT_DIR" "model=SeqSeg:$TRAIN_DATASET" "prompts=$PROMPTS_DIR" "cases=${CASES:-ALL}"

# --- auto-discover the model trainer folder anywhere under WSEARCH; auto-download if none ---
find_model () {
  local m; m="$(find "$WSEARCH" -type d -name "*Trainer*__*__3d_fullres" 2>/dev/null | grep -iE "006|aort|femo" | head -1)"
  [ -z "$m" ] && m="$(find "$WSEARCH" -type d -name "*Trainer*__*__3d_fullres" 2>/dev/null | head -1)"
  echo "$m"
}
MODEL_FOLDER="$(find_model)"
if [ -z "$MODEL_FOLDER" ]; then
  echo "[weights] no model under $WSEARCH — downloading from Zenodo into $WROOT ..."
  ( cd "$WROOT" && curl -L "$ZENODO_URL" -o nnUNet_results.zip && unzip -o -q nnUNet_results.zip )
  MODEL_FOLDER="$(find_model)"
fi
[ -n "$MODEL_FOLDER" ] || { echo "FATAL: no *Trainer*__*__3d_fullres model found/downloaded under $WSEARCH"; \
  echo "  contents:"; find "$WSEARCH" -maxdepth 5 -type d | head -40; exit 1; }
# nnUNet_results = the folder that CONTAINS DatasetXXX_* (two levels up from the trainer folder)
export nnUNet_results="$(dirname "$(dirname "$MODEL_FOLDER")")"
echo "[weights] model-folder   = $MODEL_FOLDER"
echo "[weights] nnUNet_results = $nnUNet_results"
echo "[seqseg] available configs (want one that defines ADD_RADIUS; using '$SEQSEG_CONFIG'):"
seqseg config list 2>&1 | head -20 || seqseg config --help 2>&1 | head -20 || true

ls "$PROMPTS_DIR"/*_prompts.json >/dev/null 2>&1 || { echo "FATAL: no step-1 prompts in $PROMPTS_DIR (run step 1 first)"; exit 1; }

# --- per case, per vessel: build --seed args from step-1 seeds_world_r and trace ---
for j in "$PROMPTS_DIR"/*_prompts.json; do
  c=$(basename "$j" _prompts.json)
  want_case "$c" || continue
  img="$(find_image "$c" || true)"
  [ -n "$img" ] || { echo "[skip $c] no image in any IMG_DIRS"; continue; }
  for V in $VESSELS; do
    SEEDS=$(python - "$j" "$V" <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); v = sys.argv[2]
s = j.get("structures", {}).get(v, {})
args = []
for xyzr in s.get("seeds_world_r", []):
    x, y, z, r = xyzr
    args += ["--seed", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{r:.3f}"]
print(" ".join(args))
PY
)
    [ -z "$SEEDS" ] && { echo "[skip $c/$V] no seeds"; continue; }
    out="$OUT_DIR/$c/$V"
    [ -d "$out" ] && ls "$out"/*.vt* >/dev/null 2>&1 && { echo "[done $c/$V]"; continue; }
    mkdir -p "$out"
    echo "==== $c / $V : seqseg run single ($SEEDS) ===="
    seqseg run single --image "$img" --outdir "$out" --model-folder "$MODEL_FOLDER" \
        --nnunet-type 3d_fullres --train-dataset "$TRAIN_DATASET" --fold all \
        --config-name "$SEQSEG_CONFIG" --scale "$SCALE" --unit "$UNIT" $CAPS $SEEDS \
        || echo "  [warn] seqseg failed for $c/$V (see log)"
  done
done
echo "DONE. SeqSeg traces -> $OUT_DIR/<case>/<vessel>/"

# ---- job 4: eval vs expert GT (runs only if GT_DIR given) ----
if [ -n "$GT_DIR" ]; then
  NNI_DIR="$(echo "$OUT_DIR" | sed 's/seqseg/nninteractive/')/refined"
  EVAL_OUT="$(dirname "$OUT_DIR")/eval_$(basename "$OUT_DIR")"
  EVAL_PY=/scratch/users/sastocke/conda_envs/nnunet310/bin/python   # known-good env for eval deps
  echo ""; echo "===== job 4: eval_vs_gt (GT=$GT_DIR) ====="
  echo "  nnInteractive refined: $NNI_DIR"
  "$EVAL_PY" tools/eval_vs_gt.py --gt-dir "$GT_DIR" --lcc-dir "$EVAL_LCC_DIR" \
      --nninteractive-dir "$NNI_DIR" --seqseg-dir "$OUT_DIR" --out "$EVAL_OUT" \
      ${CASES:+--cases "$CASES"} || echo "  [warn] eval_vs_gt failed (see above)"
  echo "  eval -> $EVAL_OUT/dice_vs_gt.csv"
fi
