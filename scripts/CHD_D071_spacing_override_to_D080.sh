#!/bin/bash
# =============================================================================
#  CHD_D071_spacing_override_to_D080.sh
#  HYPOTHESIS: nnU-Net fingerprints its target spacing from the TRAINING data (ImageCHD, 1mm iso).
#  At inference every Dataset080 clinical image is therefore resampled to that training spacing,
#  predicted, and resampled back. If we instead plan the training data at Dataset080's MEDIAN
#  spacing, the clinical test images are resampled ~not at all, and the network operates at the
#  resolution it will actually be deployed at. This trades resampling error on the test side for
#  resampling error on the training side.
#
#  DESIGN — ONE VARIABLE (target spacing). Same dataset, same cases, same splits, same trainer,
#  same fold. Two PLANS on the SAME dataset (nnU-Net supports this), so nothing else can drift:
#     arm A (baseline)  nnUNetResEncUNetMPlans           <- spacing fingerprinted from ImageCHD
#     arm B (override)  nnUNetResEncUNetMPlans_d080sp    <- spacing forced to D080 median
#  Both trained on fold 0 only, both evaluated on the frozen Dataset080 with native-space inference.
#  Dataset080 is NOT in training for either arm.
#
#     Phase 0  measure D080 median spacing (nnU-Net (z,y,x) order, header-only)
#     Phase 1  plan arm B with -overwrite_target_spacing + -overwrite_plans_name
#              -> prints resulting patch/median-shape/batch, checks transpose, guards disk
#     Phase 2  preprocess arm B
#     Phase 3  train fold 0: arm B, and arm A too if no matched baseline exists
#     Phase 4  predict D080 with both arms (fold 0) + Dice comparison
#
#  PLAN_ONLY=1  stop after Phase 1 (inspect the plan before spending disk/GPU)  <- recommended first
#  SKIP_DISK_GUARD=1  proceed even if the projected preprocessed size looks too big
# =============================================================================
#SBATCH --job-name=D071-d080sp
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D071-d080sp_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D071-d080sp_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: Python '+sys.version.split()[0]+' active, expected 3.10 from conda env nnunet310. The env did not activate (likely a scratch purge deleted the interpreter). Recreate the env before running; do not edit the code.')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=f          # env's include/python3.10 is missing -> torch.compile dies at Epoch 0

REPO="/scratch/users/sastocke/nnunet_CHD"; cd "$REPO"

# ===== config =====
TRAIN_ID=71
TRAIN_DS="Dataset071_ImageCHDClinicalOrientation"     # ImageCHD, myocardium intact
TEST_DS="Dataset080_ClinicalCaseSanjibDetailed"       # frozen clinical test set we evaluate on
PLANNER="nnUNetPlannerResEncM"
PLANS_BASE="nnUNetResEncUNetMPlans"                   # arm A: fingerprinted spacing
PLANS_OVR="nnUNetResEncUNetMPlans_d080sp"             # arm B: D080 median spacing
FULLRES="3d_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"
FOLD=0
EVAL="/scratch/users/sastocke/chd_refinement/d071_spacing_probe"
# ==================

CKPT="${nnUNet_results}/${TRAIN_DS}/.checkpoints/d080sp"; mkdir -p "$CKPT" "$REPO/logs" "$EVAL"
[ -d "${nnUNet_raw}/${TRAIN_DS}" ] || { echo "FATAL: training dataset not found: ${nnUNet_raw}/${TRAIN_DS}"; exit 1; }
[ -d "${nnUNet_raw}/${TEST_DS}"  ] || { echo "FATAL: test dataset not found: ${nnUNet_raw}/${TEST_DS}"; exit 1; }
source scripts/_provenance.sh
stamp_provenance "D071-spacing-override-to-D080" "$EVAL" \
    "train=${TRAIN_DS}" "test=${TEST_DS}" "plans_base=${PLANS_BASE}" "plans_ovr=${PLANS_OVR}" \
    "trainer=${TRAINER}" "fold=${FOLD}" "design=one_variable(target_spacing)"

echo "=============================================================="
echo " train: ${TRAIN_DS}   ($(ls "${nnUNet_raw}/${TRAIN_DS}/labelsTr" | wc -l) labelled cases)"
echo " test : ${TEST_DS}    ($(ls "${nnUNet_raw}/${TEST_DS}/labelsTr" 2>/dev/null | wc -l) expert cases)"
echo "=============================================================="

# ---- Phase 0: measure D080 median spacing (nnU-Net (z,y,x) order) ----
SPACING_JSON="${EVAL}/d080_median_spacing.json"
if [ ! -f "$SPACING_JSON" ]; then
  echo ""
  echo "[Phase 0] measuring ${TEST_DS} median spacing"
  python tools/median_spacing.py --images-dir "${nnUNet_raw}/${TEST_DS}/imagesTr" --out-json "$SPACING_JSON"
else echo "[Phase 0] spacing already measured -> $SPACING_JSON"; fi

read -r SZ SY SX <<<"$(python - "$SPACING_JSON" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["median_zyx"]
print(m[0], m[1], m[2])
PY
)"
echo "[Phase 0] target spacing (z y x) = ${SZ} ${SY} ${SX}"

# ---- Phase 1: plan arm B with the overridden spacing ----
if [ ! -f "$CKPT/01_plan.done" ]; then
  echo ""
  echo "[Phase 1] planning ${PLANS_OVR} with -overwrite_target_spacing ${SZ} ${SY} ${SX}"
  nnUNetv2_plan_experiment -d "$TRAIN_ID" -pl "$PLANNER" \
      -overwrite_target_spacing "$SZ" "$SY" "$SX" \
      -overwrite_plans_name "$PLANS_OVR"
  touch "$CKPT/01_plan.done"
else echo "[Phase 1] plan already written — skipping"; fi

# report the two plans side by side + guard against a transpose flip and a disk blow-up
python - "${nnUNet_preprocessed}/${TRAIN_DS}" "$PLANS_BASE" "$PLANS_OVR" "$FULLRES" "$nnUNet_preprocessed" <<'PY'
import json, os, shutil, sys
import numpy as np
pre, base, ovr, cfg, preroot = sys.argv[1:6]

def load(name):
    p = os.path.join(pre, name + ".json")
    return json.load(open(p)) if os.path.isfile(p) else None

b, o = load(base), load(ovr)
if o is None:
    sys.exit("FATAL: overridden plans not found: %s" % os.path.join(pre, ovr + ".json"))

def row(tag, pl):
    if pl is None:
        print("  %-28s (not present — arm A baseline will be planned/trained below)" % tag); return None
    c = pl["configurations"][cfg]
    print("  %-28s spacing=%s" % (tag, [round(float(x), 4) for x in c["spacing"]]))
    print("  %-28s patch=%s  batch=%s  median_shape=%s  transpose_fwd=%s"
          % ("", c["patch_size"], c["batch_size"],
             [int(round(float(x))) for x in c["median_image_size_in_voxels"]], pl["transpose_forward"]))
    return c

print("\n=== plans comparison (%s) ===" % cfg)
cb = row("A baseline  " + base, b)
co = row("B override  " + ovr, o)

# rigor guard: overriding the spacing also re-derives transpose_forward (argmax of target spacing).
# If the two arms disagree the comparison is no longer ONE variable.
if b is not None and b["transpose_forward"] != o["transpose_forward"]:
    print("\n  [!!] transpose_forward DIFFERS between the arms: %s vs %s" % (b["transpose_forward"], o["transpose_forward"]))
    print("       The override changed the axis ordering as well as the spacing, so the comparison")
    print("       would confound TWO variables. nnU-Net derives transpose from argmax(target_spacing)")
    print("       (default_experiment_planner.py:220-225). Fix by using a planner with")
    print("       suppress_transpose=True, or keep the argmax axis unchanged.")
    sys.exit(3)

# disk guard: finer spacing => more voxels => much larger preprocessed .npy/.b2nd
import glob
nvox = float(np.prod([float(x) for x in co["median_image_size_in_voxels"]]))
raw = os.environ["nnUNet_raw"]; ds = os.path.basename(pre)
ncases = len(glob.glob(os.path.join(raw, ds, "labelsTr", "*.nii.gz")))
# data (float32) + seg (int8) per case, plus nnU-Net keeps an extra copy during processing
per_case_gb = nvox * (4 + 1) / 1024**3
est_gb = per_case_gb * max(ncases, 1)
print("\n=== projected preprocessed footprint (arm B) ===")
print("  median voxels/case : %.3g" % nvox)
if cb is not None:
    nvox_b = float(np.prod([float(x) for x in cb["median_image_size_in_voxels"]]))
    print("  vs baseline        : %.3g  (%.1fx more voxels)" % (nvox_b, nvox / max(nvox_b, 1)))
print("  cases              : %d" % ncases)
print("  estimate           : %.1f GB  (~%.2f GB/case, data float32 + seg int8)" % (est_gb, per_case_gb))
free_gb = shutil.disk_usage(preroot).free / 1024**3
print("  free on %-10s : %.1f GB" % (os.path.basename(preroot) or "preproc", free_gb))
if est_gb > 0.8 * free_gb and os.environ.get("SKIP_DISK_GUARD", "0") != "1":
    print("\n  [!!] projected %.1f GB exceeds 80%% of the %.1f GB free." % (est_gb, free_gb))
    print("       Preprocessing would likely fill the filesystem (you hit ENOSPC before).")
    print("       Options: coarsen the target spacing (e.g. match only the z axis and keep in-plane"
          "\n       at the ImageCHD value), free space, or re-run with SKIP_DISK_GUARD=1.")
    sys.exit(4)
print("")
PY

if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "[PLAN_ONLY] stopping after Phase 1 as requested. Inspect the plan above, then re-submit without PLAN_ONLY."
  exit 0
fi

# ---- Phase 2: preprocess arm B ----
if [ ! -f "$CKPT/02_preprocess.done" ]; then
  echo "[Phase 2] preprocessing ${TRAIN_DS} with ${PLANS_OVR}"
  nnUNetv2_preprocess -d "$TRAIN_ID" -plans_name "$PLANS_OVR" -c "$FULLRES" -np 8
  touch "$CKPT/02_preprocess.done"
else echo "[Phase 2] already preprocessed — skipping"; fi

# ---- Phase 3: train fold 0 for both arms (matched trainer/fold/splits) ----
train_arm () {   # $1 = plans name, $2 = label
  local PL="$1" TAG="$2"
  local OUT="${nnUNet_results}/${TRAIN_DS}/${TRAINER}__${PL}__${FULLRES}/fold_${FOLD}"
  if [ -f "${OUT}/checkpoint_final.pth" ]; then echo "[Phase 3] ${TAG}: fold ${FOLD} already complete — skipping"; return 0; fi
  if [ ! -f "${nnUNet_preprocessed}/${TRAIN_DS}/${PL}.json" ]; then
    echo "[Phase 3] ${TAG}: plans ${PL}.json missing -> planning + preprocessing it now (baseline arm)"
    nnUNetv2_plan_experiment -d "$TRAIN_ID" -pl "$PLANNER"
    nnUNetv2_preprocess -d "$TRAIN_ID" -plans_name "$PL" -c "$FULLRES" -np 8
  fi
  local CONT=""; [ -f "${OUT}/checkpoint_latest.pth" ] && CONT="--c"
  echo "[Phase 3] ${TAG}: training ${TRAINER} / ${PL} / fold ${FOLD} ${CONT}"
  nnUNetv2_train "$TRAIN_ID" "$FULLRES" "$FOLD" -tr "$TRAINER" -p "$PL" ${CONT}
}
train_arm "$PLANS_OVR"  "arm B override"
train_arm "$PLANS_BASE" "arm A baseline"

# ---- Phase 4: predict the frozen D080 with both arms (fold 0, native space) + compare ----
IMG="${nnUNet_raw}/${TEST_DS}/imagesTr"
GT="${nnUNet_raw}/${TEST_DS}/labelsTr"
PRED_ARGS=()
for entry in "A_base:${PLANS_BASE}" "B_d080sp:${PLANS_OVR}"; do
  TAG="${entry%%:*}"; PL="${entry##*:}"
  MODEL="${nnUNet_results}/${TRAIN_DS}/${TRAINER}__${PL}__${FULLRES}/fold_${FOLD}"
  [ -f "${MODEL}/checkpoint_final.pth" ] || { echo "[Phase 4] skip ${TAG}: fold ${FOLD} not trained"; continue; }
  OUTD="${EVAL}/${TAG}_f${FOLD}"
  if [ ! -d "$OUTD" ] || [ -z "$(ls -A "$OUTD" 2>/dev/null)" ]; then
    echo "[Phase 4] predicting ${TEST_DS} with ${TAG} (fold ${FOLD})"
    mkdir -p "$OUTD"
    nnUNetv2_predict -i "$IMG" -o "$OUTD" -d "$TRAIN_ID" -c "$FULLRES" -tr "$TRAINER" -p "$PL" \
        -f "$FOLD" --disable_tta || echo "  [warn] ${TAG} prediction failed"
  else echo "[Phase 4] ${TAG} predictions exist — skipping"; fi
  PRED_ARGS+=( --pred "${TAG}=${OUTD}" )
done

if [ ${#PRED_ARGS[@]} -ge 1 ]; then
  echo ""
  echo "===== Dice on the frozen ${TEST_DS} (baseline arm = A_base, fold ${FOLD} vs fold ${FOLD}) ====="
  python tools/dice_analysis_d080.py --gt-dir "$GT" "${PRED_ARGS[@]}" --baseline A_base --out-dir "${EVAL}/analysis"
fi

echo "=============================================================="
echo "DONE. spacing probe -> ${EVAL}/analysis"
echo "  arm A ${PLANS_BASE}  (ImageCHD-fingerprinted spacing)"
echo "  arm B ${PLANS_OVR}  (forced to ${TEST_DS} median ${SZ} ${SY} ${SX})"
echo "  SINGLE FOLD: there is no within-arm spread here, so treat a delta smaller than the"
echo "  per-fold spread seen previously on D080 (~0.02-0.05 WH Dice) as not yet a result."
echo "=============================================================="
