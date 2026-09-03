#!/bin/bash
# =============================================================================
#  CHD_predict_dataset080_d092.sh
#  Incremental evaluation on the FROZEN Dataset080 expert test set:
#     D090  baseline pseudo-label run 1
#     D091  + QC-promoted pseudo-labels          (the previous increment)
#     D092  = D091 cases, REFINED labels          -> isolates LABEL QUALITY
#     D093  = D092 + newly-usable cases           -> isolates ADDITIONAL DATA
#  Every arm uses the SAME trainer/plans/folds, and Dataset080 is in NO arm's training set.
#  Predicts NATIVE spacing (no resize) per fold AND as the 5-fold ensemble: the per-fold spread is
#  what tells you whether a Dice bump exceeds fold noise rather than just looking like an improvement.
# =============================================================================
#SBATCH --job-name=D080-eval-incr
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D080-eval-incr_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D080-eval-incr_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 'FATAL: python <3.9 active — env did not activate')"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
REPO="/scratch/users/sastocke/nnunet_CHD"; cd "$REPO"
source scripts/_provenance.sh

PLANS="nnUNetResEncUNetMPlans"; FULLRES="3d_fullres"; TRAINER="nnUNetTrainerDA5_200epochs"
D080="$nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed"
IMG="$D080/imagesTr"; GT="$D080/labelsTr"
EVAL="/scratch/users/sastocke/chd_refinement/d080_incremental"
PER_FOLD="${PER_FOLD:-1}"
mkdir -p "$EVAL" "$REPO/logs"
stamp_provenance "d080-incremental-eval" "$EVAL" "trainer=$TRAINER" "arms=D090,D091,D092,D093" "per_fold=$PER_FOLD"

declare -a ARMS=( "D090:Dataset090_ImageCHDPseudoCombined"
                  "D091:Dataset091_ImageCHDPseudoCombinedV2"
                  "D092:Dataset092_ImageCHDRefined"
                  "D093:Dataset093_ImageCHDRefinedPlus" )

PRED_ARGS=()
for entry in "${ARMS[@]}"; do
  NAME="${entry%%:*}"; DS="${entry##*:}"
  MODEL="$nnUNet_results/$DS/${TRAINER}__${PLANS}__${FULLRES}"
  if [ ! -d "$MODEL" ]; then echo "[skip $NAME] no model at $MODEL"; continue; fi
  ID=$(echo "$DS" | sed 's/^Dataset0*\([0-9]*\)_.*/\1/')

  # 5-fold ensemble (headline number)
  ENS="$EVAL/${NAME}_ens"
  if [ ! -d "$ENS" ] || [ -z "$(ls -A "$ENS" 2>/dev/null)" ]; then
    echo "==== $NAME ensemble (folds 0-4) ===="
    mkdir -p "$ENS"
    nnUNetv2_predict -i "$IMG" -o "$ENS" -d "$ID" -c "$FULLRES" -tr "$TRAINER" -p "$PLANS" \
        -f 0 1 2 3 4 --disable_tta || echo "  [warn] $NAME ensemble prediction failed"
  else echo "[done] $NAME ensemble exists"; fi
  PRED_ARGS+=( --pred "${NAME}=${ENS}" )

  # per-fold (fold spread = the noise floor the bump must beat)
  if [ "$PER_FOLD" = "1" ]; then
    for F in 0 1 2 3 4; do
      [ -f "$MODEL/fold_$F/checkpoint_final.pth" ] || continue
      FD="$EVAL/${NAME}_f${F}"
      if [ ! -d "$FD" ] || [ -z "$(ls -A "$FD" 2>/dev/null)" ]; then
        echo "---- $NAME fold $F ----"; mkdir -p "$FD"
        nnUNetv2_predict -i "$IMG" -o "$FD" -d "$ID" -c "$FULLRES" -tr "$TRAINER" -p "$PLANS" \
            -f "$F" --disable_tta || echo "  [warn] $NAME fold $F failed"
      fi
      PRED_ARGS+=( --pred "${NAME}_f${F}=${FD}" )
    done
  fi
done

[ ${#PRED_ARGS[@]} -gt 0 ] || { echo "FATAL: no arms had a trained model — nothing to evaluate"; exit 1; }

echo ""
echo "===== Dice vs the frozen Dataset080 expert GT (baseline = D091) ====="
python tools/dice_analysis_d080.py --gt-dir "$GT" "${PRED_ARGS[@]}" --baseline D091 --out-dir "$EVAL/analysis"

echo "=============================================================="
echo "DONE. Incremental comparison -> $EVAL/analysis"
echo "  Read it as: D091->D092 = refinement effect (labels only, same cases)"
echo "              D092->D093 = additional-data effect"
echo "  A bump is only real if it exceeds the per-fold spread within each arm."
echo "=============================================================="
