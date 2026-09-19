#!/bin/bash
# Submit the complete four-arm study, then matched D080 prediction/scoring.
# Run from the Sherlock login node: bash scripts/CHD_refinement_submit.sh
# FOLDS=0 selects a matched fold-0 smoke run; default is five folds + ensemble.
set -euo pipefail
export REPO="${REPO:-/scratch/users/sastocke/nnunet_CHD}"
export RUN="${RUN:-$REPO/refinement_runs/accepted50_conservative_v3}"
export FOLDS="${FOLDS:-0,1,2,3,4}"
cd "$REPO"
mkdir -p logs "$RUN"
case "$FOLDS" in
  0) TRAIN_TASKS=5,10,15 ;;
  0,1,2,3,4) TRAIN_TASKS=5-19 ;;
  *) echo 'FOLDS must be 0 or 0,1,2,3,4' >&2; exit 2 ;;
esac
# Refuse accidental duplicate submission. Resume explicitly after checking old jobs.
if [ -f "$RUN/submitted_jobs.txt" ] && [ "${RESUBMIT:-0}" != 1 ]; then
  echo "Jobs already recorded in $RUN/submitted_jobs.txt; inspect them before RESUBMIT=1." >&2
  exit 2
fi
RECORD="$RUN/submitted_jobs.txt"
submit () {
  local job
  job=$(sbatch --parsable "$@")
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $job" >&2; return 1; }
  printf '%s %s\n' "$job" "$*" >> "$RECORD"
  printf '%s' "$job"
}
PREP=$(submit --time=72:00:00 scripts/CHD_refinement_four_arm.sh prepare)
PRE=$(submit --dependency="afterok:$PREP" --array=1-3%2 scripts/CHD_refinement_four_arm.sh preprocess)
TRAIN=$(submit --dependency="afterok:$PRE" --time=72:00:00 --array="$TRAIN_TASKS%2" scripts/CHD_refinement_four_arm.sh train)
EVAL=$(submit --dependency="afterok:$TRAIN" --array=1-3%2 scripts/CHD_refinement_four_arm.sh predict)
printf 'Queued prepare=%s preprocess=%s train=%s evaluate=%s\n' "$PREP" "$PRE" "$TRAIN" "$EVAL"
printf 'Run: %s\nLogs: %s/logs/refine4_*\n' "$RUN" "$REPO"
printf 'Final masks and metrics: nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/predictions/ds095_* through ds097_* (baseline: existing ds090_*)\n'
