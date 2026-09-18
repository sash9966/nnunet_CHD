#!/usr/bin/env bash
set -euo pipefail
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
mode="${1:-both}"
case "$mode" in all|cv|both) ;; *) echo 'Usage: bash submit.sh [all|cv|both]' >&2; exit 2;; esac

# One workflow at a time prevents competing preprocessing / same-fold jobs.
existing="$(squeue -h -u "$USER" -o '%j' | awk '/^d200_/{print}')"
[[ -z "$existing" ]] || { echo "Dataset200 jobs already queued/running: $existing" >&2; exit 1; }
# Snapshot before submission: later git pulls cannot change queued training code.
export RUN_MODE="$mode"
export RUN_DIR="$nnUNet_results/_run_records/Dataset200_MRI/$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$NNUNET_ENV/bin/python" "$SCRIPT_DIR/snapshot.py" "$SCRIPT_DIR" "$CHD_REPO" "$RUN_DIR"
export SCRIPT_DIR="$RUN_DIR/scripts"
export CHD_REPO="$RUN_DIR/code"
mkdir -p "$SCRIPT_DIR/logs"
log() { printf '%s\n' "$*" | tee -a "$RUN_DIR/submission.log"; }
log "Run record: $RUN_DIR"
common=(--parsable --export=ALL --chdir="$SCRIPT_DIR" --cpus-per-task=8 --mem=64G)
prep="$(sbatch "${common[@]}" --partition="$CPU_PARTITION" --time="$PREP_TIME" --job-name=d200_prep --output="$SCRIPT_DIR/logs/prep-%j.log" "$SCRIPT_DIR/preprocess.sh")"
prep="${prep%%;*}"
log "Preprocessing job: $prep"
dep="$prep"
if [[ "$mode" == all || "$mode" == both ]]; then
  job="$(sbatch "${common[@]}" --partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$TRAIN_TIME" --dependency="afterok:$prep" --job-name=d200_all --output="$SCRIPT_DIR/logs/all-%j.log" "$SCRIPT_DIR/train.sh" all)"
  dep="${job%%;*}"
  log "All-case training job: $dep"
fi
if [[ "$mode" == cv || "$mode" == both ]]; then
  # Fold 0 finishes first to avoid concurrent first-time dataset unpacking.
  job="$(sbatch "${common[@]}" --partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$TRAIN_TIME" --dependency="afterok:$dep" --job-name=d200_cv0 --output="$SCRIPT_DIR/logs/cv0-%j.log" "$SCRIPT_DIR/train.sh" 0)"
  fold0="${job%%;*}"
  job="$(sbatch "${common[@]}" --partition="$GPU_PARTITION" --gres="$GPU_GRES" --time="$TRAIN_TIME" --dependency="afterok:$fold0" --array="1-4%$CV_CONCURRENCY" --job-name=d200_cv --output="$SCRIPT_DIR/logs/cv-%A-%a.log" "$SCRIPT_DIR/train.sh")"
  log "Five-fold jobs: $fold0 and ${job%%;*}"
fi
