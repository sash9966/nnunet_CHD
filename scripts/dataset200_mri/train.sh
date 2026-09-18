#!/usr/bin/env bash
set -euo pipefail
fold="${1:-${SLURM_ARRAY_TASK_ID:-}}"
case "$fold" in all|0|1|2|3|4) ;; *) echo 'Expected fold all or 0–4' >&2; exit 2;; esac
source "${SCRIPT_DIR:?Submit using submit.sh}/environment.sh"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: check GPU allocation and environment'
print('GPU:',torch.cuda.get_device_name())
PY
out="$nnUNet_results/$DATASET/${TRAINER}__${PLANS}__${CONFIGURATION}/fold_$fold"
if [[ -n "${RUN_DIR:-}" ]]; then
  mkdir -p "$out/run_records"
  printf '%s\n' "$RUN_DIR" > "$out/run_records/${SLURM_JOB_ID:-interactive}.path"
  for name in dataset.json plans.json dataset_fingerprint.json splits_final.json; do
    if [[ -f "$nnUNet_preprocessed/$DATASET/$name" ]]; then
      cp "$nnUNet_preprocessed/$DATASET/$name" "$RUN_DIR/${SLURM_JOB_ID:-interactive}-$name"
    fi
  done
  cp "$nnUNet_preprocessed/$DATASET/$PLANS.json" "$RUN_DIR/${SLURM_JOB_ID:-interactive}-$PLANS.json"
fi
args=(200 "$CONFIGURATION" "$fold" -tr "$TRAINER" -p "$PLANS")
if [[ -f "$out/checkpoint_final.pth" ]]; then
  if [[ "$fold" != all && ! -f "$out/validation/summary.json" ]]; then
    set -x
    nnUNetv2_train "${args[@]}" --val
  else
    echo "Fold $fold already complete."
  fi
elif [[ -f "$out/checkpoint_latest.pth" ]]; then
  set -x
    nnUNetv2_train "${args[@]}" --c
elif [[ -f "$out/checkpoint_best.pth" ]]; then
  echo 'Partial run has no latest checkpoint; inspect before restarting.' >&2; exit 1
else
  set -x
    nnUNetv2_train "${args[@]}"
fi

# Fold 0 may create the split file during training.
if [[ -n "${RUN_DIR:-}" && -f "$nnUNet_preprocessed/$DATASET/splits_final.json" ]]; then
  cp "$nnUNet_preprocessed/$DATASET/splits_final.json" "$RUN_DIR/${SLURM_JOB_ID:-interactive}-splits_final.json"
fi
