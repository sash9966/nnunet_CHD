#!/usr/bin/env bash
set -euo pipefail
source "${SCRIPT_DIR:?Submit using submit.sh}/environment.sh"
cd "$nnUNet_raw/$DATASET"
sha256sum -c SHA256SUMS
# Existing experiments must not be silently re-preprocessed.
marker="$nnUNet_preprocessed/$DATASET/.dataset200_prepared"
if [[ -f "$marker" ]]; then
  cmp SHA256SUMS "$marker" || { echo 'Dataset changed since preprocessing; use a fresh experiment.' >&2; exit 1; }
  [[ -f "$nnUNet_preprocessed/$DATASET/$PLANS.json" ]] || exit 1
  echo 'Reusing preprocessing for the unchanged dataset.'
  exit 0
fi
[[ ! -d "$nnUNet_results/$DATASET/${TRAINER}__${PLANS}__${CONFIGURATION}" ]] || {
  echo 'Existing training results without preparation marker; inspect before reprocessing.' >&2; exit 1;
}
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity \
  -pl "$PLANNER" -c "$CONFIGURATION" -np 4
cp SHA256SUMS "$marker"
