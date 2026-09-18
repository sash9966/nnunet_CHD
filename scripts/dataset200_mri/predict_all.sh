#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage (inside a GPU allocation): bash predict_all.sh INPUT_DIR OUTPUT_DIR' >&2; exit 2; }
input="$1"; output="$2"
[[ "$input" = /* && "$output" = /* ]] || { echo 'Use absolute input/output paths.' >&2; exit 2; }
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/environment.sh"
nnUNetv2_predict -i "$input" -o "$output" -d 200 -c "$CONFIGURATION" \
  -tr "$TRAINER" -p "$PLANS" -f all -chk checkpoint_final.pth
