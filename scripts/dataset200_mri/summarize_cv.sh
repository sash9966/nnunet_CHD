#!/usr/bin/env bash
set -euo pipefail
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/environment.sh"
python "$SCRIPT_DIR/summarize_cv.py" "$nnUNet_results/$DATASET/${TRAINER}__${PLANS}__${CONFIGURATION}"
