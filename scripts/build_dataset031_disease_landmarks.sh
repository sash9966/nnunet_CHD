#!/usr/bin/env bash
# build_dataset031_disease_landmarks.sh
# -----------------------------------------------------------------------------
# Convenience wrapper for the chd_landmarks dataset build.
#
# ISOLATION CONTRACT (matches the rest of scripts/):
#   * The SOURCE dataset (Dataset030) is read ONLY — images are symlinked.
#   * A NEW dataset (Dataset031_imageCHD_DiseaseLandmarks) is created; nothing
#     existing is modified.
#   * plan_and_preprocess is run ONLY on the new Dataset031, and only if you
#     pass --preprocess (off by default). It NEVER touches Dataset030.
#
# Usage:
#   bash scripts/build_dataset031_disease_landmarks.sh            # inspect + build
#   bash scripts/build_dataset031_disease_landmarks.sh --preprocess  # + plan/preprocess D031
# -----------------------------------------------------------------------------
set -euo pipefail

: "${nnUNet_raw:?Set nnUNet_raw}"
SOURCE_DS="${SOURCE_DS:-Dataset030_imageCHD_HU}"
TARGET_ID="${TARGET_ID:-31}"
TARGET_NAME="${TARGET_NAME:-imageCHD_DiseaseLandmarks}"
METADATA="${METADATA:-${nnUNet_raw}/${SOURCE_DS}/imageCHD_dataset_info.xlsx}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== inspect source =="
python -m chd_landmarks.cli inspect-dataset \
    --nnunet-raw "${nnUNet_raw}/${SOURCE_DS}" --metadata "$METADATA"

echo "== build Dataset0${TARGET_ID} (source read-only) =="
python -m chd_landmarks.cli build-dataset \
    --source-dataset "${nnUNet_raw}/${SOURCE_DS}" \
    --target-dataset-id "$TARGET_ID" \
    --target-dataset-name "$TARGET_NAME" \
    --metadata "$METADATA" \
    --out-root "$nnUNet_raw" \
    --overwrite

if [[ "${1:-}" == "--preprocess" ]]; then
    echo "== plan + preprocess Dataset0${TARGET_ID} ONLY =="
    nnUNetv2_plan_and_preprocess -d "$TARGET_ID" -pl nnUNetPlannerResEncM \
        --verify_dataset_integrity
fi

echo "Done. Next:"
echo "  nnUNetv2_train ${TARGET_ID} 3d_fullres 0                                  # vanilla"
echo "  nnUNetv2_train ${TARGET_ID} 3d_fullres 0 -tr nnUNetTrainerDA5DiseaseLandmark_200epochs"
