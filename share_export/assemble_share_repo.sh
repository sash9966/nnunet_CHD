#!/bin/bash
# =============================================================================
#  assemble_share_repo.sh
#  Build a CLEAN, shareable export of this fork: the nnunetv2 package (minus the
#  clinical disease-landmark trio) + pyproject + LICENSE + the collaborator kit,
#  as a single fresh git commit (no internal history). Then push to a new repo.
#
#  Run from the repo root:  bash share_export/assemble_share_repo.sh [OUT_DIR]
# =============================================================================
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"          # repo root (parent of share_export/)
OUT="${1:-/tmp/nnunet_da5_share}"
echo "source: ${SRC}"
echo "export: ${OUT}"
[ -e "${OUT}" ] && { echo "ERROR: ${OUT} exists — remove it or pass a different OUT_DIR"; exit 1; }
mkdir -p "${OUT}/tools"

# ---- 1) nnunetv2 package, minus the disease-landmark trio + bytecode ----
rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'training/nnUNetTrainer/variants/mixins/disease_landmark.py' \
  --exclude 'training/nnUNetTrainer/variants/composed/nnUNetTrainerDA5DiseaseLandmark.py' \
  "${SRC}/nnunetv2/" "${OUT}/nnunetv2/"

# ---- 2) packaging + license + collaborator kit ----
cp "${SRC}/pyproject.toml" "${OUT}/pyproject.toml"
cp "${SRC}/LICENSE" "${OUT}/LICENSE"
cp "${SRC}/share_export/README.md"       "${OUT}/README.md"
cp "${SRC}/share_export/INFERENCE.md"    "${OUT}/INFERENCE.md"
cp "${SRC}/share_export/train_da5.sh"    "${OUT}/train_da5.sh"
cp "${SRC}/share_export/retag_checkpoint_to_stock.py" "${OUT}/tools/retag_checkpoint_to_stock.py"
cat > "${OUT}/.gitignore" <<'EOF'
__pycache__/
*.pyc
*.egg-info/
build/
dist/
EOF

# ---- 3) SAFETY CHECKS (abort if anything is wrong) ----
echo "--- checks ---"
# 3a) no dangling IMPORT of the removed clinical trio (comments/doc mentions are fine)
if grep -rlE "^[[:space:]]*(from|import)[[:space:]].*(chd_landmarks|disease_landmark|DiseaseLandmark)" "${OUT}/nnunetv2" >/dev/null 2>&1; then
  echo "ABORT: a file still IMPORTS the removed disease-landmark code:"
  grep -rlE "^[[:space:]]*(from|import)[[:space:]].*(chd_landmarks|disease_landmark|DiseaseLandmark)" "${OUT}/nnunetv2"
  exit 1
fi
# 3b) no stray CHD-specific tokens (cleanliness, not privacy)
if grep -rniE "BAF[0-9]|CHIPS[0-9]|Sanjib|PHICleared" "${OUT}" >/dev/null 2>&1; then
  echo "ABORT: stray CHD-specific token found in the export:"
  grep -rniE "BAF[0-9]|CHIPS[0-9]|Sanjib|PHICleared" "${OUT}" | head
  exit 1
fi
# 3c) the DA5 + CaseWeighted trainers survived
for t in nnUNetTrainerDA5_500epochs nnUNetTrainerDA5CaseWeighted_500epochs; do
  grep -rq "$t" "${OUT}/nnunetv2" || { echo "ABORT: ${t} missing from export"; exit 1; }
done
echo "  checks passed."

# ---- 4) fresh git repo, single commit ----
cd "${OUT}"
git init -q -b main
git add -A
git commit -q -m "nnU-Net v2 (DA5 fork): trainers, training template, inference + retag-to-stock kit"
echo ""
echo "=============================================================="
echo "Clean export ready at: ${OUT}   ($(git rev-list --count HEAD) commit)"
echo "Push to a NEW empty GitHub repo:"
echo "  cd ${OUT}"
echo "  git remote add origin git@github.com:<you>/<new-repo>.git"
echo "  git push -u origin main"
echo "=============================================================="
