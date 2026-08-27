#!/bin/bash
# =============================================================================
#  CHD_refine_step2_nninteractive.sh  (workstream D)
#  Refine each LCC pseudo-label with nnInteractive: key-slice LASSO (chambers, positive)
#  + negative points in adjacent structures. Saves the refined mask AND the prompts used.
#  Runs under the chd_nninteractive310 env the user set up.
#
#  NOTE: the nnInteractive Python API + lasso-crop axis convention are from the repo readme and
#  are UNVERIFIED against this install — the driver logs each step and falls back lasso->points.
#  Send me the first job's log if anything throws.
# =============================================================================
#SBATCH --job-name=refine-nnI
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/refine-nnI_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/refine-nnI_%j.err

set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/chd_nninteractive310
hash -r
export NNINTERACTIVE_MODEL_DIR=/scratch/users/sastocke/chd_refinement/models/nninteractive
echo "[env] python=$(command -v python)  $(python --version 2>&1)"
python -c "import nnInteractive, torch, nibabel, numpy; print('[env] nnInteractive import OK; cuda', torch.cuda.is_available())" \
    || { echo "FATAL: nnInteractive env not importable"; exit 1; }

REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
# ===== EDIT: LCC labels + output. CT images are searched across the source datasets below. =====
LCC_DIR="${1:-$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc}"  # <case>.nii.gz
OUT_DIR="${2:-/scratch/users/sastocke/chd_refinement/out/nninteractive_ds090}"
# native CT images live in the SOURCE datasets, not in Dataset090 — search all of these per case:
IMG_DIRS=(
  "$REPO/nnUNet_raw/Dataset012_Fanweidata/imagesTr"                       # Fanwei CT_*
  "$REPO/ClinicalImagesPHICleared/imagesTs"                              # clinical BAF*/AVSD*
  "$REPO/nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/imagesTr"      # BAF004 / CHIPS*
  "$REPO/nnUNet_raw/Dataset071_ImageCHDClinicalOrientation/imagesTr"     # ImageCHD ct_*
  "$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTr"
  "$REPO/nnUNet_raw/Dataset090_ImageCHDPseudoCombined/imagesTs"
)
# ==============================================================================================
mkdir -p "$OUT_DIR/refined" "$OUT_DIR/prompts" "$REPO/logs"

find_image () { local c="$1" d; for d in "${IMG_DIRS[@]}"; do [ -f "$d/${c}_0000.nii.gz" ] && { echo "$d/${c}_0000.nii.gz"; return 0; }; done; return 1; }

shopt -s nullglob
for lab in "$LCC_DIR"/*.nii.gz; do
  c=$(basename "$lab" .nii.gz)
  img="$(find_image "$c" || true)"
  [ -n "$img" ] || { echo "[skip $c] no image in any IMG_DIRS"; continue; }
  out="$OUT_DIR/refined/${c}.nii.gz"
  [ -f "$out" ] && { echo "[done $c] exists"; continue; }
  echo "==== $c ===="
  python tools/run_nninteractive_refine.py --image "$img" --label "$lab" --out "$out" \
      --structures LV-BP,RV-BP,LA,RA --prompt-mode lasso \
      --save-prompts "$OUT_DIR/prompts/${c}_nnI_prompts.json"
done
echo "DONE. Refined masks -> $OUT_DIR/refined ; prompts -> $OUT_DIR/prompts"
