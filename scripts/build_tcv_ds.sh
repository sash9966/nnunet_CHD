#!/bin/bash
# =============================================================================
#  build_tcv_ds.sh
#  Clean, standard nnU-Net v2 dataset builder for the TCV (total cardiac volume)
#  transplant project. Rebuilds one DatasetXXX from the OAK source with ABSOLUTE
#  symlinks (no Codex-style relative-link breakage), proper _0000 image naming,
#  a minimal dataset.json, and a built-in SimpleITK read-test of every image.
#
#  Cohorts by case-name first letter:  d = donor, r = recipient, x = undefined.
#  Default recipe = DONOR->RECIPIENT baseline: train D+X, hold out all R in imagesTs.
#  (R never enters training -> leak-free. Masks for R go to labelsTs for offline scoring.)
#
#  Run on Sherlock:  bash scripts/build_tcv_ds.sh
# =============================================================================
set -euo pipefail

# ===================== EDIT THESE =====================
SRC=/oak/stanford/groups/amarsden/elenasm/TCV_project
RAW=/scratch/users/sastocke/nnunet_CHD/nnUNet_raw
DS_ID=503
DS_NAME="Dataset${DS_ID}_TCV_DonorX_ToRecipient"
TRAIN_PREFIXES="d x"     # cohorts -> training (imagesTr/labelsTr)
INFER_PREFIXES="r"       # cohorts -> held-out inference (imagesTs, +labelsTs for scoring)
# ======================================================

IMG="$SRC/img_anon"; SEG="$SRC/seg_anon"
DST="$RAW/$DS_NAME"
[ -d "$IMG" ] || { echo "ERROR: source images dir not found: $IMG"; exit 1; }
[ -d "$SEG" ] || { echo "ERROR: source labels dir not found: $SEG"; exit 1; }

echo ">> rebuilding $DST (wiping any old copy)"
rm -rf "$DST"
mkdir -p "$DST"/imagesTr "$DST"/labelsTr "$DST"/imagesTs "$DST"/labelsTs

link_one () {  # <case> <img_out_dir> <lab_out_dir>
  local c="$1" io="$2" lo="$3"
  local img="$IMG/${c}_img_anon.nii.gz" seg="$SEG/${c}_seg_anon.nii.gz"
  [ -f "$img" ] || { echo "   MISSING image: $c"; return; }
  [ -f "$seg" ] || { echo "   MISSING label: $c"; return; }
  ln -sfn "$(readlink -f "$img")" "$io/${c}_0000.nii.gz"   # ABSOLUTE link
  ln -sfn "$(readlink -f "$seg")" "$lo/${c}.nii.gz"
}

ntr=0; nts=0
for img in "$IMG"/*_img_anon.nii.gz; do
  c=$(basename "$img" _img_anon.nii.gz); pfx=${c:0:1}
  if   [[ " $TRAIN_PREFIXES " == *" $pfx "* ]]; then link_one "$c" "$DST/imagesTr" "$DST/labelsTr"; ntr=$((ntr+1))
  elif [[ " $INFER_PREFIXES " == *" $pfx "* ]]; then link_one "$c" "$DST/imagesTs" "$DST/labelsTs"; nts=$((nts+1)); fi
done

cat > "$DST/dataset.json" <<JSON
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "TCV": 1},
  "numTraining": $ntr,
  "file_ending": ".nii.gz"
}
JSON

echo ">> train=$ntr  infer(imagesTs)=$nts"
echo ">> VERIFY: read-testing every training image with SimpleITK..."
python - "$DST" <<'PY'
import sys, glob, os
import SimpleITK as sitk
d=sys.argv[1]; bad=0; n=0
for f in sorted(glob.glob(os.path.join(d,"imagesTr","*.nii.gz"))):
    n+=1
    try: sitk.ReadImage(f)
    except Exception as e: bad+=1; print("  UNREADABLE:", os.path.basename(f), "|", repr(e)[:90])
print(f">> read-test: {n-bad}/{n} images OK" + ("" if bad==0 else f"  ({bad} BAD -- see above)"))
PY

echo ">> DONE. If the read-test is all OK, preprocess with:"
echo "   nnUNetv2_plan_and_preprocess -d $DS_ID -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity"
