#!/bin/bash
# =============================================================================
#  fix_d090_symlinks.sh
#  Repair dangling symlinks in Dataset090 after a /scratch purge: re-point them
#  at reuploaded source data, and TRACE anything that still can't be found.
#
#  Pure bash (ls/readlink/ln) — no GPU, no conda, safe to run on the login node.
#  SAFE BY DEFAULT: dry-run (reports only). Re-run with APPLY=1 to actually relink:
#      bash scripts/fix_d090_symlinks.sh              # dry-run, paste this output
#      APPLY=1 bash scripts/fix_d090_symlinks.sh      # do the relinking
# =============================================================================
set -uo pipefail
shopt -s nullglob

APPLY="${APPLY:-0}"
RAW="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
REPO="/scratch/users/sastocke/nnunet_CHD"
D090="${RAW}/Dataset090_ImageCHDPseudoCombined"

# Source dirs to relink FROM (first match by basename wins). Add paths here as we
# find them from the "old:" column below (e.g. the ds071 LCC label dir).
SRC_DIRS=(
  "${RAW}/Dataset012_Fanweidata/imagesTr"                    # Fanwei images (reuploaded)
  "${RAW}/Dataset071_ImageCHDClinicalOrientation/imagesTr"   # ImageCHD images
  "${RAW}/Dataset071_ImageCHDClinicalOrientation/labelsTr"   # ImageCHD labels
  "${REPO}/ClinicalImagesPHICleared/imagesTs"                # clinical images
  "${REPO}/ClinicalImagesPHICleared/imagesTr"
  "${RAW}/Dataset080_ClinicalCaseSanjibDetailed/imagesTr"     # Dataset080 clinical test images (held-out in D090)
)

echo "======================================================================"
echo "APPLY=${APPLY}   (0=dry-run/report only, 1=actually relink)"
echo "D090 = ${D090}"
[ -d "${D090}" ] || { echo "ERROR: Dataset090 not found at ${D090}"; exit 1; }
echo "source dirs:"
for s in "${SRC_DIRS[@]}"; do printf "   %-9s %s\n" "$([ -d "$s" ] && echo '[ok]' || echo '[MISSING]')" "$s"; done
echo "======================================================================"

find_src () {                       # echo first source path that has basename $1
  local b="$1" s
  for s in "${SRC_DIRS[@]}"; do [ -f "$s/$b" ] && { echo "$s/$b"; return 0; }; done
  return 1
}

dangling=0; fixable=0; relinked=0; missing=0
declare -a MISSING_LIST=()

for d in imagesTr labelsTr imagesTs; do
  [ -d "${D090}/$d" ] || continue
  for f in "${D090}/$d/"*.nii.gz; do
    [ -e "$f" ] && continue                          # resolves fine (or real file) -> skip
    dangling=$((dangling+1))
    b="$(basename "$f")"
    old="$(readlink "$f" 2>/dev/null)"
    if src="$(find_src "$b")"; then
      fixable=$((fixable+1))
      if [ "$APPLY" = "1" ]; then
        ln -sfn "$src" "$f" && relinked=$((relinked+1))
        printf "RELINKED  %-9s %-36s -> %s\n" "$d" "$b" "$src"
      else
        printf "FIXABLE   %-9s %-36s -> %s\n              (old: %s)\n" "$d" "$b" "$src" "$old"
      fi
    else
      missing=$((missing+1)); MISSING_LIST+=("$d/$b|$old")
      printf "MISSING   %-9s %-36s\n              (old: %s)\n" "$d" "$b" "$old"
    fi
  done
done

echo
echo "==================== SUMMARY ===================="
echo "dangling symlinks : $dangling"
echo "fixable from srcs : $fixable"
[ "$APPLY" = "1" ] && echo "relinked          : $relinked"
echo "still MISSING     : $missing"
if [ "$missing" -gt 0 ]; then
  echo "--- still-missing, classified (source needs reloading/relinking) ---"
  printf '%s\n' "${MISSING_LIST[@]}" | grep -E "ct_[0-9]+_image" >/dev/null && echo "  * ImageCHD (ct_*): reload/relink Dataset071 source"
  printf '%s\n' "${MISSING_LIST[@]}" | grep -E "/CT_[0-9]"       >/dev/null && echo "  * Fanwei (CT_*): case not in your reupload"
  printf '%s\n' "${MISSING_LIST[@]}" | grep -E "BAF|AVSD|CHIPS"  >/dev/null && echo "  * clinical: reload ClinicalImagesPHICleared"
  echo "  (the 'old:' path on each MISSING line above is exactly where that file used to live)"
fi
echo "================================================="
[ "$APPLY" != "1" ] && echo && echo ">>> DRY-RUN. To relink:  APPLY=1 bash scripts/fix_d090_symlinks.sh"
