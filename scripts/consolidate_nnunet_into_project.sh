#!/bin/bash
# =============================================================================
#  consolidate_nnunet_into_project.sh
#  Make nnUNet_raw / nnUNet_preprocessed / nnUNet_results REAL directories that
#  physically live UNDER the project (/scratch/users/sastocke/nnunet_CHD), instead
#  of symlinks that point out into /scratch/users/sastocke/nnUNet.
#
#  Both trees are on the same /scratch filesystem, so this is an INSTANT `mv`
#  (rename) -- NO 1.5 TB copy, NO extra space. (If a target were on a different
#  filesystem it falls back to rsync + verify.)
#
#  Run ON THE SERVER (login node) with NO training/predict jobs active.
#  DRY-RUN by default; APPLY=1 to actually do it.
#
#    bash scripts/consolidate_nnunet_into_project.sh            # dry-run
#    APPLY=1 bash scripts/consolidate_nnunet_into_project.sh    # do it
# =============================================================================
set -euo pipefail
PROJ="/scratch/users/sastocke/nnunet_CHD"
APPLY="${APPLY:-0}"

echo "project root: ${PROJ}"
echo "APPLY=${APPLY}  (0 = dry-run, 1 = perform)"
echo

for name in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  p="${PROJ}/${name}"
  echo "==================== ${p} ===================="
  if [ -L "${p}" ]; then
    tgt="$(readlink -f "${p}")"
    echo "  currently a SYMLINK -> ${tgt}"
    case "${tgt}/" in
      "${PROJ}/"*) echo "  target already inside the project — nothing to do"; echo; continue;;
    esac
    if [ ! -d "${tgt}" ]; then
      echo "  WARNING: symlink target does not exist — skipping"; echo; continue
    fi
    src_files=$(find "${tgt}/" -type f 2>/dev/null | wc -l)
    src_size=$(du -sh "${tgt}" 2>/dev/null | cut -f1)
    same_fs=0
    if [ "$(stat -c %d "${tgt}")" = "$(stat -c %d "${PROJ}")" ]; then same_fs=1; fi
    echo "  source: ${src_files} files, ${src_size}  | same_filesystem=${same_fs}"

    if [ "${same_fs}" = "1" ]; then
      echo "  plan: INSTANT move  (rm the symlink; mv ${tgt} -> ${p}) — no copy, no extra space"
      if [ "${APPLY}" = "1" ]; then
        rm "${p}"                       # remove ONLY the symlink
        mv "${tgt}" "${p}"              # atomic rename on the same filesystem
        echo "  DONE: ${p} is now a REAL directory (data moved in place)."
      else
        echo "  [dry-run] set APPLY=1 to perform the move"
      fi
    else
      staging="${PROJ}/.${name}.staging"
      echo "  plan: DIFFERENT filesystem -> rsync ${tgt}/ -> ${staging}/ , verify, swap"
      if [ "${APPLY}" = "1" ]; then
        rm -rf "${staging}"; mkdir -p "${staging}"
        rsync -a "${tgt}/" "${staging}/"
        dst_files=$(find "${staging}/" -type f | wc -l)
        echo "  copied ${dst_files} files (source ${src_files})"
        [ "${src_files}" = "${dst_files}" ] || { echo "  MISMATCH — aborting, staging kept"; exit 1; }
        rm "${p}"; mv "${staging}" "${p}"
        echo "  DONE (copied). Old data still at ${tgt} — delete after verifying."
      else
        echo "  [dry-run] set APPLY=1 to perform rsync + swap"
      fi
    fi
  elif [ -d "${p}" ]; then
    echo "  already a REAL directory inside the project — OK"
  elif [ -e "${p}" ]; then
    echo "  exists but not dir/symlink — inspect: $(ls -ld "${p}")"
  else
    echo "  missing — creating a real directory"
    if [ "${APPLY}" = "1" ]; then mkdir -p "${p}"; echo "  created"; else echo "  [dry-run] would mkdir -p ${p}"; fi
  fi
  echo
done

echo "=============================================================="
echo "After APPLY=1: nnUNet_raw / preprocessed / results are REAL dirs under ${PROJ}."
echo "All sbatch scripts export those exact paths, so nothing else changes."
echo "Check: ls ${PROJ}/nnUNet_raw/Dataset071_ImageCHDClinicalOrientation"
echo "The old /scratch/users/sastocke/nnUNet/ dirs are now empty (moved) or removable."
echo "=============================================================="
