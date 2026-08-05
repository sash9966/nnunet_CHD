#!/bin/bash
# =============================================================================
#  consolidate_nnunet_into_project.sh
#  Make nnUNet_raw / nnUNet_preprocessed / nnUNet_results REAL directories that
#  physically live UNDER the project (/scratch/users/sastocke/nnunet_CHD), instead
#  of symlinks that point out into /scratch/users/sastocke/nnUNet.
#
#  Run this ON THE SERVER (login node), with NO training/predict jobs active.
#  It is DRY-RUN by default (prints what it would do). Re-run with APPLY=1 to act.
#  Nothing is deleted from the old location — after you verify, you remove that
#  yourself.
#
#    # 1. see what's going on:
#    bash scripts/consolidate_nnunet_into_project.sh
#    # 2. actually consolidate:
#    APPLY=1 bash scripts/consolidate_nnunet_into_project.sh
# =============================================================================
set -euo pipefail
PROJ="/scratch/users/sastocke/nnunet_CHD"
APPLY="${APPLY:-0}"

echo "project root: ${PROJ}"
echo "APPLY=${APPLY}  (0 = dry-run, 1 = perform rsync + swap)"
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
      echo "  WARNING: symlink target does not exist (${tgt}) — skipping"; echo; continue
    fi
    staging="${PROJ}/.${name}.staging"
    echo "  plan: rsync ${tgt}/  ->  ${staging}/ , verify, then replace the symlink with the real dir"
    src_files=$(find "${tgt}/" -type f 2>/dev/null | wc -l)
    src_size=$(du -sh "${tgt}" 2>/dev/null | cut -f1)
    echo "  source: ${src_files} files, ${src_size}"
    if [ "${APPLY}" != "1" ]; then
      echo "  [dry-run] set APPLY=1 to perform it"; echo; continue
    fi
    echo "  [apply] rsync (this can take a while for preprocessed/results)..."
    rm -rf "${staging}"; mkdir -p "${staging}"
    rsync -a "${tgt}/" "${staging}/"
    dst_files=$(find "${staging}/" -type f | wc -l)
    echo "  copied: ${dst_files} files (source had ${src_files})"
    if [ "${src_files}" != "${dst_files}" ]; then
      echo "  FILE-COUNT MISMATCH — aborting, nothing swapped. Staging left at ${staging}"; exit 1
    fi
    rm "${p}"                     # remove ONLY the symlink (not the data it points to)
    mv "${staging}" "${p}"
    echo "  DONE: ${p} is now a REAL directory inside the project."
    echo "  Old copy still at ${tgt} — delete it yourself after you've confirmed everything works."
  elif [ -d "${p}" ]; then
    echo "  already a REAL directory inside the project — OK"
  elif [ -e "${p}" ]; then
    echo "  exists but is not a dir/symlink (?) — inspect manually: $(ls -ld "${p}")"
  else
    echo "  missing — creating a real directory"
    [ "${APPLY}" = "1" ] && mkdir -p "${p}" && echo "  created ${p}" || echo "  [dry-run] would: mkdir -p ${p}"
  fi
  echo
done

echo "=============================================================="
echo "After APPLY=1: nnUNet_raw / preprocessed / results are real dirs under ${PROJ}."
echo "All sbatch scripts already export those exact paths, so nothing else changes."
echo "Verify a dataset is intact (e.g. ls ${PROJ}/nnUNet_raw/Dataset071_ImageCHDClinicalOrientation),"
echo "then remove the old /scratch/users/sastocke/nnUNet copies to reclaim space."
echo "=============================================================="
