#!/usr/bin/env python3
"""
lcc_postprocess.py  (workstream D)
Per-label largest-connected-component (LCC) cleanup on multi-label masks — the same idea as the
seed-generation LCC, but applied AFTER promptable refinement, to test whether nnInteractive/SeqSeg
leave spurious disconnected islands.

Reports, per case/label, how many components existed and what fraction of voxels was dropped, so the
"refined" vs "refined+LCC" variants can be compared quantitatively.

CAUTION: for tubular structures a per-label LCC can delete a GENUINELY separate distal branch
(e.g. a left/right pulmonary branch not voxel-connected to the trunk). Use --skip-labels to exempt
vessels if that shows up.

Usage:
  python tools/lcc_postprocess.py --in-dir <refined> --out-dir <refined_lcc> [--skip-labels 6,7]
"""
import argparse, glob, os, sys
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

FE = ".nii.gz"


def lcc_per_label(arr, skip=()):
    out = np.zeros_like(arr)
    report = []
    for sid in sorted(set(int(v) for v in np.unique(arr)) - {0}):
        m = arr == sid
        tot = int(m.sum())
        if sid in skip:
            out[m] = sid
            report.append((sid, -1, tot, 0))          # -1 = skipped
            continue
        lab, n = ndi.label(m)
        if n <= 1:
            out[m] = sid
            report.append((sid, n, tot, 0))
            continue
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        kept = lab == keep
        out[kept] = sid
        report.append((sid, n, tot, tot - int(kept.sum())))
    return out, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-labels", default="", help="comma-separated label ids to leave untouched (e.g. 6,7 for vessels)")
    args = ap.parse_args()
    skip = {int(s) for s in args.skip_labels.split(",") if s.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, "*" + FE)))
    if not files:
        sys.exit("no masks in " + args.in_dir)
    print("skip-labels (left untouched): %s" % (sorted(skip) or "none"))
    grand = 0
    for f in files:
        im = nib.load(f)
        arr = np.rint(np.asanyarray(im.dataobj)).astype(np.int16)
        out, rep = lcc_per_label(arr, skip)
        nib.save(nib.Nifti1Image(out, im.affine), os.path.join(args.out_dir, os.path.basename(f)))
        dropped = sum(d for _, _, _, d in rep)
        grand += dropped
        detail = " ".join("L%d:n=%s,-%d" % (s, ("skip" if n == -1 else n), d) for s, n, _, d in rep)
        print("  %-16s dropped=%8d  %s" % (os.path.basename(f)[:-len(FE)], dropped, detail))
    print("[done] %d cases, %d voxels dropped total -> %s" % (len(files), grand, args.out_dir))


if __name__ == "__main__":
    main()
