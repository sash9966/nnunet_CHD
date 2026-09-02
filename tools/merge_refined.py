#!/usr/bin/env python3
"""
merge_refined.py  (workstream D)
STITCH the refinement outputs into one finalized multi-label mask:
  nnInteractive (all 7 structures)  +  SeqSeg vessel masks (Aorta=6, Pulmonary=7)

Vessels are UNIONed, not replaced: nnInteractive holds the root/proximal vessel, SeqSeg extends
distally, so the union is how they "attach and extend one another". The union is only written where
the voxel is background or already that same vessel — chambers/myocardium keep priority, so an
over-reaching vessel trace can never eat a chamber.

Usage:
  python tools/merge_refined.py --nninteractive-dir <refined> --seqseg-nifti-dir <from seqseg_to_nifti.py> \
      --out-dir <merged> [--vessels Aorta=6,Pulmonary=7]
"""
import argparse, glob, os
import numpy as np
import nibabel as nib

FE = ".nii.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nninteractive-dir", required=True)
    ap.add_argument("--seqseg-nifti-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vessels", default="Aorta=6,Pulmonary=7")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    vmap = {}
    for kv in args.vessels.split(","):
        k, v = kv.split("="); vmap[k.strip()] = int(v)

    files = sorted(glob.glob(os.path.join(args.nninteractive_dir, "*" + FE)))
    print("%-16s %-10s %10s %10s %10s" % ("case", "vessel", "nnI_vox", "seqseg_vox", "added"))
    tot_added = 0
    for f in files:
        case = os.path.basename(f)[:-len(FE)]
        im = nib.load(f)
        lab = np.rint(np.asanyarray(im.dataobj)).astype(np.int16)
        changed = False
        for vname, vid in vmap.items():
            sp = os.path.join(args.seqseg_nifti_dir, "%s_%s%s" % (case, vname, FE))
            if not os.path.isfile(sp):
                continue
            sm = np.asanyarray(nib.load(sp).dataobj) > 0
            if sm.shape != lab.shape:
                print("  [%s/%s] shape mismatch %s vs %s -> skipped" % (case, vname, sm.shape, lab.shape))
                continue
            nni_vox = int((lab == vid).sum())
            # union only into background or the same vessel: chambers/myo keep priority
            addable = sm & ((lab == 0) | (lab == vid))
            added = int((addable & (lab != vid)).sum())
            lab[addable] = vid
            tot_added += added; changed = changed or added > 0
            print("%-16s %-10s %10d %10d %10d" % (case, vname, nni_vox, int(sm.sum()), added))
        nib.save(nib.Nifti1Image(lab, im.affine), os.path.join(args.out_dir, case + FE))
    print("[done] %d cases, %d vessel voxels added by SeqSeg -> %s" % (len(files), tot_added, args.out_dir))


if __name__ == "__main__":
    main()
