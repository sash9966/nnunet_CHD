#!/usr/bin/env python3
"""
eval_vs_gt.py  (workstream D — validation on expert-GT cases, e.g. Dataset080)
Score three segmentation versions against expert ground truth, per structure + whole-heart:
  1. LCC          — the raw pseudo-label seed (resize->predict->backproject+LCC)
  2. nnInteractive— re-prompted refinement (all 7 structures)
  3. nnI + SeqSeg — same, but vessels (Ao/PA) replaced by SeqSeg's native trace where available

Dice per structure (1..7) + whole-heart (union 1..7). NOTE: on vessels the +SeqSeg Dice may DROP vs GT
because SeqSeg traces distal vessel the manual GT never drew — judge vessels visually, chambers/myo by Dice.

Usage:
  python tools/eval_vs_gt.py --gt-dir <D080/labelsTr> --lcc-dir <ds090__grid2native_lcc> \
     --nninteractive-dir <out/nninteractive_*/refined> [--seqseg-dir <out/seqseg_*>] \
     --out <eval_dir> [--cases BAF004,CHIPS002]
"""
import argparse, os, glob, csv, sys
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_to_prompts import DEFAULT_STRUCTURES, VESSELS

FE = ".nii.gz"
IDS = DEFAULT_STRUCTURES                      # name -> id (1..7)
VESSEL_IDS = {IDS[n]: n for n in VESSELS}     # {6:Aorta, 7:Pulmonary}


def load(p):
    return np.rint(np.asanyarray(nib.load(p).dataobj)).astype(np.int16)


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else round(2.0 * int(np.logical_and(a, b).sum()) / s, 4)


def seqseg_vessel_mask(seqseg_dir, case, vname, shape):
    """Load a SeqSeg vessel mask as a boolean volume if it wrote a NIfTI on the native grid.
    Returns None if absent or shape-mismatched (e.g. SeqSeg emitted only a surface)."""
    if not seqseg_dir:
        return None
    cands = glob.glob(os.path.join(seqseg_dir, case, vname, "*" + FE))
    for c in cands:
        m = load(c)
        if m.shape == shape:
            return m > 0
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--lcc-dir", required=True)
    ap.add_argument("--nninteractive-dir", required=True, help="the refined/ folder (<case>.nii.gz)")
    ap.add_argument("--seqseg-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    want = {c.strip() for c in args.cases.split(",")} if args.cases else None

    gts = sorted(glob.glob(os.path.join(args.gt_dir, "*" + FE)))
    rows = []
    agg = {}   # (structure, method) -> [dice,...]

    for gp in gts:
        case = os.path.basename(gp)[:-len(FE)]
        if want and case not in want:
            continue
        gt = load(gp)
        lp = os.path.join(args.lcc_dir, case + FE)
        npi = os.path.join(args.nninteractive_dir, case + FE)
        if not os.path.isfile(lp):
            print("[skip %s] no LCC seed" % case); continue
        lcc = load(lp)
        nni = load(npi) if os.path.isfile(npi) else None
        if nni is not None and nni.shape != gt.shape:
            print("[warn %s] nnI shape != GT; skipping nnI/fused" % case); nni = None

        # fused = nnI, but vessels replaced by SeqSeg trace where available
        fused = nni.copy() if nni is not None else None
        if fused is not None:
            for vid, vname in VESSEL_IDS.items():
                sm = seqseg_vessel_mask(args.seqseg_dir, case, vname, gt.shape)
                if sm is not None:
                    fused[fused == vid] = 0
                    fused[sm] = vid

        for name, sid in IDS.items():
            g = gt == sid
            if g.sum() == 0:
                continue                                    # structure not in GT for this case
            d_lcc = dice(lcc == sid, g)
            d_nni = dice(nni == sid, g) if nni is not None else ""
            d_fus = dice(fused == sid, g) if fused is not None else ""
            rows.append({"case": case, "structure": name, "dice_lcc": d_lcc,
                         "dice_nnI": d_nni, "dice_nnI_seqseg": d_fus})
            for meth, val in (("LCC", d_lcc), ("nnI", d_nni), ("nnI+SeqSeg", d_fus)):
                if val != "":
                    agg.setdefault((name, meth), []).append(val)
        # whole-heart (union 1..7)
        whg = np.isin(gt, list(IDS.values()))
        d_lcc = dice(np.isin(lcc, list(IDS.values())), whg)
        d_nni = dice(np.isin(nni, list(IDS.values())), whg) if nni is not None else ""
        d_fus = dice(np.isin(fused, list(IDS.values())), whg) if fused is not None else ""
        rows.append({"case": case, "structure": "WholeHeart", "dice_lcc": d_lcc,
                     "dice_nnI": d_nni, "dice_nnI_seqseg": d_fus})
        for meth, val in (("LCC", d_lcc), ("nnI", d_nni), ("nnI+SeqSeg", d_fus)):
            if val != "":
                agg.setdefault(("WholeHeart", meth), []).append(val)

    if not rows:
        sys.exit("no GT cases scored (check --cases / LCC coverage)")

    csv_path = os.path.join(args.out, "dice_vs_gt.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n=== per-case Dice vs GT (LCC | nnI | nnI+SeqSeg) ===")
    for r in rows:
        print("  %-12s %-11s  %-6s %-6s %-6s" % (r["case"], r["structure"],
              r["dice_lcc"], r["dice_nnI"], r["dice_nnI_seqseg"]))

    print("\n=== MEAN Dice by structure (n cases) ===")
    print("  %-11s %8s %8s %12s" % ("structure", "LCC", "nnI", "nnI+SeqSeg"))
    order = list(IDS.keys()) + ["WholeHeart"]
    for name in order:
        def m(meth):
            v = agg.get((name, meth)); return "%.3f(%d)" % (float(np.mean(v)), len(v)) if v else "-"
        if any((name, x) in agg for x in ("LCC", "nnI", "nnI+SeqSeg")):
            print("  %-11s %8s %8s %12s" % (name, m("LCC"), m("nnI"), m("nnI+SeqSeg")))
    print("\nCSV -> " + csv_path)
    print("Reminder: vessel (Aorta/Pulmonary) Dice can DROP with +SeqSeg by design (traces beyond GT) — judge those visually.")


if __name__ == "__main__":
    main()
