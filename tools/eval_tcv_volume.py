#!/usr/bin/env python3
"""
eval_tcv_volume.py  (workstream C — TCV transplant matching)
Score predicted Total Cardiac Volume against expert masks with the CLINICALLY meaningful metrics.
Dice alone does not answer the transplant-matching question: what matters is whether the predicted
VOLUME is right and unbiased.

Per case: GT TCV (mL), predicted TCV (mL), signed error, absolute percent error, Dice.
Aggregate: MAPE, mean bias (signed - does the model systematically over/under-estimate?), SD,
Bland-Altman limits of agreement (bias +/- 1.96 SD), and mean/median Dice.

NOTE: the donor/recipient TCV-RATIO error is not computed here because it requires the clinical
transplant pairing (which donor went to which recipient), which is not in the imaging data.

Usage:
  python tools/eval_tcv_volume.py --gt-dir <labelsTs> --pred-dir <predictions> --out <dir> [--label 1]
"""
import argparse, csv, glob, os, sys
import numpy as np
import nibabel as nib

FE = ".nii.gz"


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--gt-dir", required=True)
    a.add_argument("--pred-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--label", type=int, default=1, help="TCV foreground label id")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for gp in sorted(glob.glob(os.path.join(args.gt_dir, "*" + FE))):
        case = os.path.basename(gp)[:-len(FE)]
        pp = os.path.join(args.pred_dir, case + FE)
        if not os.path.isfile(pp):
            print("  [skip %s] no prediction" % case); continue
        gi = nib.load(gp)
        g = np.rint(np.asanyarray(gi.dataobj)).astype(np.int16) == args.label
        p = np.rint(np.asanyarray(nib.load(pp).dataobj)).astype(np.int16) == args.label
        if g.shape != p.shape:
            print("  [skip %s] shape mismatch %s vs %s" % (case, g.shape, p.shape)); continue
        ml = float(np.prod(gi.header.get_zooms()[:3])) / 1000.0     # mm^3 -> mL
        gv, pv = float(g.sum()) * ml, float(p.sum()) * ml
        inter = int(np.logical_and(g, p).sum()); s = int(g.sum()) + int(p.sum())
        rows.append({"case": case, "cohort": case[0],
                     "gt_mL": round(gv, 1), "pred_mL": round(pv, 1),
                     "err_mL": round(pv - gv, 1),
                     "abs_pct_err": round(100.0 * abs(pv - gv) / gv, 2) if gv > 0 else "",
                     "signed_pct_err": round(100.0 * (pv - gv) / gv, 2) if gv > 0 else "",
                     "dice": round(2.0 * inter / s, 4) if s else 1.0})
    if not rows:
        sys.exit("no cases scored")

    with open(os.path.join(args.out, "tcv_volume_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n%-14s %-6s %9s %9s %9s %9s %7s" % ("case", "cohort", "gt_mL", "pred_mL", "err_mL", "abs%err", "dice"))
    for r in rows:
        print("%-14s %-6s %9s %9s %9s %9s %7s" % (r["case"], r["cohort"], r["gt_mL"], r["pred_mL"],
                                                  r["err_mL"], r["abs_pct_err"], r["dice"]))
    ape = np.array([r["abs_pct_err"] for r in rows if r["abs_pct_err"] != ""], float)
    spe = np.array([r["signed_pct_err"] for r in rows if r["signed_pct_err"] != ""], float)
    diff = np.array([r["err_mL"] for r in rows], float)
    dice = np.array([r["dice"] for r in rows], float)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    print("\n=== aggregate (n=%d) ===" % len(rows))
    print("  MAPE (abs %% volume error) : %.2f %%" % ape.mean())
    print("  median abs %% error        : %.2f %%" % np.median(ape))
    print("  bias (signed %%)           : %+.2f %%   <- systematic over(+)/under(-) estimation" % spe.mean())
    print("  bias (mL)                 : %+.1f mL" % bias)
    print("  Bland-Altman 95%% limits   : %+.1f to %+.1f mL" % (bias - 1.96 * sd, bias + 1.96 * sd))
    print("  Dice mean / median        : %.3f / %.3f" % (dice.mean(), np.median(dice)))
    print("\n  Reference (Deep learning for automated TCV): Dice 0.94+/-0.03, TCV MAPE 5.5%%")
    print("  (that paper reported ~10.5%% MAPE on diseased/pre-transplant vs ~4.5%% on normal hearts)")
    print("\nCSV -> %s" % os.path.join(args.out, "tcv_volume_eval.csv"))


if __name__ == "__main__":
    main()
