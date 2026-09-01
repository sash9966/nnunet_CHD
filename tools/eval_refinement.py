#!/usr/bin/env python3
"""
eval_refinement.py  (workstream D — promptable refinement)
Consolidate a refinement run into one report: per case per structure, compare the LCC seed label vs
the nnInteractive-refined mask (volumes in mL + Dice), summarize the prompts used (centerline / lasso /
seed counts), and note SeqSeg vessel outputs. Flags likely over/under-segmentation for QC triage.

Usage:
  python tools/eval_refinement.py \
    --lcc-dir     /scratch/.../nnUNet_raw/Dataset090_ImageCHDPseudoCombined/predictions/ds090__grid2native_lcc \
    --refined-dir /scratch/.../chd_refinement/out/nninteractive_ds090/refined \
    --prompts-dir /scratch/.../chd_refinement/prompts/ds090 \
    --seqseg-dir  /scratch/.../chd_refinement/out/seqseg_ds090 \
    --out         /scratch/.../chd_refinement/eval_ds090
"""
import argparse, json, os, glob, csv, sys
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_to_prompts import DEFAULT_STRUCTURES, VESSELS, CHAMBERS

FE = ".nii.gz"


def load(p):
    im = nib.load(p)
    a = np.rint(np.asanyarray(im.dataobj)).astype(np.int16)
    ml_per_vox = float(np.prod(im.header.get_zooms()[:3])) / 1000.0   # mm^3 -> mL
    return a, ml_per_vox


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int(np.logical_and(a, b).sum()) / s


def flag(vl, vr, d):
    if np.isnan(d):
        return ""
    if d < 0.5:
        return "CHECK(low-dice)"
    if vl > 0 and (vr / vl > 2.0 or vr / vl < 0.5):
        return "CHECK(volume)"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lcc-dir", required=True)
    ap.add_argument("--refined-dir", required=True, help="nnInteractive refined masks (<case>.nii.gz)")
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--seqseg-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    per_struct_dice = {}
    lccs = sorted(glob.glob(os.path.join(args.lcc_dir, "*" + FE)))
    if not lccs:
        sys.exit("no LCC labels in " + args.lcc_dir)

    for lp in lccs:
        case = os.path.basename(lp)[:-len(FE)]
        lab, mlv = load(lp)
        rp = os.path.join(args.refined_dir, case + FE)
        ref = load(rp)[0] if os.path.isfile(rp) else None
        ref_ok = ref is not None and ref.shape == lab.shape
        pj = os.path.join(args.prompts_dir, case + "_prompts.json")
        prompts = json.load(open(pj)) if os.path.isfile(pj) else {"structures": {}}

        for name, sid in DEFAULT_STRUCTURES.items():
            vl = float((lab == sid).sum()) * mlv
            in_lcc = (lab == sid).sum() > 0
            pr = prompts.get("structures", {}).get(name, {})
            ncl = len(pr.get("centerline_voxel", []))
            nseed = len(pr.get("seeds_world_r", []))
            nlasso = sum(len(v) for v in pr.get("lasso_slices", {}).values())
            # SeqSeg output presence for vessels
            seq = ""
            if name in VESSELS and args.seqseg_dir:
                vdir = os.path.join(args.seqseg_dir, case, name)
                seq = str(len(glob.glob(os.path.join(vdir, "*.vt*")) + glob.glob(os.path.join(vdir, "*" + FE)))) if os.path.isdir(vdir) else "-"

            if name in CHAMBERS:
                # nnInteractive refines chambers -> compare volume + Dice vs the LCC seed
                if not in_lcc and not (ref_ok and (ref == sid).sum() > 0):
                    continue
                vr = float((ref == sid).sum()) * mlv if ref_ok else float("nan")
                d = dice(lab == sid, ref == sid) if ref_ok else float("nan")
                vr_disp = round(vr, 2) if ref_ok else ""
                d_disp = round(d, 4) if ref_ok else ""
                fl = flag(vl, vr, d) if ref_ok else "no-refined"
                if ref_ok:
                    per_struct_dice.setdefault(name, []).append(d)
            elif name in VESSELS:
                # SeqSeg handles vessels (not nnInteractive) -> report seed prompts + SeqSeg output presence
                if not in_lcc:
                    continue
                vr_disp = ""; d_disp = ""
                fl = ("seqseg:" + seq) if seq not in ("", "-") else ("no-seqseg" if args.seqseg_dir else "seeds-only")
            else:
                if not in_lcc:
                    continue
                vr_disp = ""; d_disp = ""; fl = "myo(lcc)"

            rows.append({"case": case, "structure": name, "vol_lcc_ml": round(vl, 2),
                         "vol_refined_ml": vr_disp, "dice": d_disp,
                         "centerline_pts": ncl, "seeds": nseed, "lasso_pts": nlasso,
                         "seqseg_files": seq, "flag": fl})

    csv_path = os.path.join(args.out, "refinement_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # printed report
    print("\n=== per-case / per-structure ===")
    hdr = "%-14s %-10s %9s %9s %7s %4s %5s %6s %6s  %s" % (
        "case", "struct", "vol_lcc", "vol_ref", "dice", "cl", "seed", "lasso", "seqsg", "flag")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print("%-14s %-10s %9s %9s %7s %4s %5s %6s %6s  %s" % (
            r["case"], r["structure"], r["vol_lcc_ml"], r["vol_refined_ml"], r["dice"],
            r["centerline_pts"], r["seeds"], r["lasso_pts"], r["seqseg_files"], r["flag"]))

    print("\n=== chamber Dice (LCC seed vs nnInteractive-refined) ===")
    for name in [n for n in DEFAULT_STRUCTURES if n in per_struct_dice]:
        ds = per_struct_dice[name]
        print("  %-8s n=%2d  mean=%.3f  median=%.3f  min=%.3f" % (
            name, len(ds), float(np.mean(ds)), float(np.median(ds)), float(np.min(ds))))
    n_check = sum(1 for r in rows if str(r["flag"]).startswith("CHECK"))
    n_ref = len({r["case"] for r in rows if r["vol_refined_ml"] != ""})
    print("\ncases with a refined mask: %d / %d LCC cases" % (n_ref, len(lccs)))
    print("rows flagged for QC: %d" % n_check)
    print("CSV -> " + csv_path)


if __name__ == "__main__":
    main()
