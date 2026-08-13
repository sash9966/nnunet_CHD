#!/usr/bin/env python3
"""
dice_analysis_d080.py
=====================

Per-case / per-label / whole-heart Dice for Dataset080 predictions vs GT, matching the
methodology in SegmentationDetailStandard/dice_analysis.ipynb:
  labels LV-BP..PA = 1..7,  WH = union(1..7),  Dice = 2|P∩G| / (|P|+|G|)  (empty↔empty = 1.0).

Emits per-method CSVs, mean/median summary tables, Δ-median table + improvement flags,
a violin/strip comparison plot, and an RdYlGn Δ-median heatmap.

Usage:
  python tools/dice_analysis_d080.py --gt-dir <D080/labelsTr> \
      --pred "Dataset090 (f0)=<...>/ds090_fold0" \
      --pred "Dataset091 (f0)=<...>/ds091_fold0" \
      --baseline "Dataset090 (f0)" --out-dir <...>/dice_fold0
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _SNS = True
except Exception:
    _SNS = False

LABEL_MAPPING = {"LV-BP": 1, "RV-BP": 2, "LA": 3, "RA": 4, "Myo": 5, "Ao": 6, "PA": 7}
CLASSES = list(LABEL_MAPPING.keys())
WH_LABELS = list(LABEL_MAPPING.values())
COLS = ["WH"] + CLASSES
FE = ".nii.gz"


def pid(name):
    return re.sub(r"(_image|_0000)?(\.nii(\.gz)?)?$", "", name)


def _dice(pred_mask, gt_mask):
    inter = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    return 1.0 if union == 0 else float(2.0 * inter / union)


def compute_row(pred, gt):
    row = {"WH": _dice(np.isin(pred, WH_LABELS).astype(np.float32),
                       np.isin(gt, WH_LABELS).astype(np.float32))}
    for name, lbl in LABEL_MAPPING.items():
        row[name] = _dice((pred == lbl).astype(np.float32), (gt == lbl).astype(np.float32))
    return row


def dice_df(gt_dir, pred_dir):
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    recs = []
    for gt_f in sorted(gt_dir.glob(f"*{FE}")):
        pred_f = pred_dir / gt_f.name
        if not pred_f.exists():
            print(f"  [WARN] missing prediction: {gt_f.name}"); continue
        pred = nib.load(str(pred_f)).get_fdata().astype(np.int32)
        gt = nib.load(str(gt_f)).get_fdata().astype(np.int32)
        if pred.shape != gt.shape:
            print(f"  [WARN] shape mismatch {gt_f.name}: pred={pred.shape} gt={gt.shape}"); continue
        recs.append({"Patient_ID": pid(gt_f.name), **compute_row(pred, gt)})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    methods = {}
    for spec in args.pred:
        name, d = spec.split("=", 1)
        methods[name.strip()] = d.strip()
    baseline = args.baseline or list(methods)[0]

    dfs = {}
    for name, d in methods.items():
        print(f"[compute] {name}  <- {d}")
        df = dice_df(args.gt_dir, d)
        if df.empty:
            print(f"  [skip] no cases for {name}"); continue
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(out / f"dice_{safe}.csv", index=False)
        dfs[name] = df
    if not dfs:
        sys.exit("no dice computed (no matching predictions found)")

    summ = pd.DataFrame([{"Method": n, "N": len(df), **{c: round(df[c].mean(), 3) for c in COLS}}
                         for n, df in dfs.items()]).set_index("Method")
    med = pd.DataFrame([{"Method": n, **{c: round(df[c].median(), 3) for c in COLS}}
                        for n, df in dfs.items()]).set_index("Method")
    print("\n== mean Dice ==\n" + summ.to_string())
    print("\n== median Dice ==\n" + med.to_string())
    summ.to_csv(out / "summary_mean.csv"); med.to_csv(out / "summary_median.csv")

    delta = None
    chal = [m for m in dfs if m != baseline]
    if baseline in dfs and chal:
        delta = med.loc[chal].subtract(med.loc[baseline], axis=1)
        print(f"\n== Δ median Dice (challenger − {baseline}) ==\n" + delta.round(3).to_string())
        delta.to_csv(out / "delta_median.csv")
        print("\n== improvements (Δ median > 0.01) ==")
        found = False
        for m in chal:
            for c in COLS:
                dv = float(delta.loc[m, c])
                if dv > 0.01:
                    b = float(med.loc[baseline, c]); found = True
                    print(f"  + {m}  {c}: +{dv:.3f}  ({b:.3f} -> {b + dv:.3f})")
        if not found:
            print("  (no structure improves by > 0.01 median)")
        # per-case paired WH
        paired = pd.DataFrame({"Patient_ID": dfs[baseline]["Patient_ID"],
                               f"WH_{baseline}": dfs[baseline]["WH"].values})
        for m in chal:
            paired[f"WH_{m}"] = dfs[m].set_index("Patient_ID").reindex(paired["Patient_ID"])["WH"].values
        paired.to_csv(out / "per_case_WH.csv", index=False)

    # ---- violin + strip comparison ----
    melt = pd.concat([df[COLS].assign(Method=n) for n, df in dfs.items()]).melt(
        id_vars="Method", value_vars=COLS, var_name="Class", value_name="Dice")
    fig, ax = plt.subplots(figsize=(13, 6))
    if _SNS:
        sns.set(style="whitegrid")
        sns.violinplot(data=melt, x="Class", y="Dice", hue="Method", cut=0,
                       inner="quartile", linewidth=1.4, alpha=0.85, ax=ax)
        sns.stripplot(data=melt, x="Class", y="Dice", hue="Method", dodge=True,
                      color="black", alpha=0.5, size=3, ax=ax)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[:len(dfs)], l[:len(dfs)], title="Method", loc="lower left")
    else:
        classes = COLS; xs = np.arange(len(classes)); w = 0.8 / max(len(dfs), 1)
        for i, (n, df) in enumerate(dfs.items()):
            ax.boxplot([df[c].values for c in classes],
                       positions=xs + i * w - 0.4 + w / 2, widths=w * 0.9, labels=None)
        ax.set_xticks(xs); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.02)
    ax.set_title("Dataset080 held-out Dice (fold 0): Dataset090 vs Dataset091")
    fig.tight_layout(); fig.savefig(out / "dice_violin_090_vs_091.png", dpi=140); plt.close(fig)

    # ---- Δ-median heatmap ----
    if delta is not None:
        fig, ax = plt.subplots(figsize=(max(7, len(chal) * 2.4), 4.6))
        data = delta.T
        if _SNS:
            sns.heatmap(data, annot=True, fmt=".3f", center=0, vmin=-0.15, vmax=0.15,
                        cmap="RdYlGn", linewidths=0.5, linecolor="white", ax=ax,
                        cbar_kws={"label": "Δ median Dice (challenger − baseline)"})
        else:
            im = ax.imshow(data.values, cmap="RdYlGn", vmin=-0.15, vmax=0.15, aspect="auto")
            ax.set_xticks(range(len(data.columns))); ax.set_xticklabels(data.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(data.index))); ax.set_yticklabels(data.index)
            for (yy, xx), v in np.ndenumerate(data.values):
                ax.text(xx, yy, f"{v:.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="Δ median Dice")
        ax.set_title(f"Δ median Dice vs {baseline} (green = better)")
        fig.tight_layout(); fig.savefig(out / "delta_heatmap.png", dpi=140); plt.close(fig)

    print(f"\n[done] CSVs + plots -> {out}")


if __name__ == "__main__":
    main()
