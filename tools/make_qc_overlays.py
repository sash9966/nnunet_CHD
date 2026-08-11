#!/usr/bin/env python3
"""
make_qc_overlays.py
==================

Quick visual QC sheets: for each prediction (or label) volume, render 3 orthogonal mid-slices of
the CT with the 7-class segmentation overlaid, one PNG per case, plus an index.html contact sheet.

Usage:
  python tools/make_qc_overlays.py --image-dir <imagesTr or clinic images> \
      --label-dir <predictions dir> --output-dir <qc out> [--alpha 0.45]

Images are <case>_0000.nii.gz; labels are <case>.nii.gz. Memory-safe (nibabel).
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

FE = ".nii.gz"
# 0=bg (transparent), 1..7 fixed colors (LV-BP, RV-BP, LA, RA, Myo, Ao, PA)
_COLORS = [(0, 0, 0, 0), (0.20, 0.65, 0.32, 1), (0.95, 0.80, 0.30, 1), (0.78, 0.45, 0.30, 1),
           (0.30, 0.70, 0.90, 1), (0.90, 0.30, 0.30, 1), (0.95, 0.50, 0.25, 1), (0.35, 0.85, 0.45, 1)]
_CMAP = ListedColormap(_COLORS)
_NORM = BoundaryNorm(list(range(9)), _CMAP.N)


def _load(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj)


def _panel(ax, ct_slice, lab_slice, alpha, title):
    ax.imshow(np.rot90(ct_slice), cmap="gray")
    m = np.ma.masked_where(lab_slice == 0, lab_slice)
    ax.imshow(np.rot90(m), cmap=_CMAP, norm=_NORM, alpha=alpha, interpolation="nearest")
    ax.set_title(title, fontsize=8); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    img_dir, lab_dir, out = Path(args.image_dir), Path(args.label_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labs = sorted(lab_dir.glob(f"*{FE}"))
    if not labs: sys.exit(f"no label/prediction files in {lab_dir}")

    made = []
    for lab in labs:
        cid = lab.name[:-len(FE)]
        img = img_dir / f"{cid}_0000{FE}"
        if not img.is_file(): print(f"  [skip {cid}] no image {img}"); continue
        try:
            ct, seg = _load(img), _load(lab)
        except Exception as e:
            print(f"  [skip {cid}] read error {e!r}"); continue
        if ct.shape[:3] != seg.shape[:3]:
            print(f"  [skip {cid}] shape mismatch {ct.shape} vs {seg.shape}"); continue
        cx, cy, cz = (s // 2 for s in seg.shape[:3])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        _panel(axes[0], ct[:, :, cz], seg[:, :, cz], args.alpha, f"{cid}  axial z={cz}")
        _panel(axes[1], ct[:, cy, :], seg[:, cy, :], args.alpha, f"coronal y={cy}")
        _panel(axes[2], ct[cx, :, :], seg[cx, :, :], args.alpha, f"sagittal x={cx}")
        fig.tight_layout()
        png = out / f"{cid}.png"
        fig.savefig(png, dpi=110, bbox_inches="tight"); plt.close(fig)
        made.append(cid)
        print(f"  [qc] {png}")

    # contact sheet
    if made:
        html = ["<!doctype html><meta charset=utf-8><title>QC overlays</title>",
                "<h2>QC overlays</h2>",
                "<p>7-class: LV-BP RV-BP LA RA Myo Ao PA. Clinic-facing all-data model — qualitative review only.</p>"]
        for cid in made:
            html.append(f"<div style='margin:14px 0'><b>{cid}</b><br>"
                        f"<img src='{cid}.png' style='max-width:100%'></div>")
        (out/"index.html").write_text("\n".join(html))
    print(f"[qc] {len(made)} sheets -> {out}  (open index.html)")


if __name__ == "__main__":
    main()
