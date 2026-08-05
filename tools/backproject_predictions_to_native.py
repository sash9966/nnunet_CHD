#!/usr/bin/env python3
"""
backproject_predictions_to_native.py
=====================================

Map grid512 predictions BACK to each case's native CT grid, so you get
native-FOV / native-spacing labels that overlay the ORIGINAL image -- the seed
for a clinical labeled dataset + correct-and-retrain loop.

Two steps per case:
  1. INVERSE of the index-space resize (tools/resize_to_imagechd_grid.py): the
     prediction lives on the 512x512x221 grid; we zoom it back to the native image
     array shape with NEAREST-NEIGHBOUR (order=0, preserves label ids), then stamp
     the native image's spacing/origin/direction so it aligns to the original CT.
     (The forward resize was a non-physical stretch, so we invert the stretch --
     not a physical resample.)
  2. Tiny post-processing: per foreground label, keep only the LARGEST connected
     component (26-connectivity). Removes speckle + stray blobs. Small disconnected
     vessel bits get dropped -- accepted: clean, low-false-positive seed labels.

Matching: prediction <case>.nii.gz  <->  native image <case>_0000.nii.gz.

Usage:
  python tools/backproject_predictions_to_native.py \
    --pred-dir   .../ClinicalImagesPHICleared/predictions/ds071 \
    --native-dir .../ClinicalImagesPHICleared/imagesTs \
    --output-dir .../ClinicalImagesPHICleared/predictions/ds071__native_labels
  # add --no-lcc to skip the connected-component cleanup
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def zoom_to_shape(arr: np.ndarray, target_shape_zyx, order: int) -> np.ndarray:
    """Zoom arr to (approximately) target shape, then center-crop/pad to EXACT
    target -- the same crop/pad convention resize_to_imagechd_grid.py uses, so the
    inverse lines up with the forward."""
    zoom = tuple(target_shape_zyx[i] / arr.shape[i] for i in range(3))
    out = ndi.zoom(arr, zoom, order=order, mode="nearest", prefilter=False)
    slices = []
    for cur, tgt in zip(out.shape, target_shape_zyx):
        if cur > tgt:
            start = (cur - tgt) // 2
            slices.append(slice(start, start + tgt))
        else:
            slices.append(slice(0, cur))
    out = out[tuple(slices)]
    pads = []
    for cur, tgt in zip(out.shape, target_shape_zyx):
        before = max((tgt - cur) // 2, 0)
        pads.append((before, max(tgt - cur - before, 0)))
    if any(b or a for b, a in pads):
        out = np.pad(out, pads, mode="edge")
    return out


def largest_cc_per_label(lab: np.ndarray, connectivity: int = 3):
    """Keep only the largest connected component of EACH foreground label id."""
    st = ndi.generate_binary_structure(3, connectivity)
    out = np.zeros_like(lab)
    removed = {}
    for lid in np.unique(lab):
        if lid == 0:
            continue
        mask = lab == lid
        cc, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        counts = np.bincount(cc.ravel())
        counts[0] = 0  # ignore background of this labeling
        largest = int(counts.argmax())
        keep = cc == largest
        out[keep] = lid
        removed[int(lid)] = int(mask.sum() - keep.sum())
    return out, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="grid512 predictions (<case>.nii.gz)")
    ap.add_argument("--native-dir", required=True, help="native images (<case>_0000.nii.gz)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-lcc", action="store_true", help="skip largest-connected-component cleanup")
    ap.add_argument("--connectivity", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pred_dir, native_dir, out_dir = Path(args.pred_dir), Path(args.native_dir), Path(args.output_dir)
    for d in (pred_dir, native_dir):
        if not d.is_dir():
            raise SystemExit(f"ERROR: not a dir: {d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "reports"; report_dir.mkdir(exist_ok=True)

    rows, failures = [], []
    for pf in sorted(pred_dir.glob("*.nii.gz")):
        case = pf.name[: -len(".nii.gz")]
        native = native_dir / f"{case}_0000.nii.gz"
        if not native.is_file():
            failures.append(f"{case}: no native image {native.name}"); continue
        dst = out_dir / pf.name
        if dst.exists() and not args.overwrite:
            failures.append(f"{case}: {dst} exists (use --overwrite)"); continue

        nat = sitk.ReadImage(str(native))
        pred = sitk.ReadImage(str(pf))
        P = sitk.GetArrayFromImage(pred)                       # (z,y,x) on grid512
        native_shape_zyx = sitk.GetArrayViewFromImage(nat).shape
        ids_before = sorted(int(x) for x in np.unique(P))

        # 1) inverse resize -> native array shape (nearest-neighbour keeps label ids)
        P_nat = zoom_to_shape(P.astype(np.int16), native_shape_zyx, order=0).astype(np.uint8)

        # 2) largest-CC cleanup per label
        removed = {}
        if not args.no_lcc:
            P_nat, removed = largest_cc_per_label(P_nat, args.connectivity)

        out_img = sitk.GetImageFromArray(P_nat)
        out_img.CopyInformation(nat)                           # exact native geometry
        sitk.WriteImage(out_img, str(dst), True)

        ids_after = sorted(int(x) for x in np.unique(P_nat))
        rows.append({
            "case": case,
            "native_size_xyz": "x".join(map(str, nat.GetSize())),
            "native_spacing_xyz": ",".join(f"{v:.3f}" for v in nat.GetSpacing()),
            "grid_pred_size_xyz": "x".join(map(str, pred.GetSize())),
            "label_ids_before": ",".join(map(str, ids_before)),
            "label_ids_after": ",".join(map(str, ids_after)),
            "lcc_voxels_removed": json.dumps(removed),
            "output": str(dst),
        })
        print(f"[{case}] grid{pred.GetSize()} -> native{nat.GetSize()} | ids {ids_after} | "
              f"lcc removed {removed if removed else '(off)'}")

    if rows:
        with (report_dir / "backproject_report.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (report_dir / "summary.json").write_text(json.dumps(
        {"pred_dir": str(pred_dir), "native_dir": str(native_dir), "output_dir": str(out_dir),
         "n_cases": len(rows), "lcc": (not args.no_lcc), "connectivity": args.connectivity,
         "failures": failures}, indent=2))
    print(f"\n[backproject] wrote {len(rows)} native-space label(s) -> {out_dir}")
    if failures:
        print(f"[backproject] {len(failures)} skipped/failed:"); [print("   -", f) for f in failures]
    print("Overlay these on the ORIGINAL CT (native imagesTs) in Slicer, correct, then add to training.")


if __name__ == "__main__":
    main()
