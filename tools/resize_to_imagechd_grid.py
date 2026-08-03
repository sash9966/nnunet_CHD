#!/usr/bin/env python3
"""
Resize clinical nnU-Net images to an ImageCHD-like voxel grid.

Diagnostic / inference front-end. Does an INDEX-SPACE resize of each image to a
fixed voxel array size, then writes a NIfTI with ImageCHD-like spacing. This is
NOT a physically honest resampling; it intentionally stretches the clinical FOV
into the voxel/grid scale an ImageCHD-trained model expects. Empirically this is
what makes inference work on clinical CT (the model segments well once anatomy is
at the trained grid scale).

Default grid:
  size    = 512 x 512 x 221
  spacing = 1.0 x 1.0 x 1.9059633016586304 mm

Pass --input / --output explicitly (no personal defaults baked in).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


LPS_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def parse_triplet(text: str, cast=float):
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three values, got {text!r}")
    return tuple(cast(p) for p in parts)


def orientation(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())


def percentiles(array: np.ndarray) -> dict:
    values = np.percentile(array[np.isfinite(array)], [1, 50, 99])
    return {"p1": float(values[0]), "p50": float(values[1]), "p99": float(values[2])}


def resize_array_to_zyx(array_zyx: np.ndarray, target_shape_zyx) -> np.ndarray:
    zoom = tuple(target_shape_zyx[i] / array_zyx.shape[i] for i in range(3))
    resized = ndimage.zoom(
        array_zyx.astype(np.float32, copy=False),
        zoom, order=1, mode="nearest", prefilter=False,
    )
    slices = []
    for current, target in zip(resized.shape, target_shape_zyx):
        if current > target:
            start = (current - target) // 2
            slices.append(slice(start, start + target))
        else:
            slices.append(slice(0, current))
    result = resized[tuple(slices)]
    pads = []
    for current, target in zip(result.shape, target_shape_zyx):
        before = max((target - current) // 2, 0)
        after = max(target - current - before, 0)
        pads.append((before, after))
    if any(before or after for before, after in pads):
        result = np.pad(result, pads, mode="edge")
    if result.shape != target_shape_zyx:
        raise RuntimeError(f"Unexpected output shape {result.shape}, expected {target_shape_zyx}")
    return result.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize images to an ImageCHD-like fixed grid.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", default="512,512,221", help="Target size as x,y,z voxels.")
    parser.add_argument("--spacing", default="1,1,1.9059633016586304", help="Output spacing as x,y,z mm.")
    parser.add_argument("--direction-mode", choices=["preserve", "lps-identity"], default="preserve")
    parser.add_argument("--origin-mode", choices=["preserve", "imagechd"], default="preserve")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_size_xyz = parse_triplet(args.size, int)
    target_spacing_xyz = parse_triplet(args.spacing, float)
    target_shape_zyx = (target_size_xyz[2], target_size_xyz[1], target_size_xyz[0])

    args.output.mkdir(parents=True, exist_ok=True)
    report_dir = args.output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sorted(args.input.glob("*.nii.gz")):
        dst = args.output / src.name
        if dst.exists() and not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        image = sitk.ReadImage(str(src))
        array_before = sitk.GetArrayFromImage(image)
        array_after = resize_array_to_zyx(array_before, target_shape_zyx)
        output = sitk.GetImageFromArray(array_after)
        output.SetSpacing(tuple(float(v) for v in target_spacing_xyz))
        output.SetDirection(
            LPS_IDENTITY_DIRECTION if args.direction_mode == "lps-identity" else image.GetDirection()
        )
        output.SetOrigin((-1.0, -1.0, 1.0) if args.origin_mode == "imagechd" else image.GetOrigin())
        output = sitk.Cast(output, sitk.sitkFloat32)
        sitk.WriteImage(output, str(dst), True)
        b, a = percentiles(array_before.astype(np.float32, copy=False)), percentiles(array_after)
        rows.append({
            "case": src.name,
            "orientation_before": orientation(image), "orientation_after": orientation(output),
            "spacing_before_xyz": json.dumps([round(float(x), 6) for x in image.GetSpacing()]),
            "spacing_after_xyz": json.dumps([round(float(x), 6) for x in output.GetSpacing()]),
            "size_before_xyz": json.dumps(list(image.GetSize())),
            "size_after_xyz": json.dumps(list(output.GetSize())),
            "p1_before": round(b["p1"], 3), "p50_before": round(b["p50"], 3), "p99_before": round(b["p99"], 3),
            "p1_after": round(a["p1"], 3), "p50_after": round(a["p50"], 3), "p99_after": round(a["p99"], 3),
            "output": str(dst),
        })

    if rows:
        with (report_dir / "resize_imagechd_grid_report.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    summary = {
        "input_dir": str(args.input), "output_dir": str(args.output),
        "target_size_xyz": list(target_size_xyz), "target_spacing_xyz_mm": list(target_spacing_xyz),
        "direction_mode": args.direction_mode, "origin_mode": args.origin_mode, "case_count": len(rows),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
