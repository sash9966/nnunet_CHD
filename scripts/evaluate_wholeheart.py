#!/usr/bin/env python3
"""
evaluate_wholeheart.py
----------------------
Evaluate a binary whole-heart prediction directory against ground-truth labels,
with topology proxies that matter for the Stage-2 decomposition decision.

Per-case metrics
----------------
Volumetric : Dice, IoU, false-positive voxels, false-negative voxels,
             pred volume, GT volume, volume ratio (pred / GT).
Surface    : Hausdorff 95th percentile (mm), mean surface distance (mm).
             Computed via scipy.ndimage.distance_transform_edt with spacing.
Topology   : connected-component count, largest-component fraction, hole count
             (binary_fill_holes diff), skeleton branch count (optional, requires
             scikit-image — gracefully skipped with a warning if missing).

Comparison mode (`--compare-to MULTICLASS_PRED_DIR`)
----------------------------------------------------
Reads the same cases from a Dataset030-style multiclass prediction folder,
binarizes the predictions (`>0`) on the fly, and emits a wide CSV with one
column per metric per model so the binary heart network can be compared to
the existing multiclass model collapsed to a binary mask.

CLI
---
    python scripts/evaluate_wholeheart.py
        --pred-dir   /path/to/predictions          # binary heart predictions (Dataset040)
        --gt-dir     /path/to/labelsTs             # binary heart ground truth (Dataset040)
        --out        eval_wholeheart.csv
        [--compare-to /path/to/multiclass_preds]   # optional: Dataset030 multiclass preds
        [--compare-label binary_heart]             # label name in the comparison CSV
        [--spacing 1.0 1.0 1.0]                    # override spacing from NIfTI header
        [--skip-surface]                           # skip Hausdorff + MSD (faster)

The script writes `<out>.csv` (one row per case + a summary mean/median row).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import warnings
from pathlib import Path
from typing import Optional

import nibabel
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

try:
    from acvl_utils.morphology.morphology_helper import label_with_component_sizes
    HAS_ACVL = True
except Exception:
    HAS_ACVL = False

try:
    from skimage.morphology import skeletonize_3d
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def dice_iou_volumes(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_v = int(pred.sum())
    gt_v = int(gt.sum())
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = (2.0 * inter / (pred_v + gt_v)) if (pred_v + gt_v) > 0 else float("nan")
    iou = (inter / union) if union > 0 else float("nan")
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return {
        "dice": dice,
        "iou": iou,
        "pred_vox": pred_v,
        "gt_vox": gt_v,
        "fp_vox": fp,
        "fn_vox": fn,
        "vol_ratio": (pred_v / gt_v) if gt_v > 0 else float("nan"),
    }


def surface_distances(
    pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
) -> dict:
    """Hausdorff 95-th percentile and mean surface distance using EDT.

    For each binary mask, compute the distance transform of its complement to
    get distance-to-surface for every voxel outside; sample these distances at
    the other mask's boundary to get the symmetric surface distances.
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return {"hd95_mm": float("nan"), "msd_mm": float("nan")}

    # Boundary voxels (binary erosion gap) — using a simple definition: voxels in
    # the mask that have at least one zero neighbour.  Faster than morphology
    # for our purpose: a voxel is a boundary if it's in the mask AND the
    # distance-transform of the OTHER mask at this position is > 0.
    # Distance from each voxel to the nearest zero in the OTHER mask:
    dt_pred = distance_transform_edt(np.logical_not(pred), sampling=spacing)
    dt_gt = distance_transform_edt(np.logical_not(gt), sampling=spacing)

    # Surface voxels of pred = pred voxels adjacent to ¬pred (dt_pred == 0 for
    # all pred voxels, so use dt_gt at the pred surface instead).
    # Cheap surface extraction: a voxel is on the surface if it's in the mask
    # and ANY 6-neighbour is outside the mask.
    pred_surf = _boundary_voxels(pred)
    gt_surf = _boundary_voxels(gt)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    hd95 = float(np.percentile(all_d, 95))
    msd = float(all_d.mean())
    return {"hd95_mm": hd95, "msd_mm": msd}


def _boundary_voxels(mask: np.ndarray) -> np.ndarray:
    """Return a boolean array marking voxels on the boundary of a 3D mask."""
    inside = mask.astype(bool)
    if not inside.any():
        return inside
    # Shift in each cardinal direction and find mask voxels with at least one
    # zero neighbour.  Pad-with-zero so the volume edge counts as boundary.
    pad = np.pad(inside, 1, mode="constant", constant_values=False)
    has_zero_neighbour = (
        ~pad[:-2, 1:-1, 1:-1] | ~pad[2:, 1:-1, 1:-1]
        | ~pad[1:-1, :-2, 1:-1] | ~pad[1:-1, 2:, 1:-1]
        | ~pad[1:-1, 1:-1, :-2] | ~pad[1:-1, 1:-1, 2:]
    )
    return inside & has_zero_neighbour


def topology_proxies(pred: np.ndarray) -> dict:
    """Number of connected components, largest-component fraction, hole count,
    skeleton branch count (optional)."""
    total_fg = int(pred.sum())
    if total_fg == 0:
        return {
            "n_components": 0,
            "largest_component_fraction": float("nan"),
            "n_holes": 0,
            "skeleton_branch_count": float("nan") if HAS_SKIMAGE else float("nan"),
        }

    # Connected components
    if HAS_ACVL:
        _, comp_sizes = label_with_component_sizes(pred.astype(bool), connectivity=3)
        sizes = list(comp_sizes.values())
    else:
        # Fallback to scipy
        from scipy.ndimage import label as ndi_label
        labeled, n = ndi_label(pred.astype(bool))
        sizes = [int((labeled == i).sum()) for i in range(1, n + 1)]
    n_components = len(sizes)
    largest_fraction = (max(sizes) / total_fg) if sizes else float("nan")

    # Holes: voxels that become foreground after binary_fill_holes minus original
    filled = binary_fill_holes(pred.astype(bool))
    hole_voxels = int(filled.sum()) - total_fg
    # Translate hole-voxel count into a hole "object" count via CC on the filled-minus-original
    if hole_voxels > 0:
        if HAS_ACVL:
            _, hole_sizes = label_with_component_sizes(
                (filled & ~pred.astype(bool)), connectivity=3
            )
            n_holes = len(hole_sizes)
        else:
            from scipy.ndimage import label as ndi_label
            _, n_holes = ndi_label(filled & ~pred.astype(bool))
    else:
        n_holes = 0

    if HAS_SKIMAGE:
        # Skeleton branch count: degree-3+ nodes in the 1-voxel-thick skeleton.
        skel = skeletonize_3d(pred.astype(bool))
        # Count degree of each skeleton voxel by summing the 26-neighbour skel value.
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0
        degree = convolve(skel.astype(int), kernel, mode="constant", cval=0)
        # Branch points are skel voxels with > 2 neighbours.
        branch_count = int(((skel > 0) & (degree > 2)).sum())
        skel_voxels = int((skel > 0).sum())
    else:
        branch_count = float("nan")
        skel_voxels = float("nan")

    return {
        "n_components": n_components,
        "largest_component_fraction": largest_fraction,
        "n_holes": n_holes,
        "skeleton_branch_count": branch_count,
        "skeleton_voxels": skel_voxels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_binary(path: Path, threshold_above_zero: bool = True) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI labelmap as a bool array + voxel spacing in mm."""
    img = nibabel.load(str(path))
    data = np.asanyarray(img.dataobj)
    if threshold_above_zero:
        binary = data > 0
    else:
        binary = data.astype(bool)
    # Header zooms come as (sx, sy, sz) in voxel units; convert to a 3-tuple.
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return binary, spacing


def _pair_cases(pred_dir: Path, gt_dir: Path) -> list[tuple[str, Path, Path]]:
    """Match prediction files to GT files by case ID (filename without .nii.gz)."""
    preds = {p.name.removesuffix(".nii.gz"): p
             for p in pred_dir.iterdir() if p.suffix == ".gz" and p.name.endswith(".nii.gz")}
    gts = {p.name.removesuffix(".nii.gz"): p
           for p in gt_dir.iterdir() if p.suffix == ".gz" and p.name.endswith(".nii.gz")}
    common = sorted(preds.keys() & gts.keys())
    if not common:
        raise SystemExit(
            f"ERROR: no matching case IDs between {pred_dir} and {gt_dir}.\n"
            f"  pred IDs (first 5): {sorted(preds.keys())[:5]}\n"
            f"  gt IDs   (first 5): {sorted(gts.keys())[:5]}"
        )
    missing_in_pred = sorted(gts.keys() - preds.keys())
    missing_in_gt = sorted(preds.keys() - gts.keys())
    if missing_in_pred:
        warnings.warn(f"GT cases without a prediction: {missing_in_pred}")
    if missing_in_gt:
        warnings.warn(f"Prediction cases without a GT: {missing_in_gt}")
    return [(c, preds[c], gts[c]) for c in common]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(
    case_id: str,
    pred_path: Path,
    gt_path: Path,
    skip_surface: bool,
    multiclass_pred_path: Optional[Path],
) -> dict:
    pred, spacing = _load_binary(pred_path)
    gt, _ = _load_binary(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(
            f"shape mismatch for {case_id}: pred {pred.shape} vs gt {gt.shape}"
        )

    row: dict = {"case_id": case_id, "spacing_mm": "x".join(f"{s:.3f}" for s in spacing)}
    row.update({f"binary__{k}": v for k, v in dice_iou_volumes(pred, gt).items()})
    if not skip_surface:
        row.update({f"binary__{k}": v for k, v in surface_distances(pred, gt, spacing).items()})
    row.update({f"binary__{k}": v for k, v in topology_proxies(pred).items()})

    if multiclass_pred_path is not None:
        if multiclass_pred_path.is_file():
            mc_pred, _ = _load_binary(multiclass_pred_path, threshold_above_zero=True)
            if mc_pred.shape != gt.shape:
                warnings.warn(f"multiclass-pred shape mismatch for {case_id}; skipping comparison")
            else:
                row.update({f"mc__{k}": v for k, v in dice_iou_volumes(mc_pred, gt).items()})
                if not skip_surface:
                    row.update({f"mc__{k}": v for k, v in surface_distances(mc_pred, gt, spacing).items()})
                row.update({f"mc__{k}": v for k, v in topology_proxies(mc_pred).items()})
        else:
            warnings.warn(f"multiclass prediction not found for {case_id}: {multiclass_pred_path}")
    return row


def _summary_row(rows: list[dict]) -> dict:
    """Compute mean and median across all numeric columns."""
    summary: dict = {"case_id": "MEAN", "spacing_mm": ""}
    for key in rows[0].keys():
        if key in ("case_id", "spacing_mm"):
            continue
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                  and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
        summary[key] = (statistics.mean(values) if values else float("nan"))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate binary whole-heart predictions against GT with topology metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pred-dir", required=True, help="Directory of binary whole-heart predictions (.nii.gz).")
    p.add_argument("--gt-dir", required=True, help="Directory of binary whole-heart ground truth (.nii.gz).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--compare-to", default=None,
                   help="Optional directory of multiclass predictions (binarized via `>0`) to compare against.")
    p.add_argument("--skip-surface", action="store_true",
                   help="Skip Hausdorff and mean surface distance (faster on large volumes).")
    args = p.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    mc_dir = Path(args.compare_to).expanduser().resolve() if args.compare_to else None

    if not pred_dir.is_dir():
        raise SystemExit(f"ERROR: --pred-dir not a directory: {pred_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"ERROR: --gt-dir not a directory: {gt_dir}")
    if mc_dir is not None and not mc_dir.is_dir():
        raise SystemExit(f"ERROR: --compare-to not a directory: {mc_dir}")

    pairs = _pair_cases(pred_dir, gt_dir)
    print(f"Evaluating {len(pairs)} cases from {pred_dir}")
    if mc_dir is not None:
        print(f"  comparing against multiclass predictions in {mc_dir}")
    if not HAS_ACVL:
        print("  [INFO] acvl_utils not available — using scipy.ndimage.label fallback for components.")
    if not HAS_SKIMAGE:
        print("  [INFO] scikit-image not available — skeleton branch count will be NaN.")

    rows: list[dict] = []
    for case_id, pred_path, gt_path in pairs:
        mc_path = (mc_dir / f"{case_id}.nii.gz") if mc_dir else None
        try:
            row = evaluate_one(case_id, pred_path, gt_path, args.skip_surface, mc_path)
        except Exception as e:
            print(f"  [FAIL] {case_id}: {e}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"  [OK] {case_id}  "
              f"Dice={row['binary__dice']:.4f}  "
              f"CC={row['binary__n_components']}  "
              f"largest={row['binary__largest_component_fraction']:.3f}")

    if not rows:
        raise SystemExit("ERROR: no cases successfully evaluated.")

    summary = _summary_row(rows)
    rows.append(summary)

    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}  ({len(rows) - 1} cases + 1 mean row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
