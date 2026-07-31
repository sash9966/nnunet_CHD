#!/usr/bin/env python3
"""
build_dataset072_multifov.py
============================

Build Dataset072_ImageCHDMultiFOV from an already-LPS nnU-Net dataset (default
Dataset071_ImageCHDClinicalOrientation) by ADDING cardiac-centered field-of-view
variants alongside the untouched originals. Motivation: pure ImageCHD trains well
but generalizes poorly to clinical CT whose FOV is tight around the heart; giving
nnU-Net a range of cardiac FOVs shifts its shape/spacing fingerprint toward the
clinical domain -- WITHOUT inventing interpolated detail.

For every input imagesTr/labelsTr pair it emits (deterministic, no random aug):
  1. <case>_full      -- exact copy (image + label untouched, native HU/spacing)
  2. <case>_bbox60mm  -- crop to (nonzero-label bbox + 60 mm/side)
  3. <case>_bbox30mm  -- crop to (nonzero-label bbox + 30 mm/side, tighter)

Cropping is a pure sub-region extraction: spacing, orientation, HU and label
values are preserved exactly; NO resampling, NO upsampling to finer spacing. FOV
diversity comes from the crop extent, not from interpolation. nnU-Net does its own
resampling to a single target spacing at plan_and_preprocess time.

Outputs: imagesTr/, labelsTr/, dataset.json, and QA reports (per-case CSV +
aggregate summary). All generated cases are training cases (no test set here).

Usage (cluster):
  python tools/build_dataset072_multifov.py \
    --nnunet-raw $nnUNet_raw \
    --source-dataset Dataset071_ImageCHDClinicalOrientation \
    --margins-mm 60,30 --overwrite

NOTE ON CROSS-VALIDATION: the 3 variants of one patient share anatomy. If you let
nnU-Net auto-generate splits, variants of the same case can land in different folds
=> validation leakage. Use a GROUPED split (all <case>_* in the same fold). This
script writes case_groups.json to make that easy; --help for how to apply it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CHANNEL_RE = re.compile(r"_(\d{4})\.nii\.gz$")


def orient_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def nz_hist(arr: np.ndarray) -> dict:
    """Histogram of NONZERO (foreground) label values -- order/crop invariant."""
    fg = arr[arr != 0]
    v, c = np.unique(fg, return_counts=True)
    return {int(x): int(y) for x, y in zip(v, c)}


def label_value_set(arr: np.ndarray) -> list:
    return sorted(int(x) for x in np.unique(arr))


def geom(img: sitk.Image) -> tuple:
    return (img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())


def geom_matches(a: sitk.Image, b: sitk.Image, atol: float = 1e-4) -> bool:
    ga, gb = geom(a), geom(b)
    return (ga[0] == gb[0]
            and np.allclose(ga[1], gb[1], atol=atol)
            and np.allclose(ga[2], gb[2], atol=atol)
            and np.allclose(ga[3], gb[3], atol=atol))


def physical_extent_mm(img: sitk.Image) -> tuple:
    return tuple(round(s * sp, 2) for s, sp in zip(img.GetSize(), img.GetSpacing()))


def crop_region_to_label_bbox(img: sitk.Image, lab: sitk.Image, margin_mm: float):
    """Return (index, size) in sitk (x,y,z) order for the nonzero-label bbox grown
    by margin_mm per side (clamped to the volume). None if the label is empty."""
    la = sitk.GetArrayViewFromImage(lab)          # numpy order: (z, y, x)
    nz = np.argwhere(la != 0)
    if nz.size == 0:
        return None
    mn = nz.min(axis=0)                            # (z, y, x)
    mx = nz.max(axis=0)
    spacing = img.GetSpacing()                     # sitk (x, y, z)
    size = img.GetSize()                           # sitk (x, y, z)
    index, roi = [], []
    for sitk_ax in range(3):                       # 0=x, 1=y, 2=z
        arr_ax = 2 - sitk_ax                        # x->arr2, y->arr1, z->arr0
        m = int(np.ceil(margin_mm / spacing[sitk_ax]))
        lo = max(int(mn[arr_ax]) - m, 0)
        hi = min(int(mx[arr_ax]) + m, size[sitk_ax] - 1)
        index.append(lo)
        roi.append(hi - lo + 1)
    return index, roi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--source-dataset", default="Dataset071_ImageCHDClinicalOrientation")
    ap.add_argument("--target-id", type=int, default=72)
    ap.add_argument("--target-name", default="ImageCHDMultiFOV")
    ap.add_argument("--margins-mm", default="60,30",
                    help="comma-separated crop margins per side (mm); each -> a bbox<M>mm variant")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.nnunet_raw:
        sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    raw = Path(args.nnunet_raw).resolve()
    src = raw / args.source_dataset
    if not (src / "imagesTr").is_dir() or not (src / "labelsTr").is_dir():
        sys.exit(f"ERROR: {src} missing imagesTr/labelsTr")
    margins = [float(x) for x in args.margins_mm.split(",") if x.strip() != ""]

    sjson = json.loads((src / "dataset.json").read_text())
    fe = sjson.get("file_ending", ".nii.gz")
    channel_names = sjson.get("channel_names", {"0": "CT"})
    labels = sjson["labels"]

    target_folder = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target_folder
    if dst.exists() and not args.overwrite:
        sys.exit(f"ERROR: {dst} exists (use --overwrite)")
    for sub in ("imagesTr", "labelsTr"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # gather source cases: case_id -> [channel image paths]
    cases: dict[str, list] = {}
    for f in sorted((src / "imagesTr").glob(f"*{fe}")):
        m = CHANNEL_RE.search(f.name)
        if not m:
            continue
        cases.setdefault(f.name[: m.start()], []).append(f)
    ids = sorted(cases)
    if args.limit:
        ids = ids[: args.limit]
    print(f"[d072] source {src.name}: {len(ids)} case(s) | margins/side (mm): {margins}")

    variants = [("full", None)] + [(f"bbox{int(m)}mm", m) for m in margins]
    qa_rows, failures = [], []
    orient_counts = Counter()
    spacing_counts = Counter()
    extents = defaultdict(list)          # variant -> [extent tuples]
    groups = defaultdict(list)           # source case -> [output case ids] (for grouped CV)
    n_out = 0

    for cid in ids:
        lab_path = src / "labelsTr" / f"{cid}{fe}"
        if not lab_path.is_file():
            failures.append(f"{cid}: missing label"); continue
        lab0 = sitk.ReadImage(str(lab_path))
        imgs0 = [sitk.ReadImage(str(p)) for p in cases[cid]]
        src_nz = nz_hist(sitk.GetArrayViewFromImage(lab0))
        src_vals = label_value_set(sitk.GetArrayViewFromImage(lab0))
        src_orient = orient_code(lab0)
        src_spacing = tuple(round(x, 4) for x in lab0.GetSpacing())

        for vname, margin in variants:
            if margin is None:
                lab_v = lab0
                imgs_v = imgs0
                region = None
            else:
                region = crop_region_to_label_bbox(imgs0[0], lab0, margin)
                if region is None:
                    failures.append(f"{cid}_{vname}: empty label, cannot crop"); continue
                idx, roi = region
                lab_v = sitk.RegionOfInterest(lab0, roi, idx)
                imgs_v = [sitk.RegionOfInterest(im, roi, idx) for im in imgs0]

            out_id = f"{cid}_{vname}"
            la_v = sitk.GetArrayViewFromImage(lab_v)
            out_vals = label_value_set(la_v)
            out_orient = orient_code(lab_v)
            out_spacing = tuple(round(x, 4) for x in lab_v.GetSpacing())

            # ---- QA checks (write the case only if it passes) ----
            fg_preserved = (nz_hist(la_v) == src_nz)          # all foreground voxels retained
            no_upsample = (out_spacing == src_spacing)        # crop must NOT change spacing
            gmatch = all(geom_matches(im, lab_v) for im in imgs_v)
            orient_ok = (out_orient == src_orient)
            ok = fg_preserved and no_upsample and gmatch and orient_ok
            if not ok:
                why = []
                if not fg_preserved: why.append("foreground label voxels changed")
                if not no_upsample:  why.append(f"spacing changed {src_spacing}->{out_spacing}")
                if not gmatch:       why.append("image/label geometry mismatch")
                if not orient_ok:    why.append(f"orientation {out_orient}!={src_orient}")
                failures.append(f"{out_id}: " + "; ".join(why)); continue

            for im, p in zip(imgs_v, cases[cid]):
                ch = CHANNEL_RE.search(p.name).group(1)
                sitk.WriteImage(im, str(dst / "imagesTr" / f"{out_id}_{ch}{fe}"))
            sitk.WriteImage(lab_v, str(dst / "labelsTr" / f"{out_id}{fe}"))

            ext = physical_extent_mm(lab_v)
            orient_counts[out_orient] += 1
            spacing_counts[out_spacing] += 1
            extents[vname].append(ext)
            groups[cid].append(out_id)
            n_out += 1
            qa_rows.append({
                "source_case": cid, "variant": vname, "output_case": out_id,
                "size": "x".join(map(str, lab_v.GetSize())),
                "spacing_mm": ",".join(f"{x:.3f}" for x in lab_v.GetSpacing()),
                "extent_mm": ",".join(f"{x:.1f}" for x in ext),
                "orientation": out_orient,
                "label_values_before": ",".join(map(str, src_vals)),
                "label_values_after": ",".join(map(str, out_vals)),
                "geometry_match": gmatch,
                "fg_labels_preserved": fg_preserved,
                "spacing_preserved": no_upsample,
            })

    if n_out == 0:
        sys.exit("ERROR: no output cases generated (check failures above)")

    # ---- dataset.json ----
    ds_json = {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": {k: int(v) for k, v in labels.items()},
        "numTraining": n_out,
        "file_ending": fe,
        "name": target_folder,
        "description": (f"Multi-FOV derived from {args.source_dataset}: per case a full copy + "
                        f"cardiac-bbox crops at {margins} mm/side. Crop-only (no resample/HU change)."),
        "source_dataset": args.source_dataset,
        "variants": [v for v, _ in variants],
        "margins_mm": margins,
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    (dst / "case_groups.json").write_text(json.dumps(groups, indent=1))

    # ---- QA: per-case CSV ----
    with open(dst / "qa_percase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qa_rows[0].keys())); w.writeheader(); w.writerows(qa_rows)

    # ---- QA: aggregate summary ----
    def ext_stats(vname):
        arr = np.array(extents[vname]) if extents[vname] else np.zeros((0, 3))
        if arr.size == 0:
            return None
        return {ax: {"min": round(float(arr[:, i].min()), 1),
                     "median": round(float(np.median(arr[:, i])), 1),
                     "max": round(float(arr[:, i].max()), 1)}
                for i, ax in enumerate(("x", "y", "z"))}
    summary = {
        "n_source_cases": len(ids),
        "n_generated_cases": n_out,
        "variants": [v for v, _ in variants],
        "orientation_counts": dict(orient_counts),
        "spacing_distribution_mm": {",".join(f"{x:.3f}" for x in k): v for k, v in spacing_counts.items()},
        "physical_fov_mm_by_variant": {v: ext_stats(v) for v, _ in variants},
        "label_value_preservation": {
            "all_foreground_preserved": all(r["fg_labels_preserved"] for r in qa_rows),
            "all_spacing_preserved": all(r["spacing_preserved"] for r in qa_rows),
            "all_geometry_match": all(r["geometry_match"] for r in qa_rows),
        },
        "failures": failures,
    }
    (dst / "qa_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[d072] wrote {n_out} case(s) -> {dst}")
    print(f"  orientation: {dict(orient_counts)}")
    print(f"  spacing distribution (mm -> #cases): "
          f"{ {','.join(f'{x:.2f}' for x in k): v for k, v in spacing_counts.items()} }")
    for v, _ in variants:
        s = ext_stats(v)
        if s:
            print(f"  FOV {v:9s} median extent mm: "
                  f"x={s['x']['median']} y={s['y']['median']} z={s['z']['median']}")
    print(f"  QA: qa_percase.csv, qa_summary.json, case_groups.json")
    if failures:
        print(f"[d072] {len(failures)} FAILURE(S):")
        for f in failures:
            print("   -", f)
        sys.exit(1)
    print("[d072] OK — all outputs LPS-consistent, spacing preserved (no upsampling), "
          "foreground labels + geometry intact.")
    print(f"[d072] NEXT: nnUNetv2_plan_and_preprocess -d {args.target_id} -pl nnUNetPlannerResEncM "
          f"-c 3d_fullres --verify_dataset_integrity")
    print("[d072] CV: use a GROUPED split (case_groups.json) so a patient's variants don't "
          "straddle train/val.")


if __name__ == "__main__":
    main()
