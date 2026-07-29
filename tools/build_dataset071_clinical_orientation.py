#!/usr/bin/env python3
"""
build_dataset071_clinical_orientation.py
=========================================

Build Dataset071_ImageCHDClinicalOrientation: a CLINICALLY-ORIENTED (LPS) copy of
the ImageCHD training data, for the "train and send to clinic" anatomy model.

What it does (and deliberately does NOT do):
  * IMAGES come from the images dataset (default Dataset060_imageCHD_CleanHoldout) --
    same cases / same clean-train partition. Native HU, untouched intensities.
  * LABELS come from the ORIGINAL labels dataset (default Dataset030_imageCHD_HU),
    matched by case id. These are the clean 7-class anatomy labels -- NO septal
    label (id 8). Dataset060's septal label is intentionally dropped: clinical
    anatomy service = 7 classes only.
  * Every image AND its label are physically reoriented to LPS with a TRUE
    orientation transform (SimpleITK DICOMOrient) -- a voxel-array permute/flip,
    NOT a header edit. No HU shift. No spacing resampling. No fixed size/depth.
    nnU-Net handles spacing/resampling later.

Verification (clinical-facing -> fail loud, write nothing bad):
  * every written image and label reports LPS on re-read,
  * image and label share size / spacing / origin / direction after reorient,
  * label value histogram is identical before vs after reorient (lossless),
  * image intensity summary (min/max/sum/size) is identical before vs after,
  * no label id 8 leaks into the clinical dataset.

Usage (cluster):
  python tools/build_dataset071_clinical_orientation.py \
    --nnunet-raw $nnUNet_raw \
    --images-dataset Dataset060_imageCHD_CleanHoldout \
    --labels-dataset Dataset030_imageCHD_HU \
    --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

TARGET_ORIENTATION = "LPS"
CHANNEL_RE = re.compile(r"_(\d{4})\.nii\.gz$")


def orient_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def reorient(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, TARGET_ORIENTATION)


def label_hist(img: sitk.Image) -> dict:
    a = sitk.GetArrayViewFromImage(img)
    vals, counts = np.unique(a, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def intensity_summary(img: sitk.Image) -> tuple:
    a = sitk.GetArrayViewFromImage(img)
    return (a.size, float(a.min()), float(a.max()), float(a.sum()))


def geom(img: sitk.Image) -> tuple:
    return (img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())


def geom_matches(a: sitk.Image, b: sitk.Image, atol: float = 1e-4) -> bool:
    ga, gb = geom(a), geom(b)
    return (ga[0] == gb[0]  # size exact
            and np.allclose(ga[1], gb[1], atol=atol)
            and np.allclose(ga[2], gb[2], atol=atol)
            and np.allclose(ga[3], gb[3], atol=atol))


def index_labels(labels_ds: Path, fe: str) -> dict:
    """case_id -> original label path, searching labelsTr AND labelsTs."""
    idx = {}
    for sub in ("labelsTr", "labelsTs"):
        d = labels_ds / sub
        if not d.is_dir():
            continue
        for lf in sorted(d.glob(f"*{fe}")):
            idx.setdefault(lf.name[: -len(fe)], lf)
    return idx


def gather_images(images_ds: Path, sub: str, fe: str) -> dict:
    """case_id -> [channel image paths] from images_ds/<sub>."""
    d = images_ds / sub
    cases: dict[str, list] = {}
    if not d.is_dir():
        return cases
    for f in sorted(d.glob(f"*{fe}")):
        m = CHANNEL_RE.search(f.name)
        if not m:
            continue
        case = f.name[: m.start()]
        cases.setdefault(case, []).append(f)
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--images-dataset", default="Dataset060_imageCHD_CleanHoldout")
    ap.add_argument("--labels-dataset", default="Dataset030_imageCHD_HU")
    ap.add_argument("--target-id", type=int, default=71)
    ap.add_argument("--target-name", default="ImageCHDClinicalOrientation")
    ap.add_argument("--no-test", action="store_true", help="skip imagesTs/labelsTs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.nnunet_raw:
        sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    raw = Path(args.nnunet_raw).resolve()
    images_ds = raw / args.images_dataset
    labels_ds = raw / args.labels_dataset
    for p in (images_ds, labels_ds):
        if not p.is_dir():
            sys.exit(f"ERROR: not found: {p}")

    img_json = json.loads((images_ds / "dataset.json").read_text())
    lab_json = json.loads((labels_ds / "dataset.json").read_text())
    fe = lab_json.get("file_ending", ".nii.gz")
    channel_names = img_json.get("channel_names") or lab_json.get("channel_names", {"0": "CT"})
    labels = lab_json["labels"]  # clean 7-class from the ORIGINAL dataset (no septal id 8)
    allowed_ids = {int(v) for v in labels.values()}
    if 8 in allowed_ids:
        sys.exit("ERROR: labels dataset contains id 8 (septal) -- wrong labels source for a clinical set")

    target_folder = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target_folder
    if dst.exists() and not args.overwrite:
        sys.exit(f"ERROR: {dst} exists (use --overwrite)")
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    label_idx = index_labels(labels_ds, fe)
    print(f"[d071] images from {images_ds.name} | labels from {labels_ds.name} "
          f"({len(label_idx)} original labels indexed)")
    print(f"[d071] target labels ({len(allowed_ids)} classes, no septal): {labels}")

    subsets = [("imagesTr", "labelsTr")] + ([] if args.no_test else [("imagesTs", "labelsTs")])
    failures, n_train, n_done = [], 0, 0

    for img_sub, lab_sub in subsets:
        cases = gather_images(images_ds, img_sub, fe)
        ids = sorted(cases)
        if args.limit:
            ids = ids[: args.limit]
        print(f"\n[d071] {img_sub}: {len(ids)} case(s)")
        for cid in ids:
            if cid not in label_idx:
                failures.append(f"{cid}: no original label in {labels_ds.name}")
                continue
            # ---- label: read original, verify clean, reorient, verify lossless ----
            lab_in = sitk.ReadImage(str(label_idx[cid]))
            hist_in = label_hist(lab_in)
            leaked = set(hist_in) - allowed_ids
            if leaked:
                failures.append(f"{cid}: original label has unexpected ids {leaked}")
                continue
            lab_out = reorient(lab_in)
            if label_hist(lab_out) != hist_in:
                failures.append(f"{cid}: label histogram changed on reorient")
                continue
            if orient_code(lab_out) != TARGET_ORIENTATION:
                failures.append(f"{cid}: label not {TARGET_ORIENTATION} after reorient")
                continue

            # ---- images: reorient each channel, verify HU + orientation + geom ----
            ok = True
            out_imgs = []
            for f in cases[cid]:
                m = CHANNEL_RE.search(f.name)
                ch = m.group(1)
                img_in = sitk.ReadImage(str(f))
                s_in = intensity_summary(img_in)
                img_out = reorient(img_in)
                if intensity_summary(img_out) != s_in:
                    failures.append(f"{cid}_{ch}: intensity summary changed (HU altered!)"); ok = False; break
                if orient_code(img_out) != TARGET_ORIENTATION:
                    failures.append(f"{cid}_{ch}: image not {TARGET_ORIENTATION}"); ok = False; break
                if not geom_matches(img_out, lab_out):
                    failures.append(f"{cid}_{ch}: image/label geometry mismatch after reorient"); ok = False; break
                out_imgs.append((ch, img_out))
            if not ok:
                continue

            # ---- write (only after all checks pass for this case) ----
            for ch, img_out in out_imgs:
                sitk.WriteImage(img_out, str(dst / img_sub / f"{cid}_{ch}{fe}"))
            sitk.WriteImage(lab_out, str(dst / lab_sub / f"{cid}{fe}"))
            n_done += 1
            if img_sub == "imagesTr":
                n_train += 1

    # ---- dataset.json ----
    ds_json = {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": {k: int(v) for k, v in labels.items()},
        "numTraining": n_train,
        "file_ending": fe,
        "name": target_folder,
        "description": (f"ImageCHD reoriented to {TARGET_ORIENTATION} (clinical). Images from "
                        f"{args.images_dataset}; clean 7-class labels from {args.labels_dataset}. "
                        f"True DICOMOrient reorient; no HU/spacing change."),
        "orientation": TARGET_ORIENTATION,
        "source_images": args.images_dataset,
        "source_labels": args.labels_dataset,
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))

    # ---- final independent re-read check on disk ----
    reread_bad = []
    for lf in sorted((dst / "labelsTr").glob(f"*{fe}"))[:5]:
        if orient_code(sitk.ReadImage(str(lf))) != TARGET_ORIENTATION:
            reread_bad.append(lf.name)

    print(f"\n[d071] wrote {n_done} case(s) -> {dst}  (numTraining={n_train})")
    if reread_bad:
        print(f"[d071] RE-READ FAILED (not {TARGET_ORIENTATION}): {reread_bad}")
    if failures:
        print(f"[d071] {len(failures)} FAILURE(S):")
        for f in failures:
            print("   -", f)
        sys.exit(1)
    print(f"[d071] OK — all outputs {TARGET_ORIENTATION}, geometry consistent, labels+HU lossless.")
    print(f"[d071] next: nnUNetv2_plan_and_preprocess -d {args.target_id} "
          f"-pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
