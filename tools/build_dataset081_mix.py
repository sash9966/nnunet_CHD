#!/usr/bin/env python3
"""
build_dataset081_mix.py
=======================

Build Dataset081 = ImageCHD (Dataset071, clean myo LPS) + clinical (Dataset080),
with the clinical cases OVERSAMPLED by duplication so they carry real weight
against the ~90 ImageCHD cases.

Layout:
  * ImageCHD cases from Dataset071 -> symlinked into imagesTr/labelsTr unchanged.
  * Each clinical case from Dataset080 -> the original + (dup_factor-1) duplicates
    (symlinks to the same files), ids <case>, <case>_dup01, <case>_dup02, ...
  * Clinical cases (all instances) are meant for TRAINING ONLY. The split written
    later (by the training script) puts them in every fold's train and NEVER in
    val -> no duplicate leakage; ImageCHD does the 5-fold validation. The real
    clinical evaluation is external inference, not an internal held-out fold.

Writes: imagesTr/, labelsTr/, dataset.json, split_meta.json
  split_meta.json = {"imagechd": [case ids], "clinical_instances": [all clinical ids]}

Self-verifies: clinical labels use the SAME ids as ImageCHD (7-class, no septal 8)
and are LPS; aborts otherwise.

Usage (cluster):
  python tools/build_dataset081_mix.py --nnunet-raw $nnUNet_raw \
    --imagechd-dataset Dataset071_ImageCHDClinicalOrientation \
    --clinical-dataset Dataset080_ClincalCaseSanjibDetailed \
    --dup-factor 8 --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CHANNEL_RE = re.compile(r"_(\d{4})\.nii\.gz$")


def orient_code(img):
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def label_ids(path):
    a = sitk.GetArrayViewFromImage(sitk.ReadImage(str(path)))
    return {int(x) for x in np.unique(a)}


def symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def gather_cases(images_dir: Path, labels_dir: Path, fe: str):
    """case_id -> {'channels': [paths], 'label': path}."""
    cases = {}
    for f in sorted(images_dir.glob(f"*{fe}")):
        m = CHANNEL_RE.search(f.name)
        if not m:
            continue
        cid = f.name[: m.start()]
        cases.setdefault(cid, {"channels": [], "label": labels_dir / f"{cid}{fe}"})["channels"].append(f)
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--imagechd-dataset", default="Dataset071_ImageCHDClinicalOrientation")
    ap.add_argument("--clinical-dataset", default="Dataset080_ClincalCaseSanjibDetailed")
    ap.add_argument("--target-id", type=int, default=81)
    ap.add_argument("--target-name", default="ImageCHDplusClinical")
    ap.add_argument("--dup-factor", type=int, default=8, help="clinical copies in training (incl. original)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.nnunet_raw:
        sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    raw = Path(args.nnunet_raw).resolve()
    chd = raw / args.imagechd_dataset
    clin = raw / args.clinical_dataset
    for p in (chd, clin):
        if not (p / "imagesTr").is_dir() or not (p / "labelsTr").is_dir():
            sys.exit(f"ERROR: {p} missing imagesTr/labelsTr")

    chd_json = json.loads((chd / "dataset.json").read_text())
    clin_json = json.loads((clin / "dataset.json").read_text())
    fe = chd_json.get("file_ending", ".nii.gz")
    labels = chd_json["labels"]
    ref_ids = {int(v) for v in labels.values()} | {0}
    if 8 in ref_ids:
        sys.exit("ERROR: ImageCHD labels contain id 8 (septal) -- use the clean 7-class Dataset071")
    # clinical must declare the same label scheme
    if {k: int(v) for k, v in clin_json["labels"].items()} != {k: int(v) for k, v in labels.items()}:
        sys.exit(f"ERROR: clinical labels {clin_json['labels']} != ImageCHD labels {labels} "
                 f"(fine-tuning/mixing requires identical id mapping)")

    target_folder = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target_folder
    if dst.exists() and not args.overwrite:
        sys.exit(f"ERROR: {dst} exists (use --overwrite)")
    for sub in ("imagesTr", "labelsTr"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    imagechd_ids, clinical_instances, failures = [], [], []

    # ---- ImageCHD: symlink unchanged ----
    chd_cases = gather_cases(chd / "imagesTr", chd / "labelsTr", fe)
    for cid, c in chd_cases.items():
        for f in c["channels"]:
            symlink(f, dst / "imagesTr" / f.name)
        symlink(c["label"], dst / "labelsTr" / f"{cid}{fe}")
        imagechd_ids.append(cid)

    # ---- Clinical: verify, then symlink original + duplicates ----
    clin_cases = gather_cases(clin / "imagesTr", clin / "labelsTr", fe)
    if not clin_cases:
        sys.exit(f"ERROR: no clinical cases found in {clin}/imagesTr")
    for cid, c in clin_cases.items():
        if not c["label"].is_file():
            failures.append(f"{cid}: missing label"); continue
        lab_img = sitk.ReadImage(str(c["label"]))
        ids = {int(x) for x in np.unique(sitk.GetArrayViewFromImage(lab_img))}
        if not ids <= ref_ids:
            failures.append(f"{cid}: label ids {ids} not subset of {sorted(ref_ids)}"); continue
        if orient_code(lab_img) != "LPS":
            failures.append(f"{cid}: label orientation {orient_code(lab_img)} != LPS"); continue
        img0 = sitk.ReadImage(str(c["channels"][0]))
        if orient_code(img0) != "LPS":
            failures.append(f"{cid}: image orientation {orient_code(img0)} != LPS"); continue
        if img0.GetSize() != lab_img.GetSize():
            failures.append(f"{cid}: image/label size mismatch {img0.GetSize()} vs {lab_img.GetSize()}"); continue
        for rep in range(args.dup_factor):
            out_id = cid if rep == 0 else f"{cid}_dup{rep:02d}"
            for f in c["channels"]:
                ch = CHANNEL_RE.search(f.name).group(1)
                symlink(f, dst / "imagesTr" / f"{out_id}_{ch}{fe}")
            symlink(c["label"], dst / "labelsTr" / f"{out_id}{fe}")
            clinical_instances.append(out_id)

    if failures:
        print("[d081] FAILURES:"); [print("   -", f) for f in failures]
        sys.exit(1)

    n_train = len(imagechd_ids) + len(clinical_instances)
    ds_json = {
        "channel_names": {str(k): v for k, v in chd_json.get("channel_names", {"0": "CT"}).items()},
        "labels": {k: int(v) for k, v in labels.items()},
        "numTraining": n_train, "file_ending": fe, "name": target_folder,
        "description": (f"ImageCHD ({args.imagechd_dataset}) + clinical ({args.clinical_dataset}) "
                        f"oversampled {args.dup_factor}x. Clinical = train-only (no val)."),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    (dst / "split_meta.json").write_text(json.dumps(
        {"imagechd": sorted(imagechd_ids), "clinical_instances": sorted(clinical_instances),
         "clinical_sources": sorted(clin_cases), "dup_factor": args.dup_factor}, indent=1))

    print(f"[d081] wrote {dst}")
    print(f"  ImageCHD cases: {len(imagechd_ids)} | clinical sources: {len(clin_cases)} "
          f"x{args.dup_factor} = {len(clinical_instances)} instances | numTraining={n_train}")
    print(f"  labels ({len(ref_ids)} incl bg, no septal): {labels}")
    print(f"  split_meta.json written (clinical -> train only, ImageCHD -> 5-fold val)")
    print(f"  NEXT: plan_and_preprocess -d {args.target_id}, then write the clinical-always-train split")


if __name__ == "__main__":
    main()
