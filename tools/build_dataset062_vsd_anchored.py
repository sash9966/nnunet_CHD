#!/usr/bin/env python3
"""
build_dataset062_vsd_anchored.py
================================

Dataset062 — the "purest" septal-defect dataset, with the v3 VSD-ANCHORED
derivation (see chd_landmarks.derived_regions.build_septal_defect_anchored).

Partition (same idea as Dataset060): no missing-myocardium case in TRAINING.
  * TRAIN (imagesTr/labelsTr) = all clean (myo-present) cases; septal_defect (id 8)
    derived reliably (v3 anchored).
  * TEST  (imagesTs/labelsTs) = the missing-myo cases (+ diagnosis-stratified topup
    to ~test_frac). The septal_defect label IS ALSO derived here so you can
    INSPECT it — but for missing-myo cases it is degraded (no myocardium to
    anchor the VSD cleanly); confidence is marked low and flagged in the report.

No myocardium hole-filling. Works with ImageCHD as-is. This is a NEW partition,
not comparable to Dataset030/050/051.

Usage (cluster):
  python tools/build_dataset062_vsd_anchored.py \
    --source-dataset $nnUNet_raw/Dataset030_imageCHD_HU \
    --target-id 62 --target-name imageCHD_VSDanchored \
    --metadata $REPO/imageCHD_diagnosis_june21.csv --out-root $nnUNet_raw
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chd_landmarks import io as cio
from chd_landmarks.derived_label_builder import DerivedLabelBuilder
from chd_landmarks.disease_rules import load_rules
from chd_landmarks.labels import load_label_map
from chd_landmarks.metadata import CaseMetadata, load_disease_flags, normalize_case_key
from chd_landmarks.nnunet_dataset_builder import _prepare_target, _write_dataset_json


def _gather(src: Path, fe: str):
    cases = {}
    for lab_sub, img_sub in [("labelsTr", "imagesTr"), ("labelsTs", "imagesTs")]:
        lab_dir, img_dir = src / lab_sub, src / img_sub
        if not lab_dir.is_dir():
            continue
        for lf in sorted(lab_dir.glob(f"*{fe}")):
            cid = lf.name[: -len(fe)]
            cases[cid] = {"label": lf,
                          "image": sorted(img_dir.glob(f"{cid}_*{fe}")) if img_dir.is_dir() else []}
    return cases


def _link(s: Path, d: Path, copy: bool):
    if d.exists() or d.is_symlink():
        d.unlink()
    (__import__("shutil").copy2 if copy else os.symlink)(s.resolve(), d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dataset", required=True)
    ap.add_argument("--target-id", type=int, default=62)
    ap.add_argument("--target-name", default="imageCHD_VSDanchored")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label-map", default="configs/chd_label_map.yaml")
    ap.add_argument("--rules", default="configs/chd_disease_rules.yaml")
    ap.add_argument("--derived-config", default="configs/chd_derived_labels.yaml")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = cio.resolve_dataset_dir(args.source_dataset)
    ds_json = cio.read_dataset_json(src)
    fe = ds_json.get("file_ending", ".nii.gz")
    channel_names = ds_json.get("channel_names", {"0": "CT"})
    target_folder = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = (Path(args.out_root or os.environ.get("nnUNet_raw") or src.parent) / target_folder).resolve()

    label_map = load_label_map(args.label_map, dataset_dir=str(src))
    ruleset = load_rules(args.rules)
    derived_cfg = cio.load_yaml(args.derived_config)
    builder = DerivedLabelBuilder(label_map, ruleset, derived_cfg, septal_mode="anchored")  # <-- v3
    meta_all = load_disease_flags(args.metadata, ruleset.flag_columns)
    myo_id = label_map.id_of("myocardium")

    cases = _gather(src, fe)
    if args.limit:
        cases = dict(list(cases.items())[: args.limit])
    ids = sorted(cases)

    missing_myo, clean = [], []
    for cid in ids:
        lab = cio.read_label(cases[cid]["label"])
        (clean if (myo_id is not None and (lab.data == myo_id).any()) else missing_myo).append(cid)
    print(f"[d062] pooled {len(ids)} | missing-myo {len(missing_myo)} | clean {len(clean)}")

    # partition: missing-myo -> test (+stratified topup to test_frac); clean -> train
    target_test = math.ceil(args.test_frac * len(ids))
    test = list(missing_myo)
    if len(test) < target_test:
        rng = random.Random(args.seed)
        def has_dx(c):
            m = meta_all.get(normalize_case_key(c)); return bool(m and any(m.flags.values()))
        pos = [c for c in clean if has_dx(c)]; neg = [c for c in clean if not has_dx(c)]
        rng.shuffle(pos); rng.shuffle(neg)
        need = target_test - len(test); n_pos = min(len(pos), round(need * len(pos) / max(len(clean), 1)))
        topup = pos[:n_pos] + neg[:need - n_pos]
        test += topup; clean = [c for c in clean if c not in set(topup)]
    train, test_set = clean, set(test)
    print(f"[d062] FINAL train {len(train)} | test {len(test)} ({len(test)/len(ids)*100:.0f}%)  [septal=v3 anchored]")

    _prepare_target(dst, args.overwrite)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    report, n_matched = [], 0
    for cid in ids:
        lab = cio.read_label(cases[cid]["label"])
        mkey = normalize_case_key(cid)
        meta = meta_all.get(mkey, CaseMetadata(case_id=mkey, flags={}))
        n_matched += int(mkey in meta_all)
        in_test = cid in test_set
        # derive v3 anchored septal on BOTH train and test (test = for inspection)
        d = builder.build_for_case(lab.data, meta, cid, affine=lab.affine, spacing=lab.spacing)
        sub_img, sub_lab = ("imagesTs", "labelsTs") if in_test else ("imagesTr", "labelsTr")
        cio.write_like(lab, d.merged_label_map, dst / sub_lab / f"{cid}{fe}", dtype=np.uint8)
        for img in cases[cid]["image"]:
            _link(img, dst / sub_img / img.name, args.copy)
        sd = d.region_meta.get("septal_defect_proxy", {})
        report.append({"case": cid, "assignment": "test" if in_test else "train",
                       "myo_present": cid not in missing_myo,
                       "septal_conf": sd.get("confidence", "none"),
                       "septal_voxels": sd.get("voxels", 0),
                       "flags": ";".join(meta.active_diseases())})

    if n_matched == 0:
        raise SystemExit("ERROR: 0 metadata matches — check --metadata / naming.")
    _write_dataset_json(dst, channel_names, builder.merged_dataset_labels(),
                        len(train), fe, target_folder, args.source_dataset)
    with open(dst / "partition_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)
    cio.save_json({"septal_derivation": "v3_vsd_anchored", "train": train, "test": test,
                   "missing_myo": missing_myo}, dst / "partition.json")
    print(f"[d062] wrote {dst}  (septal derivation = v3 VSD-anchored)")
    print(f"  inspect labelsTr/ (clean) + labelsTs/ (incl. missing-myo, degraded septal)")
    print(f"  next: nnUNetv2_plan_and_preprocess -d {args.target_id} -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
