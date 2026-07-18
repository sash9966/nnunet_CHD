#!/usr/bin/env python3
"""
build_dataset060_clean_holdout.py
=================================

Build Dataset060 with a CLEAN-TRAIN / DIRTY-HOLDOUT partition:

  * TEST (held-out imagesTs/labelsTs): every case whose GT is missing the
    myocardium label (low quality for our septal/outflow derivation), topped up
    with diagnosis-stratified clean cases to reach ~test_frac (default 10%) if
    the missing-myo pile is smaller. Never puts a missing-myo case in train.
  * TRAIN (imagesTr/labelsTr): all remaining (myo-present, clean) cases, with
    the septal_defect label (v2, ASD-fixed) derived — reliable because every
    training case has myocardium.

Rationale: maximise training data QUALITY (clinical model), and keep the
imperfect cases as a reported ImageCHD holdout that clinicians can re-segment.
This is a NEW partition — NOT comparable to the Dataset030/050/051 split.

Source: Dataset030 (pools imagesTr+labelsTr AND imagesTs+labelsTs = all cases).
nnU-Net auto-generates the fold split from the clean labelsTr.

Usage (on the cluster, where all labels exist):
  python tools/build_dataset060_clean_holdout.py \
    --source-dataset $nnUNet_raw/Dataset030_imageCHD_HU \
    --target-id 60 --target-name imageCHD_CleanHoldout \
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


def _gather_cases(src: Path, file_ending: str):
    """Return {case_id: {'image':[chan paths], 'label':path}} from imagesTr/Ts + labelsTr/Ts."""
    cases = {}
    for lab_sub, img_sub in [("labelsTr", "imagesTr"), ("labelsTs", "imagesTs")]:
        lab_dir, img_dir = src / lab_sub, src / img_sub
        if not lab_dir.is_dir():
            continue
        for lf in sorted(lab_dir.glob(f"*{file_ending}")):
            cid = lf.name[: -len(file_ending)]
            imgs = sorted(img_dir.glob(f"{cid}_*{file_ending}")) if img_dir.is_dir() else []
            cases[cid] = {"label": lf, "image": imgs}
    return cases


def _link(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil; shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dataset", required=True)
    ap.add_argument("--target-id", type=int, default=60)
    ap.add_argument("--target-name", default="imageCHD_CleanHoldout")
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
    out_root = Path(args.out_root or os.environ.get("nnUNet_raw") or src.parent)
    dst = (out_root / target_folder).resolve()
    if dst == src:
        raise SystemExit("target must differ from source")

    label_map = load_label_map(args.label_map, dataset_dir=str(src))
    ruleset = load_rules(args.rules)
    derived_cfg = cio.load_yaml(args.derived_config)
    builder = DerivedLabelBuilder(label_map, ruleset, derived_cfg)
    meta_all = load_disease_flags(args.metadata, ruleset.flag_columns)
    myo_id = label_map.id_of("myocardium")

    cases = _gather_cases(src, fe)
    if args.limit:
        cases = dict(list(cases.items())[: args.limit])
    ids = sorted(cases)
    print(f"[d060] pooled {len(ids)} cases from {src}")

    # myo presence per case
    missing_myo, clean = [], []
    for cid in ids:
        lab = cio.read_label(cases[cid]["label"])
        (clean if (myo_id is not None and (lab.data == myo_id).any()) else missing_myo).append(cid)
    print(f"[d060] missing-myo: {len(missing_myo)} | clean: {len(clean)}")

    # partition: test = missing-myo (+ stratified topup to test_frac); train = rest
    target_test = math.ceil(args.test_frac * len(ids))
    test = list(missing_myo)
    if len(test) < target_test:
        need = target_test - len(test)
        rng = random.Random(args.seed)
        # stratify by "has any disease flag" to keep test representative
        def has_dx(cid):
            m = meta_all.get(normalize_case_key(cid))
            return bool(m and any(m.flags.values()))
        pos = [c for c in clean if has_dx(c)]; neg = [c for c in clean if not has_dx(c)]
        rng.shuffle(pos); rng.shuffle(neg)
        frac_pos = len(pos) / max(len(clean), 1)
        n_pos = min(len(pos), round(need * frac_pos)); n_neg = min(len(neg), need - n_pos)
        topup = pos[:n_pos] + neg[:n_neg]
        test += topup
        clean = [c for c in clean if c not in set(topup)]
        print(f"[d060] topped up test with {len(topup)} stratified clean cases")
    train = clean
    test_set = set(test)
    print(f"[d060] FINAL: train {len(train)} | test {len(test)} ({len(test)/len(ids)*100:.0f}%)")

    # build
    from chd_landmarks.nnunet_dataset_builder import _prepare_target, _write_dataset_json
    _prepare_target(dst, args.overwrite)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs", "derived_masksTr"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    report = []
    n_matched = 0
    for cid in ids:
        lab = cio.read_label(cases[cid]["label"])
        in_test = cid in test_set
        reason = "missing_myo" if cid in missing_myo else ("topup_test" if in_test else "train")
        mkey = normalize_case_key(cid)
        meta = meta_all.get(mkey, CaseMetadata(case_id=mkey, flags={}))
        if mkey in meta_all:
            n_matched += 1
        if in_test:
            # test: images + ORIGINAL anatomy label (septal derived on-the-fly in eval)
            for img in cases[cid]["image"]:
                _link(img, dst / "imagesTs" / img.name, args.copy)
            cio.write_like(lab, lab.data, dst / "labelsTs" / f"{cid}{fe}", dtype=np.uint8)
        else:
            # train: derive septal label + write
            d = builder.build_for_case(lab.data, meta, cid, affine=lab.affine, spacing=lab.spacing)
            cio.write_like(lab, d.merged_label_map, dst / "labelsTr" / f"{cid}{fe}", dtype=np.uint8)
            for img in cases[cid]["image"]:
                _link(img, dst / "imagesTr" / img.name, args.copy)
        report.append({"case": cid, "myo_present": cid not in missing_myo,
                       "assignment": "test" if in_test else "train", "reason": reason,
                       "flags": ";".join(meta.active_diseases())})

    if n_matched == 0:
        raise SystemExit("ERROR: 0 cases matched a metadata row — check --metadata / naming.")

    _write_dataset_json(dst, channel_names, builder.merged_dataset_labels(),
                        len(train), fe, target_folder, args.source_dataset)
    with open(dst / "partition_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)
    cio.save_json({"train": train, "test": test, "missing_myo": missing_myo},
                  dst / "partition.json")
    print(f"[d060] wrote {dst}\n  next: nnUNetv2_plan_and_preprocess -d {args.target_id} "
          f"-pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
