#!/usr/bin/env python3
"""
filter_splits_to_dataset.py
===========================

Copy a source `splits_final.json` (e.g. Dataset030's) into a target dataset's
preprocessed folder, but DROP any case id not present in the target dataset's
labelsTr. Needed when the target dataset excluded cases (e.g. missing-myo cases
excluded via `build-dataset --require-myo`): the fold's train/val lists must not
reference cases that don't exist, or nnU-Net errors.

The held-out TEST set is unaffected (predictions run on the source imagesTs).

Usage:
  python tools/filter_splits_to_dataset.py \
     --source-splits $nnUNet_preprocessed/Dataset030_imageCHD_HU/splits_final.json \
     --target-dataset $nnUNet_raw/Dataset051_imageCHD_DiseaseLandmarksV2 \
     --out $nnUNet_preprocessed/Dataset051_imageCHD_DiseaseLandmarksV2/splits_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-splits", required=True)
    ap.add_argument("--target-dataset", required=True, help="raw dataset dir with labelsTr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--file-ending", default=".nii.gz")
    args = ap.parse_args()

    labels_dir = Path(args.target_dataset) / "labelsTr"
    present = {f.name[: -len(args.file_ending)] for f in labels_dir.glob(f"*{args.file_ending}")}
    splits = json.load(open(args.source_splits))

    dropped = set()
    out_splits = []
    for fold in splits:
        tr = [c for c in fold["train"] if c in present]
        va = [c for c in fold["val"] if c in present]
        dropped |= (set(fold["train"]) - present) | (set(fold["val"]) - present)
        out_splits.append({"train": tr, "val": va})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_splits, open(args.out, "w"), indent=1)
    print(f"[filter_splits] target cases present: {len(present)}")
    print(f"[filter_splits] dropped from splits (not in target): {len(dropped)} -> {sorted(dropped)[:10]}")
    for i, (src, dst) in enumerate(zip(splits, out_splits)):
        print(f"  fold {i}: train {len(src['train'])}->{len(dst['train'])}  val {len(src['val'])}->{len(dst['val'])}")
    print(f"[filter_splits] wrote {args.out}")


if __name__ == "__main__":
    main()
