#!/usr/bin/env python3
"""
convert_to_mask_conditioned.py
------------------------------
Build a 2-channel 7-class nnU-Net dataset (default Dataset041_ImageCHD_HU_MaskCond)
from an existing multiclass ImageCHD dataset (default Dataset030_imageCHD_HU) plus
a binary heart-mask source.

Stage-2 Approach A: the network sees (CT, binary_heart_prior) and predicts the
seven anatomical classes. The localisation problem is handed to it for free;
training focuses on semantic assignment.

Behaviour
---------
imagesTr/{case}_0000.nii.gz   — CT, symlinked (or copied with --copy) from source.
imagesTr/{case}_0001.nii.gz   — binary heart mask (channel 1).
                                * --mask-source gt  (default):  Dataset030 labelsTr binarised
                                  on the fly and written as a fresh uint8 NIfTI.
                                * --mask-source predicted:      symlinked/copied from
                                  --mask-dir-tr, matched by case ID.
labelsTr/{case}.nii.gz        — 7-class label, symlinked (or copied) from source.
imagesTs / labelsTs           — same logic; --mask-dir-ts is required iff
                                --mask-source predicted.
dataset.json                  — channel_names = {0:"CT", 1:"nonorm"};
                                labels copied from the source dataset.json.
conversion_summary.csv        — per-case sanity: image_ok, channel1_ok, label_ok,
                                affine_preserved, source of channel 1.

CLI
---
    python experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py
        [--source-dataset Dataset030_imageCHD_HU]
        [--target-id 41]
        [--target-name Dataset041_ImageCHD_HU_MaskCond]
        [--mask-source {gt,predicted}]          (default: gt)
        [--mask-dir-tr PATH]                    (required for --mask-source predicted)
        [--mask-dir-ts PATH]                    (required for --mask-source predicted, if imagesTs present)
        [--raw-root PATH]
        [--copy | --symlink]                    (default: symlink for images + labels;
                                                 channel 1 in gt mode is always written fresh)
        [--overwrite]
        [--dry-run]

Examples
--------
    # GT-binarised channel 1 (no Stage-1 dependency). Safest first-pass.
    python convert_to_mask_conditioned.py --dry-run
    python convert_to_mask_conditioned.py

    # Predicted-mask channel 1 from a finished Stage-1 model.
    python convert_to_mask_conditioned.py \
        --mask-source predicted \
        --mask-dir-tr $nnUNet_results/Dataset040_WH_ImageCHD_HU_Detail/predictions_wholeheart/DA5_cascade_imagesTr \
        --mask-dir-ts $nnUNet_results/Dataset040_WH_ImageCHD_HU_Detail/predictions_wholeheart/DA5_cascade

Why two modes
-------------
GT mode lets you start training immediately; the model learns a 2-channel prior
that is *clean*. At inference time channel 1 will be noisier (Stage-1 predictions),
so re-running with --mask-source predicted (using the same model architecture
but re-trained on noisy channel 1) is the production path. Compare the two to
quantify the train/test domain gap for the channel-1 prior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import nibabel
import numpy as np

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_raw_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("nnUNet_raw")
    if not env:
        raise SystemExit(
            "ERROR: nnUNet_raw not set in environment and --raw-root not given."
        )
    return Path(env).expanduser().resolve()


def _scan_files(folder: Path, suffix: str) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(suffix))


def _link_or_copy(src: Path, dst: Path, copy: bool, dry_run: bool, role: str) -> None:
    if dry_run:
        op = "COPY" if copy else "SYMLINK"
        print(f"    [{op:8}] {role:10}  {src.name}")
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _binarize_to(src_label: Path, dst_channel1: Path, dry_run: bool) -> tuple[Counter, int, bool]:
    """Read multiclass label, write binary mask as a new uint8 NIfTI (channel 1)."""
    img = nibabel.load(str(src_label))
    data = np.asanyarray(img.dataobj)
    unique, counts = np.unique(data, return_counts=True)
    counter = Counter({int(v): int(c) for v, c in zip(unique, counts)})
    binary = (data > 0).astype(np.uint8)
    bin_fg = int(binary.sum())

    if dry_run:
        return counter, bin_fg, True

    new_img = nibabel.Nifti1Image(binary, img.affine, img.header)
    new_img.set_data_dtype(np.uint8)
    nibabel.save(new_img, str(dst_channel1))
    rt = nibabel.load(str(dst_channel1))
    affine_ok = bool(np.allclose(rt.affine, img.affine, atol=1e-8))
    return counter, bin_fg, affine_ok


def _prepare_dst_folder(folder: Path, overwrite: bool, dry_run: bool, role: str) -> None:
    if folder.exists() and any(folder.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: {folder} is not empty. Pass --overwrite to wipe it."
            )
        if dry_run:
            print(f"  [DRY-RUN] would wipe {role}: {folder}")
        else:
            print(f"  wiping {role}: {folder}")
            for entry in folder.iterdir():
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
    elif not folder.exists() and not dry_run:
        folder.mkdir(parents=True, exist_ok=True)


def _index_mask_dir(mask_dir: Path) -> dict[str, Path]:
    """Index a mask directory by case ID. Accepts {case}.nii.gz or
    {case}_0000.nii.gz naming (nnUNet predict output uses the bare {case}.nii.gz form)."""
    if not mask_dir.is_dir():
        raise SystemExit(f"ERROR: mask directory not found: {mask_dir}")
    out: dict[str, Path] = {}
    for f in mask_dir.iterdir():
        if not (f.is_file() and f.name.endswith(".nii.gz")):
            continue
        stem = f.name[:-len(".nii.gz")]
        if stem.endswith("_0000"):
            stem = stem[:-len("_0000")]
        out[stem] = f
    if not out:
        raise SystemExit(f"ERROR: no *.nii.gz files in {mask_dir}")
    return out


def _load_source_labels(src_dataset_json: Path) -> dict:
    """Pull the labels dict from the source dataset.json so we re-emit it verbatim."""
    if not src_dataset_json.is_file():
        raise SystemExit(f"ERROR: source dataset.json not found: {src_dataset_json}")
    with src_dataset_json.open() as fp:
        meta = json.load(fp)
    if "labels" not in meta:
        raise SystemExit(f"ERROR: source dataset.json has no 'labels' field: {src_dataset_json}")
    return meta["labels"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a 2-channel (CT + binary heart prior) Dataset041 for Stage-2 Approach A.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source-dataset", default="Dataset030_imageCHD_HU")
    p.add_argument("--target-id", type=int, default=41)
    p.add_argument("--target-name", default="Dataset041_ImageCHD_HU_MaskCond")
    p.add_argument("--mask-source", choices=["gt", "predicted"], default="gt",
                   help="How to populate channel 1. 'gt' binarises the source "
                        "labels on the fly (default). 'predicted' takes NIfTIs "
                        "from --mask-dir-tr / --mask-dir-ts.")
    p.add_argument("--mask-dir-tr", default=None,
                   help="Directory of predicted binary masks for the training set. "
                        "Required when --mask-source predicted.")
    p.add_argument("--mask-dir-ts", default=None,
                   help="Directory of predicted binary masks for the test set. "
                        "Required when --mask-source predicted and imagesTs present.")
    p.add_argument("--raw-root", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true")
    mode.add_argument("--symlink", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    copy_mode = bool(args.copy)

    expected_prefix = f"Dataset{args.target_id:03d}_"
    if not args.target_name.startswith(expected_prefix):
        raise SystemExit(
            f"ERROR: --target-name must start with '{expected_prefix}'."
        )

    raw_root = _resolve_raw_root(args.raw_root)
    src_root = raw_root / args.source_dataset
    dst_root = raw_root / args.target_name
    if not src_root.is_dir():
        raise SystemExit(f"ERROR: source dataset not found: {src_root}")

    src_images_tr = src_root / "imagesTr"
    src_labels_tr = src_root / "labelsTr"
    src_images_ts = src_root / "imagesTs"
    src_labels_ts = src_root / "labelsTs"
    src_dataset_json = src_root / "dataset.json"

    dst_images_tr = dst_root / "imagesTr"
    dst_labels_tr = dst_root / "labelsTr"
    dst_images_ts = dst_root / "imagesTs"
    dst_labels_ts = dst_root / "labelsTs"

    # ── Resolve mask sources ────────────────────────────────────────────
    mask_tr_index: dict[str, Path] = {}
    mask_ts_index: dict[str, Path] = {}
    if args.mask_source == "predicted":
        if not args.mask_dir_tr:
            raise SystemExit("ERROR: --mask-dir-tr is required when --mask-source predicted.")
        mask_tr_index = _index_mask_dir(Path(args.mask_dir_tr).expanduser().resolve())
        if (src_images_ts.is_dir() and any(src_images_ts.iterdir())):
            if not args.mask_dir_ts:
                raise SystemExit(
                    "ERROR: --mask-dir-ts is required when --mask-source predicted "
                    "and the source has an imagesTs split."
                )
            mask_ts_index = _index_mask_dir(Path(args.mask_dir_ts).expanduser().resolve())

    # ── Source labels (re-emit verbatim) ────────────────────────────────
    src_labels = _load_source_labels(src_dataset_json)

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  Mask-conditioned conversion  ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print("=" * 72)
    print(f"  source       : {src_root}")
    print(f"  target       : {dst_root}")
    print(f"  channel 1    : {args.mask_source}"
          + (f"  ({len(mask_tr_index)} tr + {len(mask_ts_index)} ts predicted masks)"
             if args.mask_source == "predicted" else "  (GT labels binarised on the fly)"))
    print(f"  CT + labels  : {'copy' if copy_mode else 'symlink'} from source")
    print(f"  labels dict  : {list(src_labels.keys())}")
    print(f"  overwrite    : {args.overwrite}")
    print()

    image_files = _scan_files(src_images_tr, "_0000.nii.gz")
    label_files = _scan_files(src_labels_tr, ".nii.gz")
    test_image_files = _scan_files(src_images_ts, "_0000.nii.gz")
    test_label_files = _scan_files(src_labels_ts, ".nii.gz")

    if not image_files:
        raise SystemExit(f"ERROR: no *_0000.nii.gz files in {src_images_tr}")
    if not label_files:
        raise SystemExit(f"ERROR: no *.nii.gz files in {src_labels_tr}")

    case_ids = sorted({f.name.removesuffix("_0000.nii.gz") for f in image_files})
    label_ids = sorted({f.name.removesuffix(".nii.gz") for f in label_files})
    if case_ids != label_ids:
        only_imgs = sorted(set(case_ids) - set(label_ids))[:5]
        only_lbls = sorted(set(label_ids) - set(case_ids))[:5]
        raise SystemExit(
            f"ERROR: images/labels mismatch. only_in_images={only_imgs} "
            f"only_in_labels={only_lbls}"
        )

    test_case_ids = sorted({f.name.removesuffix("_0000.nii.gz") for f in test_image_files})

    # If predicted mode, every case must have a mask.
    if args.mask_source == "predicted":
        missing_tr = [c for c in case_ids if c not in mask_tr_index]
        if missing_tr:
            raise SystemExit(
                f"ERROR: {len(missing_tr)} training cases have no predicted mask. "
                f"First few: {missing_tr[:5]}"
            )
        if test_case_ids:
            missing_ts = [c for c in test_case_ids if c not in mask_ts_index]
            if missing_ts:
                raise SystemExit(
                    f"ERROR: {len(missing_ts)} test cases have no predicted mask. "
                    f"First few: {missing_ts[:5]}"
                )

    print(f"  imagesTr cases : {len(case_ids)}")
    print(f"  imagesTs cases : {len(test_case_ids)}")
    print()

    # ── Prepare destination folders ─────────────────────────────────────
    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
    for folder, role in [(dst_images_tr, "imagesTr"), (dst_labels_tr, "labelsTr")]:
        _prepare_dst_folder(folder, args.overwrite, args.dry_run, role)
    if test_image_files:
        _prepare_dst_folder(dst_images_ts, args.overwrite, args.dry_run, "imagesTs")
    if test_label_files:
        _prepare_dst_folder(dst_labels_ts, args.overwrite, args.dry_run, "labelsTs")

    # ── Per-case conversion ─────────────────────────────────────────────
    summary_rows: list[dict] = []
    affine_failures: list[str] = []

    def _convert_split(
        split: str,
        case_ids_split: list[str],
        src_img_dir: Path,
        src_lbl_dir: Path,
        dst_img_dir: Path,
        dst_lbl_dir: Path | None,
        mask_index: dict[str, Path] | None,
    ) -> None:
        nonlocal affine_failures
        print(f"-- {split} ({len(case_ids_split)} cases) --")
        for case_id in case_ids_split:
            src_ct = src_img_dir / f"{case_id}_0000.nii.gz"
            src_lbl = src_lbl_dir / f"{case_id}.nii.gz"
            dst_ct = dst_img_dir / f"{case_id}_0000.nii.gz"
            dst_ch1 = dst_img_dir / f"{case_id}_0001.nii.gz"

            # Channel 0: CT
            _link_or_copy(src_ct.resolve(), dst_ct, copy_mode, args.dry_run, "CT")

            # Channel 1: binary mask
            affine_ok = True
            ch1_source = "?"
            if args.mask_source == "gt":
                if not src_lbl.is_file():
                    raise SystemExit(f"ERROR: missing GT label for {case_id}: {src_lbl}")
                counter, bin_fg, affine_ok = _binarize_to(src_lbl, dst_ch1, args.dry_run)
                ch1_source = f"gt_binarised(fg={bin_fg})"
                if not affine_ok:
                    affine_failures.append(f"{split}/{case_id}")
            else:
                pred_mask = mask_index[case_id]  # validated above
                _link_or_copy(
                    pred_mask.resolve(), dst_ch1, copy_mode, args.dry_run, "mask"
                )
                ch1_source = f"predicted({pred_mask.name})"

            # Labels: symlink/copy from source (only if we have a labels split)
            label_ok = True
            if dst_lbl_dir is not None:
                if not src_lbl.is_file():
                    if split == "Tr":
                        raise SystemExit(f"ERROR: missing label for {case_id}: {src_lbl}")
                    label_ok = False
                else:
                    _link_or_copy(
                        src_lbl.resolve(),
                        dst_lbl_dir / f"{case_id}.nii.gz",
                        copy_mode,
                        args.dry_run,
                        "label",
                    )

            summary_rows.append({
                "split": split,
                "case_id": case_id,
                "channel1_source": ch1_source,
                "affine_preserved": affine_ok,
                "label_present": label_ok,
            })
            print(f"  [OK] {case_id:>16}  ch1={ch1_source}  affine_ok={affine_ok}  label={label_ok}")

    _convert_split(
        "Tr", case_ids,
        src_images_tr, src_labels_tr,
        dst_images_tr, dst_labels_tr,
        mask_tr_index if args.mask_source == "predicted" else None,
    )
    if test_case_ids:
        _convert_split(
            "Ts", test_case_ids,
            src_images_ts, src_labels_ts,
            dst_images_ts, dst_labels_ts if test_label_files else None,
            mask_ts_index if args.mask_source == "predicted" else None,
        )

    # ── dataset.json ────────────────────────────────────────────────────
    print()
    print("-- dataset.json --")
    if args.dry_run:
        print(f"  [DRY-RUN] would write {dst_root / 'dataset.json'}  "
              f"(channels=2, labels={list(src_labels.keys())}, numTraining={len(case_ids)})")
    else:
        generate_dataset_json(
            output_folder=str(dst_root),
            channel_names={0: "CT", 1: "nonorm"},
            labels=src_labels,
            num_training_cases=len(case_ids),
            file_ending=".nii.gz",
            dataset_name=args.target_name,
            description=(
                f"Mask-conditioned 7-class segmentation derived from {args.source_dataset}. "
                "Channel 0 = CT (CTNormalization). Channel 1 = binary heart prior "
                f"({'GT-binarised' if args.mask_source == 'gt' else 'Stage-1 predicted'}, "
                "noNorm). 7-class labels from the source dataset."
            ),
            reference=f"Derived from {args.source_dataset}",
            converted_by="convert_to_mask_conditioned.py",
            license="See source dataset license.",
        )
        print(f"  wrote {dst_root / 'dataset.json'}")

    # ── conversion_summary.csv ──────────────────────────────────────────
    summary_path = dst_root / "conversion_summary.csv"
    if args.dry_run:
        print(f"  [DRY-RUN] would write {summary_path}  ({len(summary_rows)} rows)")
    else:
        with summary_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["split", "case_id", "channel1_source", "affine_preserved", "label_present"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  wrote {summary_path}")

    # ── final report ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    if affine_failures:
        print("  COMPLETED WITH WARNINGS")
        print(f"    affine drift on {len(affine_failures)} cases: "
              f"{affine_failures[:5]}{'...' if len(affine_failures) > 5 else ''}")
        return 1
    print("  CONVERSION COMPLETE")
    print("=" * 72)
    print()
    print("Next step:")
    print(f"  sbatch scripts/CHD_Dataset{args.target_id:03d}_mask_constrained.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
