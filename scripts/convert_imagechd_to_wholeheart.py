#!/usr/bin/env python3
"""
convert_imagechd_to_wholeheart.py
---------------------------------
Derive a binary whole-heart nnU-Net dataset (default Dataset040_WH_ImageCHD_HU_Detail)
from an existing multiclass ImageCHD dataset (default Dataset030_imageCHD_HU).

Behaviour
---------
imagesTr / imagesTs   — symlinked (or copied with --copy) from the source dataset.
                        Same affine / spacing / orientation as the source.
labelsTr / labelsTs   — read each labelmap, write (arr > 0).astype(uint8) with
                        the source's affine + header preserved (same binarization
                        for both splits; evaluate_wholeheart.py compares directly).
dataset.json          — built via `generate_dataset_json()` with the binary label scheme.
disease_map.json      — NOT copied here. The companion SLURM launcher regenerates
                        it from `imageCHD_dataset_info.xlsx` after preprocessing.
conversion_summary.csv — per-case sanity check: original labels present, original
                        fg-voxel count, binary fg-voxel count, delta (must be 0),
                        affine-match flag.

CLI
---
    python scripts/convert_imagechd_to_wholeheart.py
        [--source-dataset Dataset030_imageCHD_HU]
        [--target-id 40]
        [--target-name Dataset040_WH_ImageCHD_HU_Detail]
        [--raw-root $nnUNet_raw]
        [--copy | --symlink]      (default: --symlink)
        [--overwrite]
        [--dry-run]

Examples
--------
    # Dry run: print the plan, write nothing.
    python scripts/convert_imagechd_to_wholeheart.py --dry-run

    # Real conversion with the defaults.
    python scripts/convert_imagechd_to_wholeheart.py

    # Different source/target.
    python scripts/convert_imagechd_to_wholeheart.py \
        --source-dataset Dataset001_all_imageCHD --target-id 41 \
        --target-name Dataset041_imageCHD_WH
"""
from __future__ import annotations

import argparse
import csv
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
    """Resolve the nnUNet_raw root from CLI or environment."""
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("nnUNet_raw")
    if not env:
        raise SystemExit(
            "ERROR: nnUNet_raw not set in environment and --raw-root not given. "
            "Export nnUNet_raw or pass --raw-root /path/to/nnUNet_raw."
        )
    return Path(env).expanduser().resolve()


def _scan_files(folder: Path, suffix: str) -> list[Path]:
    """List files with the given suffix, sorted by name (stable order)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(suffix))


def _link_or_copy(src: Path, dst: Path, copy: bool, dry_run: bool) -> None:
    """Symlink or copy src → dst; refuse to overwrite (caller pre-cleans)."""
    if dry_run:
        op = "COPY" if copy else "SYMLINK"
        print(f"    [{op}] {src.name}")
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        # Absolute symlinks survive moves of the target dir.
        os.symlink(src, dst)


def _binarize_label(src: Path, dst: Path, dry_run: bool) -> tuple[Counter, int, int, bool]:
    """Read a multiclass NIfTI label, write a binary copy.

    Returns
    -------
    label_counter, original_fg_voxels, binary_fg_voxels, affine_preserved
    """
    img = nibabel.load(str(src))
    data = np.asanyarray(img.dataobj)
    unique, counts = np.unique(data, return_counts=True)
    label_counter = Counter({int(v): int(c) for v, c in zip(unique, counts)})

    binary = (data > 0).astype(np.uint8)
    original_fg = int(label_counter.values().__iter__().__next__()) if False else int(
        sum(c for v, c in label_counter.items() if v != 0)
    )
    binary_fg = int(binary.sum())

    if not dry_run:
        new_img = nibabel.Nifti1Image(binary, img.affine, img.header)
        # Force dtype on the header to uint8 (binary masks are small + correct).
        new_img.set_data_dtype(np.uint8)
        nibabel.save(new_img, str(dst))
        # Re-read to confirm affine round-tripped exactly.
        round_tripped = nibabel.load(str(dst))
        affine_ok = np.allclose(round_tripped.affine, img.affine, atol=1e-8)
    else:
        affine_ok = True

    return label_counter, original_fg, binary_fg, affine_ok


def _prepare_dst_folder(
    folder: Path,
    overwrite: bool,
    dry_run: bool,
    role: str,
) -> None:
    """Make/clean a destination subfolder."""
    if folder.exists() and any(folder.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: {folder} is not empty. Pass --overwrite to wipe it, or "
                f"choose a different --target-id / --target-name."
            )
        if dry_run:
            print(f"  [DRY-RUN] would wipe existing {role} folder: {folder}")
        else:
            print(f"  wiping existing {role} folder: {folder}")
            for entry in folder.iterdir():
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
    elif not folder.exists() and not dry_run:
        folder.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert a multiclass ImageCHD nnU-Net dataset into a binary whole-heart dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source-dataset", default="Dataset030_imageCHD_HU",
                   help="Source dataset folder name under nnUNet_raw.")
    p.add_argument("--target-id", type=int, default=40,
                   help="Target dataset ID (default 40 → Dataset040_*).")
    p.add_argument("--target-name", default="Dataset040_WH_ImageCHD_HU_Detail",
                   help="Target dataset folder name under nnUNet_raw. "
                        "Must start with 'Dataset{ID:03d}_' to satisfy nnU-Net's naming.")
    p.add_argument("--raw-root", default=None,
                   help="Path to nnUNet_raw. Defaults to $nnUNet_raw.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true",
                      help="Copy image files instead of symlinking (default is --symlink).")
    mode.add_argument("--symlink", action="store_true",
                      help="Symlink image files (default).")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe and rebuild the target dataset folder if it already has content.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and per-case summary without writing any files.")
    args = p.parse_args()

    copy_mode = bool(args.copy)  # default = False → symlink

    expected_prefix = f"Dataset{args.target_id:03d}_"
    if not args.target_name.startswith(expected_prefix):
        raise SystemExit(
            f"ERROR: --target-name must start with '{expected_prefix}' (got '{args.target_name}'). "
            "nnU-Net requires the folder name to match its ID."
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

    dst_images_tr = dst_root / "imagesTr"
    dst_labels_tr = dst_root / "labelsTr"
    dst_images_ts = dst_root / "imagesTs"
    dst_labels_ts = dst_root / "labelsTs"

    print("=" * 68)
    print(f"  Whole-heart conversion  ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print("=" * 68)
    print(f"  source : {src_root}")
    print(f"  target : {dst_root}")
    print(f"  mode   : {'copy' if copy_mode else 'symlink'} images, binarize labels")
    print(f"  overwrite : {args.overwrite}")
    print()

    image_files = _scan_files(src_images_tr, ".nii.gz")
    label_files = _scan_files(src_labels_tr, ".nii.gz")
    test_image_files = _scan_files(src_images_ts, ".nii.gz")
    test_label_files = _scan_files(src_labels_ts, ".nii.gz")

    if not image_files:
        raise SystemExit(f"ERROR: no .nii.gz files in {src_images_tr}")
    if not label_files:
        raise SystemExit(f"ERROR: no .nii.gz files in {src_labels_tr}")

    # Pair imagesTr (case_id_0000.nii.gz) ↔ labelsTr (case_id.nii.gz).
    case_ids = sorted({f.name.removesuffix("_0000.nii.gz") for f in image_files})
    label_ids = sorted({f.name.removesuffix(".nii.gz") for f in label_files})
    if case_ids != label_ids:
        only_in_images = sorted(set(case_ids) - set(label_ids))
        only_in_labels = sorted(set(label_ids) - set(case_ids))
        raise SystemExit(
            f"ERROR: images / labels mismatch.\n"
            f"  only in imagesTr: {only_in_images[:5]}{'...' if len(only_in_images) > 5 else ''}\n"
            f"  only in labelsTr: {only_in_labels[:5]}{'...' if len(only_in_labels) > 5 else ''}"
        )

    print(f"  imagesTr cases : {len(case_ids)}")
    print(f"  imagesTs cases : {len(test_image_files)}")
    print(f"  labelsTs cases : {len(test_label_files)}")
    print()

    # ── Prepare destination folders ──────────────────────────────────────
    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
    for folder, role in [
        (dst_images_tr, "imagesTr"),
        (dst_labels_tr, "labelsTr"),
    ]:
        _prepare_dst_folder(folder, args.overwrite, args.dry_run, role)
    if test_image_files:
        _prepare_dst_folder(dst_images_ts, args.overwrite, args.dry_run, "imagesTs")
    if test_label_files:
        _prepare_dst_folder(dst_labels_ts, args.overwrite, args.dry_run, "labelsTs")

    # ── imagesTr ─────────────────────────────────────────────────────────
    print(f"-- imagesTr ({len(image_files)} files) --")
    for f in image_files:
        _link_or_copy(f.resolve(), dst_images_tr / f.name, copy_mode, args.dry_run)

    # ── imagesTs ─────────────────────────────────────────────────────────
    if test_image_files:
        print(f"-- imagesTs ({len(test_image_files)} files) --")
        for f in test_image_files:
            _link_or_copy(f.resolve(), dst_images_ts / f.name, copy_mode, args.dry_run)

    # ── labelsTr + labelsTs: binarize both ──────────────────────────────
    summary_rows: list[dict] = []
    affine_failures: list[str] = []
    delta_failures: list[str] = []

    def _binarize_split(files: list[Path], dst_dir: Path, split: str) -> None:
        print(f"-- {split} ({len(files)} files): binarizing --")
        for f in files:
            case_id = f.name.removesuffix(".nii.gz")
            dst = dst_dir / f.name
            try:
                counter, orig_fg, bin_fg, affine_ok = _binarize_label(
                    f.resolve(), dst, args.dry_run
                )
            except Exception as e:
                print(f"  [FAIL] {case_id}: {e}", file=sys.stderr)
                raise

            delta = bin_fg - orig_fg
            unique_labels = sorted(counter.keys())
            ok = (delta == 0) and affine_ok
            flag = "OK" if ok else "WARN"
            print(
                f"  [{flag}] {case_id:>16}  labels={unique_labels}  "
                f"orig_fg={orig_fg}  bin_fg={bin_fg}  delta={delta}  affine_ok={affine_ok}"
            )

            if delta != 0:
                delta_failures.append(f"{split}/{case_id}")
            if not affine_ok:
                affine_failures.append(f"{split}/{case_id}")

            summary_rows.append({
                "split": split,
                "case_id": case_id,
                "original_labels_present": ",".join(str(v) for v in unique_labels),
                "original_fg_voxels": orig_fg,
                "binary_fg_voxels": bin_fg,
                "fg_voxel_delta": delta,
                "affine_preserved": affine_ok,
            })

    _binarize_split(label_files, dst_labels_tr, "labelsTr")
    if test_label_files:
        _binarize_split(test_label_files, dst_labels_ts, "labelsTs")

    # ── dataset.json ─────────────────────────────────────────────────────
    print()
    print("-- dataset.json --")
    if args.dry_run:
        print(f"  [DRY-RUN] would write {dst_root / 'dataset.json'} "
              f"(labels={{'background':0,'heart':1}}, numTraining={len(case_ids)})")
    else:
        generate_dataset_json(
            output_folder=str(dst_root),
            channel_names={0: "CT"},
            labels={"background": 0, "heart": 1},
            num_training_cases=len(case_ids),
            file_ending=".nii.gz",
            dataset_name=args.target_name,
            description=(
                f"Binary whole-heart segmentation derived from {args.source_dataset} "
                "by collapsing all 7 foreground labels (LV-BP, RV-BP, LA, RA, Myo, AO, PA) "
                "into a single 'heart' class. Test set + imagesTr are "
                f"{'copied' if copy_mode else 'symlinked'} from the source dataset."
            ),
            reference=f"Derived from {args.source_dataset}",
            converted_by="convert_imagechd_to_wholeheart.py",
            license="See source dataset license.",
        )
        print(f"  wrote {dst_root / 'dataset.json'}")

    # ── conversion_summary.csv ───────────────────────────────────────────
    summary_path = dst_root / "conversion_summary.csv"
    if args.dry_run:
        print(f"  [DRY-RUN] would write {summary_path}  ({len(summary_rows)} rows)")
    else:
        with summary_path.open("w", newline="") as fp:
            fieldnames = ["split", "case_id", "original_labels_present",
                          "original_fg_voxels", "binary_fg_voxels",
                          "fg_voxel_delta", "affine_preserved"]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  wrote {summary_path}")

    # ── final report ─────────────────────────────────────────────────────
    print()
    print("=" * 68)
    if delta_failures or affine_failures:
        print("  COMPLETED WITH WARNINGS")
        if delta_failures:
            print(f"    fg-voxel-delta != 0 on {len(delta_failures)} cases: "
                  f"{delta_failures[:5]}{'...' if len(delta_failures) > 5 else ''}")
        if affine_failures:
            print(f"    affine drift on {len(affine_failures)} cases: "
                  f"{affine_failures[:5]}{'...' if len(affine_failures) > 5 else ''}")
        print("  Inspect the cases above before training.")
        return 1
    print("  CONVERSION COMPLETE — all sanity checks passed")
    print("=" * 68)
    print()
    print("Next step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.target_id} "
          "-pl nnUNetPlannerResEncM -c 3d_fullres 3d_lowres 3d_cascade_fullres "
          "--verify_dataset_integrity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
