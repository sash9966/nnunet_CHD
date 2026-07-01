"""
chd_landmarks.nnunet_dataset_builder
=====================================

Builds a NEW nnU-Net raw dataset (default Dataset031_imageCHD_DiseaseLandmarks)
from an existing source dataset + CHD metadata + configs.

SAFETY
  * The source dataset is read ONLY (images are symlinked, never modified).
  * The target dataset id/name are CLI-controlled and must differ from the
    source. The builder refuses to write into a non-empty target unless
    --overwrite is given. It never calls plan_and_preprocess.

Layout produced:
    <out_root>/Dataset0XX_<name>/
        imagesTr/                 (symlinks to source image channels)
        labelsTr/                 (merged integer label maps)
        derived_masksTr/<case>/   (one NIfTI per auxiliary/derived region)
        case_metadata/<case>.json (active flags, derived regions, confidence)
        dataset.json
        chd_derivation_report.json
        chd_derivation_report.csv
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import io
from .derived_label_builder import DerivedLabelBuilder
from .disease_rules import load_rules
from .labels import load_label_map
from .metadata import CaseMetadata, load_disease_flags, normalize_case_key, summarize


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _prepare_target(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists() and any(dst_root.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: target {dst_root} is not empty. Pass --overwrite to rebuild, "
                f"or choose a different --target-dataset-id/name."
            )
        for entry in dst_root.iterdir():
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)
    dst_root.mkdir(parents=True, exist_ok=True)


def build_dataset(
    source_dataset: str,
    target_dataset_id: int,
    target_dataset_name: str,
    metadata_path: str,
    label_map_cfg: str,
    rules_cfg: str,
    derived_cfg_path: str,
    out_root: Optional[str] = None,
    raw_root: Optional[str] = None,
    copy_images: bool = False,
    overwrite: bool = False,
    id_prefix: str = "ct_",
    limit: Optional[int] = None,
) -> int:
    src_root = io.resolve_dataset_dir(source_dataset, raw_root)
    expected_prefix = f"Dataset{target_dataset_id:03d}_"
    target_folder = target_dataset_name
    if not target_folder.startswith("Dataset"):
        target_folder = f"{expected_prefix}{target_dataset_name}"
    if not target_folder.startswith(expected_prefix):
        raise SystemExit(
            f"ERROR: target name must start with '{expected_prefix}' (got '{target_folder}')."
        )

    out_root_path = Path(out_root or raw_root or os.environ.get("nnUNet_raw") or src_root.parent)
    dst_root = (out_root_path / target_folder).resolve()
    if dst_root == src_root:
        raise SystemExit("ERROR: target dataset must differ from source dataset.")

    # --- load configs + metadata ---------------------------------------
    ds_json = io.read_dataset_json(src_root)
    file_ending = ds_json.get("file_ending", ".nii.gz")
    channel_names = ds_json.get("channel_names", {"0": "CT"})

    label_map = load_label_map(label_map_cfg, dataset_dir=str(src_root))
    ruleset = load_rules(rules_cfg)
    derived_cfg = io.load_yaml(derived_cfg_path)
    builder = DerivedLabelBuilder(label_map, ruleset, derived_cfg)

    meta_all = load_disease_flags(metadata_path, ruleset.flag_columns, id_prefix=id_prefix)

    # --- enumerate cases (source labelsTr) -----------------------------
    src_images = src_root / "imagesTr"
    src_labels = src_root / "labelsTr"
    label_files = sorted(p for p in src_labels.iterdir()
                         if p.name.endswith(file_ending)) if src_labels.is_dir() else []
    if not label_files:
        raise SystemExit(f"ERROR: no label files in {src_labels}")
    if limit:
        label_files = label_files[:limit]

    print("=" * 70)
    print(f"  CHD disease-landmark dataset build")
    print(f"  source : {src_root}  (READ ONLY)")
    print(f"  target : {dst_root}")
    print(f"  labels : merged scheme {builder.merged_dataset_labels()}")
    print(f"  cases  : {len(label_files)}")
    print("=" * 70)
    print(f"  disease flag summary: {summarize(meta_all)}")
    for w in label_map.warnings:
        print(f"  [label-map warning] {w}")

    # --- prepare target -------------------------------------------------
    _prepare_target(dst_root, overwrite)
    (dst_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst_root / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dst_root / "derived_masksTr").mkdir(parents=True, exist_ok=True)
    (dst_root / "case_metadata").mkdir(parents=True, exist_ok=True)

    report_rows: List[dict] = []
    full_report: Dict[str, dict] = {}
    n_matched = 0  # cases whose id matched a metadata row

    for lf in label_files:
        case_id = io.case_id_from_label_file(lf.name, file_ending)   # keeps _image (file naming)
        meta_key = normalize_case_key(case_id)                        # bare id for flag lookup
        meta = meta_all.get(meta_key, CaseMetadata(case_id=meta_key, flags={}))
        if meta_key in meta_all:
            n_matched += 1
        else:
            io.warn(f"{case_id} (key '{meta_key}'): no metadata row found -> treated as no-disease")

        lab = io.read_label(lf)
        derivation = builder.build_for_case(
            lab.data, meta, case_id, affine=lab.affine, spacing=lab.spacing)

        # write merged label map
        io.write_like(lab, derivation.merged_label_map,
                      dst_root / "labelsTr" / f"{case_id}{file_ending}", dtype=np.uint8)

        # symlink (or copy) image channels
        for img in sorted(src_images.glob(f"{case_id}_*{file_ending}")):
            _link_or_copy(img, dst_root / "imagesTr" / img.name, copy_images)

        # write auxiliary / derived masks
        case_mask_dir = dst_root / "derived_masksTr" / case_id
        if derivation.auxiliary_masks:
            case_mask_dir.mkdir(parents=True, exist_ok=True)
            for name, mask in derivation.auxiliary_masks.items():
                io.write_mask(lab, mask, case_mask_dir / f"{name}{file_ending}")
        # hard audit map
        if derivation.hard_label_map.any():
            case_mask_dir.mkdir(parents=True, exist_ok=True)
            io.write_like(lab, derivation.hard_label_map,
                          case_mask_dir / f"_hard_label_map{file_ending}", dtype=np.uint8)

        # per-case metadata json
        case_record = {
            "case_id": case_id,
            "active_disease_flags": meta.active_diseases(),
            "all_flags": meta.flags,
            "derived_regions": {n: m for n, m in derivation.region_meta.items()
                                if derivation.auxiliary_masks.get(n) is not None},
            "annotation_status": derivation.annotation_status.to_dict(),
            "merged_labels": builder.merged_dataset_labels(),
            "warnings": derivation.warnings,
        }
        io.save_json(case_record, dst_root / "case_metadata" / f"{case_id}.json")
        full_report[case_id] = case_record

        report_rows.append({
            "case_id": case_id,
            "active_flags": ";".join(meta.active_diseases()),
            "derived_regions": ";".join(sorted(derivation.auxiliary_masks.keys())),
            "merged_hard_labels": ";".join(sorted(derivation.annotation_status.derived.keys())),
            "unknown_absence": ";".join(sorted(set(derivation.annotation_status.unknown))),
            "n_warnings": len(derivation.warnings),
        })
        print(f"  [OK] {case_id:>12}  flags={meta.active_diseases() or '-'}  "
              f"derived={sorted(derivation.auxiliary_masks.keys()) or '-'}")

    # --- guard: a naming mismatch that matched ZERO metadata rows means no
    #     disease flags were read, so NO disease labels were derived -> abort
    #     loudly instead of silently shipping a defect-free dataset.
    if n_matched == 0:
        example = io.case_id_from_label_file(label_files[0].name, file_ending)
        raise SystemExit(
            f"ERROR: no label case id matched a metadata row (0/{len(label_files)}).\n"
            f"  Example case id from labels: '{example}' -> lookup key "
            f"'{normalize_case_key(example)}'.\n"
            f"  Metadata keys look like: {list(meta_all)[:3]}.\n"
            f"  Check --metadata ({metadata_path}) and the id-prefix / naming. "
            f"No disease labels would be derived, so the build is aborted.")
    print(f"  matched metadata for {n_matched}/{len(label_files)} cases")

    # --- dataset.json ---------------------------------------------------
    _write_dataset_json(dst_root, channel_names, builder.merged_dataset_labels(),
                        len(label_files), file_ending, target_folder, source_dataset)

    # --- reports --------------------------------------------------------
    io.save_json(full_report, dst_root / "chd_derivation_report.json")
    with open(dst_root / "chd_derivation_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    print("=" * 70)
    print(f"  DONE. New dataset at {dst_root}")
    print(f"  Next: nnUNetv2_plan_and_preprocess -d {target_dataset_id} "
          f"-pl nnUNetPlannerResEncM --verify_dataset_integrity")
    print("=" * 70)
    return 0


def _write_dataset_json(dst_root: Path, channel_names: dict, labels: dict,
                        num_training: int, file_ending: str, name: str, source: str) -> None:
    """Write a plain integer-label dataset.json (region-based is a separate step)."""
    try:
        from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
        generate_dataset_json(
            output_folder=str(dst_root),
            channel_names={int(k): v for k, v in channel_names.items()},
            labels=labels,
            num_training_cases=num_training,
            file_ending=file_ending,
            dataset_name=name,
            description=(f"CHD disease-landmark dataset derived from {source}. "
                         "Adds conservative disease-proxy integer labels to the 7 "
                         "anatomy labels. Images symlinked from source."),
            converted_by="chd_landmarks.nnunet_dataset_builder",
            reference=f"Derived from {source}",
        )
    except Exception as e:  # noqa: BLE001
        io.warn(f"generate_dataset_json unavailable ({e}); writing minimal dataset.json")
        io.save_json({
            "channel_names": channel_names,
            "labels": labels,
            "numTraining": num_training,
            "file_ending": file_ending,
            "name": name,
            "description": f"CHD disease-landmark dataset derived from {source}",
        }, dst_root / "dataset.json")
