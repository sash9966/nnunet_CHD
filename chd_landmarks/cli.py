"""
chd_landmarks.cli
================

Command-line entry point:

    python -m chd_landmarks.cli inspect-dataset   --nnunet-raw <dir> --metadata <csv>
    python -m chd_landmarks.cli derive-case       --image ... --label ... --metadata ... --case-id ...
    python -m chd_landmarks.cli build-dataset      --source-dataset ... --target-dataset-id 031 ...
    python -m chd_landmarks.cli make-region-dataset-json --dataset ... --region-config ...
    python -m chd_landmarks.cli evaluate-disease-metrics --pred ... --gt ... --case-id ...

Run any subcommand with -h for its options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import io
from .derived_label_builder import DerivedLabelBuilder
from .disease_rules import load_rules
from .labels import load_label_map
from .metadata import CaseMetadata, load_disease_flags, summarize

_DEFAULT_CFG = {
    "label_map": "configs/chd_label_map.yaml",
    "rules": "configs/chd_disease_rules.yaml",
    "derived": "configs/chd_derived_labels.yaml",
    "region": "configs/chd_region_training.yaml",
    "metric": "configs/chd_metric_config.yaml",
}


# ---------------------------------------------------------------------------
def cmd_inspect_dataset(args) -> int:
    ds_dir = io.resolve_dataset_dir(args.nnunet_raw)
    ds_json = io.read_dataset_json(ds_dir)
    label_map = load_label_map(args.label_map, dataset_dir=str(ds_dir))
    ruleset = load_rules(args.rules)

    print("=" * 70)
    print(f"  Dataset: {ds_dir}")
    print(f"  file_ending : {ds_json.get('file_ending')}")
    print(f"  channels    : {ds_json.get('channel_names')}")
    print(f"  raw labels  : {ds_json.get('labels')}")
    print("-" * 70)
    print("  Resolved structures (name -> id, None=absent):")
    for s, i in label_map.structure_to_id.items():
        print(f"    {s:18s} -> {i}")
    if label_map.warnings:
        print("  Warnings:")
        for w in label_map.warnings:
            print(f"    - {w}")

    if args.metadata:
        meta = load_disease_flags(args.metadata, ruleset.flag_columns)
        print("-" * 70)
        print(f"  Metadata: {args.metadata}")
        print(f"  Flag summary: {summarize(meta)}")
        print("  Derivable diseases (flag + anatomy present):")
        sample = next(iter(meta.values())) if meta else None
        for disease, rule in ruleset.rules.items():
            missing = [l for l in rule.relevant_labels if not label_map.has(l)]
            status = "DERIVABLE" if not missing and not rule.requires_missing_anatomy else f"NEEDS {missing or 'extra annotation'}"
            print(f"    {disease:20s} {status}")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
def cmd_derive_case(args) -> int:
    label_map = load_label_map(args.label_map,
                               dataset_dir=str(Path(args.label).parent.parent) if args.dataset_dir is None else args.dataset_dir)
    ruleset = load_rules(args.rules)
    derived_cfg = io.load_yaml(args.derived_config)
    builder = DerivedLabelBuilder(label_map, ruleset, derived_cfg)

    meta_all = load_disease_flags(args.metadata, ruleset.flag_columns)
    meta = meta_all.get(args.case_id, CaseMetadata(case_id=args.case_id, flags={}))

    lab = io.read_label(args.label)
    d = builder.build_for_case(lab.data, meta, args.case_id, affine=lab.affine, spacing=lab.spacing)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_like(lab, d.merged_label_map, out_dir / f"{args.case_id}_merged.nii.gz", dtype="uint8")
    io.write_like(lab, d.hard_label_map, out_dir / f"{args.case_id}_hard.nii.gz", dtype="uint8")
    for name, mask in d.auxiliary_masks.items():
        io.write_mask(lab, mask, out_dir / f"{args.case_id}_{name}.nii.gz")
    io.save_json({
        "case_id": args.case_id,
        "active_flags": meta.active_diseases(),
        "region_meta": d.region_meta,
        "annotation_status": d.annotation_status.to_dict(),
        "merged_labels": builder.merged_dataset_labels(),
        "warnings": d.warnings,
    }, out_dir / f"{args.case_id}_derivation.json")

    print(f"[derive-case] {args.case_id}: flags={meta.active_diseases() or '-'}")
    print(f"[derive-case] derived regions: {sorted(d.auxiliary_masks.keys()) or '-'}")
    print(f"[derive-case] merged hard labels: {sorted(d.annotation_status.derived.keys()) or '-'}")
    print(f"[derive-case] outputs -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
def cmd_build_dataset(args) -> int:
    from .nnunet_dataset_builder import build_dataset
    return build_dataset(
        source_dataset=args.source_dataset,
        target_dataset_id=int(args.target_dataset_id),
        target_dataset_name=args.target_dataset_name,
        metadata_path=args.metadata,
        label_map_cfg=args.label_map,
        rules_cfg=args.rules,
        derived_cfg_path=args.derived_config,
        out_root=args.out_root,
        raw_root=args.raw_root,
        copy_images=args.copy,
        overwrite=args.overwrite,
        limit=args.limit,
    )


# ---------------------------------------------------------------------------
def cmd_make_region_json(args) -> int:
    from .region_based_dataset_json import build_region_based_json
    build_region_based_json(
        dataset_dir=args.dataset,
        label_map_cfg=args.label_map,
        region_cfg_path=args.region_config,
        apply=args.apply,
    )
    return 0


# ---------------------------------------------------------------------------
def cmd_evaluate(args) -> int:
    import numpy as np
    from . import metrics as M

    ds_dir = args.dataset_dir
    label_map = load_label_map(args.label_map, dataset_dir=ds_dir)
    ruleset = load_rules(args.rules)
    metric_cfg = io.load_yaml(args.metric_config)

    pred = io.read_label(args.pred)
    gt = io.read_label(args.gt)

    active = []
    if args.metadata and args.case_id:
        meta = load_disease_flags(args.metadata, ruleset.flag_columns).get(args.case_id)
        if meta:
            active = meta.active_diseases()

    derived_gt = {}
    if args.derived_gt_dir:
        dgt = Path(args.derived_gt_dir)
        for f in dgt.glob("*.nii.gz"):
            name = f.name[:-len(".nii.gz")].replace(f"{args.case_id}_", "")
            derived_gt[name] = io.read_label(f).data

    res = M.evaluate_case(pred.data, gt.data, label_map, pred.spacing,
                          affine=pred.affine, active_diseases=active,
                          derived_gt=derived_gt, metric_cfg=metric_cfg)
    res["case_id"] = args.case_id
    res["active_diseases"] = active
    io.save_json(res, args.out)
    print(f"[evaluate] {args.case_id}: wrote {args.out}")
    print(f"[evaluate] active diseases: {active or '-'}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chd_landmarks.cli",
                                description="CHD disease-landmark preprocessing toolkit.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_cfg(sp):
        sp.add_argument("--label-map", default=_DEFAULT_CFG["label_map"])
        sp.add_argument("--rules", default=_DEFAULT_CFG["rules"])

    sp = sub.add_parser("inspect-dataset", help="Report labels + derivable diseases.")
    sp.add_argument("--nnunet-raw", required=True, help="Dataset dir or name under $nnUNet_raw.")
    sp.add_argument("--metadata", default=None)
    add_cfg(sp)
    sp.set_defaults(func=cmd_inspect_dataset)

    sp = sub.add_parser("derive-case", help="Derive landmarks for a single case.")
    sp.add_argument("--image", default=None)
    sp.add_argument("--label", required=True)
    sp.add_argument("--metadata", required=True)
    sp.add_argument("--case-id", required=True)
    sp.add_argument("--dataset-dir", default=None, help="Source dataset dir for label resolution.")
    sp.add_argument("--derived-config", default=_DEFAULT_CFG["derived"])
    sp.add_argument("--out-dir", required=True)
    add_cfg(sp)
    sp.set_defaults(func=cmd_derive_case)

    sp = sub.add_parser("build-dataset", help="Build a NEW nnU-Net dataset with derived labels.")
    sp.add_argument("--source-dataset", required=True)
    sp.add_argument("--target-dataset-id", required=True)
    sp.add_argument("--target-dataset-name", default="imageCHD_DiseaseLandmarks")
    sp.add_argument("--metadata", required=True)
    sp.add_argument("--derived-config", default=_DEFAULT_CFG["derived"])
    sp.add_argument("--out-root", default=None)
    sp.add_argument("--raw-root", default=None)
    sp.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--limit", type=int, default=None, help="Process only first N cases (debug).")
    add_cfg(sp)
    sp.set_defaults(func=cmd_build_dataset)

    sp = sub.add_parser("make-region-dataset-json", help="Generate region-based dataset.json.")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--region-config", default=_DEFAULT_CFG["region"])
    sp.add_argument("--apply", action="store_true",
                    help="Install as dataset.json (backs up the integer version).")
    add_cfg(sp)
    sp.set_defaults(func=cmd_make_region_json)

    sp = sub.add_parser("evaluate-disease-metrics", help="Disease-aware metrics for one case.")
    sp.add_argument("--pred", required=True)
    sp.add_argument("--gt", required=True)
    sp.add_argument("--derived-gt-dir", default=None)
    sp.add_argument("--metadata", default=None)
    sp.add_argument("--case-id", required=True)
    sp.add_argument("--dataset-dir", default=None)
    sp.add_argument("--metric-config", default=_DEFAULT_CFG["metric"])
    sp.add_argument("--out", required=True)
    add_cfg(sp)
    sp.set_defaults(func=cmd_evaluate)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
