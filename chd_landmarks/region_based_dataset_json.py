"""
chd_landmarks.region_based_dataset_json
=======================================

Generate an nnU-Net v2 *region-based* dataset.json for overlapping /
hierarchical targets, driven by configs/chd_region_training.yaml.

In region-based training, `labels` maps a region NAME to either a single
integer or a LIST of integers (the region = union of those label values in the
stored integer segmentation). `regions_class_order` is REQUIRED and gives, for
each non-background region in order, the integer label value that region is
painted as when converting predicted regions back to a segmentation. Regions
are painted in the listed order, so LATER regions overwrite earlier ones.

ORDER MATTERS. This generator derives `regions_class_order` from the `priority`
field (low priority = painted first = lowest precedence) and prints the result
for review. Composite (multi-member) regions are flagged with a warning because
their representative class is ambiguous and may need manual adjustment.

By default it writes a SEPARATE file (`dataset_region_based.json`) and does NOT
touch the existing integer `dataset.json`. Pass apply=True to install it as
`dataset.json` (the original is backed up to `dataset.json.integer_backup`).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import io
from .labels import load_label_map


def build_region_based_json(
    dataset_dir: str,
    label_map_cfg: str,
    region_cfg_path: str,
    apply: bool = False,
) -> dict:
    dataset_dir = Path(io.resolve_dataset_dir(dataset_dir))
    ds_json = io.read_dataset_json(dataset_dir)
    region_cfg = io.load_yaml(region_cfg_path)
    label_map = load_label_map(label_map_cfg, dataset_dir=str(dataset_dir))

    # Resolve a member NAME to an integer id: anatomy structure or an existing
    # derived label name already present in the dataset's integer labels.
    int_labels: Dict[str, int] = {
        k: int(v) for k, v in ds_json.get("labels", {}).items()
        if isinstance(v, int) or (isinstance(v, str) and str(v).isdigit())
    }
    int_labels_norm = {k.lower(): v for k, v in int_labels.items()}

    def member_id(name: str) -> Optional[int]:
        if name in label_map.structure_to_id and label_map.id_of(name) is not None:
            return label_map.id_of(name)
        if name.lower() in int_labels_norm:
            return int_labels_norm[name.lower()]
        return None

    regions_cfg: Dict[str, dict] = region_cfg.get("regions", {})
    warnings: List[str] = []

    # Build (priority, name, member_ids) records
    records = []
    for name, spec in regions_cfg.items():
        members = spec.get("members", [])
        ids = []
        for m in members:
            mid = member_id(m)
            if mid is None:
                warnings.append(f"region '{name}': member '{m}' not in dataset labels -> dropped")
            else:
                ids.append(mid)
        if not ids:
            warnings.append(f"region '{name}': no resolvable members -> skipped")
            continue
        records.append((int(spec.get("priority", 0)), name, sorted(set(ids))))

    # Sort by priority ascending (painted first -> lowest precedence)
    records.sort(key=lambda r: (r[0], r[1]))

    labels_out: Dict[str, object] = {"background": label_map.background_id}
    class_order: List[int] = []
    for prio, name, ids in records:
        labels_out[name] = ids[0] if len(ids) == 1 else ids
        # representative class for regions_class_order
        rep = ids[0] if len(ids) == 1 else min(ids)
        class_order.append(int(rep))
        if len(ids) > 1:
            warnings.append(
                f"region '{name}' is composite {ids}; representative class set to {rep}. "
                f"VERIFY regions_class_order — composite regions overlap.")

    out = dict(ds_json)
    out["labels"] = labels_out
    out["regions_class_order"] = class_order

    for w in warnings:
        io.warn(w)
    print("[region-based] labels:", labels_out)
    print("[region-based] regions_class_order:", class_order)
    print("[region-based] (painted in this order; later regions overwrite earlier)")

    sep_path = dataset_dir / "dataset_region_based.json"
    io.save_json(out, sep_path)
    print(f"[region-based] wrote {sep_path}")

    if apply:
        ds_path = dataset_dir / "dataset.json"
        backup = dataset_dir / "dataset.json.integer_backup"
        if ds_path.is_file() and not backup.is_file():
            shutil.copy2(ds_path, backup)
            print(f"[region-based] backed up integer dataset.json -> {backup}")
        io.save_json(out, ds_path)
        print(f"[region-based] installed region-based dataset.json (training will use regions)")
    else:
        print("[region-based] NOT applied (pass --apply to install as dataset.json)")

    return out
