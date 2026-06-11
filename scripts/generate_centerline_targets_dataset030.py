#!/usr/bin/env python
"""
Offline great-vessel centerline (skeleton) precompute for a CHD dataset.

For each training label map, computes the 3D skeleton of the merged great-vessel
mask (AO ∪ PA, resolved by name from dataset.json) and saves it as a compressed
``<case>_centerline.npz`` (key ``centerline``, uint8) next to a manifest.

This is an OPTIONAL artifact:
  * ``nnUNetTrainerDA5CenterlineAux`` trains with an *on-the-fly* soft skeleton
    computed from each cropped patch (correct by construction, no setup needed).
  * These precomputed full-volume skeletons are useful for (a) inspection /
    QA of which voxels get up-weighted, and (b) future dataloader integration
    that crops a precomputed skeleton alongside the patch.

Usage
-----
    python scripts/generate_centerline_targets_dataset030.py -d 30
    python scripts/generate_centerline_targets_dataset030.py -d 30 --out /tmp/cl

Requires: SimpleITK, scikit-image, numpy.  Reads from ``$nnUNet_raw``.
"""
from __future__ import annotations

import argparse
import json
import os
from os.path import isdir, isfile, join

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:  # pragma: no cover
    raise SystemExit("SimpleITK is required: pip install SimpleITK") from e

try:
    from skimage.morphology import skeletonize
except ImportError as e:  # pragma: no cover
    raise SystemExit("scikit-image is required: pip install scikit-image") from e


_AO_ALIASES = {"ao", "aorta"}
_PA_ALIASES = {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary trunk"}


def resolve_vessel_ids(dataset_json: dict) -> list:
    ids = []
    for name, idx in dataset_json.get("labels", {}).items():
        key = str(name).lower().strip()
        if key in _AO_ALIASES or key in _PA_ALIASES:
            ids.append(int(idx))
    return sorted(set(ids))


def find_dataset_dir(raw_root: str, dataset_id: int) -> str:
    prefix = f"Dataset{dataset_id:03d}"
    for name in sorted(os.listdir(raw_root)):
        if name.startswith(prefix) and isdir(join(raw_root, name)):
            return join(raw_root, name)
    raise FileNotFoundError(f"No dataset under {raw_root} matching {prefix}*")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset_id", type=int, required=True,
                    help="nnU-Net dataset id, e.g. 30")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: <dataset>/centerlinesTr)")
    args = ap.parse_args()

    raw_root = os.environ.get("nnUNet_raw")
    if not raw_root or not isdir(raw_root):
        raise SystemExit("Set $nnUNet_raw to your raw data root.")

    ds_dir = find_dataset_dir(raw_root, args.dataset_id)
    labels_dir = join(ds_dir, "labelsTr")
    with open(join(ds_dir, "dataset.json")) as f:
        dataset_json = json.load(f)
    file_ending = dataset_json.get("file_ending", ".nii.gz")

    vessel_ids = resolve_vessel_ids(dataset_json)
    if not vessel_ids:
        raise SystemExit("Could not resolve AO/PA label ids from dataset.json.")
    print(f"[centerline] dataset={os.path.basename(ds_dir)}  vessel label ids={vessel_ids}")

    out_dir = args.out or join(ds_dir, "centerlinesTr")
    os.makedirs(out_dir, exist_ok=True)

    cases = sorted(f for f in os.listdir(labels_dir) if f.endswith(file_ending))
    manifest = {}
    for i, fname in enumerate(cases):
        case_id = fname[: -len(file_ending)]
        img = sitk.ReadImage(join(labels_dir, fname))
        seg = sitk.GetArrayFromImage(img)                       # (Z, Y, X) int
        vessel = np.isin(seg, vessel_ids)
        if vessel.sum() == 0:
            print(f"[{i+1}/{len(cases)}] {case_id}: no vessel voxels, skipping")
            continue
        skel = skeletonize(vessel)                              # bool, same shape
        out_path = join(out_dir, f"{case_id}_centerline.npz")
        np.savez_compressed(out_path, centerline=skel.astype(np.uint8))
        manifest[case_id] = {
            "vessel_voxels": int(vessel.sum()),
            "skeleton_voxels": int(skel.sum()),
            "file": os.path.basename(out_path),
        }
        print(f"[{i+1}/{len(cases)}] {case_id}: vessel={int(vessel.sum())} "
              f"skeleton={int(skel.sum())}")

    with open(join(out_dir, "centerline_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[centerline] wrote {len(manifest)} skeletons + manifest to {out_dir}")


if __name__ == "__main__":
    main()
