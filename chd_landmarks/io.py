"""
chd_landmarks.io
================

Thin I/O layer: affine-preserving NIfTI read/write, YAML/JSON config loading,
and nnU-Net dataset.json reading. Kept dependency-light (nibabel + pyyaml).

All NIfTI writes preserve the source affine + header so derived masks line up
voxel-for-voxel with the original anatomy segmentation.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Volume container
# ---------------------------------------------------------------------------
@dataclass
class Volume:
    """A 3D (or 2D) image/label volume plus the geometry needed to write it back."""
    data: np.ndarray
    affine: np.ndarray
    header: Any              # nibabel header (opaque; carried through on write)
    spacing: Tuple[float, ...]   # physical voxel size, mm, in array-axis order

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml is required to read config files. `pip install pyyaml`.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ---------------------------------------------------------------------------
# NIfTI I/O
# ---------------------------------------------------------------------------
def _spacing_from_affine(affine: np.ndarray, ndim: int) -> Tuple[float, ...]:
    """Physical voxel size (mm) per array axis = column norms of the affine."""
    spac = np.sqrt((np.asarray(affine[:3, :3]) ** 2).sum(axis=0))
    spac = tuple(float(s) for s in spac[:ndim])
    if len(spac) < ndim:  # 2D padding safety
        spac = spac + (1.0,) * (ndim - len(spac))
    return spac


def read_volume(path: str | os.PathLike, dtype: Optional[np.dtype] = None) -> Volume:
    """Read a NIfTI volume, preserving affine + header. `dtype` optionally casts data."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    spacing = _spacing_from_affine(img.affine, data.ndim)
    return Volume(data=data, affine=np.asarray(img.affine), header=img.header, spacing=spacing)


def read_label(path: str | os.PathLike) -> Volume:
    """Read a segmentation label map as int32 (background 0)."""
    return read_volume(path, dtype=np.int32)


def write_like(reference: Volume, data: np.ndarray, path: str | os.PathLike,
               dtype: np.dtype = np.uint8) -> None:
    """Write `data` as NIfTI using the reference volume's affine + header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = data.astype(dtype)
    img = nib.Nifti1Image(out, reference.affine, reference.header)
    img.set_data_dtype(dtype)
    nib.save(img, str(path))


def write_mask(reference: Volume, mask: np.ndarray, path: str | os.PathLike) -> None:
    """Write a binary mask (0/1) as uint8."""
    write_like(reference, (mask > 0).astype(np.uint8), path, dtype=np.uint8)


# ---------------------------------------------------------------------------
# nnU-Net dataset helpers
# ---------------------------------------------------------------------------
def read_dataset_json(dataset_dir: str | os.PathLike) -> Dict[str, Any]:
    p = Path(dataset_dir) / "dataset.json"
    if not p.is_file():
        raise FileNotFoundError(f"dataset.json not found in {dataset_dir}")
    return load_json(p)


def case_id_from_label_file(filename: str, file_ending: str = ".nii.gz") -> str:
    """labelsTr file 'ct_1001.nii.gz' -> 'ct_1001'."""
    name = os.path.basename(filename)
    if name.endswith(file_ending):
        name = name[: -len(file_ending)]
    return name


def resolve_dataset_dir(dataset: str | os.PathLike, raw_root: Optional[str] = None) -> Path:
    """
    Resolve a dataset directory. `dataset` may be an absolute path, or a folder
    name to look up under `raw_root` / $nnUNet_raw.
    """
    p = Path(dataset)
    if p.is_dir():
        return p.resolve()
    root = raw_root or os.environ.get("nnUNet_raw")
    if root:
        cand = Path(root) / str(dataset)
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset}'. Pass an absolute path or set "
        f"$nnUNet_raw / --raw-root."
    )


def warn(msg: str) -> None:
    """Emit a uniform, greppable warning."""
    warnings.warn(f"[chd_landmarks] {msg}", stacklevel=2)
