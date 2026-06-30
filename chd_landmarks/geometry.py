"""
chd_landmarks.geometry
======================

Voxel/physical geometry helpers used by the region builders and metrics.
All distances are in millimetres. `spacing` is the physical voxel size in
array-axis order (z, y, x for nnU-Net's typical NIfTI arrays); the functions
are axis-order-agnostic as long as `spacing` matches `mask`'s axes.

Pure numpy + scipy. No torch, no nnU-Net imports (locally testable).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from scipy import ndimage as ndi


def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    v = 1.0
    for s in spacing:
        v *= float(s)
    return v


def volume_mm3(mask: np.ndarray, spacing: Sequence[float]) -> float:
    return float(mask.astype(bool).sum()) * voxel_volume_mm3(spacing)


def voxel_to_physical(coord: Sequence[float], affine: np.ndarray) -> np.ndarray:
    """Map an array index (i, j, k) to physical (x, y, z) via the affine."""
    c = np.asarray(list(coord) + [1.0], dtype=float)
    return (np.asarray(affine) @ c)[:3]


def physical_centroid(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    """Centroid of a binary mask in physical (mm) coordinates, or None if empty."""
    if not mask.any():
        return None
    idx = np.array(ndi.center_of_mass(mask.astype(bool)))
    return voxel_to_physical(idx, affine)


def centroid_distance_mm(mask_a: np.ndarray, mask_b: np.ndarray, affine: np.ndarray) -> Optional[float]:
    ca = physical_centroid(mask_a, affine)
    cb = physical_centroid(mask_b, affine)
    if ca is None or cb is None:
        return None
    return float(np.linalg.norm(ca - cb))


def equivalent_diameter_from_volume(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Diameter (mm) of a sphere with the same volume as the mask."""
    vol = volume_mm3(mask, spacing)
    if vol <= 0:
        return 0.0
    return float(2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))


def distance_transform_radius(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Euclidean distance (mm) from each foreground voxel to the nearest background.
    Inside a tube this approximates the local radius."""
    if not mask.any():
        return np.zeros_like(mask, dtype=float)
    return ndi.distance_transform_edt(mask.astype(bool), sampling=spacing)


def approximate_minimum_diameter(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """
    Approximate the minimum local diameter (mm) of a tubular structure as
    2 * (smallest local radius along the structure's medial region).
    Uses the distance transform; robust without explicit centerline.
    """
    if not mask.any():
        return 0.0
    dt = distance_transform_radius(mask, spacing)
    # medial voxels = local maxima of the distance transform
    medial = dt > (0.5 * dt.max())
    radii = dt[medial]
    if radii.size == 0:
        radii = dt[dt > 0]
    return float(2.0 * radii.min()) if radii.size else 0.0


def local_cross_sectional_area_proxy(
    mask: np.ndarray, roi: np.ndarray, spacing: Sequence[float]
) -> float:
    """Cross-section-area proxy (mm^2): mean over the ROI of pi * r^2 using the
    distance-transform radius. Crude but monotone with true area."""
    sub = mask.astype(bool) & roi.astype(bool)
    if not sub.any():
        return 0.0
    dt = distance_transform_radius(mask, spacing)
    r = dt[sub]
    r = r[r > 0]
    if r.size == 0:
        return 0.0
    return float(np.pi * (r.mean() ** 2))


def bbox_from_mask(
    mask: np.ndarray, margin_mm: float = 0.0, spacing: Optional[Sequence[float]] = None
) -> Optional[Tuple[slice, ...]]:
    """Bounding-box slices around a mask, optionally expanded by margin_mm."""
    if not mask.any():
        return None
    coords = np.array(np.nonzero(mask))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    if margin_mm and spacing is not None:
        for ax in range(mask.ndim):
            pad = int(np.ceil(margin_mm / max(spacing[ax], 1e-6)))
            mins[ax] = max(0, mins[ax] - pad)
            maxs[ax] = min(mask.shape[ax], maxs[ax] + pad)
    return tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))


# ---------------------------------------------------------------------------
# Surface distance metrics (HD95, ASSD)
# ---------------------------------------------------------------------------
def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from surface of a -> b and b -> a.

    Computed on the combined bounding box (+ margin) of the two masks so the
    distance transforms scale with the structure size, not the full volume.
    The margin bounds the largest distance reported (research approximation for
    HD95 when the two surfaces are far apart)."""
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() or not b.any():
        return np.array([]), np.array([])
    union = a | b
    coords = np.array(np.nonzero(union))
    margin = 16
    mins = np.maximum(coords.min(axis=1) - margin, 0)
    maxs = np.minimum(coords.max(axis=1) + 1 + margin, a.shape)
    sl = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
    a = a[sl]
    b = b[sl]
    # surface = voxels with at least one background neighbour
    struct = ndi.generate_binary_structure(a.ndim, 1)
    a_surf = a ^ ndi.binary_erosion(a, struct, border_value=0)
    b_surf = b ^ ndi.binary_erosion(b, struct, border_value=0)
    if not a_surf.any() or not b_surf.any():
        return np.array([]), np.array([])
    dt_b = ndi.distance_transform_edt(~b_surf, sampling=spacing)
    dt_a = ndi.distance_transform_edt(~a_surf, sampling=spacing)
    return dt_b[a_surf], dt_a[b_surf]


def surface_distance_metrics(
    pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], percentile: float = 95
) -> dict:
    """Return {'hd95': mm, 'assd': mm}; None values if a mask is empty."""
    d_ab, d_ba = _surface_distances(pred, gt, spacing)
    if d_ab.size == 0 or d_ba.size == 0:
        return {"hd95": None, "assd": None}
    hd95 = float(max(np.percentile(d_ab, percentile), np.percentile(d_ba, percentile)))
    assd = float((d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size))
    return {"hd95": hd95, "assd": assd}
