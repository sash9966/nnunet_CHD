"""
chd_landmarks.topology
======================

Topology utilities: connected components, skeletons, label adjacency, contact
surfaces, and (optionally) Betti numbers / Euler characteristic.

Heavy / optional dependencies (gudhi, cripser, skimage) are imported lazily.
If a dependency is missing the function emits a warning and returns None / a
graceful fallback rather than crashing the pipeline.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


def _warn(msg: str) -> None:
    warnings.warn(f"[chd_landmarks.topology] {msg}", stacklevel=2)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------
def _structure(ndim: int):
    return ndi.generate_binary_structure(ndim, 1)


def connected_component_count(mask: np.ndarray, min_voxels: int = 1) -> int:
    m = mask.astype(bool)
    if not m.any():
        return 0
    lbl, n = ndi.label(m, structure=_structure(m.ndim))
    if min_voxels <= 1:
        return int(n)
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    return int((sizes >= min_voxels).sum())


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    lbl, n = ndi.label(m, structure=_structure(m.ndim))
    if n <= 1:
        return m
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def top_k_components(mask: np.ndarray, k: int) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any() or k <= 0:
        return np.zeros_like(m)
    lbl, n = ndi.label(m, structure=_structure(m.ndim))
    if n <= k:
        return m
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    keep = set(int(i) + 1 for i in np.argsort(sizes)[::-1][:k])
    out = np.zeros_like(m)
    for c in keep:
        out |= (lbl == c)
    return out


def largest_cc_fraction(mask: np.ndarray) -> Optional[float]:
    m = mask.astype(bool)
    total = int(m.sum())
    if total == 0:
        return None
    return float(largest_connected_component(m).sum()) / total


# ---------------------------------------------------------------------------
# Label adjacency / contact
# ---------------------------------------------------------------------------
def _dilate_mm(mask: np.ndarray, spacing: Sequence[float], dilation_mm: float) -> np.ndarray:
    if dilation_mm <= 0:
        return mask.astype(bool)
    iters = max(1, int(round(dilation_mm / max(min(spacing), 1e-6))))
    return ndi.binary_dilation(mask.astype(bool), _structure(mask.ndim), iterations=iters)


def labels_touch(seg: np.ndarray, label_a: int, label_b: int,
                 spacing: Sequence[float], dilation_mm: float = 0.0) -> bool:
    a = seg == label_a
    b = seg == label_b
    if not a.any() or not b.any():
        return False
    a_d = _dilate_mm(a, spacing, dilation_mm) if dilation_mm > 0 else \
        ndi.binary_dilation(a, _structure(seg.ndim))
    return bool((a_d & b).any())


def contact_surface(mask_a: np.ndarray, mask_b: np.ndarray,
                    spacing: Sequence[float], dilation_mm: float = 1.0) -> np.ndarray:
    """Voxels of either mask lying within `dilation_mm` of the other mask.
    Returns the near-contact band (union side).

    Computed on the combined bounding box (+ margin) of the two masks so cost
    scales with the local region, not the full volume.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.zeros_like(a)
    iters = max(1, int(round(dilation_mm / max(min(spacing), 1e-6)))) if dilation_mm > 0 else 1
    union = a | b
    coords = np.array(np.nonzero(union))
    mins = np.maximum(coords.min(axis=1) - (iters + 1), 0)
    maxs = np.minimum(coords.max(axis=1) + 1 + (iters + 1), a.shape)
    sl = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
    ac, bc = a[sl], b[sl]
    if dilation_mm > 0:
        a_d = ndi.binary_dilation(ac, _structure(ac.ndim), iterations=iters)
        b_d = ndi.binary_dilation(bc, _structure(bc.ndim), iterations=iters)
    else:
        a_d = ndi.binary_dilation(ac, _structure(ac.ndim))
        b_d = ndi.binary_dilation(bc, _structure(bc.ndim))
    out = np.zeros_like(a)
    out[sl] = (a_d & bc) | (b_d & ac)
    return out


def label_adjacency_graph(seg: np.ndarray, labels: List[int],
                          spacing: Sequence[float], dilation_mm: float = 1.0) -> Dict[Tuple[int, int], int]:
    """Return {(la, lb): contact_voxel_count} for label pairs that touch."""
    graph: Dict[Tuple[int, int], int] = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            band = contact_surface(seg == la, seg == lb, spacing, dilation_mm)
            c = int(band.sum())
            if c > 0:
                graph[(la, lb)] = c
    return graph


# ---------------------------------------------------------------------------
# Skeletons / centerlines
# ---------------------------------------------------------------------------
def _bbox_slices(m: np.ndarray, margin: int = 2):
    coords = np.array(np.nonzero(m))
    mins = np.maximum(coords.min(axis=1) - margin, 0)
    maxs = np.minimum(coords.max(axis=1) + 1 + margin, m.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def skeletonize_3d_safe(mask: np.ndarray) -> Optional[np.ndarray]:
    """Binary skeleton via skimage; returns None (with warning) if unavailable.

    Operates on the mask's bounding box (with a small margin) and pastes the
    result back, so cost scales with the structure size, not the full volume.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    try:
        from skimage.morphology import skeletonize
    except Exception as e:  # noqa: BLE001
        _warn(f"skimage unavailable ({e}); skeleton-based metrics return None")
        return None
    try:
        sl = _bbox_slices(m, margin=2)
        out = np.zeros_like(m)
        out[sl] = skeletonize(m[sl]).astype(bool)
        return out
    except Exception as e:  # noqa: BLE001
        _warn(f"skeletonize failed ({e}); returning None")
        return None


def skeleton_branch_count(mask: np.ndarray) -> Optional[int]:
    """Count skeleton branch voxels (endpoints + junctions) as a crude branch proxy."""
    skel = skeletonize_3d_safe(mask)
    if skel is None:
        return None
    if not skel.any():
        return 0
    neigh = ndi.convolve(skel.astype(np.uint8), np.ones((3,) * skel.ndim, dtype=np.uint8),
                         mode="constant") - skel.astype(np.uint8)
    endpoints = int(((neigh == 1) & skel).sum())
    junctions = int(((neigh >= 3) & skel).sum())
    return endpoints + junctions


def centerline_graph_from_skeleton(skel: np.ndarray) -> Dict[str, int]:
    """Summarise a skeleton: voxel count, endpoints, junctions."""
    if skel is None or not skel.any():
        return {"voxels": 0, "endpoints": 0, "junctions": 0}
    neigh = ndi.convolve(skel.astype(np.uint8), np.ones((3,) * skel.ndim, dtype=np.uint8),
                         mode="constant") - skel.astype(np.uint8)
    return {
        "voxels": int(skel.sum()),
        "endpoints": int(((neigh == 1) & skel).sum()),
        "junctions": int(((neigh >= 3) & skel).sum()),
    }


# ---------------------------------------------------------------------------
# Betti / Euler (optional)
# ---------------------------------------------------------------------------
def betti0(mask: np.ndarray, min_voxels: int = 1) -> int:
    """Betti-0 = number of connected components (always available)."""
    return connected_component_count(mask, min_voxels=min_voxels)


def euler_characteristic_binary(mask: np.ndarray) -> Optional[int]:
    """Euler characteristic via skimage.measure.euler_number; None if unavailable."""
    m = mask.astype(bool)
    if not m.any():
        return 0
    try:
        from skimage.measure import euler_number
    except Exception as e:  # noqa: BLE001
        _warn(f"skimage.measure.euler_number unavailable ({e}); returning None")
        return None
    conn = m.ndim  # full connectivity
    try:
        return int(euler_number(m, connectivity=conn))
    except Exception as e:  # noqa: BLE001
        _warn(f"euler_number failed ({e}); returning None")
        return None


def betti_numbers_binary(mask: np.ndarray) -> Optional[Dict[int, int]]:
    """
    Full Betti numbers via cripser/gudhi if installed. Returns
    {0: b0, 1: b1, 2: b2} or None if no persistent-homology lib is available.
    """
    m = mask.astype(bool)
    if not m.any():
        return {0: 0, 1: 0, 2: 0}
    # Try cripser (cubical ripser) first
    try:
        import cripser  # type: ignore
        pd = cripser.computePH(np.where(m, 0.0, 1.0).astype(np.float64))
        betti = {0: 0, 1: 0, 2: 0}
        for row in pd:
            dim = int(row[0])
            birth, death = row[1], row[2]
            if dim in betti and (death > birth):
                betti[dim] += 1
        return betti
    except Exception:  # noqa: BLE001
        pass
    try:
        import gudhi  # type: ignore  # noqa: F401
        _warn("gudhi present but cubical-complex Betti extraction not implemented; "
              "returning b0 only via scipy")
    except Exception:  # noqa: BLE001
        _warn("no persistent-homology lib (cripser/gudhi) found; b1/b2 unavailable")
    return None


def local_topology_signature(mask: np.ndarray, roi: np.ndarray,
                             min_voxels: int = 1) -> Dict[str, Optional[int]]:
    """Topology summary restricted to an ROI: b0 (+ Euler if available)."""
    sub = mask.astype(bool) & roi.astype(bool)
    return {
        "betti0": betti0(sub, min_voxels=min_voxels),
        "euler": euler_characteristic_binary(sub),
    }
