"""
chd_landmarks.derived_regions
=============================

Per-region builders that turn (anatomy_seg + disease_flags) into derived
landmark / ROI masks. Every builder is CONSERVATIVE:

  * it returns an empty region (confidence 'none') when the diagnosis flag or
    the geometry does not support the landmark — it never hallucinates;
  * it attaches a `confidence` and a human-readable `reason` so the merge step
    and the audit report can decide whether a region is trustworthy enough to
    become a hard integer label.

All masks share the input segmentation's grid (voxel-aligned). Distances are
in mm via `spacing`. Pure numpy/scipy/skimage (locally testable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

from . import geometry as geo
from . import topology as topo
from .labels import LabelMap


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DerivedRegion:
    name: str
    mask: np.ndarray
    confidence: str = "none"        # high | medium | low | none
    reason: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return self.mask is not None and bool(np.any(self.mask))


def _region(name: str, mask: np.ndarray, spacing, affine, confidence: str, reason: str,
            extra: Optional[Dict] = None) -> DerivedRegion:
    mask = (mask > 0).astype(np.uint8)
    meta: Dict = {
        "confidence": confidence,
        "reason": reason,
        "voxels": int(mask.sum()),
        "volume_mm3": geo.volume_mm3(mask, spacing),
        "n_components": topo.connected_component_count(mask),
    }
    if mask.any() and affine is not None:
        c = geo.physical_centroid(mask, affine)
        meta["centroid_mm"] = None if c is None else [float(x) for x in c]
        bb = geo.bbox_from_mask(mask)
        meta["bbox"] = None if bb is None else [[int(s.start), int(s.stop)] for s in bb]
    if extra:
        meta.update(extra)
    return DerivedRegion(name=name, mask=mask, confidence=confidence, reason=reason, meta=meta)


def _empty(name: str, ref_shape, reason: str) -> DerivedRegion:
    return DerivedRegion(name=name, mask=np.zeros(ref_shape, dtype=np.uint8),
                         confidence="none", reason=reason,
                         meta={"confidence": "none", "reason": reason, "voxels": 0})


def _contact_area_mm2(band: np.ndarray, spacing: Sequence[float]) -> float:
    """Approx contact area: #band voxels * area of the two smallest-spacing faces."""
    s = sorted(float(x) for x in spacing)
    face = s[0] * s[1] if len(s) >= 2 else s[0] ** 2
    return float(band.sum()) * face


# ---------------------------------------------------------------------------
# 1. Generic interface ROI
# ---------------------------------------------------------------------------
def build_interface_roi(seg: np.ndarray, label_a: int, label_b: int,
                        spacing, affine=None, dilation_mm: float = 2.0,
                        name: str = "interface_roi") -> DerivedRegion:
    """Voxels where label_a and label_b touch or lie within `dilation_mm`."""
    a = seg == label_a
    b = seg == label_b
    if not a.any() or not b.any():
        return _empty(name, seg.shape, "one or both labels absent")
    band = topo.contact_surface(a, b, spacing, dilation_mm)
    if not band.any():
        return _empty(name, seg.shape, "labels present but not within dilation")
    return _region(name, band, spacing, affine, "medium",
                   f"contact band ({dilation_mm} mm) between labels {label_a},{label_b}")


# ---------------------------------------------------------------------------
# 2 + ASD. Septal-defect orifice proxy
# ---------------------------------------------------------------------------
def _septal_defect_proxy(seg, chamber_a_id, chamber_b_id, myo_id, spacing, affine,
                         params, name, flag_ok: bool) -> DerivedRegion:
    if not flag_ok:
        return _empty(name, seg.shape, "diagnosis flag not set")
    a = seg == chamber_a_id
    b = seg == chamber_b_id
    if not a.any() or not b.any():
        return _empty(name, seg.shape, "both chambers required but one absent")

    dil = float(params.get("interface_dilation_mm", 2.0))
    band = topo.contact_surface(a, b, spacing, dil)
    if not band.any():
        return _empty(name, seg.shape, "chambers do not approach within dilation")

    # Where the septum (myocardium) is present, there is no defect. The defect
    # proxy is contact WITHOUT intervening myocardium.
    if myo_id is not None and (seg == myo_id).any():
        myo = ndi.binary_dilation(seg == myo_id, iterations=1)
        defect = band & ~myo
        myo_available = True
    else:
        defect = band
        myo_available = False

    defect = topo.largest_connected_component(defect) if defect.any() else defect
    if not defect.any():
        return _empty(name, seg.shape, "contact fully separated by myocardium (no defect proxy)")

    band_area_mm2 = _contact_area_mm2(band, spacing)        # full septal contact (context)
    defect_area_mm2 = _contact_area_mm2(defect, spacing)    # the gap itself
    eqd = geo.equivalent_diameter_from_volume(defect, spacing)

    # Confidence is judged on the DEFECT (the septal gap), not the full contact
    # band: a large band with a small gap is a normal septum with a localized
    # VSD, NOT a false merge. A defect that fills (most of) the contact and is
    # large is what indicates a global merge.
    if eqd > 30.0:
        conf, reason = "low", f"septal gap too broad (eqd={eqd:.1f} mm) -> likely a chamber merge, not a localized defect"
    elif not myo_available:
        conf, reason = "low", f"no myocardium present to confirm a true septal gap (eqd={eqd:.1f} mm)"
    elif eqd >= 1.5:
        conf, reason = "high", f"localized LV/RV gap in an otherwise-present septum (eqd={eqd:.1f} mm)"
    else:
        conf, reason = "low", f"defect too small to trust (eqd={eqd:.1f} mm)"

    return _region(name, defect, spacing, affine, conf, reason,
                   extra={"septal_contact_area_mm2": band_area_mm2,
                          "defect_area_mm2": defect_area_mm2,
                          "equivalent_diameter_mm": eqd,
                          "myocardium_available": myo_available})


def build_vsd_orifice_proxy(seg, label_map: LabelMap, spacing, affine, disease_flags,
                            params) -> DerivedRegion:
    flag_ok = bool({"VSD", "ToF", "DORV", "AVSD"} & set(disease_flags))
    return _septal_defect_proxy(
        seg, label_map.id_of("lv"), label_map.id_of("rv"), label_map.id_of("myocardium"),
        spacing, affine, params, "vsd_orifice_proxy", flag_ok)


def build_asd_orifice_proxy(seg, label_map: LabelMap, spacing, affine, disease_flags,
                            params) -> DerivedRegion:
    flag_ok = bool({"ASD", "AVSD"} & set(disease_flags))
    return _septal_defect_proxy(
        seg, label_map.id_of("la"), label_map.id_of("ra"), label_map.id_of("myocardium"),
        spacing, affine, params, "asd_orifice_proxy", flag_ok)


# ---------------------------------------------------------------------------
# Peri-defect blood-pool shell (EVALUATION construct, never a training label)
# ---------------------------------------------------------------------------
def build_peri_defect_roi(defect_mask, seg, chamber_ids, spacing, affine,
                          dilation_mm, name="peri_septal_defect_roi") -> DerivedRegion:
    """Blood-pool shell around a derived defect: ``dilate(defect) ∩ (chambers)``.

    Captures the chamber blood pool *immediately around* the septal defect — the
    locally disease-relevant tissue — as opposed to the whole (volume-dominant)
    chambers. Used for evaluation (peri-defect local Dice), not embedded in
    training. ``chamber_ids`` is the set of adjacent blood-pool labels
    (LV/RV for ventricular, LA/RA for atrial, all four for AVSD).
    """
    if defect_mask is None or not np.any(defect_mask):
        return _empty(name, seg.shape, "no defect to build a peri-region around")
    ids = [int(c) for c in chamber_ids if c is not None]
    if not ids:
        return _empty(name, seg.shape, "no chamber labels available")
    iters = max(1, int(round(dilation_mm / max(min(spacing), 1e-6))))
    shell = ndi.binary_dilation(defect_mask.astype(bool), iterations=iters)
    pool = np.isin(seg, ids)
    roi = shell & pool
    if not roi.any():
        return _empty(name, seg.shape, "no chamber blood pool within shell")
    return _region(name, roi, spacing, affine, "high",
                   f"blood-pool shell ({dilation_mm} mm) around the defect (evaluation ROI)",
                   extra={"shell_mm": dilation_mm, "chamber_ids": ids})


def build_septal_defect_proxy(vsd_region, asd_region, seg, spacing, affine) -> DerivedRegion:
    """Unify the ventricular (VSD) and atrial (ASD) septal-gap proxies into ONE
    'septal defect' label — they are the same kind of lesion (a hole in a septal
    wall). For the merged hard label we keep only components that individually
    reach high confidence; if none do, the union stays low-confidence (so the
    merge step leaves it auxiliary). The full union is still exported for eval.
    """
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    parts = [(tag, r) for tag, r in (("ventricular", vsd_region), ("atrial", asd_region))
             if r is not None and r.present]
    if not parts:
        return _empty("septal_defect_proxy", seg.shape, "no septal defect derivable")
    high = [r for _, r in parts if rank[r.confidence] >= rank["high"]]
    use = high if high else [r for _, r in parts]
    mask = np.zeros(seg.shape, dtype=np.uint8)
    for r in use:
        mask |= r.mask.astype(np.uint8)
    conf = "high" if high else max((r.confidence for _, r in parts), key=lambda c: rank[c])
    components = {tag: r.confidence for tag, r in parts}
    return _region("septal_defect_proxy", mask, spacing, affine, conf,
                   "unified septal-wall defect (ventricular ∪ atrial); "
                   + ", ".join(f"{k}={v}" for k, v in components.items()),
                   extra={"components": components})


# ---------------------------------------------------------------------------
# 3. LV/RV false-merge region
# ---------------------------------------------------------------------------
def build_lv_rv_false_merge_region(seg, label_map: LabelMap, spacing, affine, params) -> DerivedRegion:
    lv_id, rv_id = label_map.id_of("lv"), label_map.id_of("rv")
    if lv_id is None or rv_id is None:
        return _empty("lv_rv_false_merge_region", seg.shape, "LV or RV absent")
    band = topo.contact_surface(seg == lv_id, seg == rv_id, spacing, 1.0)
    if not band.any():
        return _empty("lv_rv_false_merge_region", seg.shape, "LV and RV not in contact")
    area_mm2 = _contact_area_mm2(band, spacing)
    threshold = float(params.get("false_merge_min_contact_mm2", 200.0))
    suspicious = area_mm2 > threshold
    conf = "medium" if suspicious else "low"
    reason = (f"LV/RV contact area {area_mm2:.0f} mm^2 "
              f"{'exceeds' if suspicious else 'below'} {threshold:.0f} mm^2 threshold")
    return _region("lv_rv_false_merge_region", band, spacing, affine, conf, reason,
                   extra={"contact_area_mm2": area_mm2,
                          "false_merge_suspected": bool(suspicious)})


# ---------------------------------------------------------------------------
# Generic vessel narrowing (shared by stenosis + coarctation)
# ---------------------------------------------------------------------------
def _vessel_narrowing(vessel: np.ndarray, spacing, params, drop_frac_key: str):
    """
    Returns (roi_mask, min_point_idx, info). Finds the narrowest segment of a
    tubular mask using the distance transform (and skeleton if available).
    """
    if not vessel.any():
        return None, None, {"reason": "vessel absent"}
    dt = geo.distance_transform_radius(vessel, spacing)
    skel = topo.skeletonize_3d_safe(vessel)
    drop_frac = float(params.get(drop_frac_key, 0.6))
    roi_dil = float(params.get("stenosis_roi_dilation_mm", 4.0))

    if skel is not None and skel.any():
        radii = dt[skel]
        med = float(np.median(radii[radii > 0])) if (radii > 0).any() else 0.0
        if med <= 0:
            return None, None, {"reason": "degenerate radius along centerline"}
        narrow = skel & (dt < drop_frac * med)
        min_idx = np.unravel_index(int(np.argmin(np.where(skel, dt, np.inf))), dt.shape)
        min_radius = float(dt[min_idx])
        used = "skeleton"
    else:
        medial = dt > (0.5 * dt.max())
        med = float(np.median(dt[medial])) if medial.any() else float(dt.max())
        narrow = medial & (dt < drop_frac * med)
        masked = np.where(vessel, dt, np.inf)
        min_idx = np.unravel_index(int(np.argmin(masked)), dt.shape)
        min_radius = float(dt[min_idx]) if np.isfinite(dt[min_idx]) else 0.0
        used = "distance_transform_fallback"

    # ROI = narrow voxels grown locally, clipped to the vessel
    seed = narrow.copy()
    seed[min_idx] = True
    iters = max(1, int(round(roi_dil / max(min(spacing), 1e-6))))
    roi = ndi.binary_dilation(seed, iterations=iters) & vessel
    info = {
        "median_radius_mm": med,
        "min_radius_mm": min_radius,
        "min_diameter_mm": 2.0 * min_radius,
        "drop_fraction": drop_frac,
        "method": used,
        "min_point_idx": [int(x) for x in min_idx],
    }
    return roi, min_idx, info


# ---------------------------------------------------------------------------
# 4. Pulmonary stenosis ROI
# ---------------------------------------------------------------------------
def build_pulmonary_stenosis_roi(seg, label_map: LabelMap, spacing, affine, disease_flags,
                                 params) -> DerivedRegion:
    if not ({"ToF", "Pulmonary_Stenosis"} & set(disease_flags)):
        return _empty("pulmonary_stenosis_roi", seg.shape, "no pulmonary-stenosis flag")
    pa_id = label_map.id_of("pulmonary_artery")
    if pa_id is None:
        return _empty("pulmonary_stenosis_roi", seg.shape, "pulmonary artery absent")
    vessel = seg == pa_id
    roi, min_idx, info = _vessel_narrowing(vessel, spacing, params, "stenosis_radius_drop_frac")
    if roi is None or not roi.any():
        return _empty("pulmonary_stenosis_roi", seg.shape, info.get("reason", "no narrowing found"))
    return _region("pulmonary_stenosis_roi", roi, spacing, affine, "medium",
                   f"narrowest PA segment (min diameter {info['min_diameter_mm']:.1f} mm via {info['method']})",
                   extra=info)


# ---------------------------------------------------------------------------
# 5. Aortic coarctation ROI
# ---------------------------------------------------------------------------
def build_aortic_coarctation_roi(seg, label_map: LabelMap, spacing, affine, disease_flags,
                                 params) -> Dict[str, DerivedRegion]:
    out: Dict[str, DerivedRegion] = {}
    if "Coarctation" not in set(disease_flags):
        out["aortic_narrowing_roi"] = _empty("aortic_narrowing_roi", seg.shape, "no coarctation flag")
        return out
    ao_id = label_map.id_of("aorta")
    if ao_id is None:
        out["aortic_narrowing_roi"] = _empty("aortic_narrowing_roi", seg.shape, "aorta absent")
        return out
    vessel = seg == ao_id
    roi, min_idx, info = _vessel_narrowing(vessel, spacing, params, "coarctation_radius_drop_frac")
    if roi is None or not roi.any():
        out["aortic_narrowing_roi"] = _empty("aortic_narrowing_roi", seg.shape,
                                             info.get("reason", "no narrowing found"))
        return out

    out["aortic_narrowing_roi"] = _region(
        "aortic_narrowing_roi", roi, spacing, affine, "medium",
        f"narrowest aortic segment (min diameter {info['min_diameter_mm']:.1f} mm via {info['method']})",
        extra=info)

    # minimum-radius point
    point = np.zeros_like(vessel, dtype=np.uint8)
    point[tuple(min_idx)] = 1
    point = ndi.binary_dilation(point, iterations=1) & vessel
    out["minimum_aortic_radius_point"] = _region(
        "minimum_aortic_radius_point", point, spacing, affine, "medium",
        "voxel of minimum aortic radius", extra={"min_diameter_mm": info["min_diameter_mm"]})

    # pre/post segments: remove a small sphere at the min point, take 2 largest CCs
    iters = max(1, int(round(3.0 / max(min(spacing), 1e-6))))
    cut = ndi.binary_dilation(point.astype(bool), iterations=iters)
    split = vessel & ~cut
    lbl, n = ndi.label(split, structure=ndi.generate_binary_structure(split.ndim, 1))
    if n >= 2:
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        order = np.argsort(sizes)[::-1] + 1
        seg_a = lbl == order[0]
        seg_b = lbl == order[1]
        out["pre_coarctation_aorta"] = _region("pre_coarctation_aorta", seg_a, spacing, affine,
                                               "low", "larger aortic segment beyond the narrowing")
        out["post_coarctation_aorta"] = _region("post_coarctation_aorta", seg_b, spacing, affine,
                                                "low", "second aortic segment beyond the narrowing")
    else:
        out["pre_coarctation_aorta"] = _empty("pre_coarctation_aorta", seg.shape,
                                              "aorta not separable at narrowing (stays connected)")
        out["post_coarctation_aorta"] = _empty("post_coarctation_aorta", seg.shape,
                                               "aorta not separable at narrowing (stays connected)")
    return out


# ---------------------------------------------------------------------------
# 6. Aorta/pulmonary confusion interface
# ---------------------------------------------------------------------------
def build_aorta_pulmonary_confusion_interface(seg, label_map: LabelMap, spacing, affine,
                                              disease_flags, params) -> DerivedRegion:
    if not ({"ToF", "Pulmonary_Atresia", "Truncus", "TGA", "DORV"} & set(disease_flags)):
        return _empty("aorta_pulmonary_confusion_interface", seg.shape,
                      "no great-vessel-confusion flag")
    ao_id, pa_id = label_map.id_of("aorta"), label_map.id_of("pulmonary_artery")
    if ao_id is None or pa_id is None:
        return _empty("aorta_pulmonary_confusion_interface", seg.shape, "aorta or PA absent")
    dil = float(params.get("interface_dilation_mm", 2.0))
    band = topo.contact_surface(seg == ao_id, seg == pa_id, spacing, dil)
    if not band.any():
        return _empty("aorta_pulmonary_confusion_interface", seg.shape,
                      "aorta and PA do not approach within dilation")
    return _region("aorta_pulmonary_confusion_interface", band, spacing, affine, "medium",
                   f"aorta/PA near-contact band ({dil} mm) — boundary-confusion auxiliary target")


# ---------------------------------------------------------------------------
# 7. Aorta/pulmonary connection candidate
# ---------------------------------------------------------------------------
def build_aorta_pulmonary_connection_candidate(seg, label_map: LabelMap, spacing, affine,
                                               disease_flags, params) -> DerivedRegion:
    if not ({"Pulmonary_Atresia", "Truncus"} & set(disease_flags)):
        return _empty("aorta_pulmonary_connection_candidate", seg.shape,
                      "no atresia/truncus flag")
    ao_id, pa_id = label_map.id_of("aorta"), label_map.id_of("pulmonary_artery")
    if ao_id is None or pa_id is None:
        return _empty("aorta_pulmonary_connection_candidate", seg.shape, "aorta or PA absent")
    ao, pa = seg == ao_id, seg == pa_id
    if not ao.any() or not pa.any():
        return _empty("aorta_pulmonary_connection_candidate", seg.shape, "aorta or PA empty")

    # very tight band: where the two vessels are essentially touching
    tight = topo.contact_surface(ao, pa, spacing, max(1.0, float(spacing and min(spacing))))
    if not tight.any():
        return _empty("aorta_pulmonary_connection_candidate", seg.shape,
                      "no near-contact between aorta and PA")
    area = _contact_area_mm2(tight, spacing)
    # candidate only; never asserted as true anatomy
    conf = "low" if area < 20.0 else "medium"
    return _region("aorta_pulmonary_connection_candidate", tight, spacing, affine, conf,
                   f"possible aorta/PA connection (touch area {area:.0f} mm^2) — CANDIDATE only, not true anatomy",
                   extra={"contact_area_mm2": area})


# ---------------------------------------------------------------------------
# 8. Hypoplastic structure preservation
# ---------------------------------------------------------------------------
_HYPOPLASTIC_TARGETS = {
    "HLHS": [("lv", "hypoplastic_lv_preservation_region"),
             ("aorta", "hypoplastic_aorta_region")],
    "Pulmonary_Atresia": [("pulmonary_artery", "hypoplastic_pa_preservation_region")],
    "Tricuspid_Atresia": [("rv", "hypoplastic_rv_region")],
}


def build_hypoplastic_structure_preservation_roi(seg, label_map: LabelMap, spacing, affine,
                                                 disease_flags, params) -> Dict[str, DerivedRegion]:
    out: Dict[str, DerivedRegion] = {}
    min_vox = int(params.get("min_component_voxels", 10))
    for disease, targets in _HYPOPLASTIC_TARGETS.items():
        if disease not in set(disease_flags):
            continue
        for structure, region_name in targets:
            sid = label_map.id_of(structure)
            if sid is None:
                out[region_name] = _empty(region_name, seg.shape, f"{structure} absent")
                continue
            mask = seg == sid
            if not mask.any():
                out[region_name] = _empty(region_name, seg.shape,
                                          f"{structure} not present in this case (possible true absence)")
                continue
            n_comp = topo.connected_component_count(mask, min_voxels=min_vox)
            # preserve the whole (small) structure incl. its disconnected specks
            out[region_name] = _region(
                region_name, mask, spacing, affine, "medium",
                f"preserve small/disease-relevant {structure} ({n_comp} component(s)); "
                f"do not drop in postprocessing",
                extra={"structure": structure, "n_components_kept": n_comp})
    return out
