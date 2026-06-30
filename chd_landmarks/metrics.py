"""
chd_landmarks.metrics
=====================

Disease-aware evaluation metrics beyond Dice. All functions operate on integer
label maps (pred, gt) plus a resolved LabelMap and spacing. Disease-conditioned
metrics only run when the case carries the flag and the required labels /
derived GT exist; otherwise they return None.

Pure numpy/scipy/skimage; optional topology libs degrade to None.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import geometry as geo
from . import topology as topo
from .labels import LabelMap


# ---------------------------------------------------------------------------
# Generic per-label metrics
# ---------------------------------------------------------------------------
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / denom)


def volume_difference_mm3(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    return float(geo.volume_mm3(pred, spacing) - geo.volume_mm3(gt, spacing))


def connected_component_difference(pred: np.ndarray, gt: np.ndarray, min_voxels: int = 10) -> int:
    return (topo.connected_component_count(pred, min_voxels)
            - topo.connected_component_count(gt, min_voxels))


# ---------------------------------------------------------------------------
# clDice (vessel-like labels)
# ---------------------------------------------------------------------------
def cldice(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """Centerline Dice. Uses skeletons; returns None if skeletonization unavailable."""
    sp = topo.skeletonize_3d_safe(pred)
    sg = topo.skeletonize_3d_safe(gt)
    if sp is None or sg is None:
        return None
    p, g = pred.astype(bool), gt.astype(bool)
    if not sp.any() or not sg.any():
        return 0.0 if (p.any() or g.any()) else 1.0
    tprec = (sp & g).sum() / max(sp.sum(), 1)   # pred skeleton inside gt
    tsens = (sg & p).sum() / max(sg.sum(), 1)   # gt skeleton inside pred
    if (tprec + tsens) == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def centerline_recall(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """Fraction of the GT centerline covered by the predicted mask."""
    sg = topo.skeletonize_3d_safe(gt)
    if sg is None:
        return None
    if not sg.any():
        return None
    return float((sg & pred.astype(bool)).sum() / sg.sum())


# ---------------------------------------------------------------------------
# Vessel / great-vessel metrics
# ---------------------------------------------------------------------------
def vessel_min_diameter_mm(mask: np.ndarray, spacing) -> float:
    return geo.approximate_minimum_diameter(mask, spacing)


def aorta_pulmonary_leakage_volume_mm3(pred: np.ndarray, gt: np.ndarray,
                                       ao_id: int, pa_id: int, spacing) -> float:
    """Volume where pred says aorta but gt says PA (or vice versa) — identity leakage."""
    leak = ((pred == ao_id) & (gt == pa_id)) | ((pred == pa_id) & (gt == ao_id))
    return geo.volume_mm3(leak, spacing)


def aorta_pulmonary_confusion_volume_mm3(pred: np.ndarray, gt: np.ndarray,
                                         ao_id: int, pa_id: int, spacing,
                                         band_dilation_mm: float = 2.0) -> float:
    """Volume of pred AO∪PA voxels mislabeled within the GT AO/PA contact band."""
    band = topo.contact_surface(gt == ao_id, gt == pa_id, spacing, band_dilation_mm)
    if not band.any():
        return 0.0
    wrong = band & (((pred == ao_id) & (gt == pa_id)) | ((pred == pa_id) & (gt == ao_id)))
    return geo.volume_mm3(wrong, spacing)


# ---------------------------------------------------------------------------
# Septal-defect (VSD / ASD) metrics vs derived GT proxy
# ---------------------------------------------------------------------------
def detection_tp_fp_fn(pred_mask: np.ndarray, gt_mask: np.ndarray,
                       min_overlap_voxels: int = 1) -> Dict[str, int]:
    """Simple presence-based detection: does pred overlap the derived GT region?"""
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    if g.any():
        overlap = int((p & g).sum())
        tp = 1 if overlap >= min_overlap_voxels else 0
        fn = 1 - tp
        fp = 0
    else:
        tp = 0
        fn = 0
        fp = 1 if p.any() else 0
    return {"tp": tp, "fp": fp, "fn": fn}


def vsd_centroid_error_mm(pred_mask: np.ndarray, gt_mask: np.ndarray, affine) -> Optional[float]:
    return geo.centroid_distance_mm(pred_mask, gt_mask, affine)


def vsd_equivalent_diameter_error_mm(pred_mask, gt_mask, spacing) -> Optional[float]:
    if not gt_mask.any():
        return None
    return float(geo.equivalent_diameter_from_volume(pred_mask, spacing)
                 - geo.equivalent_diameter_from_volume(gt_mask, spacing))


def lv_rv_false_merge_indicator(pred: np.ndarray, lv_id: int, rv_id: int, spacing,
                                threshold_mm2: float = 200.0) -> Dict[str, float]:
    band = topo.contact_surface(pred == lv_id, pred == rv_id, spacing, 1.0)
    s = sorted(float(x) for x in spacing)
    face = s[0] * s[1] if len(s) >= 2 else s[0] ** 2
    area = float(band.sum()) * face
    return {"contact_area_mm2": area, "false_merge_suspected": bool(area > threshold_mm2)}


# ---------------------------------------------------------------------------
# Volume preservation (hypoplastic structures)
# ---------------------------------------------------------------------------
def volume_preservation(pred: np.ndarray, gt: np.ndarray, lid: int, spacing) -> Dict[str, float]:
    pv = geo.volume_mm3(pred == lid, spacing)
    gv = geo.volume_mm3(gt == lid, spacing)
    return {
        "pred_volume_mm3": pv,
        "gt_volume_mm3": gv,
        "preserved_fraction": float(pv / gv) if gv > 0 else None,
        "present_in_pred": bool((pred == lid).any()),
        "present_in_gt": bool((gt == lid).any()),
    }


# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------
def topology_metrics(pred: np.ndarray, gt: np.ndarray, lid: int, min_voxels: int = 10) -> Dict:
    p, g = pred == lid, gt == lid
    bp = topo.betti_numbers_binary(p)
    bg = topo.betti_numbers_binary(g)
    ep = topo.euler_characteristic_binary(p)
    eg = topo.euler_characteristic_binary(g)
    return {
        "betti0_difference": (topo.betti0(p, min_voxels) - topo.betti0(g, min_voxels)),
        "betti1_difference": (None if bp is None or bg is None else bp[1] - bg[1]),
        "euler_characteristic_difference": (None if ep is None or eg is None else ep - eg),
        "skeleton_branch_count_pred": topo.skeleton_branch_count(p),
        "skeleton_branch_count_gt": topo.skeleton_branch_count(g),
        "largest_cc_fraction_pred": topo.largest_cc_fraction(p),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    label_map: LabelMap,
    spacing,
    affine=None,
    active_diseases: Optional[List[str]] = None,
    derived_gt: Optional[Dict[str, np.ndarray]] = None,
    metric_cfg: Optional[dict] = None,
) -> dict:
    """
    Compute general + disease-conditioned + topology metrics for one case.
    `derived_gt` maps derived-region name -> GT mask (from the dataset builder's
    derived_masksTr) for septal-defect / stenosis comparisons.
    """
    active_diseases = active_diseases or []
    derived_gt = derived_gt or {}
    metric_cfg = metric_cfg or {}
    params = metric_cfg.get("params", {})
    percentile = int(params.get("hd95_percentile", 95))
    min_vox = int(params.get("min_component_voxels", 10))

    result: dict = {"general": {}, "disease": {}, "topology": {}}

    # ---- general per-label -------------------------------------------
    for structure, lid in label_map.structure_to_id.items():
        if lid is None:
            continue
        p, g = pred == lid, gt == lid
        if not p.any() and not g.any():
            continue
        sd = geo.surface_distance_metrics(p, g, spacing, percentile)
        result["general"][structure] = {
            "dice": dice(p, g),
            "hd95": sd["hd95"],
            "assd": sd["assd"],
            "centroid_distance_mm": geo.centroid_distance_mm(p, g, affine) if affine is not None else None,
            "volume_difference_mm3": volume_difference_mm3(p, g, spacing),
            "connected_component_difference": connected_component_difference(p, g, min_vox),
        }

    # ---- topology for vessel labels ----------------------------------
    for structure in (params.get("vessel_labels", ["aorta", "pulmonary_artery"])):
        lid = label_map.id_of(structure)
        if lid is None:
            continue
        result["topology"][structure] = topology_metrics(pred, gt, lid, min_vox)
        result["topology"][structure]["cldice"] = cldice(pred == lid, gt == lid)
        result["topology"][structure]["centerline_recall"] = centerline_recall(pred == lid, gt == lid)

    # ---- disease-conditioned -----------------------------------------
    ao, pa = label_map.id_of("aorta"), label_map.id_of("pulmonary_artery")
    lv, rv = label_map.id_of("lv"), label_map.id_of("rv")

    def add(disease, key, value):
        result["disease"].setdefault(disease, {})[key] = value

    la, ra = label_map.id_of("la"), label_map.id_of("ra")
    _SEPTAL = ("VSD", "ASD", "AVSD", "ToF", "DORV")

    for disease in active_diseases:
        if disease in _SEPTAL and "septal_defect_proxy" in derived_gt:
            gtm = derived_gt["septal_defect_proxy"].astype(bool)
            # predicted proxy: the derived hard-label id from the trained
            # dataset.json (raw_labels), if the model was trained with it.
            pvid = label_map.raw_labels.get("septal_defect_proxy")
            predm = (pred == pvid) if pvid is not None else np.zeros_like(gtm)
            add(disease, "septal_defect_detection", detection_tp_fp_fn(predm, gtm))
            add(disease, "septal_defect_centroid_error_mm", vsd_centroid_error_mm(predm, gtm, affine))
            add(disease, "septal_defect_equivalent_diameter_error_mm",
                vsd_equivalent_diameter_error_mm(predm, gtm, spacing))
            if lv is not None and rv is not None:
                add(disease, "lv_rv_false_merge", lv_rv_false_merge_indicator(pred, lv, rv, spacing))

            # peri-septal local Dice: correctness of the blood pool AROUND the
            # defect (and the defect itself) restricted to the GT-defined shell.
            # This is the CHD-relevant headline — it ignores the volume-dominant
            # rest of the chambers that saturates global Dice.
            if "peri_septal_defect_roi" in derived_gt:
                roi = derived_gt["peri_septal_defect_roi"].astype(bool)
                locals_ = {}
                for s, lid2 in (("lv", lv), ("rv", rv), ("la", la), ("ra", ra)):
                    if lid2 is not None and (((gt == lid2) & roi).any() or ((pred == lid2) & roi).any()):
                        locals_[s] = dice((pred == lid2) & roi, (gt == lid2) & roi)
                if pvid is not None:
                    locals_["septal_defect"] = dice((pred == pvid) & roi, (gt == pvid) & roi)
                if locals_:
                    add(disease, "peri_septal_local_dice", locals_)
                    add(disease, "peri_septal_headline",
                        float(np.mean(list(locals_.values()))))

        if disease in ("ToF", "Pulmonary_Stenosis") and pa is not None:
            add(disease, "pulmonary_min_diameter_pred_mm", vessel_min_diameter_mm(pred == pa, spacing))
            add(disease, "pulmonary_min_diameter_gt_mm", vessel_min_diameter_mm(gt == pa, spacing))
            add(disease, "pa_component_count_pred", topo.connected_component_count(pred == pa, min_vox))

        if disease in ("Pulmonary_Atresia",) and pa is not None:
            add(disease, "pulmonary_label_presence_pred", bool((pred == pa).any()))
            add(disease, "pulmonary_label_presence_gt", bool((gt == pa).any()))
            add(disease, "pulmonary_volume_error_mm3", volume_difference_mm3(pred == pa, gt == pa, spacing))

        if disease in ("ToF", "TGA", "DORV", "Truncus", "Pulmonary_Atresia") and ao is not None and pa is not None:
            add(disease, "aorta_pulmonary_leakage_volume_mm3",
                aorta_pulmonary_leakage_volume_mm3(pred, gt, ao, pa, spacing))
            add(disease, "aorta_pulmonary_confusion_volume_mm3",
                aorta_pulmonary_confusion_volume_mm3(pred, gt, ao, pa, spacing,
                                                     float(params.get("confusion_band_dilation_mm", 2.0))))

        if disease in ("Coarctation",) and ao is not None:
            add(disease, "aortic_min_diameter_pred_mm", vessel_min_diameter_mm(pred == ao, spacing))
            add(disease, "aortic_min_diameter_gt_mm", vessel_min_diameter_mm(gt == ao, spacing))
            add(disease, "aortic_centerline_continuity",
                topo.largest_cc_fraction(pred == ao))

        if disease in ("HLHS",):
            if lv is not None:
                add(disease, "lv_volume_preservation", volume_preservation(pred, gt, lv, spacing))
            if ao is not None:
                add(disease, "aorta_volume_preservation", volume_preservation(pred, gt, ao, spacing))

    return result
