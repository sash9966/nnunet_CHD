"""
chd_landmarks.derived_label_builder
====================================

Orchestrates per-case derivation and applies the CONSERVATIVE merge policy
declared in chd_derived_labels.yaml.

Outputs per case (see :class:`CaseDerivation`):
  * original anatomy segmentation (unchanged),
  * a hard derived-label map (derived integer labels only, for audit),
  * a merged nnU-Net label map (anatomy + permitted hard labels),
  * auxiliary masks (every derived region, for metrics/topology/audit),
  * region-based targets (name -> id list),
  * per-region confidence metadata + annotation status,
  * warnings.

Merge rules (enforced here):
  * original anatomy labels are NEVER overwritten unless a hard label's
    overwrite_policy explicitly allows it AND its confidence >= required;
  * `do_not_overwrite_anatomy_by_default` labels (e.g. stenosis ROI) stay
    auxiliary — they get NO integer id in the default merged scheme;
  * interface-only policies paint only the named chamber interface voxels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import derived_regions as dr
from . import io
from .disease_rules import RuleSet
from .labels import LabelMap
from .metadata import AnnotationStatus, CaseMetadata

_CONF_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class CaseDerivation:
    case_id: str
    anatomy_seg: np.ndarray
    hard_label_map: np.ndarray
    merged_label_map: np.ndarray
    auxiliary_masks: Dict[str, np.ndarray]
    region_based_targets: Dict[str, List[int]]
    region_meta: Dict[str, dict]
    annotation_status: AnnotationStatus
    warnings: List[str] = field(default_factory=list)


class DerivedLabelBuilder:
    def __init__(self, label_map: LabelMap, ruleset: RuleSet, derived_cfg: dict):
        self.label_map = label_map
        self.ruleset = ruleset
        self.cfg = derived_cfg
        self.params = derived_cfg.get("construction_params", {})
        self.hard_cfg: Dict[str, dict] = derived_cfg.get("hard_integer_labels", {}) or {}
        self.aux_names: List[str] = list(derived_cfg.get("auxiliary_masks", []) or [])

        # --- deterministic id assignment for MERGEABLE hard labels ----------
        # (those whose policy is NOT do_not_overwrite_anatomy_by_default)
        self.merge_scheme: Dict[str, int] = {}
        next_id = label_map.next_free_id()
        for name, spec in self.hard_cfg.items():
            if spec.get("overwrite_policy") == "do_not_overwrite_anatomy_by_default":
                continue
            self.merge_scheme[name] = next_id
            next_id += 1

        # all hard labels (incl. auxiliary-only) get an id for the audit map
        self.hard_scheme: Dict[str, int] = {}
        aid = label_map.next_free_id()
        for name in self.hard_cfg:
            self.hard_scheme[name] = aid
            aid += 1

    # ------------------------------------------------------------------
    def merged_dataset_labels(self) -> Dict[str, int]:
        """labels dict for the merged Dataset031 dataset.json."""
        labels: Dict[str, int] = {"background": self.label_map.background_id}
        for structure, lid in self.label_map.structure_to_id.items():
            if lid is not None:
                # use the original dataset name where possible
                labels[structure.upper() if len(structure) <= 3 else structure] = lid
        for name, lid in self.merge_scheme.items():
            labels[name] = lid
        return labels

    # ------------------------------------------------------------------
    def _run_builders(self, seg, spacing, affine, diseases) -> Dict[str, dr.DerivedRegion]:
        lm, p = self.label_map, self.params
        regions: Dict[str, dr.DerivedRegion] = {}

        def add(r):
            if isinstance(r, dict):
                regions.update(r)
            else:
                regions[r.name] = r

        add(dr.build_vsd_orifice_proxy(seg, lm, spacing, affine, diseases, p))
        add(dr.build_asd_orifice_proxy(seg, lm, spacing, affine, diseases, p))
        add(dr.build_lv_rv_false_merge_region(seg, lm, spacing, affine, p))
        add(dr.build_pulmonary_stenosis_roi(seg, lm, spacing, affine, diseases, p))
        add(dr.build_aortic_coarctation_roi(seg, lm, spacing, affine, diseases, p))
        add(dr.build_aorta_pulmonary_confusion_interface(seg, lm, spacing, affine, diseases, p))
        add(dr.build_aorta_pulmonary_connection_candidate(seg, lm, spacing, affine, diseases, p))
        add(dr.build_hypoplastic_structure_preservation_roi(seg, lm, spacing, affine, diseases, p))

        # unified septal-defect hard label (v2): all abnormal chamber connections
        # (VSD LV-RV + ASD LA-RA + AVSD cross LV-RA/RV-LA), flag-gated, unioned.
        septal = dr.build_septal_defect(seg, lm, spacing, affine, diseases, p)
        regions["septal_defect_proxy"] = septal

        # peri-septal blood-pool shell (EVALUATION ROI, never merged)
        if septal.present:
            chambers = [lm.id_of(c) for c in ("lv", "rv", "la", "ra") if lm.has(c)]
            regions["peri_septal_defect_roi"] = dr.build_peri_defect_roi(
                septal.mask, seg, chambers, spacing, affine,
                float(p.get("peri_defect_shell_mm", 5.0)), "peri_septal_defect_roi")

        # near-contact auxiliary bands (flag-gated)
        dil = float(p.get("interface_dilation_mm", 2.0))
        if {"VSD", "AVSD", "DORV", "ToF"} & set(diseases) and lm.require(["lv", "rv"]):
            regions["lv_rv_near_contact"] = dr.build_interface_roi(
                seg, lm.id_of("lv"), lm.id_of("rv"), spacing, affine, dil, "lv_rv_near_contact")
        if {"ASD", "AVSD"} & set(diseases) and lm.require(["la", "ra"]):
            regions["la_ra_near_contact"] = dr.build_interface_roi(
                seg, lm.id_of("la"), lm.id_of("ra"), spacing, affine, dil, "la_ra_near_contact")
        return regions

    # ------------------------------------------------------------------
    def build_for_case(self, anatomy_seg: np.ndarray, metadata: CaseMetadata,
                       case_id: str, affine=None, spacing=None) -> CaseDerivation:
        seg = anatomy_seg.astype(np.int32)
        spacing = tuple(spacing) if spacing is not None else (1.0,) * seg.ndim
        warnings: List[str] = []

        active_rules, skipped = self.ruleset.for_case(metadata, self.label_map)
        diseases = {r.name for r in active_rules}
        for disease, reason in skipped:
            warnings.append(f"{disease}: flagged but skipped — {reason}")

        regions = self._run_builders(seg, spacing, affine, diseases)

        # ---- auxiliary masks: every PRESENT region (audit-friendly) --------
        auxiliary: Dict[str, np.ndarray] = {}
        region_meta: Dict[str, dict] = {}
        for name, reg in regions.items():
            region_meta[name] = reg.meta
            if reg.present:
                auxiliary[name] = reg.mask

        # ---- merged label map: paint permitted hard labels -----------------
        merged = seg.copy()
        hard_map = np.zeros_like(seg)
        annot = AnnotationStatus(case_id=case_id)

        for name, lid in self.merge_scheme.items():
            spec = self.hard_cfg.get(name, {})
            reg = regions.get(name)
            required = spec.get("confidence_required", "high")
            if reg is None or not reg.present:
                continue
            # always record in the hard audit map
            hard_map[reg.mask.astype(bool)] = lid
            if _CONF_RANK.get(reg.confidence, 0) < _CONF_RANK.get(required, 3):
                warnings.append(
                    f"{name}: confidence '{reg.confidence}' < required '{required}' "
                    f"-> kept auxiliary, NOT merged")
                annot.unknown.append(name)
                continue
            paint = self._apply_overwrite_policy(name, spec, reg.mask, seg)
            painted = int(paint.sum())
            merged[paint] = lid
            annot.derived[name] = reg.confidence
            region_meta[name]["merged_voxels"] = painted

        # auxiliary-only hard labels (e.g. stenosis): never merged
        for name in self.hard_cfg:
            if name in self.merge_scheme:
                continue
            reg = regions.get(name)
            if reg is not None and reg.present:
                hard_map[reg.mask.astype(bool)] = self.hard_scheme[name]
                annot.derived[name] = f"{reg.confidence} (auxiliary-only)"

        # diseases flagged but not derivable -> absence is UNKNOWN
        for disease, reason in skipped:
            annot.unknown.append(disease)
        annot.warnings = warnings

        region_targets = self._resolve_region_targets()

        return CaseDerivation(
            case_id=case_id,
            anatomy_seg=seg,
            hard_label_map=hard_map,
            merged_label_map=merged,
            auxiliary_masks=auxiliary,
            region_based_targets=region_targets,
            region_meta=region_meta,
            annotation_status=annot,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def _apply_overwrite_policy(self, name, spec, mask, seg) -> np.ndarray:
        policy = spec.get("overwrite_policy", "do_not_overwrite_anatomy_by_default")
        m = mask.astype(bool)
        if policy == "may_overwrite_lv_rv_interface_only":
            allowed = np.isin(seg, [self.label_map.id_of("lv"), self.label_map.id_of("rv")])
            return m & allowed
        if policy == "may_overwrite_la_ra_interface_only":
            allowed = np.isin(seg, [self.label_map.id_of("la"), self.label_map.id_of("ra")])
            return m & allowed
        if policy == "may_overwrite_septal_interface_only":
            ids = [self.label_map.id_of(c) for c in ("lv", "rv", "la", "ra")
                   if self.label_map.has(c)]
            return m & np.isin(seg, ids)
        if policy == "do_not_overwrite_anatomy_by_default":
            # only paint background voxels (never clobber anatomy)
            return m & (seg == self.label_map.background_id)
        return m & (seg == self.label_map.background_id)

    # ------------------------------------------------------------------
    def _resolve_region_targets(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for region, spec in (self.cfg.get("region_based_targets") or {}).items():
            ids: List[int] = []
            for member in spec.get("labels", []):
                if member in self.label_map.structure_to_id:
                    i = self.label_map.id_of(member)
                    if i is not None:
                        ids.append(i)
                elif member in self.merge_scheme:
                    ids.append(self.merge_scheme[member])
            if ids:
                out[region] = sorted(set(ids))
        return out
