"""
Disease-adaptive target mixin for composable trainers.

During training, modifies the ground-truth label map per sample based on
the active disease flags.  When a disease makes a specific boundary
genuinely ambiguous (e.g. VSD blurs the LV/RV septum, pulmonary atresia
fuses AO/PA), the affected label pair is **merged** in the training target
so the model is not penalised for the ambiguous boundary.

At validation/inference time, targets are NOT modified — evaluation uses
the original full label set to measure true per-class Dice.

This mixin loads ``disease_map.json`` independently, so it can be used
with or without network-level disease conditioning (FiLM, gated, etc.).

Default merge rules (configurable):
    VSD  → merge RV into LV   (ventricle septum is defective)
    PuA  → merge PA into AO   (pulmonary artery is atretic / fused)
"""
from __future__ import annotations

import json
from os.path import isfile, join
from typing import Dict, List, Optional, Set, Tuple

import torch

from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


# =========================================================================
# Label keyword matching
# =========================================================================

_LABEL_KEYWORDS: Dict[str, Set[str]] = {
    "LV": {"lv", "left ventricle", "leftventricle", "left_ventricle"},
    "RV": {"rv", "right ventricle", "rightventricle", "right_ventricle"},
    "LA": {"la", "left atrium", "leftatrium", "left_atrium"},
    "RA": {"ra", "right atrium", "rightatrium", "right_atrium"},
    "AO": {"ao", "aorta"},
    "PA": {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary_artery"},
    "MYO": {"myo", "myocardium"},
}

# Standard disease flag ordering.  Slots 0-7 match the original K=8
# convention; slots 8-31 are reserved for future diseases.  The model
# uses K=32 so new diseases can be added without retraining.
DEFAULT_DISEASE_FLAGS = [
    "HLHS", "ASD", "VSD", "AVSD", "DORV", "PuA", "ToF", "TGA",  # 0-7
    # 8-31: reserved for future diseases
]

# Default merge rules: disease_name → list of (from_label_name, to_label_name)
# "from" label gets replaced with "to" label in the training target.
DEFAULT_MERGE_RULES: Dict[str, List[Tuple[str, str]]] = {
    "VSD": [("RV", "LV")],   # merge RV → LV (septum is defective)
    "PuA": [("PA", "AO")],   # merge PA → AO (PA is atretic / fused)
}


def _resolve_label_idx(dataset_json: dict, label_name: str) -> Optional[int]:
    """Find the integer label index for a structure name via keyword matching."""
    keywords = _LABEL_KEYWORDS.get(label_name)
    if keywords is None:
        return None
    labels = dataset_json.get("labels", {})
    for name, idx in labels.items():
        if name.lower().strip() in keywords:
            return int(idx)
    return None


def resolve_disease_label_merges(
    dataset_json: dict,
    disease_flags: Optional[List[str]] = None,
    merge_rules: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> Dict[int, List[Tuple[int, int]]]:
    """Resolve merge rules to integer indices.

    Returns
    -------
    dict
        Maps disease flag index → list of (from_label_idx, to_label_idx).
        Only includes rules where both labels were found in dataset.json.
    """
    if disease_flags is None:
        disease_flags = DEFAULT_DISEASE_FLAGS
    if merge_rules is None:
        merge_rules = DEFAULT_MERGE_RULES

    result: Dict[int, List[Tuple[int, int]]] = {}

    for disease_name, pairs in merge_rules.items():
        if disease_name not in disease_flags:
            continue
        flag_idx = disease_flags.index(disease_name)
        resolved_pairs = []
        for from_name, to_name in pairs:
            from_idx = _resolve_label_idx(dataset_json, from_name)
            to_idx = _resolve_label_idx(dataset_json, to_name)
            if from_idx is not None and to_idx is not None:
                resolved_pairs.append((from_idx, to_idx))
        if resolved_pairs:
            result[flag_idx] = resolved_pairs

    return result


# =========================================================================
# DiseaseAdaptiveTargetMixin
# =========================================================================

class DiseaseAdaptiveTargetMixin(TrainerMixin):
    """Mixin that merges label pairs in training targets based on disease flags.

    Configurable via class attributes or ``mixin_init`` overrides:

    ==============================  ==================  =========================================
    Attribute                       Default             Description
    ==============================  ==================  =========================================
    adaptive_target_enabled         True                Master switch
    adaptive_target_merge_rules     DEFAULT_MERGE_RULES Disease → label merge rules (name-based)
    adaptive_target_disease_flags   DEFAULT_FLAGS       Ordered disease flag names
    ==============================  ==================  =========================================

    Requires ``disease_map.json`` in the preprocessed dataset folder.
    Can be used with or without ``DiseaseConditioningMixin`` — loads the
    disease map independently.
    """

    def mixin_init(self):
        super().mixin_init()
        self.adaptive_target_enabled: bool = True
        self.adaptive_target_merge_rules = DEFAULT_MERGE_RULES
        self.adaptive_target_disease_flags = DEFAULT_DISEASE_FLAGS
        # Resolved at initialize time:
        self._adaptive_disease_map: Optional[Dict[str, List[int]]] = None
        self._adaptive_label_merges: Dict[int, List[Tuple[int, int]]] = {}
        self._adaptive_merge_count: int = 0  # for logging

    def mixin_initialize(self):
        super().mixin_initialize()
        if not self.adaptive_target_enabled:
            return

        # Load disease map (reuse existing if DiseaseConditioningMixin loaded it)
        if hasattr(self, 'disease_map') and self.disease_map is not None:
            self._adaptive_disease_map = self.disease_map
        else:
            path = join(self.preprocessed_dataset_folder_base, "disease_map.json")
            if isfile(path):
                with open(path) as f:
                    self._adaptive_disease_map = json.load(f)
            else:
                self.print_to_log_file(
                    "[AdaptiveTarget] disease_map.json not found. "
                    "Adaptive targets DISABLED."
                )
                self.adaptive_target_enabled = False
                return

        # Resolve merge rules to integer indices
        self._adaptive_label_merges = resolve_disease_label_merges(
            self.dataset_json,
            self.adaptive_target_disease_flags,
            self.adaptive_target_merge_rules,
        )

        if self._adaptive_label_merges:
            parts = []
            for flag_idx, pairs in self._adaptive_label_merges.items():
                flag_name = self.adaptive_target_disease_flags[flag_idx]
                pair_strs = [f"label {f}→{t}" for f, t in pairs]
                parts.append(f"{flag_name}(flag {flag_idx}): {', '.join(pair_strs)}")
            self.print_to_log_file(
                f"[AdaptiveTarget] Active merge rules: {'; '.join(parts)}"
            )
        else:
            self.print_to_log_file(
                "[AdaptiveTarget] No merge rules resolved from dataset.json labels. "
                "Adaptive targets DISABLED."
            )
            self.adaptive_target_enabled = False

    # ------------------------------------------------------------------
    # Hook: modify target (training only)
    # ------------------------------------------------------------------
    def mixin_modify_target(self, target, batch):
        target = super().mixin_modify_target(target, batch)

        if not self.adaptive_target_enabled or not self._adaptive_label_merges:
            return target
        if self._adaptive_disease_map is None:
            return target

        # Look up disease vectors for this batch
        batch_keys = batch.get("keys")
        if batch_keys is None:
            return target

        disease_vecs = []
        for case_id in batch_keys:
            if case_id in self._adaptive_disease_map:
                disease_vecs.append(self._adaptive_disease_map[case_id])
            else:
                disease_vecs.append(None)

        # Check if any sample in the batch needs merging
        needs_merge = False
        for dv in disease_vecs:
            if dv is not None:
                for flag_idx in self._adaptive_label_merges:
                    if dv[flag_idx] == 1:
                        needs_merge = True
                        break
            if needs_merge:
                break

        if not needs_merge:
            return target

        # Clone target to avoid modifying the original batch data
        if isinstance(target, list):
            # Deep supervision: list of tensors at different resolutions
            target = [t.clone() for t in target]
        else:
            target = target.clone()

        # Apply merges per sample
        merged_count = 0
        for b, dv in enumerate(disease_vecs):
            if dv is None:
                continue
            for flag_idx, pairs in self._adaptive_label_merges.items():
                if dv[flag_idx] != 1:
                    continue
                for from_label, to_label in pairs:
                    if isinstance(target, list):
                        for t in target:
                            mask = t[b] == from_label
                            if mask.any():
                                t[b][mask] = to_label
                                merged_count += 1
                    else:
                        mask = target[b] == from_label
                        if mask.any():
                            target[b][mask] = to_label
                            merged_count += 1

        self._adaptive_merge_count += merged_count
        return target

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self.adaptive_target_enabled and self.current_epoch % 10 == 0:
            self.print_to_log_file(
                f"[AdaptiveTarget] Merges applied this epoch window: "
                f"{self._adaptive_merge_count}"
            )
            self._adaptive_merge_count = 0
