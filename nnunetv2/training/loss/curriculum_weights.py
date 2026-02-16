"""
Curriculum class-weight schedules for CrossEntropy loss.

Provides helpers to:
1. Resolve target class indices (e.g. aorta, pulmonary artery) from
   ``dataset.json`` label mappings — no hardcoded IDs.
2. Compute per-epoch CE weight vectors that upweight target classes
   early in training, then revert to uniform.

Only the CE term is weighted; Dice loss is unaffected. This is intentional:
Dice already balances per-class contributions via its per-class formulation,
while CE benefits from explicit class weighting for rare/important structures.

Schedules
---------
``step``
    weights[target_ids] = multiplier for epoch < T, else 1.0
``linear_decay``
    weights[target_ids] linearly decay from multiplier to 1.0 over [0, T],
    then 1.0 for epoch > T

Where T = ceil(curriculum_fraction * num_epochs).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import torch


# =========================================================================
# Resolve target class IDs from dataset.json
# =========================================================================

# Canonical keyword sets (case-insensitive, stripped)
_AO_KEYWORDS: Set[str] = {"ao", "aorta"}
_PA_KEYWORDS: Set[str] = {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary_artery"}


def resolve_target_class_ids(
    dataset_json: dict,
    target_names: Optional[List[str]] = None,
) -> List[int]:
    """Resolve class indices for target structures from dataset.json labels.

    Parameters
    ----------
    dataset_json : dict
        Must contain a ``"labels"`` key mapping label names to integer IDs.
    target_names : list of str, optional
        If provided, match these names (case-insensitive) instead of the
        default aorta + pulmonary keywords.

    Returns
    -------
    list of int
        Sorted list of matched class indices.

    Raises
    ------
    ValueError
        If no target classes are found.
    """
    labels: Dict[str, int] = dataset_json.get("labels", {})
    if not labels:
        raise ValueError("dataset.json has no 'labels' key or it is empty.")

    if target_names is not None:
        target_set = {n.lower().strip() for n in target_names}
        matched = []
        for name, idx in labels.items():
            if name.lower().strip() in target_set:
                matched.append(int(idx))
        if not matched:
            available = list(labels.keys())
            raise ValueError(
                f"None of the target names {target_names} found in dataset.json labels. "
                f"Available labels: {available}"
            )
        return sorted(set(matched))

    # Default: find aorta and pulmonary artery
    matched: List[int] = []
    for name, idx in labels.items():
        name_lower = name.lower().strip()
        if name_lower == "background":
            continue
        if name_lower in _AO_KEYWORDS:
            matched.append(int(idx))
        elif name_lower in _PA_KEYWORDS:
            matched.append(int(idx))

    if not matched:
        available = list(labels.keys())
        raise ValueError(
            f"Could not find aorta or pulmonary artery labels in dataset.json. "
            f"Available labels: {available}"
        )

    return sorted(set(matched))


# =========================================================================
# Compute curriculum CE weight vector
# =========================================================================

def get_curriculum_ce_weights(
    epoch: int,
    num_epochs: int,
    num_classes: int,
    target_ids: List[int],
    multiplier: float = 3.0,
    fraction: float = 0.2,
    schedule_type: str = "step",
) -> torch.Tensor:
    """Compute a CE class-weight vector for the given epoch.

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    num_epochs : int
        Total number of training epochs.
    num_classes : int
        Number of classes (including background).
    target_ids : list of int
        Class indices to upweight.
    multiplier : float
        Weight multiplier for target classes during the curriculum phase.
    fraction : float
        Fraction of total epochs during which curriculum weighting is active.
    schedule_type : str
        ``"step"`` or ``"linear_decay"``.

    Returns
    -------
    torch.Tensor
        Shape ``(num_classes,)`` float32 weight vector.
    """
    T = math.ceil(fraction * num_epochs)
    weights = torch.ones(num_classes, dtype=torch.float32)

    if epoch >= T or T == 0:
        return weights

    if schedule_type == "step":
        current_multiplier = multiplier
    elif schedule_type == "linear_decay":
        # linear decay from multiplier to 1.0 over epochs [0, T)
        current_multiplier = multiplier - (multiplier - 1.0) * (epoch / max(T - 1, 1))
    else:
        raise ValueError(f"Unknown schedule_type '{schedule_type}'. Use 'step' or 'linear_decay'.")

    for cid in target_ids:
        if cid < num_classes:
            weights[cid] = current_multiplier

    return weights
