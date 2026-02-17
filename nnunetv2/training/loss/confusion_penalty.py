"""
Confusion-pair penalty loss for targeted inter-class disambiguation.

Adds extra loss when the model predicts class B at voxels where the ground
truth is class A (and vice versa), for specified confusion pairs.  Standard
CE treats all misclassifications equally; this loss adds a targeted penalty
for the specific failure modes that matter most (e.g. AO labelled as PA).

The penalty is the mean predicted probability of the wrong class at the
relevant ground-truth voxels:

    penalty(c1, c2) = mean(P(c2) | GT=c1) + mean(P(c1) | GT=c2)

This is bounded in [0, 1] per pair and is zero when the model never
confuses the specified classes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# =========================================================================
# AO / PA keyword sets (shared with curriculum_weights.py)
# =========================================================================

_AO_KEYWORDS: Set[str] = {"ao", "aorta"}
_PA_KEYWORDS: Set[str] = {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary_artery"}


def resolve_confusion_pairs(dataset_json: dict) -> List[Tuple[int, int]]:
    """Auto-detect AO-PA confusion pair from dataset.json labels.

    Returns a list of (class_idx_1, class_idx_2) tuples.  Currently only
    detects the AO-PA pair; returns an empty list if either is missing.
    """
    labels: Dict[str, int] = dataset_json.get("labels", {})
    ao_idx: Optional[int] = None
    pa_idx: Optional[int] = None

    for name, idx in labels.items():
        name_lower = name.lower().strip()
        if name_lower in _AO_KEYWORDS:
            ao_idx = int(idx)
        elif name_lower in _PA_KEYWORDS:
            pa_idx = int(idx)

    if ao_idx is not None and pa_idx is not None:
        return [(ao_idx, pa_idx)]
    return []


def confusion_penalty_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    confusion_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    """Compute confusion penalty for specified class pairs.

    Parameters
    ----------
    pred_logits : (B, C, *spatial)
        Raw logits from the network.
    target : (B, 1, *spatial) or (B, *spatial)
        Integer ground-truth labels.
    confusion_pairs : list of (int, int)
        Pairs of class indices to penalise mutual confusion.

    Returns
    -------
    torch.Tensor
        Scalar loss (0-dim tensor on same device as pred_logits).
    """
    if not confusion_pairs:
        return torch.tensor(0.0, device=pred_logits.device)

    probs = F.softmax(pred_logits, dim=1)  # (B, C, *spatial)

    # Flatten target to (B, *spatial) for indexing
    tgt = target.squeeze(1).long() if target.ndim == pred_logits.ndim else target.long()

    loss = torch.tensor(0.0, device=pred_logits.device)
    count = 0

    for c1, c2 in confusion_pairs:
        mask_c1 = (tgt == c1)  # (B, *spatial)
        mask_c2 = (tgt == c2)

        # Penalise predicting c2 where GT is c1
        if mask_c1.any():
            loss = loss + probs[:, c2][mask_c1].mean()
            count += 1
        # Penalise predicting c1 where GT is c2
        if mask_c2.any():
            loss = loss + probs[:, c1][mask_c2].mean()
            count += 1

    if count > 0:
        loss = loss / count
    return loss
