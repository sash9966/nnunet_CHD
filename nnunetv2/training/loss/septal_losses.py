"""
Septal-defect focus losses (trainer-free, unit-testable).

The septal_defect class (id resolved by name) is tiny relative to the whole
heart, so Dice+CE barely see it. `soft_tversky_binary` is a false-negative-
weighted region loss (Tversky, beta>alpha) computed on the softmax probability
of the septal class — scale-invariant and recall-biased, ideal for a small
critical structure. Pure torch; no nnU-Net trainer import.
"""
from __future__ import annotations

from typing import Optional

import torch


def resolve_septal_label_id(dataset_json: dict) -> Optional[int]:
    """Find the integer id of the septal-defect class by NAME (never hardcoded)."""
    for name, val in (dataset_json.get("labels", {}) or {}).items():
        if "septal" in str(name).lower():
            try:
                return int(val)
            except (TypeError, ValueError):
                # region-based (list) — not expected for 051; skip
                return None
    return None


def soft_tversky_binary(prob: torch.Tensor, gt: torch.Tensor,
                        alpha: float = 0.3, beta: float = 0.7,
                        smooth: float = 1e-5) -> torch.Tensor:
    """1 - Tversky on a single binary class.

    prob : (B,1,*spatial) predicted probability of the class in [0,1].
    gt   : (B,1,*spatial) binary ground truth (0/1).
    beta > alpha penalises false negatives more (find the small defect).
    """
    prob = prob.float()
    gt = gt.float()
    dims = tuple(range(1, prob.ndim))  # sum over channel+spatial, keep batch
    tp = (prob * gt).sum(dims)
    fp = (prob * (1.0 - gt)).sum(dims)
    fn = ((1.0 - prob) * gt).sum(dims)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()
