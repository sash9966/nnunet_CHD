"""
Confusion-pair penalty mixin for composable trainers.

Adds an extra loss term that penalises inter-class confusion between
specified class pairs (auto-detected from dataset.json, default: AO-PA).

This mixin only touches the loss path and is composable with any other
mixin (disease conditioning, curriculum weights, topology, etc.).
"""
from __future__ import annotations

from typing import List, Tuple

from nnunetv2.training.loss.confusion_penalty import (
    confusion_penalty_loss,
    resolve_confusion_pairs,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class ConfusionPenaltyMixin(TrainerMixin):
    """Mixin that adds a targeted confusion penalty loss.

    Class attributes (override on concrete trainer or in mixin_init):

    ==============================  ========  ==========================================
    Attribute                       Default   Description
    ==============================  ========  ==========================================
    confusion_penalty_weight        1.0       Multiplier for the confusion penalty term
    confusion_penalty_enabled       True      Master switch
    ==============================  ========  ==========================================
    """

    def mixin_init(self):
        super().mixin_init()
        self.confusion_penalty_weight: float = 1.0
        self.confusion_penalty_enabled: bool = True
        self.confusion_pairs: List[Tuple[int, int]] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        if self.confusion_penalty_enabled:
            self.confusion_pairs = resolve_confusion_pairs(self.dataset_json)
            if self.confusion_pairs:
                self.print_to_log_file(
                    f"[ConfusionPenalty] Active for pairs: {self.confusion_pairs}, "
                    f"weight={self.confusion_penalty_weight}"
                )
            else:
                self.print_to_log_file(
                    "[ConfusionPenalty] No AO-PA pair found in dataset.json labels. "
                    "Confusion penalty DISABLED."
                )

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)

        if not self.confusion_penalty_enabled or not self.confusion_pairs:
            return extra

        # When deep supervision is active, output and target are lists.
        # Use only the full-resolution output (index 0) for the penalty.
        if isinstance(output, (list, tuple)):
            pred_logits = output[0]
            tgt = target[0] if isinstance(target, (list, tuple)) else target
        else:
            pred_logits = output
            tgt = target

        penalty = confusion_penalty_loss(pred_logits, tgt, self.confusion_pairs)
        return extra + self.confusion_penalty_weight * penalty
