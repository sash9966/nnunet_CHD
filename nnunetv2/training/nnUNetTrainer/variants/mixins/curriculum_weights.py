"""
Curriculum class-weighting mixin for composable trainers.

Upweights specific target classes (aorta, pulmonary artery) in the
CrossEntropy loss during the first P% of training epochs, then reverts
to uniform weighting.  Dice loss is NOT affected.

This mixin:
- Overrides ``_build_loss`` to inject an initial CE weight tensor.
- Uses ``mixin_on_train_epoch_start`` to update the CE weight tensor
  according to the schedule each epoch.
- Logs the current curriculum state once per epoch.

When ``curriculum_enabled=False``, behaviour is identical to baseline
(all CE weights = 1).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn

from nnunetv2.training.loss.curriculum_weights import (
    get_curriculum_ce_weights,
    resolve_target_class_ids,
)
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class CurriculumWeightsMixin(TrainerMixin):
    """Mixin that adds epoch-dependent CE class weighting.

    Configurable via class attributes or ``mixin_init`` overrides:

    ==========================  ========  ======================================
    Attribute                   Default   Meaning
    ==========================  ========  ======================================
    curriculum_enabled          True      Toggle the feature on/off
    curriculum_fraction         0.20      Fraction of epochs with upweighting
    curriculum_multiplier       3.0       CE weight multiplier for target classes
    curriculum_schedule         "step"    "step" or "linear_decay"
    curriculum_target_names     None      Custom label names; None = AO+PA auto
    ==========================  ========  ======================================
    """

    def mixin_init(self):
        super().mixin_init()
        self.curriculum_enabled: bool = True
        self.curriculum_fraction: float = 0.20
        self.curriculum_multiplier: float = 3.0
        self.curriculum_schedule: str = "step"  # "step" or "linear_decay"
        self.curriculum_target_names: Optional[List[str]] = None
        # resolved at mixin_initialize
        self._curriculum_target_ids: List[int] = []
        self._curriculum_label_names: List[str] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        if not self.curriculum_enabled:
            return
        if self.label_manager.has_regions:
            self.print_to_log_file(
                "WARNING: CurriculumWeightsMixin is not supported with region-based training. "
                "Curriculum weighting DISABLED."
            )
            self.curriculum_enabled = False
            return

        # resolve target class IDs
        self._curriculum_target_ids = resolve_target_class_ids(
            self.dataset_json,
            target_names=self.curriculum_target_names,
        )
        # build label name list for logging
        labels = self.dataset_json.get("labels", {})
        id_to_name = {int(v): k for k, v in labels.items()}
        self._curriculum_label_names = [
            id_to_name.get(cid, f"class_{cid}") for cid in self._curriculum_target_ids
        ]
        self.print_to_log_file(
            f"Curriculum CE weighting enabled: target classes = "
            f"{list(zip(self._curriculum_label_names, self._curriculum_target_ids))}, "
            f"multiplier = {self.curriculum_multiplier}, "
            f"fraction = {self.curriculum_fraction}, "
            f"schedule = {self.curriculum_schedule}"
        )

    # ------------------------------------------------------------------
    # Override _build_loss to inject initial CE weight tensor
    # ------------------------------------------------------------------
    def _build_loss(self):
        """Build loss with an initial CE class-weight tensor.

        When curriculum is disabled, falls through to the base
        ``_build_loss`` unchanged (exact baseline behaviour).
        """
        if not self.curriculum_enabled or self.label_manager.has_regions:
            return super()._build_loss()

        num_classes = len(self.label_manager.all_labels)

        # compute initial weights for epoch 0
        initial_weights = get_curriculum_ce_weights(
            epoch=0,
            num_epochs=self.num_epochs,
            num_classes=num_classes,
            target_ids=self._curriculum_target_ids,
            multiplier=self.curriculum_multiplier,
            fraction=self.curriculum_fraction,
            schedule_type=self.curriculum_schedule,
        )

        loss = DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice,
             "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {"weight": initial_weights},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            import numpy as np
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    # ------------------------------------------------------------------
    # Helper: navigate to the CE loss module
    # ------------------------------------------------------------------
    def _get_ce_loss_module(self) -> nn.Module:
        """Get the CE loss module from the (possibly DS-wrapped) loss."""
        if isinstance(self.loss, DeepSupervisionWrapper):
            return self.loss.loss.ce
        return self.loss.ce

    # ------------------------------------------------------------------
    # Hook: update CE weights each epoch
    # ------------------------------------------------------------------
    def mixin_on_train_epoch_start(self):
        super().mixin_on_train_epoch_start()
        if not self.curriculum_enabled:
            return

        ce_module = self._get_ce_loss_module()
        if ce_module.weight is None:
            return

        num_classes = ce_module.weight.shape[0]
        new_weights = get_curriculum_ce_weights(
            epoch=self.current_epoch,
            num_epochs=self.num_epochs,
            num_classes=num_classes,
            target_ids=self._curriculum_target_ids,
            multiplier=self.curriculum_multiplier,
            fraction=self.curriculum_fraction,
            schedule_type=self.curriculum_schedule,
        )
        ce_module.weight.copy_(new_weights.to(ce_module.weight.device))

        # log curriculum state
        T = math.ceil(self.curriculum_fraction * self.num_epochs)
        is_active = self.current_epoch < T
        target_w = new_weights[self._curriculum_target_ids[0]].item() if self._curriculum_target_ids else 1.0
        self.print_to_log_file(
            f"[Curriculum] epoch {self.current_epoch}/{self.num_epochs}: "
            f"active={is_active}, target_weight={target_w:.2f}, "
            f"classes={self._curriculum_label_names}"
        )
