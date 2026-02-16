"""
Topology loss mixin for composable trainers.

Adds soft-clDice topology loss on AO/PA classes, with a weight schedule
(linear warmup, cosine decay).
"""
from __future__ import annotations

from typing import List, Optional

import torch

from nnunetv2.training.loss.topology_losses import TopologyLoss, topo_weight_schedule
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopoLoss import (
    _resolve_topo_class_ids,
    _foreground_label_names,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class TopologyLossMixin(TrainerMixin):
    """Mixin that adds soft-clDice topology loss on AO and PA classes."""

    def mixin_init(self):
        super().mixin_init()
        self.topo_warmup_epochs: int = 10
        self.topo_decay_start_epoch: int = 30
        self.topo_w_high: float = 1.0
        self.topo_w_low: float = 0.1
        self.topo_num_iter: int = 10
        self.topo_class_ids: List[int] = _resolve_topo_class_ids(self.dataset_json)
        self.topo_loss: Optional[TopologyLoss] = None
        self._current_topo_weight: float = 0.0

    def mixin_initialize(self):
        super().mixin_initialize()
        if self.topo_class_ids:
            self.topo_loss = TopologyLoss(
                topo_class_ids=self.topo_class_ids,
                num_iter=self.topo_num_iter,
            ).to(self.device)
            self.print_to_log_file(
                f"Topology loss enabled for class IDs: {self.topo_class_ids}"
            )
        else:
            self.print_to_log_file(
                "WARNING: No AO/PA classes found in dataset.json labels. "
                "Topology loss DISABLED (baseline mode)."
            )
        self.logger.label_names = _foreground_label_names(self.dataset_json)
        self.logger.topo_class_ids = self.topo_class_ids if self.topo_class_ids else None

    def mixin_on_train_epoch_start(self):
        super().mixin_on_train_epoch_start()
        self._current_topo_weight = topo_weight_schedule(
            self.current_epoch,
            self.num_epochs,
            self.topo_warmup_epochs,
            self.topo_decay_start_epoch,
            self.topo_w_high,
            self.topo_w_low,
        )

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        if self.topo_loss is not None and self._current_topo_weight > 0:
            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                full_res_output = output[0]
                full_res_target = target[0] if isinstance(target, (list, tuple)) else target
            else:
                full_res_output = output
                full_res_target = target
            topo_l = self.topo_loss(full_res_output, full_res_target)
            extra = extra + self._current_topo_weight * topo_l
        return extra
