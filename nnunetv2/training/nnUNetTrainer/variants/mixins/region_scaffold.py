"""
Region-scaffold mixin for composable trainers.

Adds hierarchical region-level supervision derived from the existing multiclass
labels (whole-heart, blood-pool, chambers, ventricles, atria, great-vessels,
myocardium).  The intent is to teach stable coarse anatomy *before* fine
multiclass competition, reducing the semantic flip-flopping that Dice barely
penalises.

The region loss is computed on soft softmax probabilities (no argmax) and added
on top of the base Dice+CE loss with a stepwise-decaying weight.  The network is
left unchanged, so inference and comparison to the DA5 baseline are identical.

See :mod:`nnunetv2.training.loss.anatomy_losses` for the (trainer-free) math.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from nnunetv2.training.loss.anatomy_losses import (
    SoftRegionScaffoldLoss,
    build_region_groups,
    region_lambda_schedule,
    resolve_chd_label_ids,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class RegionScaffoldMixin(TrainerMixin):
    """Mixin that adds hierarchical region-scaffold supervision.

    Class attributes (override on the concrete trainer)
    ----------------------------------------------------
    region_lambda_boundaries : tuple[int, ...]
        Epoch boundaries for the stepwise weight schedule.  Default (100, 500).
    region_lambda_values : tuple[float, ...]
        Weights per phase (one longer than boundaries).  Default (0.3, 0.15, 0.05).
    """

    region_lambda_boundaries: tuple = (100, 500)
    region_lambda_values: tuple = (0.3, 0.15, 0.05)

    def mixin_init(self):
        super().mixin_init()
        self._region_label_ids: Dict[str, int] = resolve_chd_label_ids(self.dataset_json)
        self._region_groups: Dict[str, List[int]] = build_region_groups(self._region_label_ids)
        self.region_loss = None
        self._region_weight: float = self.region_lambda_values[0]
        # Per-epoch diagnostics.
        self._region_loss_accum: List[float] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        if self._region_groups:
            self.region_loss = SoftRegionScaffoldLoss(self._region_groups).to(self.device)
            self.print_to_log_file(
                f"[RegionScaffold] resolved labels {self._region_label_ids}; "
                f"region groups: "
                + ", ".join(f"{k}={v}" for k, v in self._region_groups.items())
            )
        else:
            self.print_to_log_file(
                "[RegionScaffold] WARNING: no CHD region groups resolved from "
                "dataset.json labels — region loss DISABLED (baseline mode)."
            )

    def mixin_on_train_epoch_start(self):
        super().mixin_on_train_epoch_start()
        self._region_weight = region_lambda_schedule(
            self.current_epoch,
            self.region_lambda_boundaries,
            self.region_lambda_values,
        )

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        if self.region_loss is None or self._region_weight <= 0:
            return extra
        if isinstance(output, (list, tuple)):
            full_res_output = output[0]
            full_res_target = target[0] if isinstance(target, (list, tuple)) else target
        else:
            full_res_output = output
            full_res_target = target
        region_l = self.region_loss(full_res_output, full_res_target)
        self._region_loss_accum.append(float(region_l.detach()))
        return extra + self._region_weight * region_l

    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self.region_loss is None:
            return
        mean_region = (
            sum(self._region_loss_accum) / len(self._region_loss_accum)
            if self._region_loss_accum else float("nan")
        )
        per_region = self.region_loss.last_per_region_loss
        self.logger.log(
            "region_scaffold",
            {
                "weight": float(self._region_weight),
                "mean_loss": mean_region,
                "per_region_loss": {k: float(v) for k, v in per_region.items()},
            },
            self.current_epoch,
        )
        parts = " | ".join(f"{k}={v:.4f}" for k, v in per_region.items())
        self.print_to_log_file(
            f"[RegionScaffold] epoch={self.current_epoch} w={self._region_weight:.3f} "
            f"mean_region={mean_region:.4f}  | {parts}"
        )
        self._region_loss_accum.clear()
