"""
Centerline-auxiliary mixin for composable trainers.

Adds a cross-entropy term up-weighted near the GT great-vessel centerline
(skeleton), pushing the network to preserve thin vessel branches without
changing the output head or label set.

The weight map ``1 + alpha * skeleton(GT vessel mask)`` is derived from the
*cropped* training target and fully detached, so it is always spatially aligned
with the patch and only the logits receive gradient (Option B in the plan:
no auxiliary head, nnU-Net-compatible).

Precomputed skeletons (from ``scripts/generate_centerline_targets_dataset030.py``)
are supported by the underlying loss for future dataloader integration, but the
default path computes the soft skeleton on the fly from the patch — correct by
construction and requiring zero extra setup.

See :mod:`nnunetv2.training.loss.anatomy_losses` for the (trainer-free) loss.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from nnunetv2.training.loss.anatomy_losses import CenterlineWeightedCELoss, resolve_chd_label_ids
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class CenterlineAuxMixin(TrainerMixin):
    """Mixin that adds centerline-weighted CE on the great vessels.

    Class attributes (override on the concrete trainer)
    ----------------------------------------------------
    centerline_lambda  : float  weight of the auxiliary term (default 0.2)
    centerline_alpha   : float  extra CE weight at the skeleton (default 3.0)
    centerline_num_iter: int    soft-skeletonization iterations (default 10)
    """

    centerline_lambda: float = 0.2
    centerline_alpha: float = 3.0
    centerline_num_iter: int = 10

    def mixin_init(self):
        super().mixin_init()
        ids = resolve_chd_label_ids(self.dataset_json)
        self._cl_vessel_ids: List[int] = sorted(
            v for k, v in ids.items() if k in ("AO", "PA")
        )
        self.centerline_loss: Optional[CenterlineWeightedCELoss] = None
        self._cl_loss_accum: List[float] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        if self._cl_vessel_ids:
            ignore = self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
            self.centerline_loss = CenterlineWeightedCELoss(
                vessel_ids=self._cl_vessel_ids,
                alpha=self.centerline_alpha,
                num_iter=self.centerline_num_iter,
                ignore_label=ignore,
            ).to(self.device)
            self.print_to_log_file(
                f"[Centerline] centerline-weighted CE on vessel ids "
                f"{self._cl_vessel_ids} (lambda={self.centerline_lambda}, "
                f"alpha={self.centerline_alpha})"
            )
        else:
            self.print_to_log_file(
                "[Centerline] WARNING: AO/PA not found in dataset.json — "
                "centerline loss DISABLED (baseline mode)."
            )

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        if self.centerline_loss is None or self.centerline_lambda <= 0:
            return extra
        if isinstance(output, (list, tuple)):
            full_res_output = output[0]
            full_res_target = target[0] if isinstance(target, (list, tuple)) else target
        else:
            full_res_output = output
            full_res_target = target
        cl_l = self.centerline_loss(full_res_output, full_res_target)
        self._cl_loss_accum.append(float(cl_l.detach()))
        return extra + self.centerline_lambda * cl_l

    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self.centerline_loss is None:
            return
        mean_cl = (
            sum(self._cl_loss_accum) / len(self._cl_loss_accum)
            if self._cl_loss_accum else float("nan")
        )
        self.logger.log(
            "centerline_aux",
            {
                "lambda": float(self.centerline_lambda),
                "mean_weighted_ce": mean_cl,
                "weight_max": float(self.centerline_loss.last_weight_max),
            },
            self.current_epoch,
        )
        self.print_to_log_file(
            f"[Centerline] epoch={self.current_epoch} lambda={self.centerline_lambda:.3f} "
            f"mean_weighted_ce={mean_cl:.4f} w_max={self.centerline_loss.last_weight_max:.2f}"
        )
        self._cl_loss_accum.clear()
