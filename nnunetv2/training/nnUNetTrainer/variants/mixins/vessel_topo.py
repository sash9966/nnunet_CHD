"""
Vessel-focused topology mixin for composable trainers.

Applies soft-clDice on the *merged* great-vessel mask (AO ∪ PA) treated as a
single connected tubular structure, rather than the per-class topology loss of
``TopologyLossMixin``.  The hypothesis: per-class clDice dilutes the continuity
signal and forces AO-vs-PA identity competition; a single binary vessel-tree
target concentrates topology pressure on *continuity*.

The weight follows a warmup → ramp → cap schedule (silent for the first
``vessel_warmup_epochs`` so it does not destabilise early training).  The network
is unchanged → inference and baseline comparison are identical.

See :mod:`nnunetv2.training.loss.anatomy_losses` for the (trainer-free) loss.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from nnunetv2.training.loss.anatomy_losses import BinaryVesselClDiceLoss, resolve_chd_label_ids
from nnunetv2.training.loss.topology_losses import topo_weight_schedule
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class VesselFocusedTopologyMixin(TrainerMixin):
    """Mixin that adds binary great-vessel (AO ∪ PA) soft-clDice.

    Class attributes (override on the concrete trainer)
    ----------------------------------------------------
    vessel_warmup_epochs : int   silent ramp start (default 50)
    vessel_ramp_end      : int   epoch the ramp reaches the cap (default 150)
    vessel_w_high        : float ramp target weight (default 0.10)
    vessel_w_cap         : float hard cap (default 0.15)
    vessel_num_iter      : int   soft-skeletonization iterations (default 10)
    """

    vessel_warmup_epochs: int = 50
    vessel_ramp_end: int = 150
    vessel_w_high: float = 0.10
    vessel_w_cap: float = 0.15
    vessel_num_iter: int = 10

    def mixin_init(self):
        super().mixin_init()
        ids = resolve_chd_label_ids(self.dataset_json)
        self._vessel_ids: List[int] = sorted(
            v for k, v in ids.items() if k in ("AO", "PA")
        )
        self.vessel_loss: Optional[BinaryVesselClDiceLoss] = None
        self._vessel_weight: float = 0.0
        self._vessel_loss_accum: List[float] = []
        self._vessel_present_accum: List[bool] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        if self._vessel_ids:
            self.vessel_loss = BinaryVesselClDiceLoss(
                vessel_ids=self._vessel_ids, num_iter=self.vessel_num_iter,
            ).to(self.device)
            self.print_to_log_file(
                f"[VesselTopo] binary great-vessel clDice on label ids "
                f"{self._vessel_ids} (cap={self.vessel_w_cap}, "
                f"warmup={self.vessel_warmup_epochs}->{self.vessel_ramp_end})"
            )
        else:
            self.print_to_log_file(
                "[VesselTopo] WARNING: AO/PA not found in dataset.json — "
                "vessel topology loss DISABLED (baseline mode)."
            )

    def mixin_on_train_epoch_start(self):
        super().mixin_on_train_epoch_start()
        # Reuse the topo schedule helper for warmup→plateau, then hard-cap.
        w = topo_weight_schedule(
            self.current_epoch,
            self.num_epochs,
            warmup_epochs=self.vessel_ramp_end,   # linear ramp up to ramp_end
            decay_start_epoch=self.num_epochs,    # no decay; stay at plateau
            w_high=self.vessel_w_high,
            w_low=self.vessel_w_high,
        )
        # Silent before warmup start.
        if self.current_epoch < self.vessel_warmup_epochs:
            w = 0.0
        self._vessel_weight = min(w, self.vessel_w_cap)

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        if self.vessel_loss is None or self._vessel_weight <= 0:
            return extra
        if isinstance(output, (list, tuple)):
            full_res_output = output[0]
            full_res_target = target[0] if isinstance(target, (list, tuple)) else target
        else:
            full_res_output = output
            full_res_target = target
        vessel_l = self.vessel_loss(full_res_output, full_res_target)
        self._vessel_present_accum.append(self.vessel_loss.last_present)
        if self.vessel_loss.last_present:
            self._vessel_loss_accum.append(self.vessel_loss.last_loss)
        return extra + self._vessel_weight * vessel_l

    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self.vessel_loss is None:
            return
        mean_loss = (
            sum(self._vessel_loss_accum) / len(self._vessel_loss_accum)
            if self._vessel_loss_accum else float("nan")
        )
        present_rate = (
            sum(self._vessel_present_accum) / len(self._vessel_present_accum)
            if self._vessel_present_accum else 0.0
        )
        self.logger.log(
            "vessel_topology",
            {
                "weight": float(self._vessel_weight),
                "mean_cldice": mean_loss,
                "present_rate": float(present_rate),
            },
            self.current_epoch,
        )
        self.print_to_log_file(
            f"[VesselTopo] epoch={self.current_epoch} w={self._vessel_weight:.4f} "
            f"mean_cldice={mean_loss:.4f} present={present_rate:.2f}"
        )
        self._vessel_loss_accum.clear()
        self._vessel_present_accum.clear()
