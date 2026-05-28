"""
Topology loss mixin for composable trainers.

Adds soft-clDice topology loss on AO/PA classes at a fixed weight
throughout training (no warmup or decay schedule).
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
    """Mixin that adds soft-clDice topology loss on AO and PA classes.

    The loss is applied at a fixed weight (``topo_weight``) from epoch 0
    through the end of training — no warmup or decay.  Subclasses can
    override ``topo_weight`` to tune the contribution.
    """

    def mixin_init(self):
        super().mixin_init()
        self.topo_weight: float = 0.2      # constant loss weight throughout training
        self.topo_num_iter: int = 10
        self.topo_class_ids: List[int] = _resolve_topo_class_ids(self.dataset_json)
        self.topo_loss: Optional[TopologyLoss] = None
        # Per-epoch accumulators for diagnostics (cleared in mixin_on_train_epoch_end).
        self._topo_loss_accum: List[float] = []
        self._topo_per_class_loss_accum: dict = {}   # cid -> list of float losses
        self._topo_per_class_present_accum: dict = {}  # cid -> list of bool

    def mixin_initialize(self):
        super().mixin_initialize()
        if self.topo_class_ids:
            self.topo_loss = TopologyLoss(
                topo_class_ids=self.topo_class_ids,
                num_iter=self.topo_num_iter,
            ).to(self.device)
            # Pre-seed accumulators so we can always emit a consistent shape per epoch.
            for cid in self.topo_class_ids:
                self._topo_per_class_loss_accum[cid] = []
                self._topo_per_class_present_accum[cid] = []
            self.print_to_log_file(
                f"Topology loss enabled for class IDs: {self.topo_class_ids} "
                f"(weight={self.topo_weight}, constant)"
            )
        else:
            self.print_to_log_file(
                "WARNING: No AO/PA classes found in dataset.json labels. "
                "Topology loss DISABLED (baseline mode)."
            )
        self.logger.label_names = _foreground_label_names(self.dataset_json)
        self.logger.topo_class_ids = self.topo_class_ids if self.topo_class_ids else None

    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        if self.topo_loss is not None:
            if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                full_res_output = output[0]
                full_res_target = target[0] if isinstance(target, (list, tuple)) else target
            else:
                full_res_output = output
                full_res_target = target
            topo_l = self.topo_loss(full_res_output, full_res_target)
            # Capture per-class diagnostics from the loss module (set inside forward).
            self._topo_loss_accum.append(float(topo_l.detach()))
            for cid in self.topo_class_ids:
                present = self.topo_loss.last_per_class_present.get(cid, False)
                self._topo_per_class_present_accum[cid].append(bool(present))
                if present:
                    self._topo_per_class_loss_accum[cid].append(
                        self.topo_loss.last_per_class_loss[cid]
                    )
            extra = extra + self.topo_weight * topo_l
        return extra

    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self.topo_loss is None:
            return

        # Aggregate per-epoch diagnostics.
        per_class_loss = {}
        per_class_present = {}
        for cid in self.topo_class_ids:
            losses = self._topo_per_class_loss_accum.get(cid, [])
            presence = self._topo_per_class_present_accum.get(cid, [])
            per_class_loss[cid] = (sum(losses) / len(losses)) if losses else float('nan')
            per_class_present[cid] = (sum(presence) / len(presence)) if presence else 0.0

        mean_topo = (sum(self._topo_loss_accum) / len(self._topo_loss_accum)
                     if self._topo_loss_accum else float('nan'))

        # Persist to logger so plot_progress_png can render a panel.
        self.logger.log(
            'topology',
            {
                'weight': float(self.topo_weight),
                'mean_loss': mean_topo,
                # JSON-safe: convert int keys to strings (per_class_loss[6] -> '6').
                'per_class_loss': {str(k): v for k, v in per_class_loss.items()},
                'per_class_present': {str(k): v for k, v in per_class_present.items()},
            },
            self.current_epoch,
        )

        # Text log: split AO / PA so a flip-flopping class is obvious.
        # Use foreground label names from dataset_json for the print line.
        label_names = _foreground_label_names(self.dataset_json) or []
        parts = []
        for cid in self.topo_class_ids:
            # cid is a label id; foreground label list excludes background (cid 0),
            # so labels[cid-1] corresponds to label id cid.
            name = label_names[cid - 1] if 0 < cid <= len(label_names) else f"cls{cid}"
            l = per_class_loss[cid]
            p = per_class_present[cid]
            parts.append(f"{name}(cid={cid}) cldice={l:.4f} present={p:.2f}")
        self.print_to_log_file(
            f"[Topo] epoch={self.current_epoch}  w={self.topo_weight:.4f}  "
            f"mean_cldice={mean_topo:.4f}  | {' | '.join(parts)}"
        )

        # Reset accumulators for next epoch.
        self._topo_loss_accum.clear()
        for cid in self.topo_class_ids:
            self._topo_per_class_loss_accum[cid].clear()
            self._topo_per_class_present_accum[cid].clear()


class TopologyLossScheduledMixin(TopologyLossMixin):
    """Like TopologyLossMixin but with a warmup → plateau → cosine-decay weight schedule.

    The topology loss weight starts at 0 and linearly ramps to ``topo_w_high``
    over ``topo_warmup_epochs`` epochs, stays at ``topo_w_high`` until
    ``topo_decay_start``, then cosine-decays to ``topo_w_low`` by the end of
    training.

    This avoids topology loss destabilising early training when the main
    Dice/CE loss has not yet converged.

    Subclasses can override the four schedule parameters as class attributes.
    """

    def mixin_init(self):
        super().mixin_init()
        self.topo_weight         = 0.0   # overwritten each epoch; start silent
        self.topo_warmup_epochs  = 10
        self.topo_decay_start    = 40
        self.topo_w_high         = 0.5
        self.topo_w_low          = 0.05

    def mixin_on_train_epoch_start(self):
        super().mixin_on_train_epoch_start()
        self.topo_weight = topo_weight_schedule(
            self.current_epoch,
            self.num_epochs,
            warmup_epochs=self.topo_warmup_epochs,
            decay_start_epoch=self.topo_decay_start,
            w_high=self.topo_w_high,
            w_low=self.topo_w_low,
        )
        if getattr(self, 'local_rank', 0) == 0:
            self.print_to_log_file(
                f"[TopoSched] epoch={self.current_epoch} topo_weight={self.topo_weight:.4f}"
            )
