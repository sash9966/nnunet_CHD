"""
Septal-defect focus mixins for the ablation study (Dataset051, septal label id 8).

Two INDEPENDENT levers so their individual + combined contribution can be
measured (see composed trainers `nnUNetTrainerDA5Septal*`):

  * SeptalOversampleMixin — biases nnU-Net's foreground patch sampling toward
    the septal class so the model actually *sees* the tiny defect. Uses the
    existing `overwrite_class` hook in the data loader's `get_bbox` (no rewrite
    of the sampling logic). NOTE: this overrides the real `get_dataloaders`
    method (a shared code path), not a mixin_* hook — flagged intentionally; it
    only swaps the loader class during construction and restores it.

  * SeptalTverskyMixin — adds a false-negative-weighted Tversky term on the
    septal class (mixin_extra_loss), on softmax probs (no argmax). Recall-biased
    for the small structure. Reuses `soft_tversky_binary`.

Both resolve the septal class id BY NAME from dataset.json; if absent they
self-disable (baseline behaviour), so the arms are safe on non-052/051 datasets.
"""
from __future__ import annotations

import numpy as np
import torch

from nnunetv2.training.loss.septal_losses import resolve_septal_label_id, soft_tversky_binary
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin
from nnunetv2.utilities.helpers import softmax_helper_dim1


class SeptalOversampleMixin(TrainerMixin):
    """Force a fraction of foreground patches to be centered on the septal class.

    Class attributes (override on the concrete trainer):
      septal_oversample_prob : of the foreground-oversampled patches, fraction
        centered on the septal class when it is present in the case. Default 0.5.
    """
    septal_oversample_prob: float = 0.5

    def _resolve_septal_id(self):
        return resolve_septal_label_id(self.dataset_json)

    def get_dataloaders(self):
        # SHARED-PATH OVERRIDE (not a mixin_* hook): swap the loader class used
        # inside DA5.get_dataloaders to a subclass that biases get_bbox toward
        # the septal class, then restore. No duplication of the DA5 transform
        # pipeline.
        sid = self._resolve_septal_id()
        if sid is None:
            self.print_to_log_file("[SeptalOversample] no septal label in dataset.json -> disabled")
            return super().get_dataloaders()

        import nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 as da5mod
        Base = da5mod.nnUNetDataLoader
        prob = float(self.septal_oversample_prob)

        class _SeptalLoader(Base):
            def get_bbox(self, data_shape, force_fg, class_locations,
                         overwrite_class=None, verbose=False):
                if force_fg and overwrite_class is None and class_locations is not None:
                    loc = class_locations.get(sid)
                    if loc is not None and len(loc) > 0 and np.random.uniform() < prob:
                        overwrite_class = sid
                return super().get_bbox(data_shape, force_fg, class_locations,
                                        overwrite_class, verbose)

        da5mod.nnUNetDataLoader = _SeptalLoader
        self.print_to_log_file(
            f"[SeptalOversample] biasing fg sampling to septal id {sid} "
            f"(prob {prob}) for oversampled patches")
        try:
            return super().get_dataloaders()
        finally:
            da5mod.nnUNetDataLoader = Base


class SeptalTverskyMixin(TrainerMixin):
    """FN-weighted Tversky on the septal class, added to the base Dice+CE loss.

    Class attributes:
      septal_tversky_weight : weight on the term. Default 1.0.
      septal_tversky_alpha  : FP weight. Default 0.3.
      septal_tversky_beta   : FN weight (>alpha to find the small defect). Default 0.7.
      septal_tversky_warmup_epochs : epochs to keep the term OFF at the start, so
        the septal class can emerge under the base Dice+CE before the recall-biased
        term acts. 0 (default) = no warmup, constant weight (legacy arms unchanged).
      septal_tversky_ramp_epochs : after warmup, epochs over which the weight ramps
        linearly 0 -> septal_tversky_weight (avoids a destabilising step). Default 30.

    WHY warmup+ramp+low weight exist (Dataset070 ablation, 2026-07): at weight 1.0
    with alpha/beta=0.3/0.7 the term COLLAPSED the septal class to 0 predicted voxels
    on the test set (both Tversky arms: 0/13 cases vs oversample-only 10/13). The
    recall bias over-fired the tiny class early, the base loss backlashed, and the
    optimiser found it cheaper to suppress class 8 entirely. Fix = small weight +
    let the class emerge first + softer bias. See nnUNetTrainerDA5SeptalTverskyV2.
    """
    septal_tversky_weight: float = 1.0
    septal_tversky_alpha: float = 0.3
    septal_tversky_beta: float = 0.7
    septal_tversky_warmup_epochs: int = 0
    septal_tversky_ramp_epochs: int = 30

    def mixin_init(self):
        super().mixin_init()
        self._septal_id = None
        self._septal_tversky_accum = []

    def mixin_initialize(self):
        super().mixin_initialize()
        self._septal_id = resolve_septal_label_id(self.dataset_json)
        if self._septal_id is None:
            self.print_to_log_file(
                "[SeptalTversky] WARNING: no septal label in dataset.json -> term DISABLED")
        else:
            self.print_to_log_file(
                f"[SeptalTversky] septal id {self._septal_id}, weight={self.septal_tversky_weight}, "
                f"alpha={self.septal_tversky_alpha}, beta={self.septal_tversky_beta}, "
                f"warmup={self.septal_tversky_warmup_epochs} ramp={self.septal_tversky_ramp_epochs}")

    def _tversky_weight_now(self) -> float:
        """Effective weight this epoch, applying optional warmup + linear ramp.

        warmup<=0 -> constant weight (legacy behaviour, byte-for-byte). Otherwise
        0 until `warmup`, then linear 0->weight over `ramp` epochs, then full.
        """
        w = self.septal_tversky_warmup_epochs
        if w <= 0:
            return self.septal_tversky_weight
        if self.current_epoch < w:
            return 0.0
        r = max(1, self.septal_tversky_ramp_epochs)
        return self.septal_tversky_weight * min(1.0, (self.current_epoch - w) / r)

    def mixin_extra_loss(self, output, target, batch: dict, **forward_kwargs) -> float:
        if self._septal_id is None:
            return 0.0
        w_now = self._tversky_weight_now()
        if w_now == 0.0:
            return 0.0  # in warmup window: term is off
        logits = output[0] if isinstance(output, (list, tuple)) else output
        tgt = target[0] if isinstance(target, (list, tuple)) else target
        probs = softmax_helper_dim1(logits)
        p = probs[:, self._septal_id:self._septal_id + 1]
        gt = (tgt == self._septal_id).float()
        if gt.sum() == 0:
            return 0.0  # positive-supervision only: don't penalise where GT has no defect
        loss = soft_tversky_binary(p, gt, self.septal_tversky_alpha, self.septal_tversky_beta)
        self._septal_tversky_accum.append(float(loss.detach()))
        return w_now * loss

    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self._septal_tversky_accum:
            self.print_to_log_file(
                f"[SeptalTversky] mean term: "
                f"{sum(self._septal_tversky_accum) / len(self._septal_tversky_accum):.4f} "
                f"(eff weight {self._tversky_weight_now():.3f})")
            self._septal_tversky_accum.clear()
        elif self.septal_tversky_warmup_epochs > 0 and self.current_epoch < self.septal_tversky_warmup_epochs:
            self.print_to_log_file(
                f"[SeptalTversky] warmup: term OFF (epoch {self.current_epoch} "
                f"< {self.septal_tversky_warmup_epochs})")
