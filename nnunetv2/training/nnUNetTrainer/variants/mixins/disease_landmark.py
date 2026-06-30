"""
Disease-landmark mixin for composable trainers.

Adds disease-aware auxiliary supervision when training on the
``Dataset031_imageCHD_DiseaseLandmarks`` dataset produced by the
``chd_landmarks`` package (derived integer labels such as ``vsd_orifice_proxy``
and ``asd_orifice_proxy`` merged onto the 7 anatomy labels).

Two auxiliary terms, both on soft probabilities (no argmax), added on top of the
base Dice+CE loss:

  1. **Rare disease-label supervision** — a soft-Dice term for each derived
     hard label, up-weighted by ``lambda_disease``. To respect design
     principle 4 (absence of a disease landmark is NOT confirmed background
     when it was not derived/verified), the term for a label is only applied
     on batches where that label is actually present in the GT — i.e. positive
     supervision only, never penalising predictions in cases whose absence is
     unverified.

  2. **Vessel clDice** — optional centreline-Dice on the merged great-vessel
     mask (aorta ∪ pulmonary artery), weighted by ``lambda_vessel_cldice``,
     reusing :class:`SoftClDiceLoss`.

The network is left UNCHANGED, so ``nnUNetv2_predict`` works unmodified and
outputs are directly comparable to the DA5 baseline.

Label ids are resolved by NAME from ``dataset.json`` — never hardcoded. If no
derived labels are found (e.g. training on the plain anatomy dataset), the
disease term silently disables itself and the trainer behaves like its base.

NOT YET WIRED (documented scaffolds — see ``docs/chd_disease_landmark_nnunet.md``):
  * disease-ROI patch oversampling (needs dataloader sampling-location plumbing),
  * case-wise masked loss via nnU-Net ``ignore_label`` (needs per-case status).
"""
from __future__ import annotations

from typing import Dict, List

import torch

from nnunetv2.training.loss.anatomy_losses import (
    _isin_labelmap,
    _soft_dice_binary,
    build_region_groups,
    resolve_chd_label_ids,
)
from nnunetv2.training.loss.topology_losses import SoftClDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin
from nnunetv2.utilities.helpers import softmax_helper_dim1


# Canonical derived hard-label names that may appear in the merged dataset.json.
# The general CHD dataset currently promotes only the unified septal-defect
# proxy; the others are listed for forward compatibility and self-skip if absent.
_DERIVED_LABEL_NAMES = ("septal_defect_proxy", "vsd_orifice_proxy",
                        "asd_orifice_proxy", "pulmonary_stenosis_roi")


class DiseaseLandmarkMixin(TrainerMixin):
    """Mixin adding disease-aware auxiliary losses for derived landmark labels.

    Class attributes (override on the concrete trainer)
    ----------------------------------------------------
    lambda_disease : float
        Weight on the rare derived-label soft-Dice term. Default 0.3.
    lambda_vessel_cldice : float
        Weight on the great-vessel clDice term (0 disables it). Default 0.1.
    cldice_num_iter : int
        Soft-skeletonization iterations for clDice. Default 10.
    """

    lambda_disease: float = 0.3
    lambda_vessel_cldice: float = 0.1
    cldice_num_iter: int = 10

    # ------------------------------------------------------------------
    def mixin_init(self):
        super().mixin_init()
        self._derived_label_ids: Dict[str, int] = {}
        self._vessel_ids: List[int] = []
        self._vessel_cldice = None
        self._disease_loss_accum: List[float] = []
        self._vessel_loss_accum: List[float] = []

    def mixin_initialize(self):
        super().mixin_initialize()
        labels = self.dataset_json.get("labels", {})
        norm = {str(k).lower(): v for k, v in labels.items()}
        for name in _DERIVED_LABEL_NAMES:
            if name in norm and isinstance(norm[name], int):
                self._derived_label_ids[name] = int(norm[name])

        groups = build_region_groups(resolve_chd_label_ids(self.dataset_json))
        self._vessel_ids = groups.get("great_vessels", [])
        if self.lambda_vessel_cldice > 0 and self._vessel_ids:
            self._vessel_cldice = SoftClDiceLoss(num_iter=self.cldice_num_iter).to(self.device)

        if self._derived_label_ids:
            self.print_to_log_file(
                f"[DiseaseLandmark] derived labels: {self._derived_label_ids} "
                f"(lambda_disease={self.lambda_disease}); "
                f"vessel clDice ids={self._vessel_ids} "
                f"(lambda={self.lambda_vessel_cldice if self._vessel_cldice else 0})")
        else:
            self.print_to_log_file(
                "[DiseaseLandmark] WARNING: no derived disease labels found in "
                "dataset.json — disease term DISABLED (baseline behaviour). "
                "Did you point this trainer at Dataset031_imageCHD_DiseaseLandmarks?")

    # ------------------------------------------------------------------
    def mixin_extra_loss(self, output, target, batch: dict, **forward_kwargs) -> float:
        # full-resolution head under deep supervision
        logits = output[0] if isinstance(output, (list, tuple)) else output
        tgt = target[0] if isinstance(target, (list, tuple)) else target

        if not self._derived_label_ids and self._vessel_cldice is None:
            return 0.0

        probs = softmax_helper_dim1(logits)
        total = logits.new_zeros(())

        # 1) rare derived-label soft Dice (positive supervision only)
        disease_terms = []
        for name, lid in self._derived_label_ids.items():
            gt = _isin_labelmap(tgt, [lid]).float()
            if gt.sum() == 0:
                continue  # absence unverified -> do NOT supervise as background
            p = probs[:, lid:lid + 1]
            disease_terms.append(_soft_dice_binary(p, gt))
        if disease_terms:
            disease_loss = torch.stack(disease_terms).mean()
            total = total + self.lambda_disease * disease_loss
            self._disease_loss_accum.append(float(disease_loss.detach()))

        # 2) great-vessel clDice (aorta ∪ PA)
        if self._vessel_cldice is not None and self._vessel_ids:
            gt_vessel = _isin_labelmap(tgt, self._vessel_ids).float()
            if gt_vessel.sum() > 0:
                p_vessel = probs[:, self._vessel_ids].sum(dim=1, keepdim=True).clamp(0, 1)
                vloss = self._vessel_cldice(p_vessel, gt_vessel)
                total = total + self.lambda_vessel_cldice * vloss
                self._vessel_loss_accum.append(float(vloss.detach()))

        return total

    # ------------------------------------------------------------------
    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self._disease_loss_accum:
            self.print_to_log_file(
                f"[DiseaseLandmark] mean disease soft-Dice loss: "
                f"{sum(self._disease_loss_accum) / len(self._disease_loss_accum):.4f}")
            self._disease_loss_accum.clear()
        if self._vessel_loss_accum:
            self.print_to_log_file(
                f"[DiseaseLandmark] mean vessel clDice loss: "
                f"{sum(self._vessel_loss_accum) / len(self._vessel_loss_accum):.4f}")
            self._vessel_loss_accum.clear()
