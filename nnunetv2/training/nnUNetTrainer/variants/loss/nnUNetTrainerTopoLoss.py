"""
nnU-Net trainer variants with topology-preserving loss (soft-clDice).

Variant A — ``nnUNetTrainerTopoLoss``:
    Standard Dice+CE loss plus soft-clDice topology loss on aorta (AO) and
    pulmonary artery (PA).  No class reweighting.

Variant B — ``nnUNetTrainerTopoLossReweight``:
    Same as Variant A, but additionally increases CE class weights for AO/PA
    during the first 20 % of training epochs to front-load attention on
    tubular structures.

Both variants resolve AO/PA label indices dynamically from dataset.json
rather than hardcoding them.  If the labels are not found, topology loss
is silently disabled and training falls back to exact baseline behaviour.

Reference
---------
Shit et al., "clDice — A Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation", CVPR 2021.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.topology_losses import TopologyLoss, topo_weight_schedule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


# ---------------------------------------------------------------------------
# Helper: resolve AO/PA class indices from dataset.json
# ---------------------------------------------------------------------------

def _resolve_topo_class_ids(dataset_json: dict) -> List[int]:
    """Find aorta and pulmonary artery label indices from dataset.json.

    Searches the ``labels`` dict for keys matching (case-insensitive):
    - aorta / ao  → aorta label
    - pulmonary / pa  → pulmonary artery label

    Returns list of int (may be empty if not found).
    """
    labels: Dict[str, int] = dataset_json.get("labels", {})
    ao_keywords = {"ao", "aorta"}
    pa_keywords = {"pa", "pulmonary", "pulmonary artery"}
    topo_ids: List[int] = []
    for name, idx in labels.items():
        name_lower = name.lower().strip()
        if name_lower == "background":
            continue
        if name_lower in ao_keywords:
            topo_ids.append(int(idx))
        elif name_lower in pa_keywords:
            topo_ids.append(int(idx))
    return sorted(set(topo_ids))


def _foreground_label_names(dataset_json: dict) -> List[str]:
    """Return ordered list of foreground label names (excluding background).

    Matches the order of ``dice_per_class_or_region`` which excludes class 0.
    """
    labels: Dict[str, int] = dataset_json.get("labels", {})
    # sort by index, skip background (index 0)
    sorted_labels = sorted(labels.items(), key=lambda kv: int(kv[1]))
    return [name for name, idx in sorted_labels if int(idx) != 0]


# =========================================================================
# Variant A: topology loss only (no class reweighting)
# =========================================================================

class nnUNetTrainerTopoLoss(nnUNetTrainer):
    """nnUNetTrainer + soft-clDice topology loss on AO and PA.

    Configurable hyperparameters (override in subclass __init__ if needed):

    =======================  ========  =========================================
    Attribute                Default   Meaning
    =======================  ========  =========================================
    topo_warmup_epochs       10        Linear warmup from 0 to topo_w_high
    topo_decay_start_epoch   30        Start cosine decay of topo weight
    topo_w_high              1.0       Peak topology loss weight
    topo_w_low               0.1       Minimum topology loss weight after decay
    topo_num_iter            10        Soft-skeletonization iterations
    =======================  ========  =========================================
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # topology loss schedule
        self.topo_warmup_epochs: int = 10
        self.topo_decay_start_epoch: int = 30
        self.topo_w_high: float = 1.0
        self.topo_w_low: float = 0.1
        self.topo_num_iter: int = 10
        # resolved at initialize()
        self.topo_class_ids: List[int] = _resolve_topo_class_ids(dataset_json)
        self.topo_loss: Optional[TopologyLoss] = None
        self._current_topo_weight: float = 0.0

    def initialize(self):
        super().initialize()
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
        # set per-class label names on the logger for nicer progress plots
        self.logger.label_names = _foreground_label_names(self.dataset_json)
        self.logger.topo_class_ids = self.topo_class_ids if self.topo_class_ids else None

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._current_topo_weight = topo_weight_schedule(
            self.current_epoch,
            self.num_epochs,
            self.topo_warmup_epochs,
            self.topo_decay_start_epoch,
            self.topo_w_high,
            self.topo_w_low,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

            # Add topology loss on full-resolution output only
            if self.topo_loss is not None and self._current_topo_weight > 0:
                if self.enable_deep_supervision and isinstance(output, (list, tuple)):
                    full_res_output = output[0]
                    full_res_target = target[0] if isinstance(target, (list, tuple)) else target
                else:
                    full_res_output = output
                    full_res_target = target
                topo_l = self.topo_loss(full_res_output, full_res_target)
                l = l + self._current_topo_weight * topo_l

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}


# =========================================================================
# Variant B: topology loss + early-epoch CE class reweighting
# =========================================================================

class nnUNetTrainerTopoLossReweight(nnUNetTrainerTopoLoss):
    """Topology loss + early-epoch CE class reweighting for AO/PA.

    During the first ``ce_reweight_fraction`` of total epochs (default 20 %),
    the cross-entropy loss applies a higher weight to the AO and PA classes
    (default 3x).  After that threshold the CE weights revert to uniform.

    The Dice loss is NOT reweighted — it already has per-class computation
    which naturally balances class contributions.

    Extra attributes
    ----------------
    ce_reweight_fraction : float
        Fraction of total epochs during which reweighting is active.
    ce_topo_class_multiplier : float
        Weight multiplier for AO/PA in the CE loss during the reweight phase.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.ce_reweight_fraction: float = 0.20
        self.ce_topo_class_multiplier: float = 3.0

    def _build_loss(self):
        """Build loss with an initial CE class-weight tensor."""
        if self.label_manager.has_regions:
            # Region-based training uses BCE, no class weights; fall back
            return super()._build_loss()

        num_classes = len(self.label_manager.all_labels)

        # Build initial weight: reweighted for the early phase
        ce_weight = torch.ones(num_classes, dtype=torch.float32, device=self.device)
        for cid in self.topo_class_ids:
            if cid < num_classes:
                ce_weight[cid] = self.ce_topo_class_multiplier

        loss = DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice,
             "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {"weight": ce_weight},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _get_ce_loss_module(self) -> nn.Module:
        """Navigate through DeepSupervisionWrapper to the CE loss module."""
        if isinstance(self.loss, DeepSupervisionWrapper):
            return self.loss.loss.ce
        return self.loss.ce

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        reweight_end = int(math.ceil(self.num_epochs * self.ce_reweight_fraction))
        ce_module = self._get_ce_loss_module()

        if ce_module.weight is None:
            return  # safety: no weight tensor registered

        num_classes = ce_module.weight.shape[0]
        if self.current_epoch < reweight_end:
            # Reweighted phase
            w = torch.ones(num_classes, dtype=torch.float32, device=self.device)
            for cid in self.topo_class_ids:
                if cid < num_classes:
                    w[cid] = self.ce_topo_class_multiplier
            ce_module.weight.copy_(w)
        else:
            # Uniform phase
            ce_module.weight.copy_(
                torch.ones(num_classes, dtype=torch.float32, device=self.device)
            )


# =========================================================================
# 100-epoch convenience subclasses
# =========================================================================

class nnUNetTrainerTopoLoss_100epochs(nnUNetTrainerTopoLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerTopoLossReweight_100epochs(nnUNetTrainerTopoLossReweight):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
