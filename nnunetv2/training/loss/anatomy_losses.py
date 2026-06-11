"""
Anatomy-aware auxiliary losses for CHD whole-heart segmentation.

These are *trainer-free* building blocks (they import only torch and the
existing soft-skeletonization utilities) so they can be unit-tested locally
without importing a full nnU-Net trainer (which is blocked by the
``acvl_utils.insert_crop_into_image`` version mismatch on dev machines).

Three losses, each consumed by a thin mixin:

* :class:`SoftRegionScaffoldLoss` — hierarchical region supervision derived
  from the existing multiclass labels (whole-heart, blood-pool, chambers,
  ventricles, atria, great-vessels, myocardium).  Computed on *soft* softmax
  probabilities (no argmax), so gradients flow.  Backs
  ``nnUNetTrainerDA5RegionScaffold``.

* :class:`BinaryVesselClDiceLoss` — soft-clDice on the merged great-vessel
  mask (AO ∪ PA) treated as a single connected structure, rather than
  per-class.  Reuses :class:`SoftClDiceLoss`.  Backs
  ``nnUNetTrainerDA5VesselFocusedTopo``.

* :class:`CenterlineWeightedCELoss` — cross-entropy up-weighted near the GT
  vessel centerline (skeleton).  The weight map is derived from the *target*
  and detached, so only the logits receive gradient.  Backs
  ``nnUNetTrainerDA5CenterlineAux``.

All operations are dimension-agnostic (2D and 3D) and operate on the
nnU-Net convention: logits ``(B, C, *spatial)`` + integer label targets
``(B, 1, *spatial)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.topology_losses import SoftSkeletonize, SoftClDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


# ---------------------------------------------------------------------------
# CHD label resolution (by name, never hardcoded indices)
# ---------------------------------------------------------------------------

# Canonical CHD structure -> accepted (lower-cased, stripped) label-name aliases.
_CHD_LABEL_ALIASES: Dict[str, set] = {
    "LV":  {"lv", "left ventricle", "leftventricle", "lv bloodpool",
            "lv blood pool", "left ventricle bloodpool"},
    "RV":  {"rv", "right ventricle", "rightventricle", "rv bloodpool",
            "rv blood pool", "right ventricle bloodpool"},
    "LA":  {"la", "left atrium", "leftatrium"},
    "RA":  {"ra", "right atrium", "rightatrium"},
    "Myo": {"myo", "myocardium", "lv myocardium", "lvmyo", "heart muscle"},
    "AO":  {"ao", "aorta"},
    "PA":  {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary trunk"},
}


def resolve_chd_label_ids(dataset_json: dict) -> Dict[str, int]:
    """Map canonical CHD structures to their integer label ids via dataset.json.

    Returns a dict like ``{"LV": 1, "RV": 2, ..., "PA": 7}`` containing only
    the structures that were actually found.  Matching is case-insensitive on
    the label *name* — indices are never hardcoded.
    """
    labels: Dict[str, int] = dataset_json.get("labels", {})
    resolved: Dict[str, int] = {}
    for name, idx in labels.items():
        key = str(name).lower().strip()
        if key == "background":
            continue
        for canonical, aliases in _CHD_LABEL_ALIASES.items():
            if key in aliases:
                resolved[canonical] = int(idx)
                break
    return resolved


def build_region_groups(label_ids: Dict[str, int]) -> Dict[str, List[int]]:
    """Build hierarchical region groups from resolved CHD label ids.

    A group is only included if *all* of its constituent structures were
    resolved, so partial label sets degrade gracefully.
    """
    def have(*names: str) -> bool:
        return all(n in label_ids for n in names)

    def ids(*names: str) -> List[int]:
        return sorted(label_ids[n] for n in names)

    groups: Dict[str, List[int]] = {}
    chambers = ("LV", "RV", "LA", "RA")
    all_fg = ("LV", "RV", "LA", "RA", "Myo", "AO", "PA")

    if have(*all_fg):
        groups["whole_heart"] = ids(*all_fg)
    if have("LV", "RV", "LA", "RA", "AO", "PA"):
        groups["blood_pool"] = ids("LV", "RV", "LA", "RA", "AO", "PA")
    if have(*chambers):
        groups["chambers"] = ids(*chambers)
    if have("LV", "RV"):
        groups["ventricles"] = ids("LV", "RV")
    if have("LA", "RA"):
        groups["atria"] = ids("LA", "RA")
    if have("AO", "PA"):
        groups["great_vessels"] = ids("AO", "PA")
    if have("Myo"):
        groups["myocardium"] = ids("Myo")
    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isin_labelmap(target: torch.Tensor, ids: List[int]) -> torch.Tensor:
    """Binary mask (float) of voxels whose label is in ``ids``.

    ``target`` is ``(B, 1, *spatial)`` integer labels.  Returned mask has the
    same shape and is in {0.0, 1.0}.
    """
    out = torch.zeros_like(target, dtype=torch.float32)
    for cid in ids:
        out = out + (target == cid).float()
    return out.clamp_(0.0, 1.0)


def _soft_dice_binary(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """1 - soft Dice for a single binary channel, averaged over the batch.

    ``p`` and ``gt`` are ``(B, 1, *spatial)``; ``p`` is a probability in [0, 1].
    """
    dims = tuple(range(2, p.ndim))  # spatial dims, keep batch separate
    num = 2.0 * (p * gt).sum(dim=dims) + smooth
    den = p.sum(dim=dims) + gt.sum(dim=dims) + smooth
    dice = num / den                      # (B, 1)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 1) Region-scaffold loss (soft, differentiable, no argmax)
# ---------------------------------------------------------------------------

class SoftRegionScaffoldLoss(nn.Module):
    """Hierarchical region supervision on soft probabilities.

    For each region group ``G`` (a set of label ids) the per-voxel region
    probability is ``sum_{c in G} softmax(logits)[:, c]`` and the region target
    is the binary union mask of those labels.  The loss is the mean over groups
    of ``softDice + BCE``.

    No argmax anywhere — the summed-softmax region probability is differentiable
    w.r.t. the logits.

    Parameters
    ----------
    region_groups : dict[str, list[int]]
        Region name -> list of label ids (see :func:`build_region_groups`).
    smooth : float
        Dice smoothing constant.
    """

    def __init__(self, region_groups: Dict[str, List[int]], smooth: float = 1e-5):
        super().__init__()
        self.region_groups = region_groups
        self.smooth = smooth
        # Diagnostics from the most recent forward (region name -> float loss).
        self.last_per_region_loss: Dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, *spatial) raw network output.
        target : (B, 1, *spatial) integer labels.

        Returns
        -------
        Scalar mean region loss (0 if no groups are configured).
        """
        if not self.region_groups:
            return torch.zeros((), device=logits.device, requires_grad=True)

        probs = softmax_helper_dim1(logits)                  # (B, C, *spatial)
        n_classes = probs.shape[1]
        self.last_per_region_loss = {}

        losses: List[torch.Tensor] = []
        for name, ids in self.region_groups.items():
            valid_ids = [c for c in ids if c < n_classes]
            if not valid_ids:
                continue
            p_region = probs[:, valid_ids].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            gt_region = _isin_labelmap(target, valid_ids)
            dice_l = _soft_dice_binary(p_region, gt_region, self.smooth)
            # Manual BCE in fp32: F.binary_cross_entropy is banned inside an
            # autocast region (the trainer calls extra losses under autocast).
            p32 = p_region.float().clamp(1e-6, 1.0 - 1e-6)
            g32 = gt_region.float()
            bce_l = -(g32 * torch.log(p32) + (1.0 - g32) * torch.log1p(-p32)).mean()
            region_l = dice_l + bce_l
            self.last_per_region_loss[name] = float(region_l.detach())
            losses.append(region_l)

        if not losses:
            return torch.zeros((), device=logits.device, requires_grad=True)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 2) Binary great-vessel clDice (AO ∪ PA as one structure)
# ---------------------------------------------------------------------------

class BinaryVesselClDiceLoss(nn.Module):
    """Soft-clDice on the merged great-vessel mask (AO ∪ PA).

    Treats the great vessels as a single connected tubular structure so the
    topology pressure targets *continuity* without forcing AO-vs-PA identity.
    Gradients flow through the predicted vessel probability; the GT skeleton is
    computed under ``no_grad`` inside :class:`SoftClDiceLoss`.

    Parameters
    ----------
    vessel_ids : list[int]
        Label ids of the great vessels (e.g. ``[6, 7]`` for AO, PA).
    num_iter : int
        Soft-skeletonization iterations.
    smooth : float
        clDice smoothing constant.
    """

    def __init__(self, vessel_ids: List[int], num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.vessel_ids = vessel_ids
        self.cldice = SoftClDiceLoss(num_iter=num_iter, smooth=smooth)
        self.last_present: bool = False
        self.last_loss: float = float("nan")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, *spatial) raw network output.
        target : (B, 1, *spatial) integer labels.

        Returns
        -------
        Scalar ``1 - clDice`` on the merged vessel mask, or 0 if absent.
        """
        self.last_present = False
        self.last_loss = float("nan")

        valid_ids = [c for c in self.vessel_ids if c < logits.shape[1]]
        if not valid_ids:
            return torch.zeros((), device=logits.device, requires_grad=True)

        probs = softmax_helper_dim1(logits)
        p_vessel = probs[:, valid_ids].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        gt_vessel = _isin_labelmap(target, valid_ids)

        if gt_vessel.sum() == 0:
            # No vessel voxels in this batch — degenerate skeleton, skip.
            return torch.zeros((), device=logits.device, requires_grad=True)

        self.last_present = True
        loss = self.cldice(p_vessel, gt_vessel)
        self.last_loss = float(loss.detach())
        return loss


# ---------------------------------------------------------------------------
# 3) Centerline-weighted cross-entropy
# ---------------------------------------------------------------------------

class CenterlineWeightedCELoss(nn.Module):
    """Cross-entropy up-weighted near the GT great-vessel centerline.

    The weight map ``1 + alpha * skeleton(GT vessel mask)`` is derived from the
    *target* and fully detached, so only the logits receive gradient.  This
    pushes the network to preserve thin vessel branches without changing the
    output head or label set.

    A precomputed soft-skeleton may be passed in (from an offline script); if
    ``None`` it is computed on the fly via :class:`SoftSkeletonize`.

    Parameters
    ----------
    vessel_ids : list[int]
        Label ids of the great vessels.
    alpha : float
        Extra weight on centerline voxels (weight = 1 + alpha at the skeleton).
    num_iter : int
        Soft-skeletonization iterations (on-the-fly path).
    ignore_label : int | None
        Optional label to exclude from the CE (weight 0 there).
    """

    def __init__(
        self,
        vessel_ids: List[int],
        alpha: float = 3.0,
        num_iter: int = 10,
        ignore_label: Optional[int] = None,
    ):
        super().__init__()
        self.vessel_ids = vessel_ids
        self.alpha = alpha
        self.skeletonize = SoftSkeletonize(num_iter)
        self.ignore_label = ignore_label
        self.last_weight_max: float = float("nan")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        precomputed_skeleton: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, *spatial) raw network output.
        target : (B, 1, *spatial) integer labels.
        precomputed_skeleton : (B, 1, *spatial) or None
            Optional detached soft skeleton in [0, 1].

        Returns
        -------
        Scalar centerline-weighted CE.
        """
        n_classes = logits.shape[1]
        valid_ids = [c for c in self.vessel_ids if c < n_classes]

        with torch.no_grad():
            if precomputed_skeleton is not None:
                skel = precomputed_skeleton.to(logits.device).clamp(0.0, 1.0)
            elif valid_ids:
                gt_vessel = _isin_labelmap(target, valid_ids)
                skel = self.skeletonize(gt_vessel).clamp(0.0, 1.0)
            else:
                skel = torch.zeros_like(target, dtype=torch.float32)
            weight = 1.0 + self.alpha * skel             # (B, 1, *spatial)

            # Build a valid integer target for CE; mask out ignore / OOB labels.
            tgt = target[:, 0].long()                    # (B, *spatial)
            valid = (tgt >= 0) & (tgt < n_classes)
            if self.ignore_label is not None:
                valid = valid & (tgt != self.ignore_label)
            tgt_safe = torch.where(valid, tgt, torch.zeros_like(tgt))
            w = weight[:, 0] * valid.float()             # (B, *spatial)
            self.last_weight_max = float(weight.max().detach())

        ce = F.cross_entropy(logits, tgt_safe, reduction="none")   # (B, *spatial)
        denom = w.sum().clamp_min(1.0)
        return (w * ce).sum() / denom


# ---------------------------------------------------------------------------
# Region-scaffold lambda schedule (step function)
# ---------------------------------------------------------------------------

def region_lambda_schedule(
    current_epoch: int,
    boundaries: Tuple[int, ...] = (100, 500),
    values: Tuple[float, ...] = (0.3, 0.15, 0.05),
) -> float:
    """Stepwise lambda for the region-scaffold loss.

    With the defaults: epochs [0,100) -> 0.3, [100,500) -> 0.15, [500,inf) -> 0.05.
    ``len(values)`` must be ``len(boundaries) + 1``.
    """
    assert len(values) == len(boundaries) + 1, "values must be one longer than boundaries"
    for i, b in enumerate(boundaries):
        if current_epoch < b:
            return values[i]
    return values[-1]
