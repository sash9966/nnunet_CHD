"""
Topology-preserving losses for tubular structure segmentation.

Implements soft-clDice (Shit et al., CVPR 2021):
    "clDice - A Novel Topology-Preserving Loss Function for
     Tubular Structure Segmentation"

The soft-clDice loss encourages topological correctness by computing a
differentiable soft skeleton of both prediction and ground truth, then
penalising broken connectivity via skeleton precision and sensitivity.

All operations are pure PyTorch and dimension-agnostic (2D and 3D).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.helpers import softmax_helper_dim1


# ---------------------------------------------------------------------------
# Soft morphological skeleton
# ---------------------------------------------------------------------------

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Min-pool via negated max-pool (kernel 3, pad 1, stride 1)."""
    if x.ndim == 4:  # (B, 1, H, W)
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    else:  # (B, 1, D, H, W)
        return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Max-pool (kernel 3, pad 1, stride 1)."""
    if x.ndim == 4:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    else:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening = erode then dilate."""
    return _soft_dilate(_soft_erode(x))


class SoftSkeletonize(nn.Module):
    """Differentiable soft skeletonization via iterative morphological opening.

    At each iteration the residual between the current map and its opening
    (erode + dilate) captures thin structures that would be removed by
    opening.  Accumulating these residuals over decreasing scales produces
    an approximation of the morphological skeleton.

    Parameters
    ----------
    num_iter : int
        Number of erosion iterations.  Higher values capture thicker
        structures but increase compute.  Default 10 works well for most
        medical-image patch sizes.
    """

    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, *spatial)  soft probability map in [0, 1].

        Returns
        -------
        skeleton : (B, 1, *spatial)  soft skeleton approximation.
        """
        skeleton = torch.zeros_like(x)
        current = x.clone()
        for _ in range(self.num_iter):
            opened = _soft_open(current)
            # Residual: thin structures removed by opening
            skeleton = skeleton + F.relu(current - opened)
            current = _soft_erode(current)
        return skeleton


# ---------------------------------------------------------------------------
# Soft centreline Dice (clDice) — single class
# ---------------------------------------------------------------------------

class SoftClDiceLoss(nn.Module):
    """Soft centreline Dice loss for a single binary class.

    Computes skeleton precision and sensitivity, then returns ``1 - clDice``.

    Parameters
    ----------
    num_iter : int
        Skeletonization iterations (see :class:`SoftSkeletonize`).
    smooth : float
        Additive smoothing to avoid division by zero.
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.skeletonize = SoftSkeletonize(num_iter)
        self.smooth = smooth

    def forward(self, pred_soft: torch.Tensor, gt_soft: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_soft : (B, 1, *spatial) predicted soft probability in [0, 1].
        gt_soft   : (B, 1, *spatial) ground-truth binary mask (0 or 1).

        Returns
        -------
        Scalar loss = 1 - clDice.
        """
        skel_pred = self.skeletonize(pred_soft)

        # GT skeleton: no need for gradients (GT is fixed)
        with torch.no_grad():
            skel_gt = self.skeletonize(gt_soft)

        # Topology precision: how much of predicted skeleton lies inside GT
        tprec = ((skel_pred * gt_soft).sum() + self.smooth) / (skel_pred.sum() + self.smooth)

        # Topology sensitivity: how much of GT skeleton lies inside prediction
        tsens = ((skel_gt * pred_soft).sum() + self.smooth) / (skel_gt.sum() + self.smooth)

        # Centreline Dice (F1 of tprec and tsens; tiny eps to avoid 0/0)
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)

        return 1.0 - cl_dice


# ---------------------------------------------------------------------------
# Multi-class topology loss wrapper
# ---------------------------------------------------------------------------

class TopologyLoss(nn.Module):
    """Compute soft-clDice only on specified target classes.

    Designed for nnU-Net: accepts raw logits ``(B, C, *spatial)`` and
    integer label targets ``(B, 1, *spatial)``.

    If the ground truth contains **zero voxels** for a target class in
    the current batch, that class is skipped for that batch (degenerate
    skeleton).

    Parameters
    ----------
    topo_class_ids : list of int
        Label indices on which to compute the topology loss
        (e.g. ``[6, 7]`` for AO and PA).
    num_iter : int
        Skeletonization iterations.
    smooth : float
        Smoothing constant for clDice.
    """

    def __init__(
        self,
        topo_class_ids: List[int],
        num_iter: int = 10,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.topo_class_ids = topo_class_ids
        self.cldice = SoftClDiceLoss(num_iter=num_iter, smooth=smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        net_output : (B, C, *spatial) logits (pre-softmax).
        target     : (B, 1, *spatial) integer labels.

        Returns
        -------
        Scalar mean clDice loss over present target classes, or 0 if none.
        """
        probs = softmax_helper_dim1(net_output)  # (B, C, *spatial)
        losses = []

        for cid in self.topo_class_ids:
            if cid >= probs.shape[1]:
                continue  # class index out of range

            pred_c = probs[:, cid : cid + 1, ...]  # (B, 1, *spatial)
            gt_c = (target == cid).float()  # (B, 1, *spatial)

            # Skip if class is absent in this batch
            if gt_c.sum() == 0:
                continue

            losses.append(self.cldice(pred_c, gt_c))

        if len(losses) == 0:
            return torch.tensor(0.0, device=net_output.device, requires_grad=True)

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Topology weight schedule
# ---------------------------------------------------------------------------

def topo_weight_schedule(
    current_epoch: int,
    num_epochs: int,
    warmup_epochs: int = 10,
    decay_start_epoch: int = 30,
    w_high: float = 1.0,
    w_low: float = 0.1,
) -> float:
    """Compute the topology loss weight for a given epoch.

    Schedule
    --------
    1. **Warmup** (epoch 0 → warmup_epochs): linear ramp from 0 to w_high.
    2. **Plateau** (warmup_epochs → decay_start_epoch): constant w_high.
    3. **Cosine decay** (decay_start_epoch → num_epochs): cosine anneal
       from w_high down to w_low.

    Parameters
    ----------
    current_epoch : int
    num_epochs : int
        Total training epochs.
    warmup_epochs : int
    decay_start_epoch : int
    w_high : float
        Peak weight during plateau.
    w_low : float
        Minimum weight after full decay.

    Returns
    -------
    float — topology loss weight for this epoch.
    """
    if current_epoch < warmup_epochs:
        # Linear ramp-up
        return w_high * (current_epoch / max(warmup_epochs, 1))
    elif current_epoch < decay_start_epoch:
        # Plateau
        return w_high
    else:
        # Cosine decay
        progress = (current_epoch - decay_start_epoch) / max(num_epochs - decay_start_epoch, 1)
        progress = min(progress, 1.0)
        return w_low + 0.5 * (w_high - w_low) * (1.0 + math.cos(math.pi * progress))
