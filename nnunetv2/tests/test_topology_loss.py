"""
Sanity tests for topology loss (soft-clDice) and related components.

Run with:
    python -m nnunetv2.tests.test_topology_loss

Tests
-----
1. Forward shape test (2D and 3D) — SoftClDiceLoss returns a scalar.
2. Gradient flow — gradients propagate through soft skeleton to input.
3. Perfect overlap — loss near 0 when pred == GT.
4. Absent class — TopologyLoss returns 0 when GT has no voxels for target.
5. Weight schedule — boundary values at warmup, plateau, and decay.
6. CE weight shape — RobustCrossEntropyLoss accepts a per-class weight tensor.
7. Dimension-agnostic — SoftSkeletonize works for both 4D and 5D tensors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.training.loss.topology_losses import (
    SoftClDiceLoss,
    SoftSkeletonize,
    TopologyLoss,
    topo_weight_schedule,
)
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


# ---- test 1: forward shape ----

def test_forward_shape():
    print("Test 1: Forward shape test (2D + 3D) ... ", end="")
    loss_fn = SoftClDiceLoss(num_iter=5, smooth=1.0)

    # 3D
    pred_3d = torch.sigmoid(torch.randn(2, 1, 16, 16, 16))
    gt_3d = (torch.randn(2, 1, 16, 16, 16) > 0).float()
    l3 = loss_fn(pred_3d, gt_3d)
    assert l3.ndim == 0, f"Expected scalar, got shape {l3.shape}"

    # 2D
    pred_2d = torch.sigmoid(torch.randn(2, 1, 32, 32))
    gt_2d = (torch.randn(2, 1, 32, 32) > 0).float()
    l2 = loss_fn(pred_2d, gt_2d)
    assert l2.ndim == 0, f"Expected scalar, got shape {l2.shape}"

    print(f"PASSED  (3D loss={l3.item():.4f}, 2D loss={l2.item():.4f})")


# ---- test 2: gradient flow ----

def test_gradient_flow():
    print("Test 2: Gradient flow test ... ", end="")
    loss_fn = SoftClDiceLoss(num_iter=5, smooth=1.0)

    logits = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    pred = torch.sigmoid(logits)
    gt = (torch.randn(2, 1, 16, 16, 16) > 0).float()

    loss = loss_fn(pred, gt)
    loss.backward()

    assert logits.grad is not None, "No gradient on input"
    assert logits.grad.abs().sum() > 0, "Zero gradient on input"
    print("PASSED")


# ---- test 3: perfect overlap ----

def test_perfect_overlap():
    print("Test 3: Perfect overlap — loss near 0 ... ", end="")
    loss_fn = SoftClDiceLoss(num_iter=5, smooth=1.0)

    # Create a simple "tube" that has interior structure
    gt = torch.zeros(1, 1, 32, 32, 32)
    gt[0, 0, 10:22, 14:18, 14:18] = 1.0  # a rectangular bar
    pred = gt.clone()

    loss = loss_fn(pred, gt)
    assert loss.item() < 0.1, f"Expected loss < 0.1 for perfect overlap, got {loss.item()}"
    print(f"PASSED  (loss={loss.item():.6f})")


# ---- test 4: absent class ----

def test_absent_class():
    print("Test 4: Absent class — TopologyLoss returns 0 ... ", end="")
    topo = TopologyLoss(topo_class_ids=[6, 7], num_iter=5)

    # 4 classes, none of them are 6 or 7
    logits = torch.randn(2, 8, 16, 16, 16)
    target = torch.zeros(2, 1, 16, 16, 16).long()  # all background (0)

    loss = topo(logits, target)
    assert loss.item() == 0.0, f"Expected 0.0 for absent classes, got {loss.item()}"
    # verify backward works
    loss.backward()
    print("PASSED")


# ---- test 5: weight schedule ----

def test_weight_schedule():
    print("Test 5: Weight schedule boundary values ... ", end="")
    num_epochs = 100
    warmup = 10
    decay_start = 30

    # epoch 0: should be near 0 (warmup start)
    w0 = topo_weight_schedule(0, num_epochs, warmup, decay_start, 1.0, 0.1)
    assert w0 == 0.0, f"Expected 0 at epoch 0, got {w0}"

    # epoch 5: midpoint of warmup
    w5 = topo_weight_schedule(5, num_epochs, warmup, decay_start, 1.0, 0.1)
    assert abs(w5 - 0.5) < 0.01, f"Expected ~0.5 at epoch 5, got {w5}"

    # epoch 10: end of warmup = plateau
    w10 = topo_weight_schedule(10, num_epochs, warmup, decay_start, 1.0, 0.1)
    assert abs(w10 - 1.0) < 0.01, f"Expected ~1.0 at epoch 10, got {w10}"

    # epoch 20: still plateau
    w20 = topo_weight_schedule(20, num_epochs, warmup, decay_start, 1.0, 0.1)
    assert abs(w20 - 1.0) < 0.01, f"Expected ~1.0 at epoch 20, got {w20}"

    # epoch 99: should be near w_low
    w99 = topo_weight_schedule(99, num_epochs, warmup, decay_start, 1.0, 0.1)
    assert w99 < 0.15, f"Expected near 0.1 at epoch 99, got {w99}"

    print(f"PASSED  (w0={w0:.2f}, w5={w5:.2f}, w10={w10:.2f}, w20={w20:.2f}, w99={w99:.4f})")


# ---- test 6: CE weight tensor ----

def test_ce_weight_tensor():
    print("Test 6: CE weight tensor shape ... ", end="")
    num_classes = 8
    weight = torch.ones(num_classes)
    weight[6] = 3.0  # AO
    weight[7] = 3.0  # PA

    ce = RobustCrossEntropyLoss(weight=weight)

    # dummy forward
    logits = torch.randn(2, num_classes, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, num_classes, (2, 1, 4, 4, 4)).float()

    loss = ce(logits, target)
    assert loss.ndim == 0, f"Expected scalar loss, got {loss.shape}"
    loss.backward()
    print(f"PASSED  (loss={loss.item():.4f})")


# ---- test 7: dimension-agnostic skeletonize ----

def test_skeletonize_dims():
    print("Test 7: Dimension-agnostic SoftSkeletonize ... ", end="")
    skel = SoftSkeletonize(num_iter=3)

    # 2D: (B, 1, H, W)
    x2d = torch.sigmoid(torch.randn(2, 1, 32, 32))
    s2d = skel(x2d)
    assert s2d.shape == x2d.shape, f"2D shape mismatch: {s2d.shape} vs {x2d.shape}"

    # 3D: (B, 1, D, H, W)
    x3d = torch.sigmoid(torch.randn(2, 1, 16, 16, 16))
    s3d = skel(x3d)
    assert s3d.shape == x3d.shape, f"3D shape mismatch: {s3d.shape} vs {x3d.shape}"

    # skeleton values should be non-negative (relu residuals)
    assert (s2d >= 0).all(), "2D skeleton has negative values"
    assert (s3d >= 0).all(), "3D skeleton has negative values"

    print(f"PASSED  (2D: {s2d.shape}, 3D: {s3d.shape})")


# ---- main ----

def main():
    print("=" * 60)
    print("Topology Loss Sanity Tests")
    print("=" * 60)
    test_forward_shape()
    test_gradient_flow()
    test_perfect_overlap()
    test_absent_class()
    test_weight_schedule()
    test_ce_weight_tensor()
    test_skeletonize_dims()
    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
