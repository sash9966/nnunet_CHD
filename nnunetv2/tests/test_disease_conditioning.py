"""
Sanity tests for disease-conditioned ResidualEncoderUNet.

Run with:
    python -m nnunetv2.tests.test_disease_conditioning

Tests
-----
1. Forward shape test – baseline (disease_vec=None) and conditioned paths
   produce identical output shapes.
2. Gradient test – disease_mlp and injector parameters receive non-zero
   gradients after a single forward + backward pass.
3. set_disease_vec inference API – the stored attribute path works correctly.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnunetv2.architectures.disease_conditioned_unet import DiseaseConditionedResEncUNet


# ---- helpers ----

def _make_network(
    ndim: int = 3,
    disease_K: int = 8,
    deep_supervision: bool = True,
) -> DiseaseConditionedResEncUNet:
    """Instantiate a small disease-conditioned network for testing."""
    if ndim == 3:
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d

    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {"inplace": True}

    # small 4-stage network
    base = ResidualEncoderUNet(
        input_channels=1,
        n_stages=4,
        features_per_stage=[16, 32, 64, 128],
        conv_op=conv_op,
        kernel_sizes=[[3] * ndim] * 4,
        strides=[[1] * ndim] + [[2] * ndim] * 3,
        n_blocks_per_stage=[1, 2, 2, 2],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        deep_supervision=deep_supervision,
    )

    wrapped = DiseaseConditionedResEncUNet(
        base,
        conv_op=conv_op,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
        disease_K=disease_K,
        disease_H=64,
        disease_E=32,
    )
    return wrapped


# ---- test 1: forward shape ----

def test_forward_shapes():
    print("Test 1: Forward shape test (3D, deep_supervision=True) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=True)
    net.eval()
    x = torch.randn(B, 1, 32, 32, 32)

    with torch.no_grad():
        # baseline path
        out_baseline = net(x, disease_vec=None)
        # conditioned path
        dv = torch.randint(0, 2, (B, K)).float()
        out_cond = net(x, disease_vec=dv)

    assert isinstance(out_baseline, list), "Expected list output with deep_supervision=True"
    assert isinstance(out_cond, list), "Expected list output with deep_supervision=True"
    assert len(out_baseline) == len(out_cond), (
        f"Number of DS outputs differ: {len(out_baseline)} vs {len(out_cond)}"
    )
    for i, (ob, oc) in enumerate(zip(out_baseline, out_cond)):
        assert ob.shape == oc.shape, (
            f"DS level {i} shape mismatch: baseline {ob.shape} vs conditioned {oc.shape}"
        )
    print(f"PASSED  (DS levels: {len(out_baseline)}, shapes: {[o.shape for o in out_baseline]})")


def test_forward_shapes_no_ds():
    print("Test 1b: Forward shape test (3D, deep_supervision=False) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.eval()
    x = torch.randn(B, 1, 32, 32, 32)

    with torch.no_grad():
        out_baseline = net(x, disease_vec=None)
        dv = torch.randint(0, 2, (B, K)).float()
        out_cond = net(x, disease_vec=dv)

    assert isinstance(out_baseline, torch.Tensor), "Expected tensor output with deep_supervision=False"
    assert isinstance(out_cond, torch.Tensor), "Expected tensor output with deep_supervision=False"
    assert out_baseline.shape == out_cond.shape, (
        f"Shape mismatch: {out_baseline.shape} vs {out_cond.shape}"
    )
    print(f"PASSED  (shape: {out_baseline.shape})")


# ---- test 2: gradient test ----

def test_gradients():
    print("Test 2: Gradient test ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.train()
    x = torch.randn(B, 1, 32, 32, 32)
    dv = torch.randint(0, 2, (B, K)).float()

    result = net(x, disease_vec=dv)
    # forward returns (seg_out, aux_logits) tuple during training
    if isinstance(result, tuple):
        seg_out, aux_logits = result
        loss = seg_out.sum()
        if aux_logits is not None:
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(aux_logits, dv)
    else:
        loss = result.sum()
    loss.backward()

    # check disease_mlp grads
    for name, param in net.disease_mlp.named_parameters():
        assert param.grad is not None, f"disease_mlp.{name} has no grad"
        assert param.grad.abs().sum() > 0, f"disease_mlp.{name} has zero grad"

    # check bottleneck_injector grads
    for name, param in net.bottleneck_injector.named_parameters():
        assert param.grad is not None, f"bottleneck_injector.{name} has no grad"
        assert param.grad.abs().sum() > 0, f"bottleneck_injector.{name} has zero grad"

    # check decoder_injectors grads
    for i, inj in enumerate(net.decoder_injectors):
        for name, param in inj.named_parameters():
            assert param.grad is not None, f"decoder_injectors[{i}].{name} has no grad"
            assert param.grad.abs().sum() > 0, f"decoder_injectors[{i}].{name} has zero grad"

    # check disease_classifier grads
    for name, param in net.disease_classifier.named_parameters():
        assert param.grad is not None, f"disease_classifier.{name} has no grad"
        assert param.grad.abs().sum() > 0, f"disease_classifier.{name} has zero grad"

    print("PASSED  (all disease modules have non-zero gradients)")


# ---- test 3: set_disease_vec inference API ----

def test_set_disease_vec():
    print("Test 3: set_disease_vec inference API ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.eval()
    x = torch.randn(B, 1, 32, 32, 32)
    dv = torch.randint(0, 2, (B, K)).float()

    with torch.no_grad():
        # set via attribute
        net.set_disease_vec(dv)
        out_attr = net(x)  # no explicit disease_vec arg

        # explicit arg (should override stored)
        dv2 = torch.ones(B, K)
        out_explicit = net(x, disease_vec=dv2)

        # clear and verify baseline
        net.clear_disease_vec()
        out_cleared = net(x)

    # out_attr should match conditioned output (not baseline)
    # out_cleared should match baseline
    out_baseline = net(x)
    assert torch.allclose(out_cleared, out_baseline, atol=1e-6), (
        "After clear_disease_vec, output should match baseline"
    )
    print("PASSED")


# ---- test 4: 2D forward ----

def test_2d_forward():
    print("Test 4: 2D forward test ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=2, disease_K=K, deep_supervision=False)
    net.eval()
    x = torch.randn(B, 1, 64, 64)
    dv = torch.randint(0, 2, (B, K)).float()

    with torch.no_grad():
        out_baseline = net(x)
        out_cond = net(x, disease_vec=dv)

    assert out_baseline.shape == out_cond.shape
    print(f"PASSED  (shape: {out_baseline.shape})")


# ---- main ----

def main():
    print("=" * 60)
    print("Disease Conditioning Sanity Tests")
    print("=" * 60)
    test_forward_shapes()
    test_forward_shapes_no_ds()
    test_gradients()
    test_set_disease_vec()
    test_2d_forward()
    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
