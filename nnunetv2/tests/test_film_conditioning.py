"""
Sanity tests for FiLM-conditioned ResidualEncoderUNet.

Run with:
    python -m nnunetv2.tests.test_film_conditioning

Tests
-----
1. Forward shape test – baseline and conditioned paths produce identical shapes.
2. Gradient test – disease_mlp and FiLM head parameters receive non-zero grads.
3. set_disease_vec inference API works correctly.
4. Zero-init identity – with all-zero disease_vec, output matches baseline.
5. Ablation smoke test – different disease vectors produce different outputs.
6. 2D forward test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet


def _make_network(
    ndim: int = 3,
    disease_K: int = 8,
    deep_supervision: bool = True,
) -> FiLMConditionedResEncUNet:
    """Instantiate a small FiLM-conditioned network for testing."""
    if ndim == 3:
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
    else:
        conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d

    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    nonlin = nn.LeakyReLU
    nonlin_kwargs = {"inplace": True}

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

    wrapped = FiLMConditionedResEncUNet(
        base,
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
        out_baseline = net(x, disease_vec=None)
        dv = torch.randint(0, 2, (B, K)).float()
        out_cond = net(x, disease_vec=dv)

    assert isinstance(out_baseline, list)
    assert isinstance(out_cond, list)
    assert len(out_baseline) == len(out_cond)
    for i, (ob, oc) in enumerate(zip(out_baseline, out_cond)):
        assert ob.shape == oc.shape, f"DS level {i}: {ob.shape} vs {oc.shape}"
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

    assert isinstance(out_baseline, torch.Tensor)
    assert isinstance(out_cond, torch.Tensor)
    assert out_baseline.shape == out_cond.shape
    print(f"PASSED  (shape: {out_baseline.shape})")


# ---- test 2: gradient test ----

def test_gradients_film_heads():
    print("Test 2a: Gradient test (FiLM heads at zero-init) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.train()
    x = torch.randn(B, 1, 32, 32, 32)
    dv = torch.randint(0, 2, (B, K)).float()

    out = net(x, disease_vec=dv)
    loss = out.sum()
    loss.backward()

    # FiLM head params get grads even at zero-init (d(loss)/d(gamma) = x, d(loss)/d(beta) = 1)
    for name, param in net.bottleneck_film.named_parameters():
        assert param.grad is not None, f"bottleneck_film.{name} has no grad"
        assert param.grad.abs().sum() > 0, f"bottleneck_film.{name} has zero grad"

    for i, film in enumerate(net.decoder_films):
        for name, param in film.named_parameters():
            assert param.grad is not None, f"decoder_films[{i}].{name} has no grad"
            assert param.grad.abs().sum() > 0, f"decoder_films[{i}].{name} has zero grad"

    # disease_mlp gets zero grads at zero-init because FiLM head weights are 0
    # (gradient is blocked: d(gamma)/d(e) = gamma_head.weight = 0).
    # This is expected; after one optimizer step moves FiLM heads from zero,
    # disease_mlp will receive gradients.
    print("PASSED  (FiLM heads have non-zero gradients)")


def test_gradients_full_flow():
    print("Test 2b: Gradient test (full flow with non-zero FiLM heads) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    # Simulate state after a few optimizer steps: give FiLM heads small non-zero weights
    for film in [net.bottleneck_film] + list(net.decoder_films):
        nn.init.normal_(film.gamma_head.weight, std=0.01)
        nn.init.normal_(film.beta_head.weight, std=0.01)
    net.train()
    x = torch.randn(B, 1, 32, 32, 32)
    dv = torch.randint(0, 2, (B, K)).float()

    out = net(x, disease_vec=dv)
    loss = out.sum()
    loss.backward()

    for name, param in net.disease_mlp.named_parameters():
        assert param.grad is not None, f"disease_mlp.{name} has no grad"
        assert param.grad.abs().sum() > 0, f"disease_mlp.{name} has zero grad"

    print("PASSED  (disease_mlp receives non-zero gradients)")


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
        net.set_disease_vec(dv)
        out_attr = net(x)

        net.clear_disease_vec()
        out_cleared = net(x)

        out_baseline = net(x, disease_vec=None)

    assert torch.allclose(out_cleared, out_baseline, atol=1e-6), (
        "After clear_disease_vec, output should match baseline"
    )
    print("PASSED")


# ---- test 4: zero-init identity ----

def test_zero_init_identity():
    print("Test 4: Zero-init identity (zeros disease_vec ≈ baseline) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.eval()
    x = torch.randn(B, 1, 32, 32, 32)

    with torch.no_grad():
        out_baseline = net(x, disease_vec=None)
        out_zeros = net(x, disease_vec=torch.zeros(B, K))

    # Because FiLM heads are zero-init and MLP maps zeros→zeros,
    # gamma=0, beta=0, so (1+0)*x + 0 = x.  Output should match baseline.
    assert torch.allclose(out_baseline, out_zeros, atol=1e-5), (
        f"Zero disease_vec output should match baseline. "
        f"Max diff: {(out_baseline - out_zeros).abs().max().item():.2e}"
    )
    print("PASSED")


# ---- test 5: ablation smoke test ----

def test_ablation_smoke():
    print("Test 5: Ablation smoke test (different disease_vecs → different outputs) ... ", end="")
    K = 8
    B = 2
    net = _make_network(ndim=3, disease_K=K, deep_supervision=False)
    net.eval()
    # use a trained-like state: randomize FiLM head weights
    for film in [net.bottleneck_film] + list(net.decoder_films):
        nn.init.normal_(film.gamma_head.weight, std=0.1)
        nn.init.normal_(film.beta_head.weight, std=0.1)

    x = torch.randn(B, 1, 32, 32, 32)
    dv_zeros = torch.zeros(B, K)
    dv_ones = torch.ones(B, K)
    dv_random = torch.randint(0, 2, (B, K)).float()

    with torch.no_grad():
        out_z = net(x, disease_vec=dv_zeros)
        out_o = net(x, disease_vec=dv_ones)
        out_r = net(x, disease_vec=dv_random)

    diff_zo = (out_z - out_o).abs().mean().item()
    diff_zr = (out_z - out_r).abs().mean().item()
    assert diff_zo > 1e-6, f"Zeros vs ones diff too small: {diff_zo:.2e}"
    assert diff_zr > 1e-6, f"Zeros vs random diff too small: {diff_zr:.2e}"
    print(f"PASSED  (mean abs diff: zeros-vs-ones={diff_zo:.4f}, zeros-vs-random={diff_zr:.4f})")


# ---- test 6: 2D forward ----

def test_2d_forward():
    print("Test 6: 2D forward test ... ", end="")
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


def main():
    print("=" * 60)
    print("FiLM Conditioning Sanity Tests")
    print("=" * 60)
    test_forward_shapes()
    test_forward_shapes_no_ds()
    test_gradients_film_heads()
    test_gradients_full_flow()
    test_set_disease_vec()
    test_zero_init_identity()
    test_ablation_smoke()
    test_2d_forward()
    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
