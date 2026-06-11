"""
Sanity tests for the anatomy-aware auxiliary losses (region scaffold,
binary great-vessel clDice, centerline-weighted CE).

These deliberately import ONLY ``nnunetv2.training.loss.anatomy_losses`` (which
in turn pulls in ``topology_losses`` and ``utilities.helpers``) — never a
trainer — so they run locally despite the ``acvl_utils`` import mismatch that
blocks full trainer import on dev machines.

Run with:
    python -m nnunetv2.tests.test_anatomy_losses
"""
from __future__ import annotations

import torch

from nnunetv2.training.loss.anatomy_losses import (
    BinaryVesselClDiceLoss,
    CenterlineWeightedCELoss,
    SoftRegionScaffoldLoss,
    build_region_groups,
    region_lambda_schedule,
    resolve_chd_label_ids,
)


# Canonical CHD dataset.json labels (LV=1 ... PA=7).
_DATASET_JSON = {
    "labels": {
        "background": 0, "LV": 1, "RV": 2, "LA": 3,
        "RA": 4, "Myo": 5, "AO": 6, "PA": 7,
    }
}


def _bar_target(B=2, C=8, shape=(24, 24, 24)):
    """Integer label map with a couple of solid structures incl. AO/PA bars."""
    t = torch.zeros(B, 1, *shape, dtype=torch.long)
    t[:, 0, 4:20, 6:10, 6:10] = 6      # AO bar
    t[:, 0, 4:20, 14:18, 14:18] = 7    # PA bar
    t[:, 0, 6:18, 6:18, 18:22] = 1     # LV blob
    return t


def _logits_favoring(target, C=8, strength=8.0):
    """Logits that strongly predict the GT label at each voxel (low loss case)."""
    B = target.shape[0]
    logits = torch.zeros(B, C, *target.shape[2:])
    onehot = torch.zeros(B, C, *target.shape[2:])
    onehot.scatter_(1, target, 1.0)
    return strength * onehot + 0.01 * torch.randn_like(logits)


# ---- test 1: label resolution + region groups ----

def test_label_resolution():
    print("Test 1: CHD label resolution + region groups ... ", end="")
    ids = resolve_chd_label_ids(_DATASET_JSON)
    assert ids == {"LV": 1, "RV": 2, "LA": 3, "RA": 4, "Myo": 5, "AO": 6, "PA": 7}, ids
    groups = build_region_groups(ids)
    assert groups["whole_heart"] == [1, 2, 3, 4, 5, 6, 7]
    assert groups["blood_pool"] == [1, 2, 3, 4, 6, 7]      # all minus Myo
    assert groups["great_vessels"] == [6, 7]
    assert groups["ventricles"] == [1, 2]
    assert groups["atria"] == [3, 4]
    assert groups["myocardium"] == [5]
    print(f"PASSED  ({len(groups)} groups resolved)")


# ---- test 2: region loss shape + grad + ordering ----

def test_region_scaffold_loss():
    print("Test 2: Region-scaffold loss shape/grad/ordering ... ", end="")
    target = _bar_target()
    groups = build_region_groups(resolve_chd_label_ids(_DATASET_JSON))
    loss_fn = SoftRegionScaffoldLoss(groups)

    # good prediction
    good = _logits_favoring(target).requires_grad_(True)
    l_good = loss_fn(good, target)
    assert l_good.ndim == 0, "region loss must be scalar"
    l_good.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all(), "bad grad"
    assert len(loss_fn.last_per_region_loss) == len(groups), "missing per-region diag"

    # random prediction should be worse than the confident-correct one
    rand = torch.randn_like(good)
    l_rand = loss_fn(rand, target)
    assert l_rand.item() > l_good.item(), \
        f"expected random({l_rand.item():.3f}) > good({l_good.item():.3f})"
    print(f"PASSED  (good={l_good.item():.3f} < random={l_rand.item():.3f})")


# ---- test 3: binary vessel clDice ----

def test_vessel_cldice():
    print("Test 3: Binary great-vessel clDice ... ", end="")
    target = _bar_target()
    loss_fn = BinaryVesselClDiceLoss(vessel_ids=[6, 7], num_iter=5)

    good = _logits_favoring(target).requires_grad_(True)
    l_good = loss_fn(good, target)
    assert loss_fn.last_present, "vessel should be present"
    l_good.backward()
    assert good.grad is not None and good.grad.abs().sum() > 0, "no grad through vessel clDice"

    rand = torch.randn_like(good)
    l_rand = loss_fn(rand, target)
    assert l_rand.item() > l_good.item(), \
        f"expected random({l_rand.item():.3f}) > good({l_good.item():.3f})"

    # absent vessels -> zero loss
    empty = torch.zeros_like(target)
    l_empty = loss_fn(_logits_favoring(empty), empty)
    assert l_empty.item() == 0.0 and not loss_fn.last_present, "absent vessel must give 0"
    print(f"PASSED  (good={l_good.item():.3f} < random={l_rand.item():.3f}, absent=0)")


# ---- test 4: centerline-weighted CE ----

def test_centerline_ce():
    print("Test 4: Centerline-weighted CE weight map + grad ... ", end="")
    target = _bar_target()
    loss_fn = CenterlineWeightedCELoss(vessel_ids=[6, 7], alpha=3.0, num_iter=5)

    logits = _logits_favoring(target).requires_grad_(True)
    loss = loss_fn(logits, target)
    assert loss.ndim == 0, "centerline CE must be scalar"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "bad grad"

    # weight map must exceed 1 somewhere (on the skeleton) and never below 1
    assert loss_fn.last_weight_max > 1.0, \
        f"expected centerline up-weighting (>1), got {loss_fn.last_weight_max}"
    # confident-correct logits should give lower CE than random
    rand = torch.randn_like(logits)
    l_rand = loss_fn(rand, target)
    assert l_rand.item() > loss.item(), \
        f"expected random({l_rand.item():.3f}) > good({loss.item():.3f})"
    print(f"PASSED  (w_max={loss_fn.last_weight_max:.2f}, "
          f"good={loss.item():.3f} < random={l_rand.item():.3f})")


# ---- test 5: lambda schedule ----

def test_lambda_schedule():
    print("Test 5: Region lambda step schedule ... ", end="")
    assert region_lambda_schedule(0) == 0.3
    assert region_lambda_schedule(99) == 0.3
    assert region_lambda_schedule(100) == 0.15
    assert region_lambda_schedule(499) == 0.15
    assert region_lambda_schedule(500) == 0.05
    assert region_lambda_schedule(999) == 0.05
    print("PASSED  (0.3 -> 0.15 -> 0.05)")


# ---- test 6: graceful degradation on partial labels ----

def test_partial_labels():
    print("Test 6: Graceful degradation w/ partial labels ... ", end="")
    partial = {"labels": {"background": 0, "AO": 1, "PA": 2}}
    ids = resolve_chd_label_ids(partial)
    assert ids == {"AO": 1, "PA": 2}, ids
    groups = build_region_groups(ids)
    # only great_vessels group can be formed
    assert "great_vessels" in groups and groups["great_vessels"] == [1, 2]
    assert "chambers" not in groups and "whole_heart" not in groups
    print(f"PASSED  (groups={list(groups)})")


# ---- test 7: autocast safety (extra losses run under autocast in train_step) ----

def test_autocast_safety():
    print("Test 7: Autocast safety (all 3 losses under autocast) ... ", end="")
    target = _bar_target()
    groups = build_region_groups(resolve_chd_label_ids(_DATASET_JSON))
    region = SoftRegionScaffoldLoss(groups)
    vessel = BinaryVesselClDiceLoss(vessel_ids=[6, 7], num_iter=5)
    center = CenterlineWeightedCELoss(vessel_ids=[6, 7], alpha=3.0, num_iter=5)
    logits = _logits_favoring(target)
    # bfloat16 CPU autocast mirrors the trainer's GPU autocast region.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        lr = region(logits, target)
        lv = vessel(logits, target)
        lc = center(logits, target)
    for name, val in (("region", lr), ("vessel", lv), ("centerline", lc)):
        assert torch.isfinite(val), f"{name} loss non-finite under autocast: {val}"
    print(f"PASSED  (region={float(lr):.3f}, vessel={float(lv):.3f}, center={float(lc):.3f})")


def main():
    print("=" * 60)
    print("Anatomy-aware Loss Sanity Tests")
    print("=" * 60)
    test_label_resolution()
    test_region_scaffold_loss()
    test_vessel_cldice()
    test_centerline_ce()
    test_lambda_schedule()
    test_partial_labels()
    test_autocast_safety()
    print("=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
