#!/usr/bin/env python3
"""
Sanity test for curriculum class-weight schedules.

Usage:
    # With a real dataset.json:
    python scripts/test_curriculum_class_weights.py --dataset_json /path/to/dataset.json

    # With a built-in dummy (CHD labels):
    python scripts/test_curriculum_class_weights.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ensure the repo root is on sys.path when running as a standalone script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import math
import sys

import torch

from nnunetv2.training.loss.curriculum_weights import (
    get_curriculum_ce_weights,
    resolve_target_class_ids,
)

# -------------------------------------------------------------------------
# Default dummy dataset.json with CHD labels
# -------------------------------------------------------------------------
DUMMY_DATASET_JSON = {
    "labels": {
        "background": 0,
        "LV": 1,
        "RV": 2,
        "LA": 3,
        "RA": 4,
        "AO": 5,
        "PA": 6,
        "WH": 7,
    }
}


def run_test(dataset_json: dict, num_epochs: int = 100, fraction: float = 0.2,
             multiplier: float = 3.0):
    print("=" * 60)
    print("Curriculum class-weight schedule test")
    print("=" * 60)

    # 1. Resolve target class IDs
    target_ids = resolve_target_class_ids(dataset_json)
    labels = dataset_json["labels"]
    id_to_name = {int(v): k for k, v in labels.items()}
    print(f"\nResolved target classes:")
    for cid in target_ids:
        print(f"  {id_to_name.get(cid, '???')} => class ID {cid}")

    num_classes = len(labels)
    T = math.ceil(fraction * num_epochs)
    print(f"\nParameters: num_epochs={num_epochs}, fraction={fraction}, T={T}, multiplier={multiplier}")
    print(f"Number of classes: {num_classes}")

    # 2. Test STEP schedule
    print(f"\n{'─' * 60}")
    print("Schedule: STEP")
    print(f"{'─' * 60}")
    test_epochs = [0, T - 1, T, num_epochs - 1]
    for epoch in test_epochs:
        w = get_curriculum_ce_weights(epoch, num_epochs, num_classes, target_ids,
                                      multiplier, fraction, "step")
        target_w = [w[cid].item() for cid in target_ids]
        other_w = w[0].item()  # background (representative of non-target)
        active = epoch < T
        print(f"  epoch {epoch:4d}: active={active}, target_weights={target_w}, other={other_w}")
        # assertions
        if epoch < T:
            for cid in target_ids:
                assert w[cid].item() == multiplier, f"epoch {epoch}: expected {multiplier}, got {w[cid].item()}"
        else:
            for cid in target_ids:
                assert w[cid].item() == 1.0, f"epoch {epoch}: expected 1.0, got {w[cid].item()}"
    print("  STEP schedule: PASSED")

    # 3. Test LINEAR_DECAY schedule
    print(f"\n{'─' * 60}")
    print("Schedule: LINEAR_DECAY")
    print(f"{'─' * 60}")
    for epoch in test_epochs:
        w = get_curriculum_ce_weights(epoch, num_epochs, num_classes, target_ids,
                                      multiplier, fraction, "linear_decay")
        target_w = [w[cid].item() for cid in target_ids]
        active = epoch < T
        print(f"  epoch {epoch:4d}: active={active}, target_weights=[{', '.join(f'{x:.2f}' for x in target_w)}]")
        # assertions
        if epoch >= T:
            for cid in target_ids:
                assert w[cid].item() == 1.0, f"epoch {epoch}: expected 1.0, got {w[cid].item()}"
        else:
            for cid in target_ids:
                assert w[cid].item() >= 1.0, f"epoch {epoch}: weight should be >= 1.0"
    # epoch 0 should be at full multiplier
    w0 = get_curriculum_ce_weights(0, num_epochs, num_classes, target_ids,
                                    multiplier, fraction, "linear_decay")
    assert w0[target_ids[0]].item() == multiplier, "epoch 0 linear_decay should start at multiplier"
    print("  LINEAR_DECAY schedule: PASSED")

    # 4. Verify baseline (disabled) — all weights = 1
    print(f"\n{'─' * 60}")
    print("Baseline check: all weights should be 1.0 when epoch >= T")
    print(f"{'─' * 60}")
    for schedule in ["step", "linear_decay"]:
        w = get_curriculum_ce_weights(T, num_epochs, num_classes, target_ids,
                                      multiplier, fraction, schedule)
        assert torch.allclose(w, torch.ones(num_classes)), f"Weights at epoch T should all be 1.0 ({schedule})"
        print(f"  {schedule}: epoch={T}, weights=all 1.0  PASSED")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Test curriculum class-weight schedules")
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="Path to dataset.json. If not provided, uses built-in dummy.")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--multiplier", type=float, default=3.0)
    args = parser.parse_args()

    if args.dataset_json:
        with open(args.dataset_json) as f:
            dataset_json = json.load(f)
        print(f"Loaded dataset.json from {args.dataset_json}")
    else:
        dataset_json = DUMMY_DATASET_JSON
        print("Using built-in dummy dataset.json (CHD labels)")

    run_test(dataset_json, args.num_epochs, args.fraction, args.multiplier)


if __name__ == "__main__":
    main()
