# Topology Loss + Per-Class Progress Plotting

Branch: `TopologyLossPlusPerClassProgress`

## Overview

This branch adds two features to nnU-Net v2:

1. **Soft-clDice topology loss** targeting aorta (AO) and pulmonary artery (PA)
   to encourage topological connectivity of tubular structures.
2. **Per-class validation Dice in progress.png** — a 4th subplot showing
   per-class pseudo Dice curves over training epochs (works for ALL trainers,
   not just topology variants).

## Topology Loss: Soft-clDice

### Citation

> Shit et al., "clDice — A Novel Topology-Preserving Loss Function for
> Tubular Structure Segmentation", CVPR 2021.

### Algorithm

The soft-clDice loss computes differentiable soft skeletons of both
prediction and ground truth, then measures overlap:

1. **Soft skeletonization** via iterative morphological opening residuals:
   - `eroded = min_pool(current, k=3)`
   - `opened = max_pool(eroded, k=3)` (opening = erosion + dilation)
   - `skeleton += relu(current - opened)` (thin structures removed by opening)
   - `current = eroded` (continue eroding)
   - Repeat for `num_iter` iterations (default: 10)

2. **Skeleton precision and sensitivity:**
   - `tprec = sum(skel_pred * gt) / sum(skel_pred)` — predicted skeleton inside GT
   - `tsens = sum(skel_gt * pred) / sum(skel_gt)` — GT skeleton inside prediction

3. **Loss = 1 - clDice**, where `clDice = 2 * tprec * tsens / (tprec + tsens)`

### Label Resolution

AO and PA class indices are resolved **dynamically** from `dataset.json`:

```json
{
  "labels": {
    "background": 0,
    "LV": 1, "RV": 2, "LA": 3, "RA": 4, "Myo": 5,
    "AO": 6,    <-- matched by keyword "ao" or "aorta"
    "PA": 7     <-- matched by keyword "pa" or "pulmonary"
  }
}
```

If no matching labels are found, topology loss is silently disabled (baseline behaviour).

### Weight Schedule

The topology loss weight follows a warmup → plateau → cosine decay schedule:

| Phase | Epochs | Weight |
|-------|--------|--------|
| Warmup | 0 → 10 | Linear ramp 0 → 1.0 |
| Plateau | 10 → 30 | Constant 1.0 |
| Cosine decay | 30 → end | Cosine anneal 1.0 → 0.1 |

Total loss: `L = L_base + w_topo(epoch) * L_topo`

Topology loss is applied **only to the full-resolution output** (not deep supervision levels)
because skeletonization on downsampled volumes loses thin-structure detail.

### Hyperparameters

All configurable via trainer class attributes:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `topo_warmup_epochs` | 10 | Warmup period |
| `topo_decay_start_epoch` | 30 | Start of cosine decay |
| `topo_w_high` | 1.0 | Peak weight |
| `topo_w_low` | 0.1 | Minimum weight |
| `topo_num_iter` | 10 | Skeletonization iterations |

## Two Trainer Variants

### Variant A: `nnUNetTrainerTopoLoss` — Topology Loss Only

Standard Dice+CE loss plus soft-clDice on AO/PA. No class reweighting.

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerTopoLoss
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerTopoLoss_100epochs
```

### Variant B: `nnUNetTrainerTopoLossReweight` — Topology + Early Reweighting

Same as Variant A, plus CE class reweighting during the first 20% of epochs:

- **Epochs 0 to 20%**: CE weight for AO and PA is 3x (other classes remain 1x)
- **Epochs 20% to end**: CE weights revert to uniform (all 1x)

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerTopoLossReweight
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerTopoLossReweight_100epochs
```

**Why CE only (not Dice)?** The Dice loss already computes per-class Dice which
naturally balances class contributions. CE reweighting is standard, well-understood,
and simpler to implement correctly.

| Attribute | Default | Description |
|-----------|---------|-------------|
| `ce_reweight_fraction` | 0.20 | Fraction of epochs with reweighting |
| `ce_topo_class_multiplier` | 3.0 | CE weight for AO/PA during reweight phase |

## Per-Class Progress Plot

The `progress.png` file now has **4 subplots** (previously 3):

1. Training/validation loss + mean pseudo Dice (unchanged)
2. Epoch duration (unchanged)
3. Learning rate (unchanged)
4. **Per-class validation pseudo Dice** (NEW)

The 4th subplot shows one curve per foreground class. When the topology
trainers are used, AO and PA are highlighted with thicker solid lines.
When label names are available from the trainer, the legend uses class names;
otherwise generic "Class 1", "Class 2", etc.

**This works for ALL trainers**, not just topology variants. The per-class
Dice data was already logged by the base nnUNetTrainer — only the plotting
was missing.

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `nnunetv2/training/loss/topology_losses.py` | NEW | `SoftSkeletonize`, `SoftClDiceLoss`, `TopologyLoss`, `topo_weight_schedule` |
| `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerTopoLoss.py` | NEW | Both trainer variants + 100-epoch subclasses |
| `nnunetv2/training/logging/nnunet_logger.py` | MODIFIED | Added per-class Dice subplot (4th panel) |
| `nnunetv2/tests/test_topology_loss.py` | NEW | 7 sanity tests |
| `docs/topology_loss_and_per_class_progress.md` | NEW | This file |

## Sanity Tests

```bash
python -m nnunetv2.tests.test_topology_loss
```

Tests:
1. Forward shape (2D + 3D)
2. Gradient flow through soft skeleton
3. Perfect overlap gives loss near 0
4. Absent class handled gracefully (returns 0)
5. Weight schedule boundary values
6. CE weight tensor accepted by loss
7. Dimension-agnostic skeletonization

## Checkpoint Compatibility

- `TopologyLoss` has **no learned parameters** — checkpoints are identical
  to base nnUNetTrainer (same network weights).
- Resuming from a base trainer checkpoint with a topology trainer works
  seamlessly. The topology weight schedule restarts from the current epoch.
- The logger change is backward-compatible: old checkpoints without
  `label_names`/`topo_class_ids` still produce valid plots with generic labels.

## Absent-Class Handling

If the ground truth in a batch contains **zero voxels** for a topology target
class (e.g. PA not present in that particular sample), the class is skipped
for that batch — no degenerate skeleton is computed. If all topology classes
are absent, the topology loss contribution is 0 for that batch.
