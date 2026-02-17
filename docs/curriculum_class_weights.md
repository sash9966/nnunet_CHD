# Curriculum Class Weights

## What it does

Dynamically upweights specific target classes (aorta, pulmonary artery) in the **CrossEntropy loss** during the first P% of training epochs, then reverts to uniform weighting for the remainder of training.

This "curriculum" approach front-loads attention on tubular structures that are small but topologically important, without permanently biasing the loss.

## Which loss term is affected

- **CrossEntropy (CE) only.** The per-class weight vector is passed to `nn.CrossEntropyLoss(weight=...)`.
- **Dice loss is NOT weighted.** Dice already provides per-class balancing through its per-class formulation. Weighting both would be redundant and could destabilize training.
- Deep supervision: the same CE weight vector is used at all output resolutions (applied to the inner `DC_and_CE_loss` before the `DeepSupervisionWrapper`).

## How target classes are detected

Target classes are resolved **dynamically from `dataset.json` labels** — no hardcoded IDs.

Default behaviour matches these label names (case-insensitive):
- **Aorta**: `ao`, `aorta`
- **Pulmonary artery**: `pa`, `pulmonary`, `pulmonary artery`, `pulmonaryartery`, `pulmonary_artery`

You can override with custom names via the `curriculum_target_names` attribute:
```python
self.curriculum_target_names = ["LV", "RV"]  # upweight these instead
```

If no matching labels are found, a `ValueError` is raised at initialization with the list of available labels.

## Parameters and defaults

| Parameter | Default | Description |
|---|---|---|
| `curriculum_enabled` | `True` | Toggle feature on/off (False = exact baseline) |
| `curriculum_fraction` | `0.20` | Fraction of epochs with upweighting (P%) |
| `curriculum_multiplier` | `3.0` | CE weight multiplier for target classes |
| `curriculum_schedule` | `"step"` | Schedule type: `"step"` or `"linear_decay"` |
| `curriculum_target_names` | `None` | Custom label names to match; `None` = auto AO+PA |
| `curriculum_custom_weights` | `None` | Explicit per-class weight list (see below) |

### Custom per-class weights

Instead of auto-resolving target classes and applying a uniform multiplier, you can pass an explicit weight vector where each position corresponds to a class index:

```python
# Example: 8-class CHD dataset
# background=1, LV=1, RV=1, LA=1, RA=1, Myo=1, AO=5, PA=5
self.curriculum_custom_weights = [1, 1, 1, 1, 1, 1, 5, 5]
```

When `curriculum_custom_weights` is set:
- The vector is used as-is as the "peak" weights during the curriculum phase (epoch < T)
- After the curriculum fraction ends (epoch >= T), all weights revert to 1.0
- The `multiplier` and `target_ids` parameters are ignored
- Schedule types (`step` / `linear_decay`) still apply — `linear_decay` interpolates from your custom vector toward uniform

To use in a custom trainer subclass:
```python
class MyTrainer(CurriculumWeightsMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    def mixin_init(self):
        super().mixin_init()
        self.curriculum_custom_weights = [1, 1, 1, 1, 1, 1, 5, 5]
```

### Schedule types

Let `T = ceil(fraction * num_epochs)`.

**Step schedule** (`"step"`):
```
epoch < T  =>  weights[target_ids] = multiplier, others = 1
epoch >= T =>  all weights = 1
```

**Linear decay** (`"linear_decay"`):
```
epoch in [0, T)  =>  weights[target_ids] = multiplier - (multiplier-1) * epoch/(T-1)
epoch >= T       =>  all weights = 1
```

## How to enable it

### Standalone (DA5 + curriculum only)
```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5Curriculum_100epochs
```

### Combined with FiLM conditioning
```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5FiLMCurriculum_100epochs
```

### Combined with FiLM + topology loss
```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5FiLMTopoCurriculum_100epochs
```

### Baseline (feature disabled)
Any trainer without the `CurriculumWeightsMixin` in its MRO behaves as exact baseline:
```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDA5_100epochs
```

## Composability

This feature is implemented as a **mixin** (`CurriculumWeightsMixin`) and composes with other mixins:

- **No network changes** — only modifies the loss path.
- **Stacks with disease conditioning** (FiLM or MLP) — disease conditioning modifies the network forward pass, curriculum modifies loss weights. No conflict.
- **Stacks with topology loss** — topology loss adds a separate loss term (soft-clDice), curriculum weights the CE term. Both operate on loss, but on different components.
- **Works with any base trainer** (nnUNetTrainer, nnUNetTrainerDA5, etc.)

MRO ordering example:
```
DiseaseConditioningMixin → TopologyLossMixin → CurriculumWeightsMixin → ComposableTrainerMixin → nnUNetTrainerDA5
```

## Files

| File | Purpose |
|---|---|
| `nnunetv2/training/loss/curriculum_weights.py` | Helper functions: `resolve_target_class_ids()`, `get_curriculum_ce_weights()` |
| `nnunetv2/training/nnUNetTrainer/variants/mixins/curriculum_weights.py` | `CurriculumWeightsMixin` |
| `nnunetv2/training/nnUNetTrainer/variants/composed/nnUNetTrainerDA5Curriculum.py` | DA5 + curriculum |
| `nnunetv2/training/nnUNetTrainer/variants/composed/nnUNetTrainerDA5FiLMCurriculum.py` | DA5 + FiLM + curriculum |
| `nnunetv2/training/nnUNetTrainer/variants/composed/nnUNetTrainerDA5FiLMTopoCurriculum.py` | DA5 + FiLM + topo + curriculum |
| `scripts/test_curriculum_class_weights.py` | Sanity/test script |

## Testing

```bash
# With built-in dummy dataset.json:
python scripts/test_curriculum_class_weights.py

# With your real dataset.json:
python scripts/test_curriculum_class_weights.py --dataset_json /path/to/dataset.json

# Custom parameters:
python scripts/test_curriculum_class_weights.py --num_epochs 100 --fraction 0.2 --multiplier 3.0
```
