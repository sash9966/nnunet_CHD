# Approach B — Mask post-processing (cheapest baseline)

**Goal.** Establish a zero-training baseline before committing to any
learned Stage-2 method.

Take the existing Dataset030 multiclass predictions, intersect them with
the Stage-1 binary heart mask, optionally re-assign voxels that fall
inside the binary mask but were labelled 0 by the multiclass model.

## Pipeline

```
binary_mask = load(Dataset040 prediction for case X)
mc_pred     = load(Dataset030 prediction for case X)

# Step 1: zero out multiclass predictions that fell outside the heart
constrained = mc_pred * binary_mask

# Step 2 (optional): for voxels in binary_mask where mc_pred == 0,
# fill with nearest-neighbour label from the constrained prediction
# (scipy.ndimage.distance_transform_edt + map_coordinates trick)
```

## When this beats the more complex methods

- Whenever the multiclass model's errors are dominated by **boundary leak**
  rather than **label flip-flop**. If predictions of Aorta voxels are
  mostly correct but bleed into the trachea, Approach B fixes it for free.
- As a sanity check: if Approach B reaches near-multiclass Dice with
  better `largest_component_fraction`, you have proof that Stage 1 helps
  even with the dumbest possible Stage 2.

## Files to add

- `apply_mask_to_multiclass.py` — pure post-processing CLI:
  ```bash
  python apply_mask_to_multiclass.py \
      --binary-dir   $nnUNet_results/Dataset040.../predictions_wholeheart/DA5_fullres \
      --mc-dir       $nnUNet_results/Dataset030.../predictions/DA5_fullres \
      --out          $nnUNet_results/Dataset040.../stage2_postprocess/
  ```

This is the only Stage-2 subfolder where the implementation is a
one-shot script — no training, no folder reshuffling. Ship it first as
a baseline.
