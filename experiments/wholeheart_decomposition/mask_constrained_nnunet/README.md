# Approach A — Mask-constrained multiclass nnU-Net

**Goal.** Train a fresh 7-class nnU-Net that receives both the CT and the
Stage-1 binary heart mask as **two input channels**, so the network's job
shrinks from "find the heart AND label it" to just "label the voxels inside
this mask". The topology of the heart is *given*; the network only solves
the semantic assignment problem.

## How to build it

1. **Predict the binary heart mask on every imagesTr case** (not just the
   ~7 fold-0 validation cases). Use `scripts/generate_cascade_preds.py` against
   the Dataset040 fullres or cascade model — it already supports this and
   writes `.b2nd` files compatible with the standard preprocessor.

2. **Create `Dataset041_ImageCHD_HU_MultiMask`** with:
   - `imagesTr/{case}_0000.nii.gz` — symlink to Dataset030's CT (channel 0)
   - `imagesTr/{case}_0001.nii.gz` — Stage-1 binary heart mask (channel 1)
   - `labelsTr/{case}.nii.gz` — Dataset030's original 7-class labels
   - `dataset.json` — `channel_names = {0: "CT", 1: "binary_heart_prior"}`,
     `labels` copied from Dataset030 (8-class including background)

3. **Run standard nnU-Net training**. nnU-Net consumes 2-channel input
   natively; the network's first conv layer just gets two input planes
   instead of one. No new trainer class needed for the basic case.

4. **Optional — constrain predictions to the binary mask at inference**.
   The simplest version: post-process by setting predicted labels to 0
   wherever the channel-1 input is 0. Cleaner: write a small inference
   wrapper that multiplies the softmax by the binary mask before argmax.

## What to vary in ablation

- **Hard prior vs. soft prior**: feed the binary mask as 0/1, or as a soft
  probability map (logits from the binary model). The soft version lets the
  network smoothly disagree where it's confident, which can help boundary
  voxels.
- **With and without topology loss (`TopologyLossMixin`)** on AO/PA. With
  the mask constraining the bulk of the heart, the topology loss can focus
  entirely on vessel continuity.
- **With and without disease conditioning (`FiLM`, `CrossAttn`)** — the
  Stage-2 ablation matrix from `CHD_Dataset030_ablation_*.sh` translates
  directly to this dataset.

## Files to add

This subfolder will eventually contain:
- `convert_to_mask_conditioned.py` — analogous to the whole-heart
  conversion script, but adds the binary mask as channel 1.
- `predict_inside_mask.py` — optional wrapper that intersects predictions
  with the binary input channel before saving.

## Why this is the default choice

It composes cleanly with everything we already have. The full ablation
matrix (`B1`–`C5`) can be re-run on Dataset041 with no new trainer code.
The binary heart mask carries far more information than a disease vector,
so the upside is large.
