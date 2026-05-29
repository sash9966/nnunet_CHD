# Approach E — MedSAM / SAM / 3D Slicer interactive

> Anatomy + HITL spec (box-prompted correction, centerline seeds,
> active-learning capture format) lives in
> [`../STAGE2_SPEC.md`](../STAGE2_SPEC.md) §5. MedSAM proposals are
> **correction candidates, not ground truth** — they go through the same
> capture format as Segment Editor corrections.

**Goal.** Use a promptable foundation model (MedSAM, SAM-Med3D, MONAI
SAM-style) to label individual chambers and vessels inside the binary
mask given a single click / scribble per structure.

This is *not* aiming for fully automatic decomposition. It is aiming for
the fastest possible human-in-the-loop pipeline for clinical use.

## Two integration paths

### 1.  MONAI Label server, queried by Slicer

```
Slicer ⇄ MONAI Label server ⇄ MedSAM checkpoint
```

The user loads a CT + the Stage-1 binary mask in Slicer's Segment Editor,
clicks one seed point per chamber/vessel, and MONAI Label returns the
refined segmentation for that prompt.

Pros: zero local installation beyond Slicer + a server. Standard MONAI
workflow.

### 2.  Standalone batch inference

Run MedSAM offline against the test set with pre-defined seed points
extracted from the rule-based chamber detection (Approach D). No human
in the loop, but uses the foundation model's prior over anatomical
shapes.

## What to build first

A small script that takes:
- a CT
- a binary mask
- a 7-row CSV of seed points `(class_id, x, y, z)`

…and emits a 7-class NIfTI via MedSAM. Once that works end-to-end on
one case, the Slicer integration is a packaging exercise.

## Files to add

- `medsam_inference.py` — wraps MedSAM with seed-point input.
- `slicer_integration/` — MONAI Label config + module manifest.
- `seed_extraction.py` — derives seed points from the rule-based
  chamber detection (Approach D).

## Why this is interesting

- It's the only approach that scales to **out-of-distribution
  pathologies** where there isn't enough training data to learn a
  classifier — the user just clicks.
- MedSAM is pretrained on a large medical corpus, so it has stronger
  priors over anatomical shape than a 73-case fine-tune ever can.

## Why it might not work

- MedSAM is 2D + heuristic-3D under the hood; for thin tubular structures
  (PA branches) the per-slice consistency is rough. SAM-Med3D is better
  but heavier.
- Adds a runtime dependency (server or large checkpoint) that complicates
  reproducibility.
