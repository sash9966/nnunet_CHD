# Approach F — DINO / pretrained-transformer features inside the mask

**Goal.** Replace task-specific encoder training with frozen pretrained
features and learn only a lightweight classification head per voxel
(or per branch, paired with Approach C).

## Why this might help

The dataset is small (~73 cases). Self-supervised pretrained models
(DINOv2, MAE-CT, MedCLIP variants) carry strong priors that small
medical datasets can't reach. The Stage-1 binary mask gives us *where*
to look; DINO features give us *what we're looking at*.

## Pipeline sketch

```
1. Project each axial slice of the CT through a DINO backbone, at the
   resolution the backbone expects (224×224 typically). Stack the
   per-slice token features into a 3-D feature volume.
2. Within the binary heart mask, sample per-voxel features.
3. Train a small MLP / per-class linear classifier on (feature_vector → 7 classes)
   using the Dataset030 GT labels (only voxels inside the binary mask).
4. At inference: same forward pass + small head, then majority-vote per
   connected component for label stability.
```

## Key choices

- **2D vs 3D backbone**: 3D backbones (SAM-Med3D, M3D) preserve through-plane
  information; 2D backbones (DINOv2) are well-pretrained on natural images
  but lose 3-D context. For CHD, 3D is preferred.
- **Feature resolution**: foundation models output coarse feature maps.
  Upsample with bilinear/bicubic before the per-voxel classifier — or
  classify at the coarse resolution and upsample the *labels*.
- **Freezing**: keep the backbone frozen for the first ablation. Fine-tuning
  on a 73-case dataset is unstable.

## Files to add

- `extract_dino_features.py` — runs the backbone, dumps `.npy` per case.
- `train_voxel_classifier.py` — small head training loop.
- `decode_predictions.py` — combines features + head + binary mask → 7-class NIfTI.

## When to favour this

When the conclusion from Approaches A/B is that Stage-2 errors are
dominated by **semantic confusion** rather than **boundary error**, and
when the dataset is too small for a fresh nnU-Net to learn the
discriminative features it needs.

## Honest risk

DINO was not trained on cardiac CT specifically — its features may be
poorly aligned to chamber/vessel discrimination. Run a small linear-probe
sanity check (per-voxel logistic regression on the frozen features) before
investing in the full pipeline.
