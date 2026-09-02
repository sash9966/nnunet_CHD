# Promptable Refinement Flywheel — Methods (paper draft)

Paper-ready, step-by-step account of the semi-supervised segmentation flywheel and its promptable
refinement stage, the scientific work it is grounded on, and the methods we derived. Companion to the
design doc `docs/promptable_refinement.md` and the clinical-flywheel record `docs/CONVENTIONS.md`.
Reproducibility: every run stamps its git commit + params (`scripts/_provenance.sh` → `logs/run_manifest.jsonl`).

## 1. Problem & motivation
Congenital-heart-disease (CHD) whole-heart CT segmentation (7 classes: LV-BP, RV-BP, LA, RA, Myo,
Aorta, Pulmonary) trained on the public **ImageCHD** cohort does not transfer directly to our clinical
CT: the clinical scans differ in **resolution, field-of-view, and voxel spacing**. Direct native-spacing
inference with the ImageCHD model fails (chambers mislabeled); the failure is one of **input scale/grid
presentation, not model weights**. Manual annotation of clinical CHD CT is expensive and scarce.

Goal: build a **semi-supervised flywheel** that (a) obtains usable clinical labels cheaply, (b) refines
them to native resolution/FOV without interpolation artifacts, and (c) recycles the good ones into
training — with expert sign-off gating each cycle.

## 2. Scientific grounding (prior work)
- **nnU-Net** (Isensee et al., *Nature Methods* 2021) — self-configuring segmentation baseline; our base
  and vessel models are nnU-Net (ResEnc-M plans, DA5 augmentation).
- **Pseudo-labeling / self-training** — standard semi-supervised recipe: predict → keep confident/expert-
  verified labels → retrain. We add per-structure QC and largest-connected-component (LCC) cleanup.
- **DeepEdit** (Diaz-Pinto et al., 2023, arXiv:2305.10655) — deep *editable* interactive 3D segmentation
  (initial prediction + click refinement, MONAI Label). The conceptual template for our "seed → refine".
- **Promptable / interactive 3D foundation models** — **nnInteractive** (Isensee et al., 2025; points/
  scribbles/box/**lasso**, 2D-interaction→3D-propagation), **MedSAM2** (arXiv:2504.03600) and **Medical
  SAM 2** (arXiv:2408.00874) (SAM2 key-slice + memory propagation), **SAM-Med3D** (arXiv:2310.15161)
  (native-3D point prompts), **SegVol** (box prompts, zoom). We use these as the refinement engine.
- **SeqSeg** (Sveinsson Cepero & Shadden, *Ann. Biomed. Eng.* 2024; arXiv:2501.15712, Marsden lab) —
  sequential local-segment vessel tracing from a seed point along centerlines with bifurcation
  detection; used for the great vessels.
- **HiPaS** (*Nat. Commun.* 2025, s41467-025-56505-6) — pulmonary artery/vein benchmark for vessel work.

## 3. Datasets
- **Dataset071** — ImageCHD reoriented to clinical LPS, clean 7-class, myo-intact (base model).
- **Dataset012 (Fanwei)**, **ClinicalImagesPHICleared**, **Dataset080** (expert-annotated clinical,
  the only GT), **Dataset090/091** (pseudo-label flywheel iterations). All clinical IDs are anonymized
  codes (no PHI).

## 4. Methods — step by step

**Stage 0 — Base model.** Train nnU-Net (DA5, ResEnc-M, 3d_fullres) on Dataset071 → 7-class ImageCHD model.

**Stage 1 — Clinical inference route (the working pipeline).** For each clinical case:
`resize CT → 512×512×221 ImageCHD grid (index-space) → nnUNetv2_predict → backproject to native grid
(inverse index-space resize, nearest-neighbour) → per-label largest-connected-component (LCC) cleanup`.
Output = a **native-geometry seed label ("LCC label")**. Rationale: the model needs the ImageCHD scale/
grid to segment correctly; backprojection restores native geometry but by nearest-neighbour upsampling,
so the LCC is **correctly located but coarse / interpolation-limited**.

**Stage 2 — Pseudo-label flywheel.** Pool Dataset071 + usable Fanwei/clinical LCC labels → **Dataset090**;
QC-promote the best predictions → **Dataset091**; retrain. Pseudo cases are train-only; expert set
(Dataset080) held out for honest evaluation.

**Stage 3 — Promptable refinement (this work).** Recover native-resolution boundaries by re-prompting
foundation/interactive models on the **original full-resolution CT**, using the LCC only as a spatial
prompt (`tools/label_to_prompts.py`):
- **Prompt derivation** (per structure, from the native LCC label):
  - *Bounding box* = min/max voxel extent.
  - *Interior foreground point* = distance-transform peak (most-interior voxel) + eroded-core samples.
  - *Negative points* = interior points of adjacent structures (disambiguation).
  - *Chambers (LV/RV/LA/RA) → adaptive lasso*: per axial slice, contour points sampled uniformly by
    arc-length, **densified where the contour nears an adjacent chamber (septal band) and at high
    curvature**. Consequence: a septal defect (VSD) — the region where LV/RV pools are closest — gets
    denser prompting **emergently, with no VSD-specific rule**.
  - *Myocardium → spread points* across the thin shell + adjacent chambers as negatives.
  - *Vessels (Aorta/Pulmonary) → centerline*: skeletonize (Lee 3D thinning) → graph tree-diameter path;
    robust **centroid-along-longest-axis fallback** when thinning collapses (even/symmetric tubes).
    Endpoints (+ per-endpoint **radius** from the distance transform) = vessel seeds and root cut-planes.
- **nnInteractive refinement (all 7 structures).** Chambers via key-slice **lasso**; myo via spread
  points + negatives; vessels via centerline points. 2D-interaction→3D-propagation. **Runaway guard**:
  if a structure segments to > 3× its LCC voxel count, keep the LCC label for it (prevents foundation-
  model flooding, observed on a high-resolution case).
- **SeqSeg vessel tracing (Aorta/Pulmonary).** Seed SeqSeg (`run single`) with the centerline endpoints
  (physical coords + radius) and trace at native resolution; recovers distal vessel the coarse LCC missed.
- **Coordinate handling (critical).** Prompts are computed via the nibabel affine (**RAS+**); SeqSeg/
  SimpleITK read images in **LPS**, so vessel seeds negate X,Y (RAS→LPS) to land in the CT frame. All
  stages operate in the case's **native geometry** — no resampling of the CT.

**Stage 4 — Validation against expert GT (Dataset080).** Per-structure and whole-heart **Dice** for
three versions vs manual GT: (i) raw LCC seed, (ii) nnInteractive-refined, (iii) nnInteractive + SeqSeg
vessels fused. Chambers/myo/whole-heart judged by Dice; **vessels judged visually** because SeqSeg traces
distal vessel the manual annotation stopped short of (so its vessel Dice vs GT can legitimately drop).

**Stage 5 — Recycle.** Expert-verified refined labels re-enter the Stage-2 training pool; repeat.

## 5. Key design decisions & rationale
- **Re-prompt on the native CT, not upsample the label** — the segmentation is produced natively by a
  scale-robust model; the LCC supplies location only → escapes nearest-neighbour interpolation artifacts.
- **Division of labour** — nnInteractive for all structures; SeqSeg specifically for great vessels, where
  its centerline tracing extends distal branches nnInteractive/LCC cannot invent.
- **Geometry-driven (not defect-conditioned) lasso density** — defects emerge from inter-chamber proximity.

## 6. Risks / threats to validity (to address in the paper)
1. **Foundation-model normalization** — models trained on mostly-normal anatomy may smooth over CHD
   defects (e.g. close a VSD). Must be verified on defect cases; planned mitigation = prompt the
   derived septal-defect label as a **negative** to force the hole open.
2. **Garbage-in on vessels** — nnInteractive refines only what the LCC centerline found; SeqSeg is the
   component that extends coverage.
3. **Limited GT** — quantitative validation only on the small expert Dataset080; elsewhere qualitative.

## 7. Preliminary result
On the clean expert case **BAF004**, nnInteractive refinement improved Dice vs the raw LCC on the
interpolation-sensitive structures: LA 0.838→0.896, Aorta 0.860→0.948, Pulmonary 0.655→0.827,
whole-heart 0.894→0.923 (myo/LV/RV ≈ flat). Consistent with the native-refinement hypothesis. Full
8-case validation pending LCC-seed generation for the remaining expert cases and SeqSeg vessel traces.

## 8. Reproducibility
All datasets built by versioned `tools/build_dataset*.py`; refinement by `tools/label_to_prompts.py`,
`tools/run_nninteractive_refine.py`, `scripts/CHD_refine_step{1,2,3}*.sh`; validation by
`tools/eval_vs_gt.py`. Each run records git commit + params to `logs/run_manifest.jsonl` and a
`PROVENANCE.txt` beside its output.
