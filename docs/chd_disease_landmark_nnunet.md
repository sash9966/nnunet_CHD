# CHD Disease-Landmark & Derived-Region System (`chd_landmarks`)

A research preprocessing + training-support toolkit that turns existing
ground-truth anatomy segmentations **plus CHD diagnosis flags** into
disease-specific landmark / region labels, auxiliary masks for metrics &
topology, and (optionally) a region-based or disease-aware training setup.

> **Not a diagnostic tool.** This code does **not** diagnose CHD. It assumes
> diagnosis flags are already provided (from `imageCHD_dataset_info.xlsx`) and
> uses them, with the GT anatomy, to *derive research labels/metrics*. It is
> deliberately conservative: a landmark is produced only when both the flag and
> the geometry support it — it never hallucinates disease labels.

It is fully isolated from upstream nnU-Net and the existing datasets: source
datasets are read **read-only** (images symlinked), and only **new** datasets
(e.g. `Dataset050_*`) are ever written. It never calls `plan_and_preprocess`.

---

## 1. Why Dice alone is insufficient for CHD

`subclass_mean` Dice is saturated on this dataset (~0.83, <1.5-pt spread across
all methods). Dice barely responds to exactly the failures CHD cares about:
- a **VSD** is a small septal gap — getting it wrong moves whole-heart Dice by
  a rounding error;
- **pulmonary stenosis / coarctation** are local narrowings — a few mislabeled
  voxels, invisible to Dice;
- **aorta/pulmonary identity swaps** (TGA/DORV/truncus) and **vessel
  fragmentation** are topology/identity errors Dice cannot see.

So we (a) derive the disease-relevant regions explicitly and (b) measure them
with disease-aware metrics beyond Dice.

## 2. Four kinds of label (know the difference)

| Kind | Where it lives | Overlap? | Used for |
|---|---|---|---|
| **Anatomy labels** | source `labelsTr` (LV…PA, ids 1–7) | no | baseline segmentation |
| **Derived hard labels** | merged into new `labelsTr` (ids 8+) | no (integer) | vanilla nnU-Net extra classes (e.g. `vsd_orifice_proxy`) |
| **Auxiliary masks** | `derived_masksTr/<case>/*.nii.gz` | yes | metrics, topology, future interactive prompting |
| **Region-based targets** | `dataset_region_based.json` | yes (hierarchical) | nnU-Net v2 region training |

Vanilla nnU-Net integer labels **cannot** represent overlap. So overlapping /
hierarchical concepts (whole-heart ⊃ ventricles ⊃ LV, or "LV + VSD") are
exported as auxiliary masks and/or region-based targets, never as a single
integer map.

## 3. How each landmark is generated (conservative geometry)

All builders operate on the GT anatomy grid (voxel-aligned), in mm via the
NIfTI affine, and attach a `confidence` (`high/medium/low/none`) + a `reason`.

- **VSD / ASD orifice proxy** — near-contact band between the two chambers
  (LV/RV or LA/RA) **minus** the intervening myocardium/septum. A *localized*
  gap in an otherwise-present septum → high confidence; a broad gap (eqd > 30 mm)
  or no myocardium to confirm → low (kept auxiliary). Merged into the integer
  map **only at high confidence**, and only over the chamber interface voxels
  (`may_overwrite_lv_rv_interface_only`).
- **LV/RV false-merge region** — large LV↔RV contact area (> threshold mm²)
  flags suspected global mislabeling vs a true local VSD. Auxiliary only.
- **Pulmonary stenosis / aortic coarctation ROI** — distance-transform radius
  along the vessel (centerline if `skimage` is available, medial fallback
  otherwise); the narrowest segment (radius < frac × median) becomes the ROI,
  plus a minimum-radius point and pre/post segments for coarctation. **Auxiliary
  by default** (a disease *subregion* of the vessel, not separate anatomy).
- **Aorta/pulmonary confusion interface & connection candidate** — near-contact
  band / tight-touch between AO and PA for ToF/PuA/truncus/TGA/DORV. Always
  auxiliary; the connection candidate is explicitly labeled *candidate, not true
  anatomy*.
- **Hypoplastic preservation ROI** — for HLHS/PuA/tricuspid-atresia: marks the
  small but disease-relevant structure (incl. disconnected specks) so
  postprocessing does not delete it. Auxiliary, used for loss weighting/metrics.

### Annotation-status tracking (design principle 4)
For every case we record which landmarks were **derived/verified** vs whose
absence is **unknown** (flagged disease but anatomy missing, or low confidence).
Absence is **never** treated as confirmed background unless explicitly derived.
This is written per-case to `case_metadata/<case>.json` and summarised in
`chd_derivation_report.{json,csv}`.

## 4. When to use what

| Situation | Use |
|---|---|
| Add a few confident disease classes (VSD/ASD proxy) as extra labels | **vanilla nnU-Net** on the merged `Dataset050` integer labels |
| Hierarchical / overlapping targets (whole-heart ⊃ ventricles, vsd_complex) | **region-based nnU-Net v2** (`make-region-dataset-json`) |
| Up-weight rare disease labels / add vessel clDice during training | **custom trainer** `nnUNetTrainerDA5DiseaseLandmark` (see §7) |
| Just want better evaluation, no retraining | **metrics-only**: `evaluate-disease-metrics` on existing predictions |

## 5. Build the dataset (`Dataset050`)

```bash
# inspect first — reports resolved labels + which diseases are derivable
python -m chd_landmarks.cli inspect-dataset \
    --nnunet-raw $nnUNet_raw/Dataset030_imageCHD_HU \
    --metadata  $nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_dataset_info.xlsx

# build a NEW dataset (source is read-only; images symlinked)
python -m chd_landmarks.cli build-dataset \
    --source-dataset   $nnUNet_raw/Dataset030_imageCHD_HU \
    --target-dataset-id 050 \
    --target-dataset-name imageCHD_DiseaseLandmarks \
    --metadata         $nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_dataset_info.xlsx \
    --out-root         $nnUNet_raw
```

Produces `Dataset050_imageCHD_DiseaseLandmarks/` with `imagesTr`, `labelsTr`
(merged), `derived_masksTr/<case>/`, `case_metadata/<case>.json`, `dataset.json`,
and `chd_derivation_report.{json,csv}`.

## 6. Train

```bash
# one-time, on the NEW dataset only (never the source)
nnUNetv2_plan_and_preprocess -d 50 -pl nnUNetPlannerResEncM --verify_dataset_integrity

# vanilla nnU-Net with the extra disease classes
nnUNetv2_train 50 3d_fullres 0

# or disease-aware trainer (rare-label soft-Dice + vessel clDice)
nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerDA5DiseaseLandmark_200epochs
```

### Region-based training (optional, overlapping targets)
```bash
python -m chd_landmarks.cli make-region-dataset-json \
    --dataset $nnUNet_raw/Dataset050_imageCHD_DiseaseLandmarks
# review printed regions_class_order, then install it:
python -m chd_landmarks.cli make-region-dataset-json \
    --dataset $nnUNet_raw/Dataset050_imageCHD_DiseaseLandmarks --apply
```
`--apply` backs the integer `dataset.json` up to `dataset.json.integer_backup`.
**Region order matters** — `regions_class_order` is derived from each region's
`priority`; composite regions are flagged for manual review.

## 7. Disease-aware trainer

`nnUNetTrainerDA5DiseaseLandmark` (mixin: `DiseaseLandmarkMixin`) adds, on soft
probabilities (no argmax), on top of the base Dice+CE:
- a soft-Dice term for each derived hard label (`lambda_disease`, default 0.3),
  applied **positive-supervision only** (a label is supervised only on batches
  where its GT is present — unverified absence is never penalised);
- great-vessel clDice on aorta ∪ PA (`lambda_vessel_cldice`, default 0.1).

Label ids are resolved by name from `dataset.json`; if no derived labels exist
the term self-disables (baseline behaviour). The network is unchanged, so
`nnUNetv2_predict` and DA5 comparison are unaffected.

**Documented scaffolds not yet wired** (need dataloader/ignore-label plumbing):
disease-ROI **patch oversampling** and **case-wise masked loss** via nnU-Net's
`ignore_label`. The derivation already emits per-case annotation status to drive
these later.

## 8. Disease-aware metrics

```bash
python -m chd_landmarks.cli evaluate-disease-metrics \
    --pred  pred/ct_1012.nii.gz --gt gt/ct_1012.nii.gz \
    --derived-gt-dir $nnUNet_raw/Dataset050_imageCHD_DiseaseLandmarks/derived_masksTr/ct_1012 \
    --metadata imageCHD_dataset_info.xlsx --case-id ct_1012 \
    --dataset-dir $nnUNet_raw/Dataset050_imageCHD_DiseaseLandmarks \
    --out metrics_ct_1012.json
```
Computes: per-label Dice/HD95/ASSD/volume/CC-diff; vessel clDice + centerline
recall + Betti-0/Euler diffs; and disease-conditioned metrics (VSD detection &
centroid/diameter error, LV/RV false-merge, pulmonary/aortic min-diameter,
aorta/pulmonary leakage & confusion volume, hypoplastic volume preservation).

## 9. Limitations & failure cases

- **No `pulmonary_veins` label** in Dataset030 → APVC/TAPVR venous-return
  regions are **not derivable** (config placeholders + warnings only).
- **No HLHS column** in the xlsx → HLHS rules stay dormant unless a flag is
  supplied from another source.
- VSD/ASD proxies are **contact-based proxies**, not annotated defect orifices;
  treat as research surrogates. A thick/fused septum or a global chamber merge
  yields low confidence → auxiliary, not a hard label.
- Stenosis/coarctation ROIs use a distance-transform radius proxy (and skeleton
  if `skimage` present); they locate the narrowing, not a calibrated diameter.
- Surface-distance metrics crop to a bounded margin (research approximation for
  HD95 when surfaces are far apart). Betti-1/Euler need optional libs
  (`cripser`/`skimage`); they degrade to `None` rather than crash.
- This is a research surrogate pipeline; **diagnosis flags are assumed given**.

## 10. Configs & tests

- Configs: `configs/chd_label_map.yaml`, `chd_disease_rules.yaml`,
  `chd_derived_labels.yaml`, `chd_region_training.yaml`, `chd_metric_config.yaml`.
- Tests: `python3 tests/test_chd_landmarks.py` (13 synthetic tests, no trainer
  import — runs locally despite the `acvl_utils` blocker).
