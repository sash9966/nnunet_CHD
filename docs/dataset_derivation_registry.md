# Dataset derivation registry (septal-defect work)

> **DIAGNOSIS SOURCE MAP (2026-07-17):** two diagnosis sources are used, and each
> dataset is pinned to exactly one.
> - **Datasets ≤ 62 (050/051/060/062) → Kaggle** `imageCHD_dataset_info.xlsx`
>   (ImageCHD 2023 release, sheet 'classification dataset'). This is the *publishing
>   baseline* — stay true to the flags Kaggle shipped with the dataset.
> - **Dataset 070 → Fanwei** `imageCHD_diagnosis_Fanwei_june21.csv` (supervisor's
>   June-21 re-review: more/higher flags — VSD 69 vs 36, AVSD 10 vs 0, +SV/HLHS/PuA,
>   splits, notes). Renumbered far from the others (070, not 063) so it is
>   unmistakably a **different diagnosis source**, not another recipe tweak.

How each derived dataset was created — the septal-defect derivation version, the
train/test partition, and the build entry point. Keep this current when a new
derivation dataset is added.

Source for all: `Dataset030_imageCHD_HU` (read-only). Labels 1=LV 2=RV 3=LA 4=RA
5=Myo 6=AO 7=PA; derived `septal_defect` = id 8.

| Dataset | septal derivation | myocardium in train | test set | septal on test labels? | build entry point | git tag |
|---|---|---|---|---|---|---|
| **050** | **v1** — LV-RV + LA-RA union, **no AVSD cross**; merges only if a ventricular component is high-confidence (pure-ASD dropped) | mixed (missing-myo NOT excluded) | Dataset030 standard split | no (predict on Dataset030 imagesTs) | `chd_landmarks.cli build-dataset` @ v1 code | `septal-v1-dataset050` |
| **051** | **v2** — union of VSD (LV-RV − myo) + ASD (LA-RA direct) + AVSD cross (LV-RA/RV-LA direct); ASD-merge FIX (any component → label 8) | `--require-myo` excludes missing-myo from train | Dataset030 standard split (filtered) | no | `scripts/CHD_Dataset051_septal_ablation.sh` | `septal-v2-dataset051` |
| **060** | **v2** (union) | clean only (missing-myo → test) | **NEW partition**: missing-myo cases (+stratified topup ~10%) | no (anatomy-only labelsTs) | `tools/build_dataset060_clean_holdout.py` / `scripts/CHD_Dataset060_clean_holdout.sh` | — |
| **062** | **v3 — VSD-ANCHORED**: VSD (LV-RV − myo) is the anchor; ASD/AVSD-cross kept only where **continuous with the VSD** (within `septal_link_mm`≈5mm), off-septum atrial contacts dropped; pure-ASD (no anchor) kept best-effort as-is. No myo hole-filling. | clean only (missing-myo → test) | **NEW partition**: missing-myo (+stratified topup ~10%) | **YES** — septal derived on test too (degraded/low-conf for missing-myo; for visual inspection) | `tools/build_dataset062_vsd_anchored.py` / `scripts/CHD_Dataset062_vsd_anchored.sh` | — |
| **070** | **v3 VSD-anchored** (identical recipe to 062) — the ONLY difference vs 062 is the diagnosis source: **Fanwei's June-21 re-review** (more/higher flags → more cases get a septal label) | clean only (missing-myo → test) | **NEW partition** like 062 | **YES** (incl. missing-myo, degraded) | `scripts/CHD_Dataset070_vsd_anchored_fanwei.sh` (calls the 062 tool, `--metadata imageCHD_diagnosis_Fanwei_june21.csv`) | — |

## Diagnosis sources
- `imageCHD_dataset_info.xlsx` — **Kaggle ImageCHD** published flags (VSD 36, AVSD 0). Publishing baseline. Used by **050/051/060/062**.
- `imageCHD_diagnosis_Fanwei_june21.csv` — **Fanwei's (supervisor) June-21 re-review** (VSD 69, AVSD 10, + SV/HLHS/PuA, test/train/validate splits, notes col). Differs from Kaggle on 55/110 cases; more complete for VSD/AVSD. Used by **Dataset070** only.

> **062 vs 070 = clean A/B on the diagnosis source.** Same source dataset, same
> v3 VSD-anchored geometry, same partition logic — only the flags differ. 070
> should produce a septal label for more cases (Fanwei flags more VSD/AVSD).

## Septal-focus ablation result (Dataset070, fold 0, 200ep)
Which lever actually helps the model *predict* the septal class (id 8), measured on
the 13-case test set + val pseudo-Dice:

| arm | val pseudo-Dice (class 8) | test cases w/ class 8 |
|---|---|---|
| DA5 baseline | ~0.16 (emerges ~ep85) | 8/13 |
| **SeptalOversample** ✅ | **~0.22** | **10/13** |
| SeptalTversky (w=1.0) ❌ | rises then **collapses to 0 @ ep~115** | **0/13** |
| SeptalOversampleTversky (w=1.0) ❌ | bump then **collapses @ ep~40** | **0/13** |

**Finding:** the weight-1.0 FN-biased Tversky term SUPPRESSES the tiny class entirely
(the optimiser finds it cheaper to predict no class 8 than to pay the recall-biased
penalty + base-loss FP backlash). Oversampling is the lever that works.
**Fix (V2 arms):** weight 0.1 + warmup (off until ep50) + linear ramp (30ep) + softer
bias (α/β 0.4/0.6) — `nnUNetTrainerDA5Septal{TverskyV2,OversampleTverskyV2}_200epochs`,
added to the 051/062/070 ablation scripts (rerun skips completed arms).

**Fair long-schedule set (250ep, `BUDGET=250`, the script default):** val loss was still
dropping and class 8 still rising at ep200. PolyLR is tied to `num_epochs`, so 200→250
*stretches the whole LR curve* — comparing a 250-epoch arm against a 200-epoch reference
would confound the loss with both the extra epochs AND a different LR-at-each-epoch. So
the 250 set reruns **every** arm at 250 (baseline + oversample + V2 arms, warmup scaled
to 100) for a clean comparison. `BUDGET=200 sbatch ...` still runs the original set.

## Derivation code
- v1: `chd_landmarks.derived_regions.build_septal_defect_proxy`
- v2: `chd_landmarks.derived_regions.build_septal_defect`
- v3: `chd_landmarks.derived_regions.build_septal_defect_anchored`
- Selected via `DerivedLabelBuilder(..., septal_mode="v2"|"anchored")`.

## Reliability tiers (of the derived septal GT)
- **VSD-bearing (50/110)** — solid (interventricular septum is myocardium).
- **AVSD-like = VSD+ASD (12/110)** — solid, VSD-anchored + continuity.
- **pure-ASD (18/110)** — approximate (atrial septum unlabeled, no anchor); v3 keeps it best-effort.
- **missing-myo** — degraded; excluded from train everywhere from 051 on; in 062 kept in test for inspection only.

## Clinical deployment track (no septal label)
- **Dataset071** `Dataset071_ImageCHDClinicalOrientation` — the "train and send to clinic"
  anatomy model, first increment. Images from Dataset060; **clean 7-class labels pulled from
  the original Dataset030 (NO septal id 8)**; physically reoriented **RAS→LPS** to match
  clinical inference (`sitk.DICOMOrient`, true voxel flip, not a header edit). No HU shift, no
  resampling — nnU-Net handles spacing. **Both sources are pooled across Tr+Ts (ImageCHD's
  original split is arbitrary) and the output is re-partitioned by myocardium presence in the
  label: myo→train, no-myo→holdout — so every training case has real myocardium.** Built by
  `tools/build_dataset071_clinical_orientation.py` (self-verifies LPS + image/label geometry +
  lossless labels/HU; `--no-myo-partition` follows the source folders instead). Rationale:
  LV/RV boundary speckle + the septal 3-way competition are research concerns, not wanted in a
  clinical anatomy product.

- **Dataset072** `Dataset072_ImageCHDMultiFOV` — clinical **domain adaptation via FOV/scale
  diversity**. From the LPS Dataset071, per case emits a full copy + cardiac-bbox crops at 60 mm
  and 30 mm/side (`_full`/`_bbox60mm`/`_bbox30mm`). Pure ImageCHD trains well but generalizes
  poorly to clinical CT, whose FOV is tight around the heart; the crops shift nnU-Net's shape
  fingerprint toward that domain. **Crop-only** (sub-region extraction): spacing, HU and label
  values preserved exactly — NO resample/upsample (no invented detail). Deterministic v1 (no
  random aug yet). `tools/build_dataset072_multifov.py`; QA reports + `case_groups.json`.
  ⚠ Use a **grouped CV split** (all `<case>_*` in one fold) or validation leaks across a
  patient's own variants.

- **Dataset080** `Dataset080_ClinicalCaseSanjibDetailed` — expert-annotated clinical cases (7-class
  LPS, same id scheme), **grown 3 → 8** (BAF004, CHIPS001/002/005/006/007/010/016) as of 2026-08.
  The real deployment domain. **Frozen test set for the D090/D091 comparison** (never trained there);
  **intentionally trained in the clinic model Dataset100** (so its Dice from D100 is NOT unbiased).
  Note the corrected spelling `Clinical` (was long misspelled `Clincal` in code; fixed 2026-08-12).
- **Dataset081** `Dataset081_ImageCHDplusClinical` — 071 + 080 with clinical **oversampled 8×**.
  Split rule: **clinical is train-only (never in val); ImageCHD does the 5-fold val** — avoids
  duplicate-leakage and uses all 3 clinical cases fully; real clinical eval is external inference.
  `tools/build_dataset081_mix.py` + `scripts/CHD_Dataset081_mix.sh`.
- **Clinical fine-tune** — `scripts/CHD_Dataset080_finetune.sh`: move 071's plans onto 080 (so the
  architecture matches), then `nnUNetTrainerDA5_finetune` (100ep, lr 1e-3, `-f all`) initialised
  from 071 fold-0 weights (`-pretrained_weights`).
- **Clinical inference** — `scripts/CHD_predict_clinical.sh` resizes clinical sets to the ImageCHD
  grid (`tools/resize_to_imagechd_grid.py`, 512×512×221) — empirically required for the model to
  segment clinical CT — then predicts with each trained model into per-model subfolders. Key open
  finding: the dominant clinical-inference failure was **input scale/grid presentation**, not the
  weights; the resize front-end is the fix.

- **Clinical bootstrap loop** — `tools/backproject_predictions_to_native.py` maps grid512
  predictions back onto each case's native CT grid (inverse of the index-space resize,
  nearest-neighbour) + per-label largest-connected-component cleanup (drops speckle/stray
  vessel bits for clean, low-false-positive seed labels). Output = native-FOV/spacing labels
  aligned to the original CT → visually correct the few cases → add to Dataset080 → rebuild
  081 / re-fine-tune → repeat. Round-trip verified (Dice ~1.0, geometry preserved).

### Still planned
- **Respacing / all-myo variant** — Dataset071 does the orientation step only. A fuller
  clinician set may additionally drop missing-myo entirely and respace to the clinical
  acquisition; deferred until the LPS model is validated end-to-end.

- **Dataset090** `Dataset090_ImageCHDPseudoCombined` — pseudo-label run 1. Combines Dataset071
  (myo-intact ImageCHD, verified label-5 present) + **usable** Fanwei (Dataset012, 45) + usable
  clinical (5) LCC native pseudo-labels (images from source datasets, labels from
  `predictions/ds071__grid2native_lcc/`). Splits: ImageCHD does the 5-fold **val** (reusing
  Dataset071's folds if present); pseudo cases are **train-only, never val** (noisy labels).
  **quick_check** cases (13) excluded from run 1 (no per-case low-weight in stock nnU-Net) —
  recorded in `split_config.{json,csv}`. **unusable** (5) + **Dataset080** held out in
  `imagesTs` (images only, no labels). Built by `tools/build_dataset090_pseudolabel.py`
  (existence + geometry + LPS + myo-intact + leak checks) / `scripts/CHD_Dataset090_pseudolabel.sh`.
  Later stage: ignore PA/RA on the pseudo-labels.
