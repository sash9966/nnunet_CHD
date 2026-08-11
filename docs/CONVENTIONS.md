# Project conventions & durable state (nnunet_CHD)

> Living record of decisions/conventions that are NOT obvious from the code and that I
> (Claude) must not lose between sessions. **Read this at the start of each session.**
> When a convention changes, update this file in the same change and say so.

## Storage / paths
- **Everything lives under `/scratch/users/sastocke/nnunet_CHD/`.** Never write to the
  neighbouring `/scratch/users/sastocke/nnUNet/`.
- `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` are **symlinks** into `../nnUNet`
  (long-standing setup). Tools must **NOT `.resolve()`** the env paths (that follows the
  symlink into the wrong tree — it broke the sibling `ClinicalImagesPHICleared` lookup once).
  Use the paths as given; create symlinks with `os.path.abspath`, not `.resolve()`.
- Data physically sits behind the symlink but is always **accessed via the `nnunet_CHD` path**.

## Label scheme (7-class, no septal in the clinical track)
`1 LV-BP · 2 RV-BP · 3 LA · 4 RA · 5 Myo · 6 Aorta · 7 Pulmonary`. **Myocardium = 5** (not 4).
ImageCHD/clinical are **LPS**; Fanwei (Dataset012) is **RAS** — kept as-is for orientation diversity.

## Clinical inference pipeline (the route that WORKS)
Direct native-spacing inference **breaks** (chambers mislabeled). Working route:
`resize case → 512×512×221 ImageCHD grid (tools/resize_to_imagechd_grid.py) → nnUNetv2_predict
→ resample back to native + largest-CC cleanup (tools/backproject_predictions_to_native.py)`.
Root cause of the failure was **input scale/grid presentation, not the model weights**.

## Dataset roles (current)
- **Dataset071_ImageCHDClinicalOrientation** — LPS clean 7-class ImageCHD base; `imagesTr` = myo-intact, `imagesTs` = no-myo holdout.
- **Dataset012_Fanweidata** — Fanwei cases (RAS).
- **ClinicalImagesPHICleared** — clinical cases (LPS): `imagesTs/` + `predictions/`.
- **Dataset080_ClincalCaseSanjibDetailed** — 3 expert-annotated clinical cases (BAF004, CHIPS002, CHIPS016).
  **FROZEN TEST SET — never train on it.**
- **Dataset090_ImageCHDPseudoCombined** — pseudo-label run 1. Train = ALL of 071 `imagesTr` (~97)
  + usable Fanwei (45) + usable clinical (5) with **LCC pseudo-labels**. `imagesTs` = held-out
  (unusable 5 + quick-check 13 + Dataset080 3), **images only, no labels**.
- **Dataset091_ImageCHDPseudoCombinedV2** — Dataset090 + 9 QC-approved ds090 pseudo-label cases
  promoted from held-out→train (8 Fanwei: CT_052_7910, CT_528_0579, CT_584_09_no, CT_731_6,
  CT_747_68, CT_754_49, CT_860_8, CT_914_49 + BAF007). Labels = `ds090__grid2native_lcc`.
  **BAF004 deliberately excluded** (it's Dataset080 test). Built by `tools/build_dataset091_from_090.py`.
- **A/B experiment:** train 090 & 091 full 5-fold (`CHD_Dataset090_train5fold.sh`,
  `CHD_Dataset091_pseudolabel.sh`), then predict Dataset080 with both (per-fold + ensemble,
  `CHD_predict_dataset080_compare.sh`, `--no-lcc`). 090 and 091 share identical ImageCHD val
  folds (pseudo is train-only), so the comparison is clean. Dice/violin/heatmap done offline.
- **Dataset100_FinalClinic** — CLINIC-FACING ALL-DATA model = Dataset091 + ALL Dataset080 cases
  in TRAINING (Dataset080 intentionally included). Trained fold `all` (no held-out), same
  trainer/plans as 091. Built by `tools/build_dataset100_finalclinic.py`, run by
  `scripts/CHD_Dataset100_finalclinic.sh` (build→preprocess→train all→export weights→predict
  clinic→QC overlays via `tools/make_qc_overlays.py`). **Deployment/qualitative review ONLY —
  Dataset080 Dice from THIS model is NOT unbiased** (it's in training). Weights export:
  `nnUNet_results/Dataset100_FinalClinic/CLINIC_MODEL_allData/`. Keep 090/091 as the unbiased eval.

## Prediction output locations
- **Preferred convention:** `<image-set>/predictions/<model>/` (e.g. `ClinicalImagesPHICleared/predictions/ds071__grid2native_lcc/`).
- **Accepted deviation:** Dataset090 Phase-3 held-out predictions go to
  `Dataset090.../predictions/ds090__grid2native_lcc/` (inside the raw dataset folder) — OK
  because they're reused as training data. Revisit if it causes confusion.
- **Folder-suffix meaning:** `__grid512` = resized-input prediction · `__native` = direct
  native-input prediction · `__grid2native_lcc` = grid512 prediction resampled back to native
  + LCC = the clean seed labels for the loop.

## Self-improvement (pseudo-label) loop
- Cycle: predict with current model (resize route) → LCC → **expert quick-fix** → add corrected
  cases to training → retrain **fold 0** → repeat. Do the full **5-fold only when the set stabilizes**.
- Keep **Dataset080 frozen** (never train) to measure each cycle honestly.
- Add only **expert-corrected** labels (never raw predictions) — avoid confirmation bias.
- Uncertain classes (PA/RA vessels): set to **`ignore`** rather than feed wrong labels
  (partial-label; ImageCHD carries them). See `docs/project_overview.html` → Literature.
- Version datasets per cycle: `Dataset091`, `092`, …

## Gotchas / lessons
- **SimpleITK use-after-free:** never `sitk.GetArrayViewFromImage(sitk.ReadImage(...))` — the
  temporary Image is freed and the view dangles → garbage values + segfault. Keep a ref and use
  `GetArrayFromImage` (copy). (This caused Dataset090's segfault + false "no myo".)
- **Metrics undersell VSD:** report detection + centroid + NSD, split VSD vs ASD; volumetric
  Dice is brutal on thin proxy structures (visually-perfect cases scored ~0.3).
