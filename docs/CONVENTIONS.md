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
- **Dataset080_ClinicalCaseSanjibDetailed** — expert-annotated clinical cases (grown 3 → 8:
  BAF004, CHIPS001/002/005/006/007/010/016). **Note the correct spelling `Clinical`** (was long
  misspelled `Clincal` in code; fixed 2026-08-12). **Frozen test set for the D090/D091 comparison —
  never trained there.** D100 (clinic model) *does* intentionally train on it, so its Dice from
  D100 is NOT unbiased.
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
- **Dataset100_FinalClinic** — CLINIC-FACING ALL-DATA model = Dataset091 + ALL Dataset080 cases in
  TRAINING (Dataset080 intentionally included). **Deployment/qualitative review ONLY — Dataset080
  Dice from THIS model is NOT unbiased** (it's in training). Built by
  `tools/build_dataset100_finalclinic.py` (writes split_meta + `case_weights.json`). Same data pool
  for both variants below; ImageCHD does the val folds, pseudo-labels + Dataset080 stay train-only —
  same split logic as 091. no-myo ImageCHD cases are excluded (they live in Dataset071/imagesTs,
  never pulled into 090/091/100). Keep 090/091 as the unbiased eval — D100 is never the eval model.
  - **CANONICAL / SHIPPED = the CASE-WEIGHTED variant:** trainer
    `nnUNetTrainerDA5CaseWeighted_500epochs` (mixin `variants/mixins/case_weight.py`), fold **`all`**,
    **500 epochs**, run by `scripts/CHD_Dataset100_weighted_all.sh`. Per-case **sampling** weights
    from `case_weights.json`: ImageCHD GT 1×, Dataset080 expert **3×**, QC'd promoted 1×, usable
    clinical pseudo 0.5×, Fanwei pseudo **0.5×** (FOV/volume diversity, not label trust). This is the
    model trained and sent to the clinical team. Weighting is a deployment choice — D090/D091 stay unweighted.
  - **Alternate (earlier, unweighted) variant:** `nnUNetTrainerDA5_200epochs`, **5-fold (0–4) +
    ensemble**, run by `scripts/CHD_Dataset100_finalclinic.sh`, export
    `nnUNet_results/Dataset100_FinalClinic/CLINIC_MODEL_5fold/`. Kept for reference; not the shipped model.
  - **Sharing to a stock nnU-Net / Slicer (clinical team):** the DA5/CaseWeighted network IS the stock
    ResEncUNet (weighting changes only train-time sampling, not the architecture), and the trainer name
    the predictor looks up lives in `checkpoint['trainer_name']` (NOT in plans.json/dataset.json). So
    **retag the checkpoints to `nnUNetTrainerDA5`** (a stock trainer) with
    `share_export/retag_checkpoint_to_stock.py --target-trainer nnUNetTrainerDA5`; ship the retagged
    sibling folder — runs on stock nnU-Net, no fork. Keep the original CaseWeighted folder as provenance.
    (There is NO 1000-epoch variant; only CaseWeighted _200/_500epochs exist.)

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
- **Conda env lives on /scratch (fast runtime) and CAN get purged.** Env:
  `/scratch/users/sastocke/conda_envs/nnunet310` (Python 3.10). Sherlock purges `/scratch`; on
  2026-08-12 the `python3.10` binary was deleted, leaving dangling `python`/`python3` symlinks so
  `python` silently fell back to system 2.7 → every f-string build died at line 69. Do NOT move the
  env to OAK (OAK I/O is too slow for training/dataloading). Instead: recreate on scratch at the
  SAME path (no script edits), and keep a **tarball backup on OAK** for fast restore
  (`tar czf` to OAK; `tar xzf` back to the same scratch path — same prefix, so it just works).
  All run scripts have an env guard that aborts with a clear message if `python` isn't ≥3.9.
- **Server Python is 3.10** (conda env `nnunet310`); local dev Python is 3.12. Do NOT use
  Python-3.12-only f-string syntax: no nested `{ }` (dict/set/comprehension) inside a replacement
  field, no nested same-type quotes, no backslash inside `{ }`. Compute into a variable first.
  Local `py_compile` (3.12) will NOT catch these — they only fail on the server as a `SyntaxError`
  (often misreported to a wrong line). Keep all repo Python 3.10-compatible.
- **SimpleITK use-after-free:** never `sitk.GetArrayViewFromImage(sitk.ReadImage(...))` — the
  temporary Image is freed and the view dangles → garbage values + segfault. Keep a ref and use
  `GetArrayFromImage` (copy). (This caused Dataset090's segfault + false "no myo".)
- **Metrics undersell VSD:** report detection + centroid + NSD, split VSD vs ASD; volumetric
  Dice is brutal on thin proxy structures (visually-perfect cases scored ~0.3).
