# Matched native-grid refinement comparison — 2026-09-16

Status: implemented and locally regression-tested; GPU refinement and training have not been run with this implementation. Historical D093 outputs are not results of these fixes. See [shared debugging memory](REFINEMENT_MEMORY.md).

## Cohort and comparison

Use the **first iteration's accepted 50 pseudo-label cases**, selected from Dataset090 `split_config.csv` where `intended_use == pseudo_label_train`: **45 Fanwei + 5 clinical** (AVSD003, BAF001, BAF002, BAF008, BAF010). Do not use all 60 or Dataset091's later promotions. Initialization requires exactly 50 cases and classwise LCC seeds on the image's physical grid.

| Arm | Reserved dataset ID | Labels changed |
|---|---|---|
| `baseline` | existing 90 | Reuse D090 model and predictions; no new training |
| `chambers` | 95 | nnInteractive LV/RV/LA/RA, labels 1–4 |
| `seqseg` | 96 | SeqSeg aorta/PA, labels 6–7; union with original vessels |
| `combined` | 97 | Same saved chamber and vessel candidates, composed together |

Myocardium (5) remains unchanged. All four comparison cohorts contain the same 97 ImageCHD + 50 accepted pseudo cases, **147 total**. The existing five ImageCHD validation splits, architecture, spacing, patch size and intensity normalization are copied from D090. No clinical-spacing experiment or replanning is included. Pseudo cases stay train-only. Existing dataset IDs are refused; choose unused refinement IDs through the lower-level builder if these IDs are occupied.

## What changed

- nnInteractive defaults to **additive chamber refinement**: union its candidate with the original chamber LCC before cleanup and QC. Every original chamber voxel is preserved, including when the model produces an empty or shrunken prediction. This cannot remove original false positives. Explicit `init --chamber-mode replace` restores the optional replacement experiment, where up to 30% seed loss is permitted.
- nnInteractive receives three separated interior lasso planes and one interior negative point from each present competing structure. Every exact raster prompt, point, order and intermediate prediction is saved. Prompts remain imperfect pseudo supervision; erosion is not proof of correctness.
- SeqSeg uses two centerline seeds: start at least 5 mm inward (one-third of path length for short paths), then walk inward to the first point with at least half the maximum interior-path radius. This avoids the thin-spur placements found in real slice checks; it can move starts substantially inward. The 0.5 fraction is a heuristic, not an anatomical trunk detector. Seeds retain actual outward local directions and measured radii in **LPS millimetres**. No fabricated +z direction or endpoint radius floor. The historical aorta/femoral checkpoint and `aorta_tutorial` configuration are retained for a controlled comparison; they are not evidence of a pulmonary-specific model. Poor PA output may still reflect model/domain mismatch.
- The process-local SeqSeg adapter allows **at most two detected bifurcation generations per seed path**, tagging both the continuing and side daughters. A third detected split terminates that path without retrying. A separate 14-branch resource ceiling is not the bifurcation rule. The adapter fails if the installed tracing code no longer matches the supported structure; it does not edit the installed package.
- This is not an anatomical branch counter: repeated outlet detections may stop early, two roots can explore different paths, and a local prediction crop may include distal stubs. Review `depth.json` and output anatomy before interpreting the cap as an annotation-extent match.
- Use binary SeqSeg MHA directly, physically mapped to the original CT grid; no surface rasterization roundtrip. Save probability assembly, binary masks before/after seed-component filtering and final native candidate to locate collapse.
- Vessels are **additive by default**: union with the seed, keep the seed-connected component, then apply QC. This preserves the existing vessel even when SeqSeg returns nothing. An empty or degenerate result is reported explicitly and is not evidence of successful refinement.
- Merge proposals simultaneously. Original territory of every competing label is protected; ambiguous background additions remain background and get a conflict mask. There is no first-writer priority and no silent consumption of myocardium/fallback labels. This conservative rule cannot fix original inter-class swaps.
- QC measures retained seed and growth after cleanup: optional replacement-mode chamber retention ≥70%, chamber growth ≤4×, vessel growth ≤20×. These are adjustable **heuristics**, not validated anatomy thresholds. Vessel growth has a separate, more permissive bound because extension is the intended operation. Rejected candidates restore the exact seed; no cases are dropped. Large growth can still be a false positive.
- Inputs, policies, model checkpoints, implementation and completed outputs are hashed. A changed run requires a new directory. Technical failures stop the stage; completed cases can resume.

## Sherlock launch (2026-09-18)

The launcher queues baseline checks/preparation → preprocessing of three new datasets → training of three new models → D080 prediction/scoring with `afterok` dependencies. The default is **all five folds and the five-fold ensemble**, matching the earlier D091/D092 workflow. Run on the login node:

```bash
cd /scratch/users/sastocke/nnunet_CHD
git pull --ff-only origin all-experiments
bash scripts/CHD_refinement_submit.sh
```

For a faster matched fold-0 test, replace the last command with:

```bash
FOLDS=0 bash scripts/CHD_refinement_submit.sh
```

Use one of those launches, not both concurrently. Run directory defaults to `$REPO/refinement_runs/accepted50_conservative_v3`. Job IDs and submitted arguments are recorded in `submitted_jobs.txt`; duplicate submission is refused. After inspecting/cancelling any old pending jobs, `RESUBMIT=1` explicitly permits a resume. Completed refinement cases, preprocessing and training checkpoints are verified/reused; incomplete stages resume. To expand a finished fold-0 study to five folds, use `RESUBMIT=1 FOLDS=0,1,2,3,4 bash scripts/CHD_refinement_submit.sh` after ensuring no prior jobs remain active.

D090 is the matching original-50 baseline; D091 added nine cases and is not a matched substitute. Existing D090 models/predictions are checked and hashed before launching refinement. No D094 dataset, baseline preprocessing, retraining or new baseline prediction is performed. D093 is preserved as historical evidence. New names:

- `Dataset095_ImageCHDRefinedChambers`
- `Dataset096_ImageCHDRefinedSeqSeg`
- `Dataset097_ImageCHDRefinedCombined`

The preparation job refuses occupied IDs in raw, preprocessed or results roots before refinement starts. Availability could not be rechecked remotely on September 18 because the Sherlock session expired. No old datasets or models are reset/deleted. A partially completed dataset build needs inspection before retry; the builder will not overwrite it.

Training uses `nnUNetTrainerDA5_200epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_N` under each dataset's usual `nnUNet_results` directory. D090 plans and splits are copied, preserving architecture, spacing and intensity normalization; fingerprint extraction and preprocessing do not replan the arms.

The evaluator uses the same resize helper/defaults as the old D090/D091 script, predicts each requested fold, and backprojects with `--no-lcc`. Five-fold mode also predicts the ensemble. Native outputs go to:

```text
nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/predictions/
.../ds090_foldN/ and ds090_ensemble/  (existing baseline, reused)
.../ds095_foldN/ and ds095_ensemble/
.../ds096_foldN/ and ds096_ensemble/
.../ds097_foldN/ and ds097_ensemble/
```

Grid predictions use the existing `predictions/_grid512/dsNNN_*` convention. Each native folder includes `metrics.csv`, `metrics_summary.json`, input/model provenance and a checksum completion marker. Missing cases/checkpoints or mismatched geometry fail explicitly. A pre-existing untracked output is refused rather than silently reused. Resized inputs are cached separately per arm inside the run directory to avoid concurrent writers; the resize method is identical across arms.

Model folder defaults are the previously verified Sherlock paths: nnInteractive `chd_refinement/models/nninteractive/models/nnInteractive_v1.0`, SeqSeg `chd_refinement/seqseg_weights/aorta_ct_mr/Dataset006_SEQAORTANDFEMOCT/nnUNetTrainer__nnUNetPlans__3d_fullres`, both under `/scratch/users/sastocke`. Override `NNI_MODEL`/`SEQSEG_MODEL` explicitly if moved. Other supported overrides include `RUN`, `REPO`, `IMAGES_DIR`, `SEEDS_DIR`, `ACCEPTED_CSV`, `REFERENCE_PLANS`, and the standard nnU-Net root variables. Retain the source masks and run directory: training datasets link to those immutable inputs.

The lower-level `CHD_refinement_four_arm.sh` remains available for individual `init`, `seqseg`, `nni`, `assemble`, `build`, `preprocess`, `train`, `predict` and diagnostic `evaluate` stages. GPU refinement/training runs have not yet validated the new workflow; 24 local tests cover core geometry/QC, dataset construction, launch dependencies and native-mask scoring.

## Output locations and review

Everything below is relative to `$RUN`:

| Path | Contents |
|---|---|
| `run.json`, `implementation.json`, `*_model.json` | Exact cohort, paths, policy and provenance |
| `nni/<case>/` | Four chamber candidates, lasso NPZs, prompt JSON, predictions after each interaction |
| `seqseg/<case>/<6-or-7>/` | Directed seeds, attempts/logs, native candidate and explicit status |
| `seqseg/<case>/<label>/depth_audit_<attempt>/` | Bifurcation events, probability assembly, before/after component filtering |
| `arms/<arm>/<case>.nii.gz` | **Exact final masks used for training and direct label evaluation** |
| `qc/<arm>/` | Per-case fallback/retention/growth, raw SeqSeg count, final added/removed voxel counts and ambiguous-background masks |
| `assembly_summary.json` | All assembled case/arm QC reports |
| `training.json` | Four dataset destinations, identical cohort/splits, frozen plans and label hashes |

Check all 50 cases remain represented, and distinguish usable preserved seeds from successful additions. Review no-addition/tiny candidates, growth, lost chamber seed, protected-class overlap and ambiguous additions. Compare before/after filtering to distinguish failed tracing from filtering losses. Do not tune the accepted cohort separately per arm.

If expert labels exist for every case in a separate diagnostic run, set `GT_DIR` and run `bash scripts/CHD_refinement_four_arm.sh evaluate`. This scores the final assembled masks; it does **not** evaluate trained nnU-Net models and does not pretend the 50 pseudo cases have ground truth. Create any annotated diagnostic run separately, using the Python CLI `init --cases ...`; never build those diagnostic cases into training.

For the trained-model comparison, use the same held-out image inputs, inference route, postprocessing and fold selection for all four datasets. Predictions should use trainer `nnUNetTrainerDA5_200epochs`, plans `nnUNetResEncUNetMPlans`, configuration `3d_fullres`; compare fold 0 against fold 0, then corresponding folds/ensembles. Do not compare a single refined fold against a baseline five-fold ensemble. D090 is reused as the baseline: the builder requires exact source-seed byte identity, source cohort and splits, and unchanged source plans; preflight verifies current model/source metadata and required checkpoints/predictions. These checks do not reconstruct historical training bytes or prediction provenance, so existing saved evaluation records remain part of the comparison evidence. Optimization randomness is not fixed by this runner.

Report per-case/per-class Dice, vessel extent and recall, plus failures/fallback counts. A vessel traced beyond the ground-truth annotation can be anatomically real but is still a disagreement in full-mask Dice. Any annotation-extent ROI must be defined consistently before comparing arms; do not crop each prediction to make its score better. Dataset080 has informed debugging, so it is a development-exposed holdout, not an untouched final test. Final claims need independent cases.

Native-grid refinement can exploit additional CT detail, but native output spacing alone does not ensure better labels: the models still have their own preprocessing and learned priors. The experiment tests improvement rather than assuming it.

## What “fallback” means

The 50 cases are a fixed cohort, not a QC score. A fallback replaces only the rejected structure's proposed refinement with its exact original classwise LCC; the case stays in every arm.

For vessels, processing order is: run SeqSeg → union with original vessel LCC → remove claims on other original classes → retain the component connected to the seed → check volume growth → resolve competing background additions. With the current defaults, growth above 20× the seed volume rejects that vessel's proposal. Since union already contains the seed, an empty SeqSeg candidate normally produces `unchanged_seed`, not a shrinkage rejection. Detached additions are discarded individually, not grounds to drop a case. A competing-background conflict also discards only that ambiguous addition.

For nnInteractive chambers, the default union mode preserves the entire original chamber LCC. Volume above 4× the seed triggers fallback; disconnected or conflicting additions are excluded. Only the explicit optional replacement mode permits shrinkage, requiring at least 70% seed retention. No seed / empty usable proposal also triggers fallback. Technical failures (wrong geometry, missing files, failed model command) stop the stage rather than count as a successful fallback.

QC now distinguishes `changed_mask`, `unchanged_seed`, and `fallback_to_seed`, and records final added/removed voxels. None of these outcome names establishes anatomical correctness. The vessel 20× and chamber 4× bounds are deliberately explicit, provisional settings, not learned or validated clinical thresholds.

### Changes from the first launcher revision

The current launcher uses `accepted50_conservative_v3` rather than the earlier v2 directory, because the chamber policy changed. If you already submitted v2 jobs, this update does not cancel or modify them; inspect those jobs before submitting v3. Only array indices 1–3 (D095–D097) are now preprocessed/evaluated; training uses indices 5–19, or 5/10/15 for fold 0. D090 baseline predictions stay at their original paths. If baseline files are missing or incompatible, preflight stops instead of silently retraining the baseline.
