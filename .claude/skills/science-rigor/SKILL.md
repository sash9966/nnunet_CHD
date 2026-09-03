---
name: science-rigor
description: Scientific-rigor check for an experiment, dataset increment, training run, or evaluation before it is built or submitted. Verifies one-variable-per-increment, split hygiene, matched comparison arms, metric validity, confounds, selection bias, noise floor, and reproducibility. Use when the user proposes or requests any change to datasets, training configuration, or evaluation on a project intended for publication — and ALWAYS auto-invoke on such requests for nnunet_CHD, rather than waiting to be asked.
---

# Scientific Rigor Check

Run this BEFORE writing the build/train/eval code, not after. Output is a short pass/fail per item
plus a single explicit recommendation. If an item fails, raise it as a BLOCKING question — do not
note the flaw and proceed to build the flawed version anyway. That specific failure has happened on
this project and is the reason this skill exists.

## 1. One variable per increment
Name the single variable this increment changes. If two or more things change at once (e.g. new
labels AND new cases, or new data AND a new trainer), the result is uninterpretable: SPLIT IT into
separate datasets/arms so each comparison isolates one effect. State the resulting ablation chain
explicitly (e.g. D091 baseline -> D092 labels -> D093 +data).

## 2. Split hygiene
- train / val / test disjoint; verify programmatically, do not assume.
- The test set is FROZEN and appears in NO arm's training set. Name it and name the guard that
  enforces it (a hard-fail in the builder, not a comment).
- Validation composition identical across arms being compared; noisy/pseudo labels train-only.
- No patient/case leakage across folds (all variants of one case in one fold).

## 3. Matched arms
Trainer, plans, epochs, folds, seed, preprocessing and augmentation identical across compared arms.
List any difference. A comparison across different budgets or fold schemes is not an ablation.

## 4. Metric validity
Does the metric actually measure the claim? State its failure mode for THIS data. Examples seen here:
volumetric Dice under-reports thin structures; Dice penalises a prediction that legitimately extends
beyond an incompletely-annotated ground truth; overlap-vs-a-previous-prediction is a change measure,
not a correctness measure when there is no ground truth. Choose or add an appropriate metric
(recall, surface/NSD, volume error, topology) and say what each is evidence for.

## 5. Noise floor
Can the observed effect be distinguished from run-to-run variation? Report per-fold spread /
seed variance / n. An improvement smaller than the fold spread is not yet a result.

## 6. Confound audit
List everything else that differs between the arms (data volume, label provenance, preprocessing
version, code commit, class balance). For each, say whether it is controlled or a caveat to report.

## 7. Selection bias
Was any selection (QC approval, case inclusion, threshold tuning, variant choice) made using
information from the test set, or using the metric being reported? If yes, it must be disclosed or
the selection redone on a held-out subset.

## 8. Reproducibility
Build tool versioned and deterministic; git commit + parameters stamped with the outputs; dataset
derivation recorded (source, method, variant); symlink-only builds so raw data is never mutated;
exact commands recorded.

## 9. Pre-register the outcome
Before running: state what result would count as an improvement, what would count as no effect, and
what would falsify the hypothesis. This prevents post-hoc reinterpretation of whatever comes out.

## Output format
A compact table (item / PASS or FAIL / one-line reason), then "RECOMMENDATION:" with the design you
would actually run, then any blocking question for the user. Be concise; this is a gate, not an essay.
