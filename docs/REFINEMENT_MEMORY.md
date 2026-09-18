# Shared refinement memory — Claude and Codex

Updated 2026-09-16. Read with [conventions](CONVENTIONS.md) and the [four-arm runbook](refinement_four_arm.md). Preserve observed facts separately from proposed fixes and future results.

## Current decision

The clinical-median-spacing retraining experiment is on hold. The requested study uses **the original accepted 50** (45 Fanwei + 5 clinical), not all 60 or the later promoted cohort. Source: D090 `split_config.csv`, `intended_use == pseudo_label_train`; clinical IDs AVSD003, BAF001, BAF002, BAF008, BAF010. All four training datasets use the same 97 ImageCHD + 50 pseudo cases and D090 splits/plans.

Arms: unchanged LCC; nnInteractive chambers only (1–4); SeqSeg vessels only (6–7); both. Preserve myocardium (5). Vessel refinement adds to the original LCC. The user requested at most two bifurcation generations, not a count of two branches or unrestricted distal tracing. Implemented local adapter caps detected generations on both continuing and side paths; it does not guarantee exact anatomical extent. Dataset IDs 094–097 are proposed and checked for collisions at build time.

## Historical D093 audit: evidence, not speculation

The local review is stored outside the repository in `../analysis/refinement_debug_review_2026-09-16.md` and `../analysis/d093_review_2026-09-16/` (inventory, SHA256 verification, downloaded outputs, tables and overlays). These local artifacts are not required for running the new code. 628 remote artifacts were verified: 588 retrieved, 40 existing CT/GT files reused. Avoid committing patient volumes.

- All 24 D093 relabelled masks exactly matched their merged sources; physical geometry matched. The confirmed failure was not a general affine mismatch.
- Historical merge consumed **raw nnInteractive** output and bypassed the cleaned LCC output. Across 24 cases, 139,098 voxels removed by that cleanup were still present upstream of the merge.
- Sequential first-writer merging could eat later labels and even fallback masks. CT_462_49's myocardium fallback lost 428,679 seed voxels (19.47%). That is a preservation failure; without expert GT it is not a measured anatomical error rate.
- On expert-labelled CHIPS016, direct refined-mask RA Dice fell from 0.8890 to 0.6787 (recall 86.46% to 52.38%); PA Dice fell from 0.7925 to 0.5014 (recall 73.55% to 34.12%). About 17.156 mL of correctly seeded RA became background; 5.206 mL of correctly seeded PA became RV and 2.725 mL became background. RV false-positive volume grew from 2.54 to 7.17 mL.
- Approximately 20% of CHIPS016's RV positive lasso lay outside expert RV. RA's three negatives all came from myocardium; the saved negatives were not inside their intended target. The sparse historical prompt generation did not consume the adaptive JSON. A plausible-looking prompt is not necessarily correct supervision.
- Exact positive raster prompts were not saved historically; bounding boxes cannot reconstruct their contents. Save exact masks, coordinates, ordering and every interaction's output.
- Both vessels used the aorta/femoral checkpoint with `aorta_tutorial`; no pulmonary-specific checkpoint was established. Correct RAS-to-LPS conversion already existed historically. Do not infer a scale bug from `--scale 0.1` alone; inspect installed physical-coordinate handling.
- Historical SeqSeg seeds discarded local tangent direction in favour of artificial +z, with a floored endpoint radius. Fix physical direction and measure radius at an interior point.
- 11 of 48 final SeqSeg MHAs were below 0.1 mL (10 PA, 1 aorta), **before surface conversion**. CT_704_49 PA was 0.013311 mL (20 converted voxels, 10 added); CT_853_56_no aorta was 0.022167 mL (294 converted, 53 added). Surface conversion alone cannot explain these tiny outputs. Probability assembly and component filtering must be inspected separately.
- Of 3,544,149 SeqSeg candidate voxels, 2,003,515 (56.5%) overlapped another existing label; only 501,785 were newly added. Overlap may indicate either a false-positive candidate or an incorrect existing class. Do not automatically give the vessel or chamber priority without evidence.
- Historical evaluation did not read the produced MHA/surface outputs correctly; “combined” scoring could effectively score only nnInteractive. Another evaluator used replacement while production used union. Always score the exact final training masks.
- Historical growth QC could reject desired extension while missing shrinkage and class confusion. Growth is not automatically bad, and retriggering is not automatically beneficial.
- Old branch limits of 3 or 7 were **not** bifurcation-generation limits. A real depth rule must propagate to both daughters, including the continuing branch.
- D093 downstream model results were not uniformly worse: whole-heart Dice D091 0.901970 → D093 0.901689; PA 0.626543 → 0.664623; LA 0.788025 → 0.808643. Do not conflate a failed direct refinement with proof that downstream training always degrades.

## Implementation rules and remaining limits

Use the new isolated workflow; historical scripts remain available and are not silently rewritten. Simultaneous merge protects original competing-label territory, abstains on conflicting background claims, and restores exact seeds on rejection. This intentionally cannot correct original inter-class swaps. Classwise seed-connected cleanup must occur before acceptance and actual training assembly, not only in an unused preview.

Vessel union preserves existing labels but can retain existing false positives. Retention/growth thresholds (0.7, chamber 4×, vessel 20×) are configurable guardrails, not clinical validation. Empty tracing is explicitly logged and preserves the seed; do not call an unchanged case a successful extension. All 50 stay in each arm regardless of fallback, avoiding arm-specific selection bias.

Plans, normalization, spacing, cohort and splits stay fixed. Model weights, input hashes, exact prompts, code and intermediate outputs must be attributable. Keep Sherlock paths under the nnunet_CHD symlink namespace; do not resolve the raw/preprocessed/results roots into the neighbouring repository.

Dataset080 remains excluded from these training datasets, but repeated use for debugging makes it a development-exposed holdout. D100 intentionally trained on Dataset080 and cannot provide an unbiased Dataset080 comparison. Same-fold comparisons do not remove random training variation. Native-grid output is not proof of recovered information or correct anatomy, and limiting branch depth cannot resolve missing distal ground-truth annotations.

## Validation state

Geometry, fallback, overlap arbitration, additive vessel preservation, seed direction, prompt coverage, immutable output checks, frozen plans/splits and both-daughter depth inheritance pass 21 local regression tests, including saved-candidate assembly/evaluation/dataset construction, in `tests/test_refinement_ablation.py`. The adapter was checked against the downloaded installed Sherlock tracer source. This is software validation only: no new GPU refinement, training or accuracy result has been produced yet. Record future run directories, checkpoint identities, failures and measured outcomes here rather than replacing these observations with assumptions.

## Follow-up seed slice check (2026-09-16)

A real-data check exposed a gap in the original synthetic tests: both current/previous points could be inside the correct LCC and correctly converted to LPS, yet a fixed 5-mm inset still landed on a thin spur. CHIPS016 aorta seed radius was only 0.317 mm. Membership alone must not be reported as adequate initialization.

Seed placement now walks inward from the initial inset until the measured radius reaches half the maximum interior radius along the same diameter path (or the best available eligible radius, explicitly recording whether the target was met). Never fix this by merely inflating a tiny measured radius. This relative-core choice is a heuristic, may move the start much farther inward, and changes where the two-generation trace begins; it does not identify the anatomical proximal trunk or guarantee distal coverage. Exact before/after seed indices, measured radii and inset distances are saved.

QC logging now separates raw SeqSeg candidate size from the union proposal, and records final additions/removals plus `changed_mask` / `unchanged_seed` / `fallback_to_seed`. An empty additive vessel candidate leaves the original seed unchanged; it is not a successful growth result or an automatic case exclusion.

Corrected-placement verification: 16 seeds across CHIPS016, CT_462_49, CT_704_49 and CT_853_56_no passed current/previous membership, connecting-segment membership, and independent physical-coordinate checks (maximum error <0.000005 mm). Twelve moved inward; all met the relative-core target. Three-plane overlays were visually inspected for all four cases. This is not all-50 validation or GPU validation. Local report: `../analysis/seed_checks_core_2026-09-16/README.md`; raw pre-fix audit retained separately. Maximum final inset was 130.28 mm, so do not describe the correction as a small fixed displacement. CT_704_49 PA remains narrow (about 0.98–1.16 mm radius).

## Launch decision — 2026-09-18

User requested GitHub push and a short server command sequence, preserving legacy training/evaluation conventions. Final IDs are D094–D097, superseding the preliminary 194–197 suggestion; D093 historical results remain intact. Names are ImageCHDPseudoBaseline / ImageCHDRefinedChambers / ImageCHDRefinedSeqSeg / ImageCHDRefinedCombined. Standard ResEncM plans filename retained, copied D090 contents frozen. The launcher chains all jobs; full five folds + ensemble by default, optional matched fold 0. D080 outputs follow existing `predictions/dsNNN_foldN` / `dsNNN_ensemble`, grid intermediates `_grid512`, backprojection without LCC, with metrics alongside masks. Sherlock authentication expired during final packaging, so runtime ID availability is enforced by fail-closed preflight rather than claimed verified. No new cluster model runs were launched from this task.
