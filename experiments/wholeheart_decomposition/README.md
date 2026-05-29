# Whole-Heart Decomposition — Stage 2

> **Read first:** [`STAGE2_SPEC.md`](STAGE2_SPEC.md) — canonical anatomy
> (class 7 = pulmonary artery, not pulmonary veins), soft topology priors,
> disease-conditioned overrides for TGA/DORV/ToF/PuA/HLHS, AO-vs-PA feature
> menu, and the HITL capture format. Machine-readable mirror in
> [`anatomy_priors.yaml`](anatomy_priors.yaml).

The Stage-1 pipeline (`scripts/CHD_Dataset040_wholeheart.sh`) produces a clean,
topology-preserving **binary heart vs. not-heart** mask for every test case.
Stage 2 takes that mask and assigns the original 7 anatomical classes
(LV-BP, RV-BP, LA, RA, Myo, Aorta, PA — see `docs/FEATURES.md`).

This folder holds **scaffolds** for the candidate Stage-2 methods. Nothing here
is wired up to actually run yet — that is intentional. Pick a direction *after*
the Stage-1 evaluation:

| Subfolder | Approach | When to choose it |
|---|---|---|
| [`mask_constrained_nnunet/`](mask_constrained_nnunet/README.md) | New nnU-Net with 2-channel input (CT + binary heart mask) → 7-class output | Default choice if Stage-1 binary masks are clean and you want a learned, topology-aware semantic head |
| [`mask_postprocessing/`](mask_postprocessing/README.md) | Take existing Dataset030 multiclass preds and intersect them with the Stage-1 binary mask | Cheapest baseline — zero training. If multiclass-inside-binary already wins, the more complex methods aren't justified |
| [`graph_based/`](graph_based/README.md) | Extract skeleton + components from the binary mask, build a graph, run a GNN classifier | If the failure mode after Approach A is still "wrong label on a continuous branch" — branch-level reasoning helps |
| [`rule_based/`](rule_based/README.md) | Heuristic propagation (largest CC = chambers; geodesic distance from chambers labels vessels) | Useful as an interpretable baseline + as a feature for graph_based |
| [`medsam_slicer/`](medsam_slicer/README.md) | MedSAM / SAM-style interactive correction inside 3D Slicer | If a clinical user needs to hand-correct cases; Stage 2 becomes assistive rather than fully automatic |
| [`dino_features/`](dino_features/README.md) | DINO/transformer per-voxel features inside the binary mask, light classification head | If pretrained representations are expected to outperform task-specific training given small dataset size |
| [`human_in_the_loop/`](human_in_the_loop/README.md) | Slicer Segment Editor workflow for manual correction of edge cases | Always relevant — these notes describe how the binary mask + Stage-2 output can be loaded for review |

## Quickstart for Approach A (current focus — chambers + bulk vessels)

```bash
# 1.  Build Dataset041 (CT + binary heart prior + 7-class labels).
#     Default uses GT-binarised channel 1 — no Stage-1 dependency.
python experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py --dry-run
python experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py

# 2.  Preprocess + train three trainers fold 0 + infer on imagesTs.
sbatch scripts/CHD_Dataset041_mask_constrained.sh

# 3.  Compare exp1 / exp2 / exp3 predictions.  Per-class Dice tells you
#     whether topology loss earns its keep; AO/PA swap-rate stratified by
#     TGA/DORV tells you whether FiLM disease conditioning helps.
```

Once Stage-1 binary masks exist for every imagesTr case, re-run with
`--mask-source predicted --mask-dir-tr ... --mask-dir-ts ...` to retrain on
the noisier mask distribution the model will see at inference time.

## Decision criteria (after Stage-1 eval)

Read `eval_compare.csv` (output of `scripts/evaluate_wholeheart.py --compare-to ...`)
and answer:

1. **Did the binary heart model beat the collapsed multiclass model on
   `largest_component_fraction` and `n_components`?**
   - If yes by a wide margin → topology is fixable in Stage 1 → Stage 2 only needs to assign labels (any approach OK).
   - If no → the binary model isn't actually more topologically correct, and Stage 2 needs to also fix continuity (favour `graph_based`).

2. **Is the binary `Dice` ≥ the Dataset030 mean Dice on the heart class?**
   - If no → Stage 1 itself needs more work before Stage 2 is meaningful (rerun with `--partition=bioe` + more epochs, or try a bigger plan).

3. **Are chamber/vessel boundaries plausible in Slicer overlays of the binary mask?**
   - If yes → `mask_constrained_nnunet` is the natural next step.
   - If no (mask "smears" between AO and PA) → `graph_based` or `medsam_slicer` (which can fix specific branches case-by-case).

## Conventions for all sub-experiments

- Inputs come from `${nnUNet_results}/Dataset040_WH_ImageCHD_HU_Detail/predictions_wholeheart/<model>/`.
- Original CT lives at `${nnUNet_raw}/Dataset030_imageCHD_HU/imagesTs/`
  (Dataset040 symlinks the same files; either path works).
- Reference 7-class GT lives at `${nnUNet_raw}/Dataset030_imageCHD_HU/labelsTs/`.
- Output a new directory `${nnUNet_results}/Dataset040_WH_ImageCHD_HU_Detail/stage2_<method>/`
  containing the recovered 7-class predictions in the same NIfTI format as Dataset030.
- Final eval should reuse the existing nnU-Net evaluation tooling against
  Dataset030's `labelsTs`.

## Topology proxies vs Dice — what to optimise for

Surface Dice goes up when the binary mask is locally accurate, but **`n_components`** and
**`largest_component_fraction`** are what catch the flip-flopping branches that motivated
this pipeline. When comparing Stage-2 methods, plot Dice + CC together — a method that
loses 0.5 Dice but reduces CC from 25 → 3 is probably better for downstream CFD work.
