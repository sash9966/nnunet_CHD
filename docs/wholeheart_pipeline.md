# Whole-Heart-First Pipeline (Dataset040)

A two-stage segmentation approach for CHD CT.
**Stage 1** trains a binary whole-heart segmenter (everything inside the heart →
class 1, everything else → 0). **Stage 2** assigns the 7 anatomical classes
(LV-BP, RV-BP, LA, RA, Myo, AO, PA) *within* the binary mask, using one of
several candidate methods (see [`experiments/wholeheart_decomposition/`](../experiments/wholeheart_decomposition/README.md)).

The hypothesis: the failure mode in the existing multiclass Dataset030 model
is semantic assignment, not anatomical localization. If true, separating the
two tasks should preserve topology much better than the joint 7-class problem.

---

## 1. Build Dataset040 from Dataset030

```bash
# Dry run first — prints the plan, writes nothing.
python scripts/convert_imagechd_to_wholeheart.py --dry-run

# Real conversion.
python scripts/convert_imagechd_to_wholeheart.py
```

**What happens.**
- `imagesTr/`, `imagesTs/`, `labelsTs/` are **symlinked** from
  `Dataset030_imageCHD_HU` (default `--symlink`; pass `--copy` to duplicate).
  This keeps the test set identical so binary Dice is comparable to the
  collapsed multiclass model.
- `labelsTr/` is rewritten: every voxel where the original label is `> 0`
  becomes `1`, everything else stays `0`. Affine and header are preserved
  exactly (`np.allclose(affine_after, affine_before, atol=1e-8)`).
- `dataset.json` is generated with `labels = {"background": 0, "heart": 1}`,
  `channel_names = {"0": "CT"}`, `file_ending = ".nii.gz"`.
- `conversion_summary.csv` captures per-case sanity: original labels present,
  fg-voxel count before and after, delta (must be 0), affine preservation flag.

CLI options:

| Flag | Default | Purpose |
|---|---|---|
| `--source-dataset` | `Dataset030_imageCHD_HU` | Where to read images and multiclass labels |
| `--target-id` | `40` | Dataset ID (folder is `Dataset0{id:03d}_*`) |
| `--target-name` | `Dataset040_ImageCHD_HU_WH` | Output folder name |
| `--raw-root` | `$nnUNet_raw` | Override the nnUNet_raw root |
| `--copy` / `--symlink` | symlink | How to handle CT files |
| `--overwrite` | off | Wipe existing target folder first |
| `--dry-run` | off | Print plan only |

---

## 2. Train binary heart models (fullres + lowres + cascade)

```bash
sbatch scripts/CHD_Dataset040_wholeheart.sh
```

The script runs **three models** so we can directly test whether a cascade
preserves small-vessel continuity better than a single-stage fullres model:

| Model | Trainer | Config | Why |
|---|---|---|---|
| **Fullres** | `nnUNetTrainerDA5_200epochs` | `3d_fullres` | Standard high-resolution patch-based training |
| **Lowres** | `nnUNetTrainerDA5_200epochs` | `3d_lowres` | Larger effective field of view; should preserve global topology better; also feeds the cascade |
| **Cascade** | `nnUNetTrainerDA5_200epochs` | `3d_cascade_fullres` | Refines the lowres prediction at full resolution. Hypothesised best for thin vessels — your specific concern |

**Resume support.** Each phase writes a `.done` marker; the script's `is_done`
guard skips completed phases on re-submission. nnU-Net itself resumes mid-epoch
from `checkpoint_latest.pth`. With three 200-epoch trainings + cascade prep,
plan on ~2–3 walltime cycles at 48 h each.

**Predictions land in:**

```
$nnUNet_results/Dataset040_ImageCHD_HU_WH/predictions_wholeheart/
  DA5_fullres/    — Stage-1 fullres binary heart masks  (Phase 1b)
  DA5_lowres/     — Stage-1 lowres binary heart masks   (Phase 2b)
  DA5_cascade/    — Stage-1 cascade-refined heart masks (Phase 4b)
```

---

## 3. Evaluate — fullres vs lowres vs cascade vs collapsed multiclass

```bash
PRED=$nnUNet_results/Dataset040_ImageCHD_HU_WH/predictions_wholeheart
GT=$nnUNet_raw/Dataset040_ImageCHD_HU_WH/labelsTs

# Each model in isolation
python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_fullres --gt-dir $GT --out eval_fullres.csv
python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_lowres  --gt-dir $GT --out eval_lowres.csv
python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_cascade --gt-dir $GT --out eval_cascade.csv

# Cascade vs collapsed multiclass baseline (Dataset030 predictions binarised on the fly)
python scripts/evaluate_wholeheart.py \
    --pred-dir $PRED/DA5_cascade \
    --gt-dir   $GT \
    --compare-to $nnUNet_results/Dataset030_imageCHD_HU/predictions/DA5_fullres \
    --out      eval_cascade_vs_multiclass.csv
```

**Metrics per case.**

| Group | Metrics | Notes |
|---|---|---|
| Volumetric | Dice, IoU, FP/FN voxels, pred/GT volume ratio | Standard |
| Surface | HD95 (mm), mean surface distance (mm) | Skip with `--skip-surface` for speed |
| Topology | `n_components`, `largest_component_fraction`, `n_holes`, `skeleton_branch_count` | The metrics that actually capture the user's "flip-flopping branches" complaint — track these alongside Dice |

`skeleton_branch_count` requires `scikit-image`; it returns NaN if not
installed.

**Expected reading of the results.**
- Cascade should give **higher `largest_component_fraction` and fewer
  `n_components`** than fullres if the hypothesis (cascade preserves vessels)
  holds. Dice might trail by a small amount — that's an acceptable trade.
- Both binary models should beat the collapsed multiclass model on topology
  metrics even if Dice is similar — that's the core motivation for the
  pipeline.

---

## 4. Stage 2 — decompose the binary mask into 7 classes

See [`experiments/wholeheart_decomposition/README.md`](../experiments/wholeheart_decomposition/README.md)
for the seven candidate approaches and a decision tree.

The default first move after Stage-1 results is
[Approach A — mask-constrained nnU-Net](../experiments/wholeheart_decomposition/mask_constrained_nnunet/README.md):
build Dataset041 with the binary mask as channel 1 and run the standard
7-class trainer. Existing ablation scripts (`CHD_Dataset030_ablation_*.sh`)
translate to that dataset with no new trainer code.

---

## 5. Expected folder layout

```
$nnUNet_raw/
  Dataset030_imageCHD_HU/         ← source
    imagesTr/  labelsTr/  imagesTs/  labelsTs/
  Dataset040_ImageCHD_HU_WH/      ← built by the conversion script
    imagesTr/  → symlinks to Dataset030/imagesTr
    imagesTs/  → symlinks to Dataset030/imagesTs
    labelsTs/  → symlinks to Dataset030/labelsTs   (multiclass; binarise on the fly)
    labelsTr/  ← binary (uint8) rewrites
    dataset.json
    conversion_summary.csv

$nnUNet_preprocessed/
  Dataset040_ImageCHD_HU_WH/      ← built by nnUNetv2_plan_and_preprocess

$nnUNet_results/
  Dataset040_ImageCHD_HU_WH/
    nnUNetTrainerDA5_200epochs__nnUNetResEncUNetMPlans__3d_fullres/
    nnUNetTrainerDA5_200epochs__nnUNetResEncUNetMPlans__3d_lowres/
    nnUNetTrainerDA5_200epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/
    predictions_wholeheart/
      DA5_fullres/   DA5_lowres/   DA5_cascade/
    .checkpoints/
      CHD_Dataset040_wholeheart/   ← resume markers per phase
      shared/                       ← preprocess + disease_map markers
```

---

## 6. Open research questions

These are the questions Stage 1's results should let us start answering:

1. Does the binary heart model beat the collapsed multiclass model on
   topology metrics (`n_components`, `largest_component_fraction`)? If yes,
   the Stage-1 → Stage-2 separation is worth it.
2. Does the cascade model beat fullres on **vessel continuity** specifically
   (manually inspect AO + PA branches in Slicer)?
3. Is the lowres binary model alone good enough to feed Stage 2? If yes we
   can skip the cascade refinement for the binary mask and save compute.
4. Does disease metadata (from `disease_map.json`) correlate with
   Stage-1 failure modes? Stratify the eval CSVs by disease and check.

The first ~3 questions are answered directly by the CSVs in §3 above.
