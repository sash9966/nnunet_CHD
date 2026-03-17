# Cascade Experiments: Low-Res → High-Res with FiLM + Topology Loss

## Motivation

The 3d_fullres model achieves ~0.95 whole-heart Dice but has two systematic failure modes:

1. **AO/PA label confusion** — in TGA and DORV cases the great vessels are in non-standard
   positions. The fullres model, which only sees a local crop of the anatomy, lacks the global
   context to determine which vessel is the aorta.

2. **VSD boundary ambiguity** — in VSD cases the inter-ventricular septum is absent or
   incomplete. The fullres model often predicts a single large LV region rather than correctly
   separating LV and RV blood pools.

The cascade pipeline addresses both by introducing a low-res stage that sees the **full cardiac
anatomy in every patch**. This gives the network global context to:
- Assign vessel labels correctly based on spatial position relative to the ventricles
- Recognise the overall ventricular topology before the high-res stage refines boundaries

---

## Label Mapping

| Index | Structure | Notes |
|-------|-----------|-------|
| 0 | Background | |
| 1 | LV-BP | Left ventricle blood pool |
| 2 | RV-BP | Right ventricle blood pool |
| 3 | LA | Left atrium |
| 4 | RA | Right atrium |
| 5 | Myo | Myocardium |
| 6 | Aorta | Great vessel; exits LV in normal anatomy |
| 7 | Pulmonary | Great vessel; exits RV in normal anatomy |

---

## Disease Vector

`disease_map.json` maps each case ID to an 8-element binary flag vector:

| Index | Flag | Description | Key anatomical implication |
|-------|------|-------------|----------------------------|
| 0 | HLHS | Hypoplastic left heart syndrome | LV may be absent/tiny; AO/PA correction unreliable |
| 1 | ASD | Atrial septal defect | LA–RA may be adjacent |
| 2 | VSD | Ventricular septal defect | LV–RV boundary ambiguous |
| 3 | AVSD | Atrioventricular septal defect | All four-chamber septal borders ambiguous |
| 4 | DORV | Double outlet right ventricle | Both AO and PA exit RV |
| 5 | PuA | Pulmonary atresia | AO/PA fused; vessel correction must be skipped |
| 6 | ToF | Tetralogy of Fallot | Overriding aorta straddles VSD; AO unconstrained |
| 7 | TGA | Transposition of the great arteries | AO exits RV, PA exits LV (reversed) |

**Example:** a TGA+VSD case has `[0, 0, 1, 0, 0, 0, 0, 1]`.

FiLM conditioning injects this vector at the bottleneck so the network learns
disease-specific vessel topology without per-disease model copies.

---

## Cascade Pipeline Architecture

```
3d_lowres stage                         3d_cascade_fullres stage
──────────────────────────────────────  ────────────────────────────────────────────
Full CT (resampled to ~3–4 mm)          Full CT (native ~1 mm) + lowres prediction
Patch covers entire heart               Patch covers local region (boundary focus)
Global label assignment                 Boundary refinement
FiLM: disease-aware topology            FiLM: disease-aware boundary decisions
Optional: topology loss (AO/PA CCs)     Same FiLM conditioning
──────────────────────────────────────  ────────────────────────────────────────────
Output: coarse whole-heart labels       Output: final refined labels
        used as prior for stage 2
```

---

## Experiment Ablation Matrix

| Experiment | Trainer (lowres) | Trainer (cascade fullres) | FiLM | Topo loss | Notes |
|------------|-----------------|--------------------------|------|-----------|-------|
| A | `nnUNetTrainerDA5CascadeFiLM` | `nnUNetTrainerDA5CascadeFullresFiLM` | ✓ | ✗ | FiLM only |
| B | `nnUNetTrainerDA5CascadeFiLMTopo` | `nnUNetTrainerDA5CascadeFullresFiLM` | ✓ | ✓ | FiLM + topology |
| C | `nnUNetTrainerDA5CascadeFiLMAdjacency` | `nnUNetTrainerDA5CascadeFullresFiLM` | ✓ | ✗ | FiLM + adjacency (pending mixin) |
| D | `nnUNetTrainerDA5CascadeTopo` | *(standard cascade fullres)* | ✗ | ✓ | Topo only — ablates FiLM |
| Baseline | `nnUNetTrainerDA5_100epochs` | *(standard cascade fullres)* | ✗ | ✗ | Plain DA5 cascade |

All trainers have `_100epochs` variants for the 100-epoch runs.

---

## Step-by-Step Workflow

### Step 1 — Experiment planning (force low-res)

The default planner only creates `3d_lowres` when the fullres patch covers <25% of the
median image. For cardiac CT this threshold is rarely met. Use the force-lowres planner:

```
nnUNetv2_plan_experiment \
    -d <DATASET_ID> \
    -pl ExperimentPlannerForceLowRes \
    -c 3d_fullres
```

> **ResEncUNet variant:** if you are using `nnUNetResEncUNetMPlans`, subclass
> `nnUNetPlannerResEncM` instead of `ExperimentPlanner` — see the comment at the
> top of `force_lowres_planner.py`.

### Step 2 — Preprocess

```
nnUNetv2_preprocess -d <DATASET_ID> -plans_name nnUNetResEncUNetMPlans \
    -c 3d_lowres 3d_cascade_fullres
```

### Step 3 — Train the low-res stage

```
nnUNetv2_train <DATASET_ID> 3d_lowres <FOLD> \
    -tr nnUNetTrainerDA5CascadeFiLM_100epochs \
    -p nnUNetResEncUNetMPlans
```

Run for all required folds (0–4 for 5-fold CV).

### Step 4 — Predict low-res on training data

The cascade fullres stage needs low-res predictions for the **training cases** as a spatial
prior. Use `nnUNetv2_predict` with the `--save_probabilities` flag, targeting the training
cases (not the test set):

```
nnUNetv2_predict \
    -i <nnUNet_raw>/<DATASET>/imagesTr \
    -o <nnUNet_results>/<DATASET>/nnUNetTrainerDA5CascadeFiLM_100epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_<FOLD>/validation \
    -d <DATASET_ID> \
    -c 3d_lowres \
    -tr nnUNetTrainerDA5CascadeFiLM_100epochs \
    -p nnUNetResEncUNetMPlans \
    -f <FOLD>
```

### Step 5 — Set up symlinks

When using different trainer classes for lowres and cascade-fullres stages, the cascade trainer
cannot automatically locate the lowres predictions. Create symlinks:

```
python scripts/setup_cascade_predictions.py \
    --lowres_trainer  nnUNetTrainerDA5CascadeFiLM_100epochs \
    --cascade_trainer nnUNetTrainerDA5CascadeFullresFiLM_100epochs \
    --dataset         Dataset001_CHD \
    --plans           nnUNetResEncUNetMPlans \
    --folds 0 1 2 3 4
```

Use `--dry_run` first to verify paths before creating symlinks.

### Step 6 — Train the cascade fullres stage

```
nnUNetv2_train <DATASET_ID> 3d_cascade_fullres <FOLD> \
    -tr nnUNetTrainerDA5CascadeFullresFiLM_100epochs \
    -p nnUNetResEncUNetMPlans
```

### Step 7 — Inference

Run lowres first, then cascade-fullres using its output as prior:

```
# Stage 1: lowres
nnUNetv2_predict -i <INPUT> -o <LOWRES_OUT> \
    -d <DATASET_ID> -c 3d_lowres \
    -tr nnUNetTrainerDA5CascadeFiLM_100epochs \
    -p nnUNetResEncUNetMPlans -f all

# Stage 2: cascade fullres (reads lowres output automatically via -prev_stage_predictions)
nnUNetv2_predict -i <INPUT> -o <FINAL_OUT> \
    -d <DATASET_ID> -c 3d_cascade_fullres \
    -tr nnUNetTrainerDA5CascadeFullresFiLM_100epochs \
    -p nnUNetResEncUNetMPlans -f all \
    -prev_stage_predictions <LOWRES_OUT>
```

---

## FiLM Conditioning Details

`FiLMConditionedResEncUNet` wraps the standard `ResidualEncoderUNet` with a FiLM layer
at the **bottleneck only**.

```
Image → Encoder → Bottleneck → FiLM(disease_vec) → Decoder → Segmentation
                                    ↑
                             disease_mlp(disease_vec) → (γ, β)
                             y = (1 + γ) * x + β
```

Key design decisions:
- **Bottleneck-only FiLM** avoids multiplicative distortion across N decoder stages
  (which would compound to `(1+γ)^N ≈ 2×` feature distortion with N=7 stages)
- **Near-identity init** (weights ~0.01 std) — training starts from the unconditioned baseline
- **2× LR multiplier** on `disease_mlp` and `bottleneck_film` parameters to learn conditioning faster
- **Classifier-free guidance dropout** (10% of batches use a zero disease vector) for robustness

---

## Topology Loss Details

`TopologyLossMixin` adds a soft-clDice loss on AO (label 6) and PA (label 7).

**Why these labels?** AO and PA are the most topologically complex structures (tube-like with
a single connected component). VSD and TGA cases are the primary failure modes where the
prediction breaks vessel connectivity.

**Weight schedule:**
```
epoch < topo_warmup_epochs (10):   weight = 0.0   (warm-up: standard loss only)
epoch in [10, 30]:                 weight ramps 0 → topo_w_high (1.0)
epoch > topo_decay_start_epoch:    weight decays toward topo_w_low (0.1)
```

The warm-up prevents the topology term from interfering during early training when
predictions are random and the soft-skeleton is meaningless.

---

## File Reference

| File | Purpose |
|------|---------|
| `nnunetv2/experiment_planning/experiment_planners/force_lowres_planner.py` | Always generate 3d_lowres + 3d_cascade_fullres |
| `nnunetv2/training/.../composed/nnUNetTrainerDA5CascadeFiLM.py` | Exp A: lowres FiLM trainer |
| `nnunetv2/training/.../composed/nnUNetTrainerDA5CascadeFiLMTopo.py` | Exp B: lowres FiLM + topo trainer |
| `nnunetv2/training/.../composed/nnUNetTrainerDA5CascadeFiLMAdjacency.py` | Exp C: lowres FiLM + adjacency (placeholder) |
| `nnunetv2/training/.../composed/nnUNetTrainerDA5CascadeTopo.py` | Exp D: lowres topo only trainer |
| `nnunetv2/training/.../composed/nnUNetTrainerDA5CascadeFullresFiLM.py` | High-res FiLM cascade trainer |
| `nnunetv2/training/.../data_augmentation/nnUNetTrainerDA5_epochs.py` | nnUNetTrainerDA5_100epochs |
| `scripts/setup_cascade_predictions.py` | Create symlinks for cross-trainer cascade |
