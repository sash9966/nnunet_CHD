# nnunet_CHD — Feature Reference

**Branch:** `all-experiments`  
**Last updated:** 2026-05-13  
**Purpose:** Authoritative inventory of every feature, trainer, script, and dataset.
Read this file at the start of any conversation to reconstruct full project state without re-scanning the codebase.

---

## 1. Mixin Library (`nnunetv2/training/nnUNetTrainer/variants/mixins/`)

| File | Class | What it does |
|---|---|---|
| `_base.py` | `TrainerMixin` | No-op hook terminators for all mixin_* hooks |
| `_base.py` | `ComposableTrainerMixin` | Dispatcher: calls all mixin_* hooks in MRO order |
| `topology_loss.py` | `TopologyLossMixin` | Fixed-weight (0.2) soft-clDice on AO+PA classes |
| `topology_loss.py` | `TopologyLossScheduledMixin` | Warmup→plateau→cosine-decay clDice; uses `topo_weight_schedule()` |
| `disease_conditioning.py` | `DiseaseConditioningMixin` | FiLM or Gated conditioning; requires `disease_map.json` |
| `curriculum_weights.py` | `CurriculumWeightsMixin` | AO/PA CE class weights upweighted in early epochs |
| `aux_diagnosis_mixin.py` | `AuxiliaryDiagnosisMixin` | Bottleneck MLP classification head (K=8); BCEWithLogitsLoss aux loss (weight 0.1); exposes 256-d `diagnosis_embedding`; 3× LR on head params |
| `cross_attention_mixin.py` | `CrossAttentionConditioningMixin` | Single-head cross-attention at every decoder stage; (B,8,64) disease token sequence; attention entropy logged every 50 steps; **mutually exclusive** with `DiseaseConditioningMixin`; composes with `AuxiliaryDiagnosisMixin` (embedding-reuse path) |

**Hook chain order:** Feature mixins → `ComposableTrainerMixin` → base trainer (DA5 / nnUNetTrainer)

**All hooks:** `mixin_init`, `mixin_initialize`, `mixin_prepare_forward`, `mixin_extra_loss`, `mixin_param_groups`, `mixin_fix_lr_after_scheduler`, `mixin_on_train_start`, `mixin_on_train_epoch_start`, `mixin_on_train_epoch_end`, `before_validation_case`, `after_validation_case`

---

## 2. Composed Trainer Inventory (`nnunetv2/training/nnUNetTrainer/variants/composed/`)

**Total classes: 84** (all include `_100epochs` and `_200epochs` variants at minimum)

### DA5 Baseline (no extras)
- `nnUNetTrainerDA5_200epochs`, `_100epochs` — in `nnUNetTrainerDA5_epochs.py`

### FiLM Disease Conditioning
| Class | File | Notes |
|---|---|---|
| `nnUNetTrainerDA5FiLMV3` + `_100e/_200e/_500e/_1000e` | `nnUNetTrainerDA5FiLMV3.py` | Bottleneck-only FiLM, K=8, LR×2 |
| `nnUNetTrainerDA5FiLMV2` + epoch variants | `nnUNetTrainerDA5FiLMV2.py` | Aliases → V3 for backward compat |
| `nnUNetTrainerDA5FiLMDropout` + `_100e/_200e` | `nnUNetTrainerDA5FiLMDropout.py` | FiLM + 10% CFG dropout |
| `nnUNetTrainerDA5FiLMAdaptive` + `_100e/_200e` | `nnUNetTrainerDA5FiLMAdaptive.py` | Adaptive LR scheduling on disease params |

### Topology Loss (no FiLM)
| Class | File |
|---|---|
| `nnUNetTrainerTopoLossV2` + `_100e/_200e/_1000e` | `nnUNetTrainerTopoLossV2.py` |
| `nnUNetTrainerDA5CascadeTopo` + `_100e/_200e` | `nnUNetTrainerDA5CascadeTopo.py` |

### Scheduled Topology Loss
| Class | File | Stage |
|---|---|---|
| `nnUNetTrainerDA5TopoScheduled` + `_100e/_200e` | `nnUNetTrainerDA5TopoScheduled.py` | 3d_fullres |
| `nnUNetTrainerDA5CascadeTopoScheduled` + `_100e/_200e` | same | 3d_lowres |
| `nnUNetTrainerDA5CascadeFullresTopoScheduled` + `_100e/_200e` | same | 3d_cascade_fullres |

### FiLM + Topology
| Class | File |
|---|---|
| `nnUNetTrainerDA5FiLMTopo` + `_100e/_200e/_500e/_1000e` | `nnUNetTrainerDA5FiLMTopo.py` |
| `nnUNetTrainerDA5CascadeFiLMTopo` + `_100e/_200e` | `nnUNetTrainerDA5CascadeFiLMTopo.py` |

### Curriculum Weighting
| Class | File |
|---|---|
| `nnUNetTrainerDA5Curriculum` + `_100e/_200e/_1000e` | `nnUNetTrainerDA5Curriculum.py` |
| `nnUNetTrainerDA5TopoCurriculum` + `_100e/_200e/_1000e` | `nnUNetTrainerDA5TopoCurriculum.py` |
| `nnUNetTrainerDA5FiLMCurriculum` + `_100e/_200e/_500e` | `nnUNetTrainerDA5FiLMCurriculum.py` |
| `nnUNetTrainerDA5FiLMTopoCurriculum` + `_100e/_200e/_500e/_1000e` | `nnUNetTrainerDA5FiLMTopoCurriculum.py` |

### Cascade Lowres Trainers
| Class | Pairs with fullres |
|---|---|
| `nnUNetTrainerDA5_200epochs` | `nnUNetTrainerDA5CascadeFullresBaseline_200epochs` |
| `nnUNetTrainerDA5CascadeFiLM_200epochs` | `nnUNetTrainerDA5CascadeFullresFiLM_200epochs` |
| `nnUNetTrainerDA5CascadeTopo_200epochs` | `nnUNetTrainerDA5CascadeFullresTopo_200epochs` |
| `nnUNetTrainerDA5CascadeFiLMTopo_200epochs` | `nnUNetTrainerDA5CascadeFullresFiLMTopo_200epochs` |
| `nnUNetTrainerDA5CascadeTopoScheduled_200epochs` | `nnUNetTrainerDA5CascadeFullresTopoScheduled_200epochs` |
| `nnUNetTrainerDA5CascadeFiLMAdjacency_200epochs` | — (adjacency loss variant) |

### Cascade Fullres Trainers (in `nnUNetTrainerDA5CascadeFullresVariants.py`)
- `nnUNetTrainerDA5CascadeFullresBaseline` + `_100e/_200e`
- `nnUNetTrainerDA5CascadeFullresTopo` + `_100e/_200e`
- `nnUNetTrainerDA5CascadeFullresFiLMTopo` + `_100e/_200e`

### Cross-Attention Conditioning
| Class | File |
|---|---|
| `nnUNetTrainerDA5CrossAttn` + `_100e/_200e` | `nnUNetTrainerDA5CrossAttn.py` |
| `nnUNetTrainerDA5CrossAttnTopo` + `_100e/_200e` | same |
| `nnUNetTrainerDA5AuxDiagCrossAttn` + `_100e/_200e` | same |

### Auxiliary Diagnosis Head
| Class | File |
|---|---|
| `nnUNetTrainerDA5AuxDiag` + `_100e/_200e` | `nnUNetTrainerDA5AuxDiag.py` |
| `nnUNetTrainerDA5AuxDiagTopo` + `_100e/_200e` | same |
| `nnUNetTrainerDA5FiLMAuxDiag` + `_100e/_200e` | same |

### Other
- `nnUNetTrainerDA5Gated` + `_100e/_200e` — Spatially-gated disease conditioning (GatedConditionedResEncUNet)
- `nnUNetTrainerDA5DiseaseVecTopo` + `_100e/_200e` — Disease vector (MLP concat) + topology
- `nnUNetTrainerDA5DiseaseVecV2` + `_100e/_200e` — Disease vector V2
- `nnUNetTrainerDA5Confusion` + `_100e/_200e` — Confusion/boundary-aware loss

---

## 3. SLURM Scripts (`scripts/`)

**See `scripts/README.md` for the scientific buildup order — that file is the canonical runbook.**

| Phase | Script | Dataset | Purpose | Epochs | Resume? |
|---|---|---|---|---|---|
| 1 | `CHD_Dataset040_wholeheart.sh` | Dataset040 (ID=40) | **Whole-heart Stage 1:** binary heart vs not-heart. Trains fullres + lowres + cascade DA5 and infers on imagesTs. See `docs/wholeheart_pipeline.md`. | 200 | Yes |
| 2 | `CHD_Dataset030_ablation_topo.sh` | Dataset030 (ID=30) | **Topology + cascade ablation:** baselines + topology hypothesis (B1, B2, T1, T2, T3) — 7 trainings, fold 0 | 200 | Yes |
| 3 | `CHD_Dataset030_ablation_disease.sh` | Dataset030 (ID=30) | **Disease conditioning alone:** D1=FiLM, D2=AuxDiag, D3=CrossAttn — 3 trainings, fold 0 | 200 | Yes |
| 4 | `CHD_Dataset030_ablation_combos.sh` | Dataset030 (ID=30) | **Combinations:** C1–C5 (FiLM/Aux/CrossAttn × Topo + embedding-reuse) — 5 trainings, fold 0 | 200 | Yes |
| C | `CHD_Dataset013_Fanwei.sh` | Dataset013 (ID=13) | Clinical baseline (no disease labels) | 200 | Yes |
| C | `CHD_Dataset020_clinical.sh` | Dataset020 (ID=20) | Clinical deployment (Fanwei + imageCHD merged) | 200 | Yes |
| — | `CHD_Dataset001_cascade_200epochs.sh` | Dataset001 (ID=1) | Historical D001 ablation — kept for reproducibility, not featured | 200 | Yes |
| — | `CHD_Dataset030_reinfer_all.sh` | Dataset030 | Operational helper: re-infer all trained models without re-training | — | — |
| — | `CHD_Dataset030_reinfer_conditioned.sh` | Dataset030 | Same but for disease-conditioned models only | — | — |

**Support scripts:**
- `scripts/make_disease_map.py` — converts imageCHD diagnosis CSV → `disease_map.json`; called as Phase 0b in all Dataset030/001 scripts (see usage below)
- `scripts/setup_cascade_predictions.py` — creates symlinks so cascade fullres trainers find lowres predictions when trainer class names differ
- `scripts/generate_cascade_preds.py` — runs a trained fold-0 lowres model over ALL training+val cases and saves `.b2nd` predictions to `predicted_next_stage/3d_cascade_fullres/`; needed because `perform_actual_validation` only saves the ~7 fold-0 val cases when training fold 0 only; called inside the ablation + wholeheart SLURM scripts at Phase 2.5
- `scripts/make_presentation.py` — generates `docs/CHD_TopologyLoss_Presentation.pptx`
- `scripts/test_curriculum_class_weights.py` — unit test for curriculum weights
- `scripts/convert_imagechd_to_wholeheart.py` — builds `Dataset040_WH_ImageCHD_HU_Detail` (binary heart) from Dataset030; symlinks images, binarises labelsTr, writes dataset.json. Used by `CHD_Dataset040_wholeheart.sh`.
- `scripts/evaluate_wholeheart.py` — per-case Dice / IoU / HD95 / MSD / connected-component / largest-component / hole / skeleton-branch metrics. Optional `--compare-to MULTICLASS_DIR` collapses Dataset030 predictions on the fly for side-by-side comparison.

**make_disease_map.py usage:**
```bash
# Via dataset ID (uses $nnUNet_raw / $nnUNet_preprocessed env vars):
python scripts/make_disease_map.py --dataset-id 30
python scripts/make_disease_map.py --dataset-id 1

# Explicit paths:
python scripts/make_disease_map.py \
    --csv  $nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_diagnosis.csv \
    --out  $nnUNet_preprocessed/Dataset030_imageCHD_HU/disease_map.json

# Dry-run (inspect without writing):
python scripts/make_disease_map.py --dataset-id 30 --dry-run
```
Supported CSV layouts:
- **Binary columns**: one column per disease (`HLHS,ASD,VSD,AVSD,DORV,PuA,ToF,TGA`), 0/1 values
- **String diagnosis column**: comma-separated tags per row (`"ASD,VSD"`)
- **Integer type column**: imageCHD integer class ID (0=Normal, 1=HLHS, … 8=TGA)

### SLURM output files
SLURM logs go to: `/scratch/users/sastocke/nnunet_CHD/logs/<job-name>_<jobid>.out/.err`  
Run before first submission: `mkdir -p /scratch/users/sastocke/nnunet_CHD/logs`

---

## 4. Checkpoint / Resume System

**How it works:**
- Each SLURM script (Dataset030, Dataset020, Dataset001 200e) has a checkpoint system
- Every individual training run (per fold per trainer) writes a `.done` file when complete
- On resubmission the script skips steps with `.done` files
- nnUNet resumes mid-epoch training automatically from `checkpoint_latest.pth`

**Checkpoint location:**
```
${nnUNet_results}/<DATASET_NAME>/.checkpoints/<script-name>/
```

**Check what ran / what is pending:**
```bash
# See completed steps
ls ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_imageCHD/*.done | sort

# Same for other datasets
ls ${nnUNet_results}/Dataset020FanweiDataandImageCHD_HU/.checkpoints/CHD_Dataset020_clinical/*.done | sort
ls ${nnUNet_results}/Dataset001_all_imageCHD/.checkpoints/CHD_Dataset001_cascade_200epochs/*.done | sort
```

**Resume after wall-time interrupt:**
Simply resubmit the same script: `sbatch scripts/CHD_Dataset030_imageCHD.sh`

**Checkpoint key naming convention:**
`p{PHASE}_{trainer-abbreviation}_{fold}` e.g. `p2_lowres_DA5CascadeTopo200e_fold3`

---

## 5. Dataset Registry

| Dataset | ID | Name | Purpose | Disease labels? | Notes |
|---|---|---|---|---|---|
| Dataset001 | 1 | `Dataset001_all_imageCHD` | Smoothed imageCHD, cascade ablation research | Yes | Primary research dataset |
| Dataset013 | 13 | `Dataset013_Fanweidatacleaned` | Fanwei cleaned clinical data — standalone baseline | No | No FiLM/topology; DA5 + cascade only |
| Dataset020 | 20 | `Dataset020FanweiDataandImageCHD_HU` | Fanwei + imageCHD combined | No | Clinical deployment model; superset of Dataset013 |
| Dataset030 | 30 | `Dataset030_imageCHD_HU` | Kaggle imageCHD with HU values | Yes (same labels) | More vessel branches, less smooth |
| Dataset040 | 40 | `Dataset040_WH_ImageCHD_HU_Detail` | **Binary whole-heart** derived from Dataset030 (all 7 fg labels collapsed to 1) | Yes (via stratified eval only) | Stage 1 of the whole-heart-first pipeline; built by `scripts/convert_imagechd_to_wholeheart.py` |

**Label scheme (all datasets use same names):**

| ID | Structure | Abbrev | Topology target |
|---|---|---|---|
| 0 | Background | BG | — |
| 1 | LV blood pool | LV-BP | — |
| 2 | RV blood pool | RV-BP | — |
| 3 | Left atrium | LA | — |
| 4 | Right atrium | RA | — |
| 5 | Myocardium | Myo | — |
| 6 | Aorta | AO | soft-clDice |
| 7 | Pulmonary artery | PA | soft-clDice |

**Disease flags (K=8):** HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA  
**Disease map location:** `${nnUNet_preprocessed}/<dataset>/disease_map.json`

---

## 6. Architecture & Key Files

**Plans:** `nnUNetResEncUNetMPlans` (planner: `nnUNetPlannerResEncM`)  
**Base network:** `ResidualEncoderUNet` (external: `dynamic-network-architectures`)

| File | Contents |
|---|---|
| `nnunetv2/training/loss/topology_losses.py` | `SoftSkeletonize`, `SoftClDiceLoss`, `TopologyLoss`, `topo_weight_schedule` |
| `nnunetv2/architectures/film_conditioned_unet.py` | `FiLMConditionedResEncUNet` |
| `nnunetv2/architectures/gated_conditioned_unet.py` | `GatedConditionedResEncUNet` |
| `nnunetv2/inference/predict_disease_conditioned.py` | Inference entry point for FiLM/Gated models |
| `nnunetv2/training/nnUNetTrainer/variants/mixins/` | All mixin implementations |
| `nnunetv2/training/nnUNetTrainer/variants/composed/` | All composed trainer classes (84 total) |
| `docs/project_overview.html` | Interactive HTML dashboard |
| `docs/FEATURES.md` | **This file** — authoritative feature reference |
| `docs/CHD_TopologyLoss_Presentation.pptx` | Project presentation |

**Topology loss note:** `topo_weight_schedule()` uses absolute epoch numbers.
For 200-epoch runs the defaults (warmup=10, decay_start=40) give: ramp 0→10, plateau 10→40, decay 40→200.
Consider adjusting `topo_decay_start` to ~80 for longer plateau in 200-epoch training.

---

## 7. Known Issues & Notes

- Local `acvl_utils` version mismatch: `insert_crop_into_image` not found → blocks trainer import locally, works fine on SLURM servers
- FiLM decoder films removed (were causing (1+γ)^N feature explosion with N=7 stages)
- `build_network_architecture` is called as `@staticmethod` by `nnUNetPredictor` — instance overrides must detect `isinstance(self, str)` and shift args
- Old monolithic trainers (`nnUNetTrainerDA5FiLM`, `nnUNetTrainerDA5DiseaseVec`, `nnUNetTrainerTopoLoss`) preserved for backward compat, not to be extended

---

## 8. Maintenance Rule

**When adding new trainers, mixins, or scripts:**
1. Update this file (`docs/FEATURES.md`) — section 2, 3, or 4 as appropriate
2. Update `docs/project_overview.html` — trainer count badge, nav, and relevant section
3. Commit both files together with the code change
