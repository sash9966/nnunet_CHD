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

### Other
- `nnUNetTrainerDA5Gated` + `_100e/_200e` — Spatially-gated disease conditioning (GatedConditionedResEncUNet)
- `nnUNetTrainerDA5DiseaseVecTopo` + `_100e/_200e` — Disease vector (MLP concat) + topology
- `nnUNetTrainerDA5DiseaseVecV2` + `_100e/_200e` — Disease vector V2
- `nnUNetTrainerDA5Confusion` + `_100e/_200e` — Confusion/boundary-aware loss

---

## 3. SLURM Scripts (`scripts/`)

| Script | Dataset | Purpose | Epochs | Resume? |
|---|---|---|---|---|
| `CHD_Dataset030_imageCHD.sh` | Dataset030 (ID=30) | 3 experiments: DA5 fullres + cascade baseline + cascade topo | 200 | Yes |
| `CHD_Dataset020_clinical.sh` | Dataset020 (ID=20) | Clinical deployment: DA5 fullres + cascade baseline, 5-fold ensemble | 200 | Yes |
| `CHD_Dataset001_cascade_200epochs.sh` | Dataset001 (ID=1) | 4 cascade pairs: DA5/FiLM/Topo/FiLMTopo | 200 | Yes |
| `CHD_Cascade_allFolds.sh` | Dataset001 (ID=1) | Same 4 cascade pairs — legacy 100-epoch version | 100 | No |
| `train_cascade_ablation.sh` | Dataset001 | Earlier ablation script | — | No |

**Support scripts:**
- `scripts/setup_cascade_predictions.py` — creates symlinks so cascade fullres trainers find lowres predictions when trainer class names differ
- `scripts/make_presentation.py` — generates `docs/CHD_TopologyLoss_Presentation.pptx`
- `scripts/test_curriculum_class_weights.py` — unit test for curriculum weights

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
| Dataset020 | 20 | `Dataset020FanweiDataandImageCHD_HU` | Fanwei + imageCHD combined | No | Clinical deployment model |
| Dataset030 | 30 | `Dataset030_imageCHD_HU` | Kaggle imageCHD with HU values | Yes (same labels) | More vessel branches, less smooth |

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
