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
| `cross_attention_mixin.py` | `CrossAttentionConditioningMixin` | Single-head cross-attention at every decoder stage; (B,8,64) disease token sequence; attention entropy logged every 50 steps; **mutually exclusive** with `DiseaseConditioningMixin`; composes with `AuxiliaryDiagnosisMixin` (embedding-reuse path). **Each `CrossAttnBlock` is identity at init (zero-init `gamma` gate; LayerNorm on the conditioning branch only, not on x); attention scores in fp32 — fixes the earlier all-stage renormalisation that collapsed Dice to 0.0.** |
| `region_scaffold.py` | `RegionScaffoldMixin` | Hierarchical region supervision (whole-heart/blood-pool/chambers/ventricles/atria/great-vessels/myocardium) on soft probs; stepwise λ 0.3→0.15→0.05; network unchanged |
| `vessel_topo.py` | `VesselFocusedTopologyMixin` | Binary great-vessel (AO∪PA) soft-clDice as one connected structure; warmup 50→150, cap 0.15; network unchanged |
| `centerline_aux.py` | `CenterlineAuxMixin` | Centerline-weighted CE on AO/PA (weight = 1+α·skeleton from GT, detached); on-the-fly soft skeleton; network unchanged |
| `septal_focus.py` | `SeptalOversampleMixin` / `SeptalTverskyMixin` | Septal-defect ablation levers: bias fg patch sampling to the septal class (via `overwrite_class`), and FN-weighted Tversky on the septal class. Individually toggleable for ablation. |
| `disease_landmark.py` | `DiseaseLandmarkMixin` | For `Dataset050_imageCHD_DiseaseLandmarks`: soft-Dice on derived hard labels (vsd/asd_orifice_proxy, resolved by name) with **positive-supervision only** (unverified absence never penalised) + optional great-vessel (AO∪PA) clDice; λ_disease 0.3, λ_vessel_cldice 0.1; self-disables if no derived labels; network unchanged |

Shared (trainer-free) loss math lives in `nnunetv2/training/loss/anatomy_losses.py` (`SoftRegionScaffoldLoss`, `BinaryVesselClDiceLoss`, `CenterlineWeightedCELoss`, `resolve_chd_label_ids`, `build_region_groups`) — unit-tested by `nnunetv2/tests/test_anatomy_losses.py` without importing a trainer.

**Hook chain order:** Feature mixins → `ComposableTrainerMixin` → base trainer (DA5 / nnUNetTrainer)

**All hooks:** `mixin_init`, `mixin_initialize`, `mixin_prepare_forward`, `mixin_extra_loss`, `mixin_param_groups`, `mixin_fix_lr_after_scheduler`, `mixin_on_train_start`, `mixin_on_train_epoch_start`, `mixin_on_train_epoch_end`, `before_validation_case`, `after_validation_case`

---

## 2. Composed Trainer Inventory (`nnunetv2/training/nnUNetTrainer/variants/composed/`)

**Total classes: 103** (all include `_100epochs` and `_200epochs` variants at minimum)

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

### Structural "beat-the-baseline" set (network unchanged → comparable to DA5 baseline)
| Class | File | Idea |
|---|---|---|
| `nnUNetTrainerDA5RegionScaffold` + `_100e/_200e/_500e` | `nnUNetTrainerDA5RegionScaffold.py` | Hierarchical region-scaffold supervision (soft region Dice+BCE) |
| `nnUNetTrainerDA5VesselFocusedTopo` + `_100e/_200e/_500e` | `nnUNetTrainerDA5VesselFocusedTopo.py` | Binary AO∪PA great-vessel clDice (continuity-focused) |
| `nnUNetTrainerDA5CenterlineAux` + `_100e/_200e/_500e` | `nnUNetTrainerDA5CenterlineAux.py` | Centerline-weighted CE on the great vessels |

### Disease-landmark set (for `Dataset050_imageCHD_DiseaseLandmarks`)
| Class | File | Idea |
|---|---|---|
| `nnUNetTrainerDA5DiseaseLandmark` + `_100e/_200e/_500e` | `nnUNetTrainerDA5DiseaseLandmark.py` | Soft-Dice on derived disease-proxy labels (positive-supervision only) + great-vessel clDice; pairs with the `chd_landmarks` package (see §9) |
| `nnUNetTrainerDA5Septal{Oversample,Tversky,OversampleTversky}` + `_200e` | `nnUNetTrainerDA5SeptalAblation.py` | Septal-defect ablation arms (Dataset051): oversampling lever, FN-weighted Tversky lever, and combined. Reference = `nnUNetTrainerDA5`. |

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
| 1b | `CHD_Dataset041_mask_constrained.sh` | Dataset041 (ID=41) | **Stage-2 Approach A:** 2-channel mask-conditioned 7-class. Three trainers fold 0: DA5 baseline, DA5+TopoScheduled, DA5+FiLM+Topo. Predictions in `predictions_stage2/{exp1_da5,exp2_da5_topo,exp3_da5_film_topo}/`. | 200 | Yes |
| 2 | `CHD_Dataset030_ablation_topo.sh` | Dataset030 (ID=30) | **Topology + cascade ablation:** baselines + topology hypothesis (B1, B2, T1, T2, T3) — 7 trainings, fold 0 | 200 | Yes |
| 3 | `CHD_Dataset030_ablation_disease.sh` | Dataset030 (ID=30) | **Disease conditioning alone:** D1=FiLM, D2=AuxDiag, D3=CrossAttn — 3 trainings, fold 0 | 200 | Yes |
| 4 | `CHD_Dataset030_ablation_combos.sh` | Dataset030 (ID=30) | **Combinations:** C1–C5 (FiLM/Aux/CrossAttn × Topo + embedding-reuse) — 5 trainings, fold 0 | 200 | Yes |
| 5 | `CHD_Dataset030_newmethods_and_crossattn.sh` | Dataset030 (ID=30) | **Part A** — beat-the-baseline structural set: N1=RegionScaffold, N2=VesselFocusedTopo, N3=CenterlineAux (3 fullres trainings + inference, fold 0). **Part B** — re-infer existing CrossAttn / CrossAttnTopo checkpoints (conditioned + plain) to localise the 0.0 to inference vs training. | 200 | Yes |
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
- `experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py` — builds `Dataset041_ImageCHD_HU_MaskCond` (CT + binary heart prior as channel 1, 7-class labels from Dataset030). Default `--mask-source gt` binarises GT labels on the fly (no Stage-1 dependency); `--mask-source predicted` symlinks NIfTIs from `--mask-dir-tr` / `--mask-dir-ts`. channel 1 uses `nonorm` so the binary mask passes through unscaled. Used by `CHD_Dataset041_mask_constrained.sh`.
- `scripts/evaluate_wholeheart.py` — per-case Dice / IoU / HD95 / MSD / connected-component / largest-component / hole / skeleton-branch metrics. Optional `--compare-to MULTICLASS_DIR` collapses Dataset030 predictions on the fly for side-by-side comparison.
- `scripts/evaluate_topology_dataset030.py` — **topology-aware** multi-method comparison: per-class Dice + subclass_mean, clDice (AO/PA), connected-component count, largest-CC fraction, false-disconnected volume, centerline recall, AO↔PA / RA↔LA junction confusion, label-alternation, and a hard-case (ct_1063) report. Takes `--pred name=dir …` and writes `summary.csv` (ranked), `topology_table.csv`, `hard_cases.csv`, `per_case.json`. Use this to judge the new structural trainers vs DA5 — Dice alone hides their target failures.
- `scripts/generate_centerline_targets_dataset030.py` — optional offline AO∪PA skeleton precompute (`skimage.skeletonize`) for inspection / future dataloader integration; `nnUNetTrainerDA5CenterlineAux` otherwise computes the skeleton on the fly from each patch.

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
| Dataset040 | 40 | `Dataset040_WH_ImageCHD_HU_Detail` | **Binary whole-heart** derived from Dataset030 (all 7 fg labels collapsed to 1) | Yes (via stratified eval only) | Stage 1 of the whole-heart-first pipeline; built by `scripts/convert_imagechd_to_wholeheart.py`. **Stage 2 spec:** `experiments/wholeheart_decomposition/STAGE2_SPEC.md` + `anatomy_priors.yaml` (anatomy clarification, soft topology priors, disease overrides for TGA/DORV/ToF/PuA/HLHS, AO-vs-PA feature menu, HITL capture format) |
| Dataset041 | 41 | `Dataset041_ImageCHD_HU_MaskCond` | **2-channel mask-conditioned** Stage-2 dataset (channel 0 = CT, channel 1 = binary heart prior with `nonorm`; labels = Dataset030 7-class) | Yes | Stage-2 Approach A. Built by `experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py` (defaults to `--mask-source gt`; switch to `--mask-source predicted --mask-dir-{tr,ts}` once Stage-1 has produced binary masks for every case). |
| Dataset050 | 50 | `Dataset050_imageCHD_DiseaseLandmarks` | **Disease-landmark v1** derived from Dataset030: 7 anatomy labels + unified `septal_defect` label (id 8) | Yes | Built by `chd_landmarks` (**v1** derivation: LV–RV + LA–RA only; AVSD fragments). Source read-only. Script `scripts/CHD_Dataset050_disease_landmarks.sh`. Git tag `septal-v1-dataset050`. |
| Dataset060 | 60 | `Dataset060_imageCHD_CleanHoldout` | **Clean-train / missing-myo-holdout** repartition of ImageCHD: all missing-myocardium cases -> held-out test (+stratified topup to ~10%); clean cases train with septal_defect label | Yes | NEW partition (not the 030/050/051 split). For a high-quality clinical model + reported holdout. `tools/build_dataset060_clean_holdout.py`, `scripts/CHD_Dataset060_clean_holdout.sh`. |
| Dataset051 | 51 | `Dataset051_imageCHD_DiseaseLandmarksV2` | **Disease-landmark v2** (AVSD fix) — same as Dataset050 but v2 septal-defect derivation | Yes | **v2** derivation: unions VSD (LV–RV minus myo) + ASD (LA–RA direct) + AVSD cross (LV–RA/RV–LA direct); one continuous label. A/B vs Dataset050. Script `scripts/CHD_Dataset051_disease_landmarks.sh`. Git tag `septal-v2-dataset051`. |

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
| `nnunetv2/training/nnUNetTrainer/variants/composed/` | All composed trainer classes (103 total) |
| `docs/project_overview.html` | Interactive HTML dashboard |
| `docs/FEATURES.md` | **This file** — authoritative feature reference |
| `docs/CHD_TopologyLoss_Presentation.pptx` | Project presentation |
| `experiments/wholeheart_decomposition/STAGE2_SPEC.md` | Canonical Stage-2 spec: anatomy (class 7 = pulmonary artery), soft priors, disease overrides, AO/PA feature menu, HITL flows |
| `experiments/wholeheart_decomposition/anatomy_priors.yaml` | Machine-readable mirror of the priors — single source of truth for graph / rule / QC code |
| `experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py` | Builds Dataset041 (2-channel CT + binary heart prior, 7-class labels). Stage-2 Approach A. |
| `scripts/CHD_Dataset041_mask_constrained.sh` | SLURM driver: preprocess Dataset041, train DA5 / DA5+Topo / DA5+FiLM+Topo on fold 0, infer on imagesTs. |

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

---

## 9. CHD Disease-Landmark Package (`chd_landmarks/`)

Isolated research package that derives disease-specific landmark/region labels
from GT anatomy + CHD diagnosis flags. **Never** mutates existing datasets;
reads source read-only and writes only new datasets. Full guide:
`docs/chd_disease_landmark_nnunet.md`.

| Module | Role |
|---|---|
| `io.py` | Affine-preserving NIfTI + YAML/JSON I/O |
| `labels.py` | Resolve anatomy structures → ids by NAME from dataset.json (never hardcoded) |
| `metadata.py` | Parse disease flags from `imageCHD_dataset_info.xlsx`; per-case annotation status |
| `disease_rules.py` | Load `chd_disease_rules.yaml`; decide active rules (flag + anatomy present) |
| `geometry.py` | mm geometry: centroids, volumes, radii, HD95/ASSD, bboxes |
| `topology.py` | Connected components, skeletons, contact surfaces, Betti/Euler (optional libs) |
| `derived_regions.py` | Builders: VSD/ASD proxy, stenosis/coarctation ROI, AO/PA interface+candidate, hypoplastic preservation |
| `derived_label_builder.py` | Per-case orchestration + conservative merge policy (confidence-gated, interface-only overwrite) |
| `nnunet_dataset_builder.py` | Build `Dataset050` (merged labels + auxiliary masks + reports) |
| `region_based_dataset_json.py` | nnU-Net v2 region-based dataset.json (non-destructive by default) |
| `metrics.py` | Disease-aware metrics beyond Dice (VSD detection, vessel min-diameter, AO/PA leakage, clDice, volume preservation, Betti diffs) |
| `cli.py` | `inspect-dataset`, `derive-case`, `build-dataset`, `make-region-dataset-json`, `evaluate-disease-metrics` |

**Configs:** `configs/chd_{label_map,disease_rules,derived_labels,region_training,metric_config}.yaml`
**Tests:** `tests/test_chd_landmarks.py` (13 synthetic, no trainer import)
**Pairs with trainer:** `nnUNetTrainerDA5DiseaseLandmark` (§2)
**Limitations:** no `pulmonary_veins` label → APVC/TAPVR not derivable; no HLHS column in xlsx → HLHS dormant. Diagnosis flags assumed given (not a diagnostic tool).
