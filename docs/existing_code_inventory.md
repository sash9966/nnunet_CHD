# Existing-code inventory (Dataset052 Phase 0)

Branch: **`all-experiments`** (has the custom work). Audited read-only 2026-07-02.

## Datasets & label scheme (confirmed from dataset.json)
- Anatomy labels (all datasets): `1=LV-BP, 2=RV-BP, 3=LA, 4=RA, 5=Myo, 6=AO, 7=PA` (bg 0). ✔ matches spec.
- **Dataset050** = disease-landmark **v1** (septal_defect id 8), **Dataset051** = **v2** (AVSD-fixed). Both derived from `Dataset030_imageCHD_HU`, source read-only. Built by the `chd_landmarks` package. Git tags `septal-v1-dataset050`, `septal-v2-dataset051`.
- Next free id for Dataset052 = **8 (septal_defect), 9 (outflow_cap)**.

## The EXACT 050/051 `septal_defect` definition (reuse target) — ⚠ differs from the spec's description
Lives in `chd_landmarks/derived_regions.py`. It is **NOT** a "fixed-physical-thickness planar disk." It is:
- **v1** (`_septal_defect_proxy` / `build_septal_defect_proxy`): `band = contact_surface(chamberA, chamberB, dilation≈2mm)`; `defect = band & ~dilate(Myo)`; keep **largest connected component**; confidence by equivalent-diameter. Flag-gated (VSD/ASD). Merged interface-only (`may_overwrite_septal_interface_only`), never overwrites other anatomy.
- **v2** (`build_septal_defect`): union of flag-gated abnormal pairs — VSD `LV–RV` (contact − myo), ASD `LA–RA` (tight 1-voxel adjacency), AVSD cross `LV–RA`/`RV–LA` (tight adjacency); specks < `min_component_voxels` dropped.
- **Implication for Phase 2:** "reuse verbatim" = the contact‑minus‑myo band, a thin interface sheet — **it does NOT guarantee the two pools become 26‑disconnected** (it is not a closing plane). The spec's *completeness enforcement* (pools no longer 26-connected after insertion) is a **new** requirement not satisfied by the current definition. Flagged.

## Losses present (`nnunetv2/training/loss/`)
| File | Symbols | Use for 052 |
|---|---|---|
| `compound_losses.py` | `DC_and_CE_loss`, `DC_and_BCE_loss`, `DC_and_topk_loss` | base loss; extend for separators |
| `topology_losses.py` | `SoftSkeletonize`, `SoftClDiceLoss`, `TopologyLoss`, `topo_weight_schedule` | clDice on PA (endpoint + loss) |
| `confusion_penalty.py` | `resolve_confusion_pairs`, `confusion_penalty_loss` | **directly relevant** — penalize Ao↔PA / LV↔RV confusion |
| `anatomy_losses.py` | `SoftRegionScaffoldLoss`, `BinaryVesselClDiceLoss`, `CenterlineWeightedCELoss` | region/vessel aux |
| `dice.py`, `robust_ce_loss.py`, `curriculum_weights.py`, `deep_supervision.py` | standard | — |

## ⚠ Spec-referenced infrastructure that does NOT exist here (reimplement per spec)
- **`CombinedLoss`** — not present. (Base is `DC_and_CE_loss`; I'll add a composable combiner or extend the mixin `mixin_extra_loss` path.)
- **Byrne / persistence loss** — not present. No `gudhi`/`cripser` on disk either → persistent-homology unavailable; must use Euler/connected-component proxies for a "closure/no-spurious-holes" penalty.
- **Tversky loss** — not present. Reimplement (FN-weighted) for the thin separator classes.
- **"ROI oversampling / existing 90%-foreground logic"** — not present as described. What exists: nnU-Net's built-in foreground oversampling (`oversample_foreground_percent`, `data_loader.py`) + `nnUNetTrainer_probabilisticOversampling` variant. Separator-class oversampling must be added on top of these.

## Trainer infrastructure (reuse)
- Composable mixin system: `variants/mixins/_base.py` (`TrainerMixin`/`ComposableTrainerMixin`), feature mixins, thin composed trainers. New 052 arms subclass `nnUNetTrainerDA5` via mixins.
- `variants/mixins/disease_landmark.py` — `DiseaseLandmarkMixin`: positive-only soft-Dice on derived labels + AO∪PA clDice. Closest existing analog to the 052 separator-loss idea.
- Local caveat: full trainer import blocked by `acvl_utils.insert_crop_into_image` mismatch → pure loss/geometry functions are locally testable; trainers only on the cluster.

## Diagnosis sources (on disk)
- `imageCHD_dataset_info.xlsx` (repo root): ASD,VSD,AVSD,ToF,TGA,DORV,CAT,CA,AAH,DAA,IAA,PA,APVC,DSVC,PDA,PAS. **No SV (single-ventricle) column.**
- `…/AlisonMarsden/Segmentations/imageCHD_dataset_WH_diagnosis_info-june21updates(Sheet1).csv`: has **`SV`** (single ventricle) + `TGA`. → `single_ventricle` flag from here; `tga_ivs` = TGA ∧ no LV–RV VSD gap.

## Libs available
cc3d ✔, SimpleITK ✔, skimage ✔, scipy ✔, nibabel ✔ · **gudhi ✘, cripser ✘** (no persistent homology → Euler/CC proxies).
