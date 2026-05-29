# Stage-2 Spec — Anatomy, Priors, AO/PA Features, HITL

Canonical reference for the Stage-2 task. All per-approach READMEs in this folder
(`mask_constrained_nnunet/`, `graph_based/`, `rule_based/`, `medsam_slicer/`,
`human_in_the_loop/`, …) should source their anatomy assumptions and prior rules
from **this file** and the machine-readable mirror at
[`anatomy_priors.yaml`](anatomy_priors.yaml).

The Stage-1 binary heart mask is assumed available
(`$nnUNet_results/Dataset040_WH_ImageCHD_HU_Detail/predictions_wholeheart/…`).
Stage-2's job is to decompose that mask into seven semantic classes.

---

## 1. Anatomical clarification — class 7 is the pulmonary artery

The ImageCHD seven-class scheme (label IDs from `docs/FEATURES.md §5`) is:

| ID | Structure | Abbrev | Code name |
|---|---|---|---|
| 1 | LV blood pool | LV-BP | `left_ventricle` |
| 2 | RV blood pool | RV-BP | `right_ventricle` |
| 3 | Left atrium | LA | `left_atrium` |
| 4 | Right atrium | RA | `right_atrium` |
| 5 | Myocardium | Myo | `myocardium` |
| 6 | Aorta | AO | `aorta` |
| 7 | Pulmonary artery / pulmonary outflow | PA | `pulmonary_artery` |

**Class 7 is the pulmonary artery (pulmonary outflow), not pulmonary veins.**
Some older documentation in this repo and in upstream ImageCHD discussion calls
it "PA" without qualifying — that ambiguity is what this section closes.

Anatomical groupings (used by graph/QC code):

- **ventricular_blood_pools** — `left_ventricle`, `right_ventricle`
- **atrial_blood_pools** — `left_atrium`, `right_atrium`
- **myocardium** — `myocardium`
- **arterial_outflow** — `aorta`, `pulmonary_artery`

The exact label indices must still be read from each dataset's `dataset.json` at
runtime. The interpretation above is what those indices *mean*.

---

## 2. Topology priors — soft, never hard

In a healthy heart:

- aorta arises from the **left ventricle**, courses superiorly with an arch
- pulmonary artery arises from the **right ventricle**, bifurcates into L/R PA
- both vessels are arterial, contrast-filled, locally similar in HU

Stage-2 must distinguish them, but in **CHD the ventriculo-arterial connections
themselves may be abnormal**. Hard-coding "AO comes from LV" is wrong for TGA
and DORV. Therefore all priors are `soft`.

### Defaults (`prior_strength: soft`)

| Vessel | Usually adjacent to | Expected behaviour |
|---|---|---|
| `aorta` | `left_ventricle`, `myocardium` | systemic outflow, arch-like course |
| `pulmonary_artery` | `right_ventricle`, `myocardium` | pulmonary outflow, bifurcation pattern if visible |

### Disease-conditioned overrides (`prior_strength: disease_conditioned`)

Keyed by the K=8 disease flags from `disease_map.json`
(`HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA`).

- **TGA** (transposition of great arteries): aorta *may* connect to RV;
  pulmonary artery *may* connect to LV. Swapped relative to normal.
- **DORV** (double outlet right ventricle): both aorta and pulmonary artery
  *may* connect to RV.
- **ToF** (tetralogy of Fallot): pulmonary artery may be small-calibre /
  stenotic, with abnormal RVOT geometry.
- **PuA** (pulmonary atresia): pulmonary outflow may be absent or replaced by
  collaterals — graph code must handle the case where PA is not present as a
  single connected component.

All overrides live in [`anatomy_priors.yaml`](anatomy_priors.yaml) under
`disease_topology_priors.<disease_key>`.

### The "no silent relabel" rule

Topology priors are **diagnostic signals, not authority**. Approaches that
detect a violation must:

1. **Log** the conflict per case (case ID, disease flags, predicted topology,
   prior that was violated).
2. **Surface uncertainty** — flag the affected voxels/branches in an
   uncertainty mask or `qc_flags.json`.
3. **Suggest** a correction (e.g. swap AO↔PA labels) and emit it as a
   *proposal*, not an applied edit.
4. **Defer** the actual relabeling to the next step in the pipeline (human
   review, a learned graph model with the prior as a feature, or an explicit
   `--apply-suggestions` flag).

No code path in this folder should silently swap labels because a heuristic
said the wrong ventricle was connected. CHD's whole point is that "wrong" can
be ground truth.

---

## 3. Features that distinguish aorta from pulmonary artery

Stage-2 approaches that need to disambiguate AO vs PA should consider these
inputs. Not all approaches use all features; this is the menu.

**Local (voxel/patch):**
- direct nnU-Net class probabilities (from a 7-class trainer)
- low-resolution prior probabilities (cascade lowres model)
- local HU statistics (mean, std, gradient magnitude)
- model–model disagreement / softmax entropy

**Structural (component/branch):**
- adjacency / contact area with LV blood pool
- adjacency / contact area with RV blood pool
- adjacency with myocardium (where the vessel emerges)
- geodesic distance to LV blood pool
- geodesic distance to RV blood pool
- vessel radius profile along the centerline (EDT max along skeleton)
- vessel length
- branch pattern (bifurcation count — PA bifurcates early, AO arches)
- centerline curvature / tortuosity
- direction vector from ventricular outflow

**Global:**
- normalised heart coordinates (superior/inferior, anterior/posterior, left/right)
- disease-code embedding (K=8 one-hot or learned)

**Cross-model:**
- agreement between direct nnU-Net, rule-based, and graph-based predictions

The graph-based approach (`graph_based/`) is the natural consumer of the
structural features; the rule-based approach (`rule_based/`) consumes the
adjacency and geodesic-distance features. Disease embeddings should be
available to both as optional inputs.

---

## 4. Graph representation (for `graph_based/`)

When building a graph from the binary mask:

- **nodes** — supervoxels, connected regions, skeleton branch points, chamber
  centroids, or vessel segments (pick one granularity per experiment)
- **edges** — anatomical adjacency, skeleton connectivity, or spatial
  neighbourhood within a radius
- **node features** — local intensity stats, region shape, normalised
  position, distance-to-chambers, per-class probabilities, disease metadata
- **edge features** — contact area, physical distance, direction, continuity
  confidence (boundary gradient)

Start with the simplest baselines before GNNs:

1. mask-constrained nnU-Net (Approach A) — voxel-level baseline
2. rule-based geodesic propagation (Approach D) — interpretable baseline
3. supervoxel graph classifier (MLP over precomputed features)
4. centerline / branch graph classifier
5. disease-conditioned GNN

Each step should beat the previous one on at least one of:
Dice on AO+PA, `n_components`, `largest_component_fraction`, AO-vs-PA swap
rate (a per-case categorical metric).

---

## 5. Human-in-the-loop — arterial-outflow focus

Most clinician-correction time is spent on the great arteries:
AO↔PA swaps, disconnected PA branches, ambiguous outflow tracts. The HITL
plumbing in `human_in_the_loop/` and the interactive correction in
`medsam_slicer/` should prioritise that surface.

### Things to expose to the clinician per case

- the CT
- the Stage-1 binary heart mask
- the Stage-2 prediction (7-class)
- a per-voxel **uncertainty map** (softmax entropy or model disagreement)
- a **disconnected-component map** for AO and PA (each component coloured by
  its index so disconnected branches are obvious)
- an **AO/PA disagreement map** — voxels where rule-based, graph-based, and
  nnU-Net predictions differ
- per-case disease flags (from `disease_map.json`) so the clinician knows
  whether to expect TGA/DORV anatomy

### Three correction modes

1. **Segment Editor baseline** — load the bundle in 3D Slicer, correct with
   Paint/Erase/Threshold, export back. Already scaffolded in
   `human_in_the_loop/`.
2. **Centerline-guided correction** — clinician drops a Slicer fiducial in
   the PA trunk or aortic root. Pipeline traces the connected branch / vessel
   centerline and proposes relabeling all connected voxels. Clinician accepts
   or rejects. Especially useful for the SeqSeg-style workflows in
   `medsam_slicer/`.
3. **Box-prompted (MedSAM)** — clinician draws a 3D box around a mislabeled
   vessel; MedSAM proposes a local segmentation, which is inserted as a
   correction *proposal* (not ground truth). See
   `medsam_slicer/README.md`.

### Active-learning capture format

Every correction should be saved in a structured way so it can feed
fine-tuning, error analysis, and disease-specific evaluation:

```
corrections/<case_id>/
  original_prediction.nii.gz       # what the pipeline produced
  corrected_labelmap.nii.gz        # what the clinician produced
  diff.nii.gz                      # voxel-level difference (uint8 mask)
  metadata.json                    # case_id, disease_flags, model_ids,
                                   # timestamp, user_id (optional),
                                   # correction_type {segment_editor,
                                   # centerline, medsam_box}, notes
```

Downstream uses:

- fine-tuning a Stage-2 model on corrected cases
- per-disease error stratification
- uncertainty calibration (does high uncertainty predict where corrections
  land?)
- regression detection across pipeline versions

---

## 6. How approaches in this folder consume this spec

| Subfolder | Uses anatomy table | Uses priors | Uses AO/PA features | Uses HITL spec |
|---|---|---|---|---|
| `mask_constrained_nnunet/` | yes | as optional eval | direct probs only | no |
| `mask_postprocessing/` | yes | no | no | no |
| `rule_based/` | yes | **yes — as soft features and QC** | adjacency + geodesic | no |
| `graph_based/` | yes | **yes — as node features** | full menu | no |
| `medsam_slicer/` | yes | as QC overlay | no | **yes — boxes / seeds** |
| `dino_features/` | yes | no | learned features | no |
| `human_in_the_loop/` | yes | as overlay | as overlay | **yes — workflow** |

Each per-approach README has a short pointer back to this file so the spec
stays single-sourced.

---

## 7. Related docs

- [`README.md`](README.md) — decision tree across the seven approaches
- [`anatomy_priors.yaml`](anatomy_priors.yaml) — machine-readable mirror
- [`../../docs/wholeheart_pipeline.md`](../../docs/wholeheart_pipeline.md) — Stage-1 spec
- [`../../docs/FEATURES.md`](../../docs/FEATURES.md) §5 — label scheme + disease flags
