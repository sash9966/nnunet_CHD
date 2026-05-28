# `scripts/` — what to run and in what order

This folder is the canonical runbook. Every SLURM script and Python utility
below has a single purpose; pick the **phase** you want to investigate and run
the corresponding scripts in order.

If you're new to the project, read top-to-bottom. The phases form a scientific
buildup: each one isolates a single question.

---

## Quick reference

| Phase | Question | Script(s) | Compute |
|---|---|---|---|
| **0** | Setup | `convert_imagechd_to_wholeheart.py`, `make_disease_map.py` | seconds |
| **1** | Does whole-heart-first preserve topology better than joint multiclass? | `CHD_Dataset040_wholeheart.sh` | ~2–3 walltime cycles |
| **2** | Does topology loss help, and where (fullres vs lowres vs cascade)? | `CHD_Dataset030_ablation_topo.sh` | ~3 walltime cycles |
| **3** | Which disease-conditioning method works best? | `CHD_Dataset030_ablation_disease.sh` | ~1 walltime cycle |
| **4** | Do combinations beat either alone? | `CHD_Dataset030_ablation_combos.sh` | ~2 walltime cycles |
| **C** | Clinical deployment baselines | `CHD_Dataset013_Fanwei.sh`, `CHD_Dataset020_clinical.sh` | separate track |

Each SLURM script resumes from `.done` markers + nnU-Net's `checkpoint_latest.pth`
after wall-time interrupts — just resubmit.

---

## Phase 0 — Setup (run once per dataset)

```bash
# Build the binary whole-heart dataset (Phase 1).  Symlinks images, binarises labels.
python scripts/convert_imagechd_to_wholeheart.py --dry-run    # inspect plan
python scripts/convert_imagechd_to_wholeheart.py              # commit
```

`make_disease_map.py` is invoked **automatically** inside the SLURM scripts that
need a `disease_map.json` — you do not normally call it by hand. If you need to
regenerate it after changing `imageCHD_dataset_info.xlsx`:

```bash
python scripts/make_disease_map.py --dataset-id 30
```

---

## Phase 1 — Whole-heart hypothesis (Dataset040)

Test whether a binary heart-vs-not-heart Stage 1 produces a topologically
cleaner mask than the joint multiclass model. Trains 3 binary-heart models on
the same data: fullres, lowres, and cascade-fullres.

```bash
sbatch scripts/CHD_Dataset040_wholeheart.sh
```

After training, evaluate with topology proxies:

```bash
PRED=$nnUNet_results/Dataset040_ImageCHD_HU_WH/predictions_wholeheart
GT=$nnUNet_raw/Dataset040_ImageCHD_HU_WH/labelsTs

python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_fullres --gt-dir $GT --out eval_fullres.csv
python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_lowres  --gt-dir $GT --out eval_lowres.csv
python scripts/evaluate_wholeheart.py --pred-dir $PRED/DA5_cascade --gt-dir $GT --out eval_cascade.csv

# Compare cascade vs collapsed multiclass baseline (Dataset030 → binarised on the fly)
python scripts/evaluate_wholeheart.py \
    --pred-dir   $PRED/DA5_cascade \
    --gt-dir     $GT \
    --compare-to $nnUNet_results/Dataset030_imageCHD_HU/predictions/DA5_fullres \
    --out        eval_cascade_vs_multiclass.csv
```

Full spec: [`docs/wholeheart_pipeline.md`](../docs/wholeheart_pipeline.md).
Stage-2 decomposition options scaffolded in
[`experiments/wholeheart_decomposition/`](../experiments/wholeheart_decomposition/README.md).

---

## Phase 2 — Topology + cascade hypothesis (Dataset030, fold 0)

5 rows in the ablation matrix:

| ID | Description |
|---|---|
| **B1** | DA5 fullres baseline — control |
| **B2** | DA5 → DA5 cascade — classic cascade control |
| **T1** | DA5 + topology loss at fullres |
| **T2** | DA5 + topology at **lowres only**, plain fullres cascade — primary hypothesis |
| **T3** | DA5 + topology at **both** lowres and cascade fullres |

```bash
sbatch scripts/CHD_Dataset030_ablation_topo.sh
```

Predictions in `$nnUNet_results/Dataset030_imageCHD_HU/predictions_ablation/`.
Compare T2 vs T3 to decide if topology at both stages is worth the compute.

---

## Phase 3 — Disease conditioning alone (Dataset030, fold 0)

Three methods, no topology, no combinations:

| ID | Trainer | Method |
|---|---|---|
| **D1** | DA5FiLMV3_200e | FiLM bottleneck scale+shift |
| **D2** | DA5AuxDiag_200e | Aux disease-classification head (training-only regulariser) |
| **D3** | DA5CrossAttn_200e | Per-stage cross-attention between decoder features and disease tokens |

```bash
sbatch scripts/CHD_Dataset030_ablation_disease.sh
```

---

## Phase 4 — Combinations (Dataset030, fold 0)

Disease conditioning × topology + embedding-reuse variants. Run after Phases
2 and 3 so you have baselines for both axes.

| ID | Trainer | Composition |
|---|---|---|
| **C1** | DA5FiLMTopo_200e | FiLM + Topo |
| **C2** | DA5AuxDiagTopo_200e | Aux + Topo |
| **C3** | DA5CrossAttnTopo_200e | CrossAttn + Topo |
| **C4** | DA5AuxDiagCrossAttn_200e | Embedding reuse: aux head supplies tokens to cross-attention |
| **C5** | DA5FiLMAuxDiag_200e | FiLM + Aux regulariser |

```bash
sbatch scripts/CHD_Dataset030_ablation_combos.sh
```

---

## Track C — Clinical deployment (separate from the research ablation)

These two scripts produce models for clinical CFD preprocessing, not for the
scientific buildup. They share preprocessing markers with the research scripts
but don't use disease conditioning (clinical data has no disease labels).

```bash
sbatch scripts/CHD_Dataset013_Fanwei.sh       # Dataset013_Fanweidatacleaned
sbatch scripts/CHD_Dataset020_clinical.sh     # Dataset020FanweiDataandImageCHD_HU
```

---

## Operational helpers (call by hand, not on a schedule)

| Script | Purpose |
|---|---|
| `CHD_Dataset030_reinfer_all.sh` | Re-run inference for previously trained Dataset030 models without retraining. Useful after changes to the inference pipeline. |
| `CHD_Dataset030_reinfer_conditioned.sh` | Same but only for disease-conditioned models (FiLM / CrossAttn / FiLMAuxDiag). |
| `setup_cascade_predictions.py` | Symlinks lowres predictions into a cascade trainer's expected location. Called inside the SLURM scripts; rarely invoked by hand. |
| `generate_cascade_preds.py` | Generates `predicted_next_stage` files for ALL training cases (perform_actual_validation only writes the ~7 fold-0 val cases). Called inside the SLURM scripts. |
| `make_presentation.py` | Builds `docs/CHD_TopologyLoss_Presentation.pptx`. Run only when you need fresh slides. |
| `test_curriculum_class_weights.py` | Unit test for the curriculum-weights mixin. |

---

## Archived / unused

`CHD_Dataset001_cascade_200epochs.sh` exists from earlier D001 research (4
cascade pairs at 200 epochs). The new Dataset030 ablation matrix supersedes
its scientific value but the script is kept for reproducibility of historical
results. Don't feature it in new comparisons.

---

## Scientific buildup at a glance

```
Phase 0  ──  build datasets
   │
   ├──  Phase 1  (Dataset040)  →  evaluate_wholeheart.py
   │       Is whole-heart-first topologically better than joint multiclass?
   │       If YES → drives Stage-2 decomposition (experiments/wholeheart_decomposition/)
   │
   └──  Phase 2  (Dataset030)  →  predictions_ablation/
           Topology + cascade
              │
              ├──  Phase 3  (Dataset030)
              │       Disease conditioning alone
              │
              └──  Phase 4  (Dataset030)
                       Combinations
```

Read CSV outputs into `docs/project_overview.html` (Cascade Ablation / Fold-0
Ablation / Whole-Heart Pipeline sections — entries persist in localStorage)
to track all of these results in one place.
