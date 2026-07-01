# TODO — Disease-Landmark System (Dataset050 + septal-defect eval)

Two workflows: **server** = one pullable bash/SLURM script (build → preprocess → train → predict),
**local** = the Jupyter notebook (in-notebook septal-defect eval, CPU, no GPU needed).

Code commit: `all-experiments`.

---

## A. SERVER — one script: build → preprocess → train (3d_fullres) → predict

Pull the repo, then submit the script. It does everything (`.done` markers make it resumable):

```bash
git checkout all-experiments && git pull
sbatch scripts/CHD_Dataset050_disease_landmarks.sh
#   (or run directly on a GPU node:  bash scripts/CHD_Dataset050_disease_landmarks.sh)
```

Phases inside the script:
- **0** build `Dataset050_imageCHD_DiseaseLandmarks` from Dataset030 (source read-only, images symlinked)
- **1** `nnUNetv2_plan_and_preprocess -d 50 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity`
- **1b** copy Dataset030 `splits_final.json` → Dataset050 (identical fold-0 train/val; test set is Dataset030/imagesTs, unchanged)
- **2** train `nnUNetTrainerDA5_200epochs` (3d_fullres, fold 0) → predict test set → `predictions/DA5_baseline`
- **3** train `nnUNetTrainerDA5DiseaseLandmark_200epochs` → predict → `predictions/DA5DiseaseLandmark`

- [ ] 1. `git pull`
- [ ] 2. `sbatch scripts/CHD_Dataset050_disease_landmarks.sh`
- [ ] 3. when done, **copy the two prediction folders back** into the Alison
      `Dataset030/predictions/` (e.g. `DA5DiseaseLandmark200e`) and add them to the notebook's
      method list (Step 2 `DATASET_CONFIG`) so the local eval picks them up.
- [ ] 4. (optional, for my inspection) copy back `Dataset050/chd_derivation_report.csv`,
      training `progress.png`, `validation/summary.json`.

Notes: SLURM logs → `/scratch/users/sastocke/nnunet_CHD/logs/D050-landmarks_%j.{out,err}`.
Edit the header paths/partition/env in the script if your cluster layout differs.
I cannot see the server FS — commit/scp the small artifacts back if you want my read.

---

## B. LOCAL — the Jupyter notebook (no bash scripts)

Folder: `.../AlisonMarsden/SegmentationDetailStandard/dice_analysis.ipynb`

The septal-defect evaluation is **Step 10** in the notebook — fully in-notebook, CPU-only
(a few minutes for a full pass). It derives the septal defect from LV/RV (+LA/RA) touching on
**both GT and each prediction** (SDF4CHD-style) and compares them. It imports only the geometric
derivation primitives from the `chd_landmarks` library.

- [ ] 1. Run cells: Step 1 (dice fns), Step 2 (settings), Step 3 (compute Dice) — the usual prereqs.
- [ ] 2. Run **Step 10** — septal-defect ranking + per-case table + bar plot render inline.
      Set `RECOMPUTE_SEPTAL = True` to re-derive; leave `False` to use cached per-case CSVs.
- [ ] 3. When the Dataset050 predictions arrive from the server, add them as methods (Step 2) and
      re-run Steps 3 + 10.

**Artifacts written (readable offline):**
- `Dataset030/dice_results/septal_defect_<Method>.csv` — per-case metrics (cache)
- `Dataset030/dice_results/septal_defect_summary.csv` + `logs/septal_defect_summary.md` — ranking
- `Dataset030/dice_results/septal_defect_ranking.png` + `logs/septal_defect_ranking.png` — plot

Metrics: `septal_dice` (derived pred vs derived GT), detection recall/TP/FP/FN, `centroid_err_mm`,
`peri_septal_dice` (blood-pool correctness in the peri-defect shell). Set `CHD_LANDMARKS_REPO` if
the nnunet_CHD repo lives elsewhere.

---

## Notes
- The septal defect is derived from where blood pools touch (SDF4CHD-style), applied to BOTH GT and
  predictions, so it evaluates all existing predictions with no retraining. The Dataset050 model
  adds a direct septal label (id 8) as an additional variant.
- Full guide: `docs/chd_disease_landmark_nnunet.md`.
