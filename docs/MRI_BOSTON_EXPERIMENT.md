# MRI clinical / Boston experiment

## Cohorts and fixed training budget

| Dataset | Training | Role |
|---|---|---|
| 200 MRI | 17 clinical | Clinical-only baseline |
| 201 ClinicalBostonMRI | Same 17 + ten reviewed Boston patients 000–009 | Added fully annotated public data |
| 202 ClinicalBostonPseudoMRI | Same 27 + individually reviewed Boston 010–059 | Six human structures plus reviewed pseudo myocardium |

Dataset numbers are identifiers, not patient counts. Maximum Dataset202 size is 77.
Legacy HVSMR and HVSMR-2.0 overlap: the ten legacy masks enrich existing patients;
they do not create ten additional independent images beyond those ten Boston patients.
Boston SVC/IVC merge into RA. Legacy label-A myocardium fills only background of the
newer annotations. This mapping cannot eliminate differences in vessel extent,
myocardial conventions, acquisition or operated anatomy. Sampled overlays were
accepted by the user on 2026-09-18; that is not exhaustive 3D expert adjudication.

All three use ResEnc-M, DA5 and 100 epochs. This is a matched, short training-budget
comparison, not proof of convergence. Inspect learning curves. nnU-Net plans each
cohort independently; spacing, patch size and batch size may differ, so this tests
whole pipelines rather than strictly isolating the number of images. Save plans.
For a stricter data-only ablation, harmonize plans and rerun every arm, including 200.

Dataset201 copies the actual Dataset200 clinical validation splits when available;
otherwise it uses nnU-Net's identical sorted-ID/KFold(seed=12345) default. Boston
is training-only in all five folds. No existing Dataset200 files are modified.
Dataset202 deliberately has no conventional clinical five-fold script: its fold_all
teacher saw all 17 clinical cases. Honest internal CV would need separate teachers
and pseudo masks within each clinical fold. The independent new clinical cohort
can evaluate the all-case models without that leakage.

## Scientific assessment

This is a reasonable low-data experiment, not a demonstrated improvement for this
institution. [STAMP](https://pmc.ncbi.nlm.nih.gov/articles/PMC10134897/) studies
teacher/student pseudo-labeling with augmentation and confidence handling for 3D
cardiac MRI. It supports investigating the approach; its performance does not
transfer automatically to CHD, these annotations, or this simpler reviewed-label
pipeline. [Cascaded Self-Supervision](https://pmc.ncbi.nlm.nih.gov/articles/PMC12383799/)
also studies successive pseudo-labeling in low-data cardiac MRI.

The counterargument is confirmation bias: wrong teacher labels become student
training targets. [Radhakrishnan et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Radhakrishnan_Design_Choices_for_Enhancing_Noisy_Student_Self-Training_WACV_2024_paper.html)
explicitly investigate reducing this failure mode. Human correction and preserving
six human annotations reduce exposure; they do not establish that the remaining
myocardium labels are correct. More acquisition variety may help or hurt local
performance. The [HVSMR-2.0 paper](https://www.nature.com/articles/s41597-024-03469-9)
describes the 60-case CHD source and its annotation conventions.

Equal per-case sampling is the default in all arms. Existing case-weight machinery
also permits clinical 10x / Boston 1x in Dataset202. With all 77 patients, that means
170/(170+60) = 73.9% expected clinical case draws, not tenfold gradients. It may
reinforce overfitting to 17 patients. Changing weighting only in arm 3 confounds the
pseudo-data comparison; treat 10x as a separate prespecified ablation, using a new
dataset ID/output so earlier weights, preprocessing and checkpoints remain intact.
Do not select the multiplier by repeatedly examining final test results.

## Evaluation and future clinical use

Score original Dataset201 predictions on the remaining Boston cases BEFORE merging
human labels. Six-class per-case Dice is saved; myocardium is unscored because no
reference exists. Merged labels must never be scored against the same human labels
and presented as model performance. After promotion those patients are training
cases, no longer an independent test set. Do not score pseudo myocardium against
itself; manual reference annotation is needed for a quantitative myocardium test.

Freeze all three models and the analysis before opening the 45 new clinical labels.
Use patient-level comparison (not slices/patches), macro Dice across the seven
structures as a prespecified primary metric, per-structure Dice, surface distances,
and review of clinically meaningful failures. Report paired patient bootstrap
intervals; account for repeated scans from the same patient. Stratify descriptively
by anatomy, surgery and acquisition rather than relying solely on the average.
If the 45 select the winning model, call them a model-selection cohort, not an
untouched final performance estimate. Reserve an untouched subset in advance or
obtain a later independent/prospective series for the selected model. A higher mean
Dice alone is not sufficient evidence for unsupervised clinical use; establish the
intended clinical task and acceptable failure rates with the clinical team.

## Run on Sherlock

Use the existing checkout and existing logs folder, on `all-experiments`:

```bash
cd /scratch/users/sastocke/nnunet_CHD
git pull
sbatch scripts/CHD_Dataset201_mri.sh
sbatch scripts/CHD_Dataset201_train5fold.sh
```

Only preparation is locked; the two jobs can train concurrently once it finishes.
The five folds run sequentially inside their job. Existing final checkpoints skip,
latest checkpoints resume. Optional PyTorch compilation remains disabled as in 200.
Do not reuse a completed output to claim a fresh run.

After the all-case run finishes:

```bash
sbatch scripts/CHD_Dataset201_predict_boston.sh
```

Predictions: `nnUNet_raw/Dataset201_ClinicalBostonMRI/predictions/ds201__native/`.
Review package: `nnUNet_results/Dataset201_ClinicalBostonMRI/Boston50_review/`.
Open its candidate labels with the corresponding Dataset201/imagesTs images in
Slicer. Inspect full volumes and correct myocardium. Six human structures are
preserved. For each accepted case, fill `decision=APPROVED`, `reviewer`, and the
SHA256 of the saved candidate file in `review_decisions.csv`. This binds approval
to the exact corrected mask. Leave rejected cases unapproved; a subset can train.

After review, activate the usual nnunet310 environment and build Dataset202:

```bash
python tools/mri_boston.py build202 \
  --dataset nnUNet_raw/Dataset201_ClinicalBostonMRI \
  --review nnUNet_results/Dataset201_ClinicalBostonMRI/Boston50_review \
  --output nnUNet_raw/Dataset202_ClinicalBostonPseudoMRI
sbatch scripts/CHD_Dataset202_mri.sh
```

The builder rejects unreviewed masks, missing myocardium, modified human labels,
geometric mismatches, duplicate patients, or changed review hashes. It refuses to
overwrite datasets. `--clinical-weight 10` is available when initially building
202 if that weighted experiment is deliberately chosen instead of the default.
All scripts stamp provenance; manifests bind data, teacher checkpoint and labels.
The weighted trainer adds only a 100-epoch alias to the existing sampling mixin.
Images, labels and patient data are not committed to GitHub.

## Local reproduction of Dataset201

```bash
python tools/mri_boston.py build201 \
  --clinical /path/to/Dataset200_MRI \
  --review /path/to/Boston10_Review \
  --boston '/path/to/HVSMR 2.0 Dataset/cropped' \
  --output /path/to/Dataset201_ClinicalBostonMRI
```

The approved legacy review package contains registered images, candidate_labels,
review_decisions.csv and review_manifest.json. Checksums cover every dataset file.
Only the six-label maps in labelsTs have no myocardium; none enters training until
reviewed myocardium has been added and Dataset202 is built.
