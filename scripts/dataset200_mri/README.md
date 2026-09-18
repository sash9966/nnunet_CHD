# Dataset200 MRI: Sherlock training scripts

These are new scripts based on the paths and trainer names in the supplied Sherlock listing. The GitHub repository was inspected on 2026-09-18. Its existing batch examples target another scheduler; these new scripts target Sherlock Slurm. These scripts have not been run on Sherlock.

Defaults: existing nnunet310 environment, CHD repository, ResEnc-M, 3d_fullres, nnUNetTrainerDA5MRI200 for all six models. This is a short 200-epoch baseline; convergence is not guaranteed. The trainer is included in this repository as a DA5 subclass setting num_epochs=200; the older server-only nnUNetTrainerDA5_200epochs was not present in GitHub and is not used. The script checks that the class can be imported and fails rather than substituting another trainer. Scheduler defaults are normal/gpu partitions, 8 CPUs, 64 GB RAM, one GPU, 8 hours for preparation and 24 hours per training job. Adjust config.sh to your allocation; choose a GPU with enough memory for ResEnc-M (roughly 11 GB or more). These resource requests have not been checked against your account limits.

## Install with Git

On Sherlock, update the existing repository with `git pull --ff-only` from `/scratch/users/sastocke/nnunet_CHD`. These scripts are then in `scripts/dataset200_mri/`. Dataset200 image/label files must already be uploaded separately to nnUNet_raw.

## Submit on Sherlock

```bash
cd /scratch/users/sastocke/nnunet_CHD/scripts/dataset200_mri
bash submit.sh both
```

`both`: CPU preprocessing/integrity check, then the all-case GPU run, then fold 0, then folds 1–4 with at most two running concurrently. Dependencies stop downstream training if preparation or an earlier job fails. Fold 0 runs first to avoid concurrent initial data unpacking. Each fold has its own GPU allocation. No array task runs multiple folds on one GPU.

Alternatively:

```bash
bash submit.sh all  # preprocessing + one model trained on every case
bash submit.sh cv   # preprocessing + five held-out folds
```

Submit one workflow at a time. Reusing unchanged preprocessing is supported. Rerunning a workflow resumes latest checkpoints and skips completed folds; completed CV folds missing final validation summaries are revalidated. If a job times out, cancel remaining dependent jobs shown by squeue before resubmitting. Existing results are never deleted. If you change data, trainer settings or plans, use a separate experiment rather than reusing checkpoints. Keep config.sh unchanged while jobs are queued or running. Other submission mechanisms are outside the scripts' duplicate-job guard.

## Results and performance

Results:
/scratch/users/sastocke/nnUNet/nnUNet_results/Dataset200_MRI/nnUNetTrainerDA5MRI200__nnUNetResEncUNetMPlans__3d_fullres/

The all-case model is fold_all/checkpoint_final.pth. The five other models live in fold_0 through fold_4. Default nnU-Net splits operate on these 17 case IDs; confirm they represent 17 independent subjects. No CT cases or pretrained CT weights are used.

After all five evaluation folds finish:

```bash
bash summarize_cv.sh
```

This writes cv_per_case_dice.csv and cv_summary.json and checks that 17 distinct cases were evaluated once each. Label order: 1 LV blood pool, 2 RV blood pool, 3 LA, 4 RA, 5 myocardium, 6 aorta, 7 pulmonary artery. Scores are averaged over held-out cases, then equally across the seven labels. The all-case model has no independent holdout; any validation that nnU-Net performs for fold_all is in-sample and must not be compared as a held-out score. Comparing the final model itself requires an independent test cohort. Five-fold validation estimates the training procedure's performance at the smaller fold training sizes.

## Use the all-case model

Inside a GPU allocation, with absolute paths and input names CASE_0000.nii.gz:

```bash
bash predict_all.sh /absolute/path/to/new_mri /absolute/path/to/predictions
```

For a handoff, preserve checkpoint_final.pth plus dataset.json, plans.json and dataset_fingerprint.json from the model directory, the custom trainer code, and the environment/version information. This script bundle does not package or transmit a clinical model. The quick baseline needs held-out evaluation and clinical review before patient-care use.

## References

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
https://www.sherlock.stanford.edu/docs/user-guide/gpu/

## Reproducibility records (added 2026-09-18)

Every submission saves a UTC-stamped directory under nnUNet_results/_run_records/Dataset200_MRI. It contains the Git commit, working-tree diff/status, a full nnunetv2 code snapshot (including server-local trainers), the script/config snapshot, dataset.json, file checksums, selected settings, submitted job IDs, per-job Python/PyTorch/CUDA/GPU information, installed package versions, and plans/splits available when each training job starts. Actual commands and output appear in the snapshot's scripts/logs directory. Training uses the captured code and scripts, so subsequent pulls do not alter queued jobs. The environment itself is recorded but not copied; do not change its packages during the workflow. nnU-Net writes its split file and training details in the normal preprocessing/results folders. Each fold contains a run_records pointer to its submission record.

Run records, MRI volumes, source manifests and model weights are not committed to GitHub. Copy the relevant run record with model files for a reproducible handoff. This captures provenance, not a guarantee of bitwise deterministic GPU training.
