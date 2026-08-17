# Running the shared weights

You'll receive a **model folder** named like:
```
nnUNetTrainerDA5_500epochs__nnUNetResEncUNetMPlans__3d_fullres/
    fold_all/            (or fold_0 … fold_4)
        checkpoint_final.pth
    plans.json
    dataset.json
    dataset_fingerprint.json   (optional)
```
Put it under `$nnUNet_results/Dataset<ID>_<name>/` (any dataset id — inference only reads this folder).

Your CT does **not** need pre-resampling: nnU-Net resamples to the model's spacing internally and
writes the mask back on your native grid. Input files must be named `<case>_0000.nii.gz`.

---

## Route A — with this fork installed (recommended)
```bash
conda activate nnunet_da5          # the env where you did `pip install -e .`
export nnUNet_results=/path/to/nnUNet_results

nnUNetv2_predict \
  -i /path/to/input_dir \          # contains <case>_0000.nii.gz
  -o /path/to/output_dir \
  -d <ID> -c 3d_fullres \
  -tr nnUNetTrainerDA5_500epochs \ # must match the model folder's trainer name
  -p nnUNetResEncUNetMPlans \
  -f all \                         # or: -f 0 1 2 3 4 for a 5-fold ensemble
  -chk checkpoint_final.pth
```
Output masks are integer NIfTI (labels 0–7) on the input grid — load straight into 3D Slicer.

---

## Route B — vanilla nnU-Net / 3D Slicer (NO fork install), via retag
Any DA5-family model (incl. `DA5CaseWeighted`) is the stock `ResEncUNet`; the weighting only changes
*training-time sampling* and has **zero effect on inference**. The only thing stock nnU-Net can't
resolve is the custom trainer *name* recorded in the checkpoint. `nnUNetTrainerDA5` **is** a stock
nnU-Net trainer, so we relabel to it. One-time conversion (run in any env with torch):
```bash
python tools/retag_checkpoint_to_stock.py \
  --model-dir /path/to/nnUNetTrainerDA5CaseWeighted_500epochs__nnUNetResEncUNetMPlans__3d_fullres
# default --target-trainer nnUNetTrainerDA5  ->  sibling: nnUNetTrainerDA5__nnUNetResEncUNetMPlans__3d_fullres/
# (use --target-trainer nnUNetTrainer for the absolute base if a target somehow lacks DA5)
```
Then, in a **plain** nnU-Net (e.g. the one bundled by the Slicer nnU-Net extension):
```bash
nnUNetv2_predict -i in -o out -d <ID> -c 3d_fullres \
  -tr nnUNetTrainerDA5 -p nnUNetResEncUNetMPlans -f all -chk checkpoint_best.pth
```
For the **Slicer nnU-Net extension**, just point it at the retagged `nnUNetTrainerDA5__…` model folder.

Notes:
- Retag is valid only for trainers that don't change the architecture — **DA5** and
  **DA5CaseWeighted** (the script refuses FiLM/Disease/CrossAttn, which do change it).
- Stock nnU-Net should be a compatible version (~2.5–2.6) so it builds the same ResEncUNet from
  `plans.json`. If it ever errors building the network, fall back to Route A.
