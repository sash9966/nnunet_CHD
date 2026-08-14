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

## Route B — vanilla nnU-Net / 3D Slicer (no fork install), via retag
The DA5 network **is** the stock `ResEncUNet`, so a DA5 checkpoint runs on stock nnU-Net once its
recorded trainer name is changed to `nnUNetTrainer`. One-time conversion:
```bash
python tools/retag_checkpoint_to_stock.py \
  --model-dir /path/to/nnUNetTrainerDA5_500epochs__nnUNetResEncUNetMPlans__3d_fullres
# -> writes a sibling: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
```
Then, in a **plain** `nnunetv2` install (e.g. the one bundled by the Slicer nnU-Net extension,
version ~2.6.x):
```bash
nnUNetv2_predict -i in -o out -d <ID> -c 3d_fullres \
  -tr nnUNetTrainer -p nnUNetResEncUNetMPlans -f all -chk checkpoint_final.pth
```
For the **Slicer nnU-Net extension**, point it at the retagged `nnUNetTrainer__…` model folder.

Notes:
- Retag works only for trainers that don't change the architecture — **DA5** and
  **DA5CaseWeighted** (the script refuses FiLM/Disease/CrossAttn models, which do change it).
- The stock nnU-Net must be a compatible version (~2.6.x) so it builds the same ResEncUNet from
  `plans.json`. If in doubt, use Route A.
