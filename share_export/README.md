# nnU-Net v2 (DA5 fork)

A fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (v2.6.3) with a family of **DA5**
trainers (heavier data augmentation) used to train the CHD cardiac-CT segmentation models. This
repo is what you need to **run the shared weights** and to **train your own DA5 models**.

Labels used by the shared models (7-class heart): `0 background · 1 LV-BP · 2 RV-BP · 3 LA · 4 RA ·
5 Myo · 6 Ao · 7 PA`.

> **Why you need this fork (not stock nnU-Net):** nnU-Net records the *trainer name* in each
> checkpoint and looks that class up at inference. The shared weights are trained with
> `nnUNetTrainerDA5*`, so stock nnU-Net raises `Unable to locate trainer class ...`. Install this
> fork and the trainers are found. (The network itself is the standard `ResEncUNet` — see
> `INFERENCE.md` for a "retag to stock" path if you'd rather run on vanilla nnU-Net / 3D Slicer.)

## Install
```bash
conda create -y -n nnunet_da5 python=3.10 && conda activate nnunet_da5
# PyTorch matching your CUDA (example CUDA 12.x):
pip install "torch>=2.1.2,<2.9.0" --index-url https://download.pytorch.org/whl/cu121
pip install -e .          # from the root of this repo
```
Set the three data roots (nnU-Net convention; add to your shell profile):
```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Trainers you'll use
| Trainer | Notes |
|---|---|
| `nnUNetTrainerDA5_100epochs` / `_200epochs` / `_500epochs` | DA5 augmentation; pick the epoch budget. The shared weights use **500** epochs. |
| `nnUNetTrainerDA5CaseWeighted_500epochs` | DA5 + per-case sampling weights from a `case_weights.json` in the dataset folder (up-weight trusted cases). Uniform if the file is absent. |

Plans: `nnUNetResEncUNetMPlans` (from planner `nnUNetPlannerResEncM`). Config: `3d_fullres`.

## Quick start
- **Run the shared weights** → `INFERENCE.md`.
- **Train your own model** → `train_da5.sh` (edit the paths at the top, then `bash train_da5.sh`).

## Attribution / license
Built on nnU-Net (Isensee et al., DKFZ), Apache-2.0 — see `LICENSE`. If you use this, please cite
nnU-Net: *Isensee, F. et al. nnU-Net: a self-configuring method for deep learning-based biomedical
image segmentation. Nature Methods 18, 203–211 (2021).*
