#!/usr/bin/env python3
"""
retag_checkpoint_to_stock.py
============================

Make a DA5-trained model loadable by *stock* nnU-Net (and the 3D Slicer nnU-Net extension).

nnU-Net records the trainer name in every checkpoint and looks that class up at inference, so a
`nnUNetTrainerDA5*` model fails on vanilla nnU-Net. But DA5 does NOT change the network (it's the
standard ResEncUNet built from plans.json), so re-tagging the checkpoints' `trainer_name` to
`nnUNetTrainer` and renaming the model folder is enough — the weights load unchanged.

Usage:
  python retag_checkpoint_to_stock.py --model-dir /path/to/nnUNetTrainerDA5_500epochs__nnUNetResEncUNetMPlans__3d_fullres
  # writes a sibling: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/

Only valid for network-preserving trainers (DA5, DA5CaseWeighted). Refuses FiLM/Disease/CrossAttn.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

STOCK = "nnUNetTrainer"
ARCH_CHANGING = ("FiLM", "Disease", "CrossAttn", "MLPembedding")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="a trained model folder: <TRAINER>__<PLANS>__<CONFIG>/")
    ap.add_argument("--out-dir", default=None, help="output folder (default: sibling nnUNetTrainer__<PLANS>__<CONFIG>)")
    args = ap.parse_args()

    src = Path(args.model_dir).resolve()
    if not src.is_dir():
        sys.exit(f"ERROR: not a directory: {src}")
    parts = src.name.split("__")
    if len(parts) != 3:
        sys.exit(f"ERROR: expected <TRAINER>__<PLANS>__<CONFIG>, got folder '{src.name}'")
    trainer, plans, config = parts
    if any(tok in trainer for tok in ARCH_CHANGING):
        sys.exit(f"ERROR: '{trainer}' changes the network architecture — retag to stock is NOT valid.\n"
                 f"       Run this model with the DA5 fork installed instead (see INFERENCE.md, Route A).")

    dst = Path(args.out_dir).resolve() if args.out_dir else src.parent / f"{STOCK}__{plans}__{config}"
    if dst.exists():
        sys.exit(f"ERROR: output already exists: {dst} (remove it or pass --out-dir)")
    dst.mkdir(parents=True)

    # copy model-level metadata verbatim
    for meta in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        p = src / meta
        if p.is_file():
            shutil.copy2(p, dst / meta)

    # per fold: copy the fold dir, re-tag every checkpoint
    n_ckpt = 0
    folds = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not folds:
        sys.exit(f"ERROR: no fold_* subfolders in {src}")
    for fold in folds:
        out_fold = dst / fold.name
        out_fold.mkdir()
        for f in sorted(fold.iterdir()):
            if f.suffix == ".pth":
                ckpt = torch.load(str(f), map_location="cpu", weights_only=False)
                old = ckpt.get("trainer_name", "?")
                ckpt["trainer_name"] = STOCK
                torch.save(ckpt, str(out_fold / f.name))
                n_ckpt += 1
                print(f"  retagged {fold.name}/{f.name}: '{old}' -> '{STOCK}'")
            elif f.is_file():
                shutil.copy2(f, out_fold / f.name)   # progress.png, debug.json, etc.

    print(f"\n[done] {n_ckpt} checkpoint(s) across {len(folds)} fold(s) -> {dst}")
    print(f"Predict on stock nnU-Net:")
    print(f"  nnUNetv2_predict -i IN -o OUT -d <ID> -c {config} -tr {STOCK} -p {plans} -f all -chk checkpoint_final.pth")


if __name__ == "__main__":
    main()
