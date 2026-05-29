#!/usr/bin/env python3
"""
Generate predicted_next_stage predictions for ALL training cases using a trained lowres model.

When training a cascade with fold 0 only, perform_actual_validation() only writes predictions
for the ~7 fold-0 validation cases.  The cascade-fullres trainer reads predicted_next_stage
at the start of every training step, so it needs predictions for ALL training+val cases.
This script fills that gap by running the fold-0 lowres model over every identifier found in
the lowres preprocessed folder.

Usage (add to CHD_Dataset030_imageCHD.sh between Phase 2 and Phase 4):

  python scripts/generate_cascade_preds.py \
      --dataset-id 30 \
      --lowres-trainer nnUNetTrainerDA5_200epochs \
      --plans nnUNetResEncUNetMPlans \
      --fold 0

Already-predicted cases are skipped, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import warnings
from os.path import isfile, join

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def generate_cascade_preds(
    dataset_name_or_id,
    lowres_trainer_name: str,
    plans_identifier: str,
    fold: int = 0,
    device: str = "cuda",
) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    torch_device = torch.device(device)

    print(f"\n=== generate_cascade_preds ===")
    print(f"  dataset    : {dataset_name}")
    print(f"  trainer    : {lowres_trainer_name}")
    print(f"  plans      : {plans_identifier}")
    print(f"  fold       : {fold}")
    print(f"  device     : {device}")

    # ------------------------------------------------------------------ #
    # 1.  Instantiate and restore the lowres trainer
    # ------------------------------------------------------------------ #
    trainer = get_trainer_from_args(
        dataset_name, "3d_lowres", fold, lowres_trainer_name, plans_identifier,
        device=torch_device,
    )

    checkpoint_path = join(trainer.output_folder, "checkpoint_final.pth")
    if not isfile(checkpoint_path):
        checkpoint_path = join(trainer.output_folder, "checkpoint_latest.pth")
    if not isfile(checkpoint_path):
        checkpoint_path = join(trainer.output_folder, "checkpoint_best.pth")
    if not isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found in {trainer.output_folder}. "
            "Run Phase 2 (lowres training) first."
        )
    print(f"\n  checkpoint : {checkpoint_path}")

    # load_checkpoint calls initialize() internally if not yet done
    trainer.load_checkpoint(checkpoint_path)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    # ------------------------------------------------------------------ #
    # 2.  Build nnUNetPredictor (replicates perform_actual_validation)
    # ------------------------------------------------------------------ #
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network,
        trainer.plans_manager,
        trainer.configuration_manager,
        None,
        trainer.dataset_json,
        trainer.__class__.__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    # ------------------------------------------------------------------ #
    # 3.  Determine next-stage name(s)
    # ------------------------------------------------------------------ #
    next_stages = trainer.configuration_manager.next_stage_names
    if not next_stages:
        print("\n  No next_stage_names in 3d_lowres config — nothing to do.")
        return

    # ------------------------------------------------------------------ #
    # 4.  Get ALL identifiers from the lowres preprocessed folder
    # ------------------------------------------------------------------ #
    all_ids = trainer.dataset_class.get_identifiers(trainer.preprocessed_dataset_folder)
    all_ids.sort()
    print(f"\n  Total identifiers in lowres preprocessed folder: {len(all_ids)}")

    # ------------------------------------------------------------------ #
    # 5.  Generate predictions for each next stage
    # ------------------------------------------------------------------ #
    for next_stage in next_stages:
        output_folder = join(trainer.output_folder_base, "predicted_next_stage", next_stage)
        maybe_mkdir_p(output_folder)
        print(f"\n  -> next stage: {next_stage}")
        print(f"     output dir : {output_folder}")

        next_stage_cfg = trainer.plans_manager.get_configuration(next_stage)
        next_prep_folder = join(
            nnUNet_preprocessed, dataset_name, next_stage_cfg.data_identifier
        )
        next_ds_class = infer_dataset_class(next_prep_folder)

        skipped, predicted = 0, 0
        for k in all_ids:
            # skip if already done (b2nd file exists)
            out_b2nd = join(output_folder, k + ".b2nd")
            if isfile(out_b2nd):
                skipped += 1
                continue

            # get target shape from the next-stage preprocessed data
            try:
                tmp_ds = next_ds_class(next_prep_folder, [k])
                d_next, _, _, _ = tmp_ds.load_case(k)
                target_shape = d_next.shape[1:]
                del d_next
            except FileNotFoundError:
                print(
                    f"     [SKIP] {k}: next-stage preprocessed file missing "
                    f"(run preprocessing for {next_stage} first)"
                )
                skipped += 1
                continue

            # load lowres data
            data, _, _, properties = trainer.dataset_class(
                trainer.preprocessed_dataset_folder, [k]
            ).load_case(k)
            data = data[:]  # blosc2 ndarray → numpy

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_tensor = torch.from_numpy(data)

            with torch.no_grad():
                prediction = predictor.predict_sliding_window_return_logits(data_tensor)
            prediction = prediction.cpu()

            output_file_truncated = join(output_folder, k)
            resample_and_save(
                prediction,
                target_shape,
                output_file_truncated,
                trainer.plans_manager,
                trainer.configuration_manager,
                properties,
                trainer.dataset_json,
                default_num_processes,
                next_ds_class,
            )
            predicted += 1
            print(f"     [OK]   {k}  ({predicted}/{len(all_ids) - skipped + predicted})")

        print(f"\n  Stage '{next_stage}' done. "
              f"Predicted: {predicted}  Skipped (already existed): {skipped}  "
              f"Total: {len(all_ids)}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Generate cascade next-stage predictions for ALL training cases."
    )
    parser.add_argument("--dataset-id", type=int, required=True,
                        help="Dataset ID (e.g. 30)")
    parser.add_argument("--lowres-trainer", type=str, required=True,
                        help="Lowres trainer class name (e.g. nnUNetTrainerDA5_200epochs)")
    parser.add_argument("--plans", type=str, default="nnUNetResEncUNetMPlans",
                        help="Plans identifier (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold to use (default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda)")
    args = parser.parse_args()

    # match nnUNet's threading setup
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    generate_cascade_preds(
        args.dataset_id,
        args.lowres_trainer,
        args.plans,
        args.fold,
        args.device,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
