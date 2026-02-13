"""
Disease-conditioned inference for nnU-Net v2.

Automatically loads ``disease_map.json`` from the model folder (copied there
during training) and sets the disease vector per case before prediction.
Uses a default all-zeros disease vector (not baseline!) when the disease map
is missing or a case is not found — this keeps the conditioned path active
since the decoder was trained with conditioning and expects modulated features.

Usage::

    python -m nnunetv2.inference.predict_disease_conditioned \\
        -i /path/to/input \\
        -o /path/to/output \\
        -m /path/to/model_folder \\
        [-f 0] \\
        [--disease_json /path/to/disease_map.json]
"""
from __future__ import annotations

import json
import os
from os.path import isfile, join, basename
from typing import Optional

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _get_unwrapped_network(network):
    """Unwrap DDP / torch.compile to get the raw module."""
    mod = network
    if hasattr(mod, 'module'):  # DDP
        mod = mod.module
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod
    return mod


def _extract_case_id(filepath: str) -> str:
    """Extract case_id from an nnU-Net input filename.

    ``ct_1001_image_0000.nii.gz``  →  ``ct_1001_image``
    ``ct_1001_0000.nii.gz``        →  ``ct_1001``
    """
    name = basename(filepath)
    # strip file extension(s)
    for ext in ('.nii.gz', '.nii', '.mha', '.nrrd'):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    # strip channel suffix (_0000, _0001, ...)
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        name = parts[0]
    return name


def predict_disease_conditioned(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    folds: tuple = (0,),
    checkpoint_name: str = 'checkpoint_final.pth',
    disease_json: Optional[str] = None,
    device: str = 'cuda',
    tile_step_size: float = 0.5,
    use_mirroring: bool = True,
    save_probabilities: bool = False,
    num_processes_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3,
):
    """Run inference with automatic disease-vector conditioning."""
    # --- load disease map ---
    disease_map = None
    # priority: explicit arg > model folder > skip
    if disease_json is not None and isfile(disease_json):
        with open(disease_json) as f:
            disease_map = json.load(f)
        print(f"Loaded disease map from {disease_json} ({len(disease_map)} entries)")
    else:
        auto_path = join(model_folder, 'disease_map.json')
        if isfile(auto_path):
            with open(auto_path) as f:
                disease_map = json.load(f)
            print(f"Auto-detected disease map in model folder ({len(disease_map)} entries)")
            print(f"  disease_map keys: {sorted(disease_map.keys())}")
        else:
            print(f"No disease_map.json found at {auto_path}")
            print("Using default all-zeros disease vector for all cases.")

    # --- set up predictor ---
    dev = torch.device(device)
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_mirroring=use_mirroring,
        device=dev,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint_name,
    )

    mod = _get_unwrapped_network(predictor.network)

    if disease_map is None:
        # WARNING: Running baseline (clear_disease_vec) would skip FiLM/injection
        # entirely, but the decoder was trained WITH conditioning active.
        # Use a default all-zeros disease vector to keep the conditioned path
        # active while providing a neutral signal.
        print("WARNING: No disease_map.json found. Using default disease vector [0]*8.")
        print("If results are poor, provide a disease_map.json with --disease_json")
        default_vec = torch.tensor([[0.0] * 8], dtype=torch.float32, device=dev)
        mod.set_disease_vec(default_vec)
        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=save_probabilities,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )
        mod.clear_disease_vec()
        return

    # --- disease-conditioned per-case prediction ---
    # gather input files
    input_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith('.nii.gz') or f.endswith('.nii') or f.endswith('.mha')
    ])

    # group by case_id (handles multi-channel: _0000, _0001, ...)
    cases = {}
    for f in input_files:
        case_id = _extract_case_id(f)
        cases.setdefault(case_id, []).append(join(input_folder, f))

    os.makedirs(output_folder, exist_ok=True)

    # Diagnostic: show case ID matching
    case_ids = sorted(cases.keys())
    matched = [c for c in case_ids if c in disease_map]
    unmatched = [c for c in case_ids if c not in disease_map]
    print(f"\nFound {len(case_ids)} cases in input folder.")
    print(f"  Matched in disease_map: {len(matched)} — {matched}")
    if unmatched:
        print(f"  NOT in disease_map:     {len(unmatched)} — {unmatched}")
        print(f"  (These will use default all-zeros disease vector)")

    # Determine disease vector length from first entry
    disease_K = len(next(iter(disease_map.values())))
    # Default vector for unknown cases: all zeros (no disease flags set).
    # IMPORTANT: We use the conditioned path with zeros instead of baseline,
    # because the decoder was trained WITH conditioning and expects modulated features.
    default_vec = [0] * disease_K

    for case_id, file_list in sorted(cases.items()):
        # set disease vec
        if case_id in disease_map:
            vec = torch.tensor([disease_map[case_id]], dtype=torch.float32, device=dev)
            mod.set_disease_vec(vec)
            print(f"\n{case_id}: disease_vec = {disease_map[case_id]}")
        else:
            vec = torch.tensor([default_vec], dtype=torch.float32, device=dev)
            mod.set_disease_vec(vec)
            print(f"\n{case_id}: NOT in disease_map, using default vec = {default_vec}")

        # predict this case
        output_file = join(output_folder, case_id + '.nii.gz')
        predictor.predict_from_files(
            [sorted(file_list)],  # list-of-lists format
            [join(output_folder, case_id)],  # truncated output name (no ext)
            save_probabilities=save_probabilities,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

    mod.clear_disease_vec()
    print("\nDone.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Disease-conditioned nnU-Net inference. Automatically uses '
                    'disease_map.json from the model folder if available.',
    )
    parser.add_argument('-i', '--input', required=True, help='Input folder with images')
    parser.add_argument('-o', '--output', required=True, help='Output folder for predictions')
    parser.add_argument('-m', '--model_folder', required=True,
                        help='Path to trained model folder '
                             '(e.g. nnUNet_results/DatasetXXX/.../)')
    parser.add_argument('-f', '--folds', nargs='+', type=int, default=[0],
                        help='Fold(s) to use (default: 0)')
    parser.add_argument('-chk', '--checkpoint', default='checkpoint_final.pth',
                        help='Checkpoint filename (default: checkpoint_final.pth)')
    parser.add_argument('--disease_json', default=None,
                        help='Explicit path to disease_map.json (overrides auto-detect)')
    parser.add_argument('-device', default='cuda', help='Device (default: cuda)')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Tile step size (default: 0.5)')
    parser.add_argument('--disable_tta', action='store_true',
                        help='Disable test-time augmentation (mirroring)')
    parser.add_argument('--save_probabilities', action='store_true')

    args = parser.parse_args()

    predict_disease_conditioned(
        input_folder=args.input,
        output_folder=args.output,
        model_folder=args.model_folder,
        folds=tuple(args.folds),
        checkpoint_name=args.checkpoint,
        disease_json=args.disease_json,
        device=args.device,
        tile_step_size=args.step_size,
        use_mirroring=not args.disable_tta,
        save_probabilities=args.save_probabilities,
    )


if __name__ == '__main__':
    main()
