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


def _extract_case_id(filepath: str, normalize_suffix: str = '') -> str:
    """Extract case_id from an nnU-Net input filename.

    ``ct_1001_image_0000.nii.gz``  ->  ``ct_1001_image``
    ``ct_1001_0000.nii.gz``        ->  ``ct_1001``

    If *normalize_suffix* is given (e.g. ``'_image'``), it is stripped from
    the disease_map keys at lookup time (see ``_normalize_disease_map``), so
    you don't need to rename your test images when train/test naming differs.
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


def _normalize_disease_map(disease_map: dict, strip_suffix: str) -> dict:
    """Return a copy of *disease_map* with *strip_suffix* removed from keys.

    Example: ``_image`` turns ``{"ct_1001_image": [...]}`` into
    ``{"ct_1001": [...]}``, so test files named ``ct_1001_0000.nii.gz``
    match without renaming.  Keys that don't end with the suffix are kept as-is.
    """
    if not strip_suffix:
        return disease_map
    return {
        (k[:-len(strip_suffix)] if k.endswith(strip_suffix) else k): v
        for k, v in disease_map.items()
    }


def predict_disease_conditioned(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    folds: tuple = (0,),
    checkpoint_name: str = 'checkpoint_final.pth',
    disease_json: Optional[str] = None,
    normalize_disease_map_suffix: str = '',
    device: str = 'cuda',
    tile_step_size: float = 0.5,
    use_mirroring: bool = True,
    save_probabilities: bool = False,
    num_processes_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3,
):
    """Run inference with automatic disease-vector conditioning.

    Works for both conditioned models (FiLM, DiseaseVec) and plain models
    (topology-loss-only, baseline).  For plain models the disease map is
    ignored and standard sliding-window prediction is used.

    Parameters
    ----------
    normalize_disease_map_suffix : str
        Strip this suffix from disease_map keys before matching.
        Use when training images had a suffix (e.g. ``_image``) that your
        test images don't: ``--normalize_disease_map_suffix _image`` turns
        ``"ct_1001_image"`` keys into ``"ct_1001"`` so ``ct_1001_0000.nii.gz``
        matches without renaming.
    """
    # --- read inference_config.json (written by DiseaseConditioningMixin) ---
    infer_cfg = {}
    cfg_path = join(model_folder, 'inference_config.json')
    if isfile(cfg_path):
        with open(cfg_path) as f:
            infer_cfg = json.load(f)
        print(f"Loaded inference_config.json: {infer_cfg}")
    needs_conditioning = infer_cfg.get('needs_disease_conditioning', None)

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
            print(f"  disease_map keys (raw): {sorted(disease_map.keys())}")
        else:
            print(f"No disease_map.json found at {auto_path}")
            print("Using default all-zeros disease vector for all cases.")

    # Apply suffix normalisation to disease_map keys (e.g. strip '_image')
    if disease_map and normalize_disease_map_suffix:
        disease_map = _normalize_disease_map(disease_map, normalize_disease_map_suffix)
        print(f"  disease_map keys (after stripping '{normalize_disease_map_suffix}'): "
              f"{sorted(disease_map.keys())}")

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

    # Detect conditioning capability from the network (overrides config if absent)
    model_has_conditioning = hasattr(mod, 'set_disease_vec')
    if needs_conditioning is None:
        needs_conditioning = model_has_conditioning
    if needs_conditioning and not model_has_conditioning:
        print("WARNING: inference_config.json says needs_disease_conditioning=True but "
              "the loaded network has no set_disease_vec(). Running without conditioning.")
        needs_conditioning = False

    # --- plain (non-conditioned) model: just run standard prediction ---
    if not needs_conditioning:
        print("Model does not use disease conditioning — running standard prediction.")
        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=save_probabilities,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )
        print("\nDone.")
        return

    # --- conditioned model: per-case prediction with disease vectors ---
    if disease_map is None:
        # Fallback: use an all-zeros vector so the conditioned path stays active.
        disease_K = infer_cfg.get('disease_K', 8)
        print(f"WARNING: No disease_map.json found. Using default [0]*{disease_K} for all cases.")
        print("If results are poor, provide a disease_map.json with --disease_json")
        default_vec = torch.tensor([[0.0] * disease_K], dtype=torch.float32, device=dev)
        mod.set_disease_vec(default_vec)
        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=save_probabilities,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )
        mod.clear_disease_vec()
        print("\nDone.")
        return

    # gather input files and group by case_id (handles multi-channel _0000/_0001)
    input_files = sorted([
        f for f in os.listdir(input_folder)
        if f.endswith('.nii.gz') or f.endswith('.nii') or f.endswith('.mha')
    ])
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

    if len(matched) == 0:
        print("\n" + "=" * 70)
        print("WARNING: ZERO cases matched the disease_map!")
        print("This likely means case IDs extracted from filenames don't match the")
        print("disease_map keys. Check naming conventions:")
        print(f"  Input case IDs:    {case_ids[:5]}{'...' if len(case_ids) > 5 else ''}")
        dm_keys = sorted(disease_map.keys())
        print(f"  disease_map keys:  {dm_keys[:5]}{'...' if len(dm_keys) > 5 else ''}")
        if not normalize_disease_map_suffix:
            print("Tip: if training images had a suffix like '_image', try "
                  "--normalize_disease_map_suffix _image")
        print("All cases will receive default all-zeros disease vector.")
        print("=" * 70 + "\n")

    disease_K = len(next(iter(disease_map.values())))
    default_vec = [0] * disease_K

    for case_id, file_list in sorted(cases.items()):
        if case_id in disease_map:
            vec = torch.tensor([disease_map[case_id]], dtype=torch.float32, device=dev)
            mod.set_disease_vec(vec)
            print(f"\n{case_id}: disease_vec = {disease_map[case_id]}")
        else:
            vec = torch.tensor([default_vec], dtype=torch.float32, device=dev)
            mod.set_disease_vec(vec)
            print(f"\n{case_id}: NOT in disease_map, using default vec = {default_vec}")

        predictor.predict_from_files(
            [sorted(file_list)],
            [join(output_folder, case_id)],
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
    parser.add_argument('--normalize_disease_map_suffix', default='', metavar='SUFFIX',
                        help='Strip this suffix from disease_map keys before matching '
                             '(e.g. "_image" if training images were named ct_xxxx_image_0000.nii.gz '
                             'but test images are ct_xxxx_0000.nii.gz)')
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
        normalize_disease_map_suffix=args.normalize_disease_map_suffix,
        device=args.device,
        tile_step_size=args.step_size,
        use_mirroring=not args.disable_tta,
        save_probabilities=args.save_probabilities,
    )


if __name__ == '__main__':
    main()
