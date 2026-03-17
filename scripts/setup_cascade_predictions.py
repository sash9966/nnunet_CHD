"""Symlink low-res predictions into the cascade fullres trainer's expected location.

Background
----------
nnU-Net's cascade pipeline requires that the low-res model's validation predictions
are available so the cascade-fullres trainer can use them as a spatial prior during
training.  When the SAME trainer class is used for both stages, nnU-Net resolves the
path automatically.  When DIFFERENT trainer classes are used (our case — e.g.,
nnUNetTrainerDA5CascadeFiLM for lowres and nnUNetTrainerDA5CascadeFullresFiLM for
fullres) the automatic path resolution fails.

This script creates the symlinks so the cascade-fullres trainer finds the lowres
predictions where it expects them.

Expected nnUNet_results layout
------------------------------
$nnUNet_results/
  <dataset>/
    <lowres_trainer>__<plans>__3d_lowres/
      fold_0/
        validation/          ← lowres predictions live here after nnUNetv2_predict
        ...
    <cascade_trainer>__<plans>__3d_cascade_fullres/
      fold_0/
        predicted_next_stage/   ← symlink target (cascade trainer looks here)
        ...

Usage
-----
python setup_cascade_predictions.py \\
    --lowres_trainer  nnUNetTrainerDA5CascadeFiLM_100epochs \\
    --cascade_trainer nnUNetTrainerDA5CascadeFullresFiLM_100epochs \\
    --dataset         Dataset001_CHD \\
    --plans           nnUNetResEncUNetMPlans \\
    [--folds 0 1 2 3 4] \\
    [--results_dir /path/to/nnUNet_results]   # defaults to $nnUNet_results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def get_results_dir(cli_arg: str | None) -> Path:
    if cli_arg:
        return Path(cli_arg)
    env = os.environ.get('nnUNet_results')
    if env:
        return Path(env)
    raise EnvironmentError(
        'nnUNet_results environment variable is not set and --results_dir was not provided.'
    )


def setup_fold(
    results_dir: Path,
    dataset: str,
    lowres_trainer: str,
    cascade_trainer: str,
    plans: str,
    fold: int,
    dry_run: bool,
) -> None:
    lowres_val = (
        results_dir
        / dataset
        / f'{lowres_trainer}__{plans}__3d_lowres'
        / f'fold_{fold}'
        / 'validation'
    )
    cascade_target = (
        results_dir
        / dataset
        / f'{cascade_trainer}__{plans}__3d_cascade_fullres'
        / f'fold_{fold}'
        / 'predicted_next_stage'
    )

    if not lowres_val.exists():
        print(f'  [WARN] fold_{fold}: lowres validation folder not found — skipping')
        print(f'         Expected: {lowres_val}')
        print(f'         Run nnUNetv2_predict on 3d_lowres first.')
        return

    if cascade_target.exists():
        if cascade_target.is_symlink():
            existing_target = Path(os.readlink(cascade_target))
            if existing_target == lowres_val:
                print(f'  fold_{fold}: symlink already correct — skipping')
                return
            print(f'  fold_{fold}: removing stale symlink → {existing_target}')
            if not dry_run:
                cascade_target.unlink()
        else:
            print(f'  [WARN] fold_{fold}: {cascade_target} exists and is NOT a symlink.')
            print(f'         Remove it manually if you want to replace it.')
            return

    print(f'  fold_{fold}:')
    print(f'    src : {lowres_val}')
    print(f'    link: {cascade_target}')

    if not dry_run:
        cascade_target.parent.mkdir(parents=True, exist_ok=True)
        cascade_target.symlink_to(lowres_val.resolve())
        print(f'    → symlink created')
    else:
        print(f'    → [DRY RUN] would create symlink')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Symlink lowres predictions into the cascade-fullres trainer directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--lowres_trainer', required=True,
        help='Trainer class name used for the 3d_lowres stage '
             '(e.g., nnUNetTrainerDA5CascadeFiLM_100epochs)',
    )
    parser.add_argument(
        '--cascade_trainer', required=True,
        help='Trainer class name used for the 3d_cascade_fullres stage '
             '(e.g., nnUNetTrainerDA5CascadeFullresFiLM_100epochs)',
    )
    parser.add_argument(
        '--dataset', required=True,
        help='Dataset folder name in nnUNet_results (e.g., Dataset001_CHD)',
    )
    parser.add_argument(
        '--plans', required=True,
        help='Plans identifier (e.g., nnUNetResEncUNetMPlans)',
    )
    parser.add_argument(
        '--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
        help='Folds to set up (default: 0 1 2 3 4)',
    )
    parser.add_argument(
        '--results_dir', default=None,
        help='Path to nnUNet_results directory. Defaults to $nnUNet_results.',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print what would be done without creating symlinks.',
    )

    args = parser.parse_args()
    results_dir = get_results_dir(args.results_dir)

    print(f'nnUNet_results : {results_dir}')
    print(f'Dataset        : {args.dataset}')
    print(f'Lowres trainer : {args.lowres_trainer}')
    print(f'Cascade trainer: {args.cascade_trainer}')
    print(f'Plans          : {args.plans}')
    print(f'Folds          : {args.folds}')
    print(f'Dry run        : {args.dry_run}')
    print()

    for fold in args.folds:
        setup_fold(
            results_dir=results_dir,
            dataset=args.dataset,
            lowres_trainer=args.lowres_trainer,
            cascade_trainer=args.cascade_trainer,
            plans=args.plans,
            fold=fold,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print('\nDone. Verify with:')
        print(f'  ls -la {results_dir}/{args.dataset}/'
              f'{args.cascade_trainer}__{args.plans}__3d_cascade_fullres/')


if __name__ == '__main__':
    main()
