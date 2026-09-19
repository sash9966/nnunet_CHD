#!/usr/bin/env python3
"""D090/D091 evaluation route for the four-arm study; same legacy output layout.

Resize to ImageCHD grid, predict each requested fold and the full ensemble, backproject
without LCC. Score every final native mask, never silently skip incomplete predictions.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

from refinement_common import ARMS, NAMES, load_label, sha256, write_json
from run_refinement_ablation import complete, is_complete, pin
from build_refinement_ablation import PLANS, TRAINER, validate_training


def score(predictions, references, cases):
    rows = []
    for case in cases:
        reference, gt = load_label(references/(case+'.nii.gz'))
        _, prediction = load_label(predictions/(case+'.nii.gz'), reference)
        for sid, name in list(NAMES.items()) + [(0, 'WholeHeart')]:
            g = gt == sid if sid else gt > 0
            p = prediction == sid if sid else prediction > 0
            tp = int(np.count_nonzero(g & p)); ng = int(g.sum()); npred = int(p.sum())
            rows.append({'case': case, 'structure': name, 'reference_present': bool(ng),
                         'dice': 2*tp/(ng+npred) if ng+npred else 1.,
                         'recall': tp/ng if ng else None, 'precision': tp/npred if npred else None,
                         'reference_voxels': ng, 'prediction_voxels': npred})
    with (predictions/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(predictions/'metrics_summary.json', {
        'cases': cases, 'postprocessing': 'backproject --no-lcc',
        'mean_dice': {name: float(np.mean([r['dice'] for r in rows if r['structure'] == name]))
                      for name in list(NAMES.values()) + ['WholeHeart']},
        'note': 'D080 is excluded from training but development-exposed; inspect absent-reference classes separately.'})


def require_cases(folder, cases, images=False):
    suffix = '_0000.nii.gz' if images else '.nii.gz'
    found = {p.name[:-len(suffix)] for p in folder.glob('*'+suffix)}
    if found != set(cases):
        raise ValueError('Incomplete/unexpected cases in '+str(folder))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True); ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--folds', choices=['0', '0,1,2,3,4'], default='0,1,2,3,4')
    args = ap.parse_args(); root = Path(args.run).absolute()
    study, entry = validate_training(root, args.arm)
    if entry.get('reused'):
        raise ValueError('Use the existing D090 baseline predictions; this evaluator runs the three new arms')
    raw = Path(entry['raw']).parent; results = Path(os.environ['nnUNet_results'])
    d080 = raw/'Dataset080_ClinicalCaseSanjibDetailed'
    images, references = d080/'imagesTr', d080/'labelsTr'
    cases = sorted(p.name[:-12] for p in images.glob('*_0000.nii.gz'))
    if not cases or set(cases) & set(study['cohort']):
        raise ValueError('Empty evaluation set or train/evaluation leakage')
    require_cases(references, cases)
    model = results/entry['dataset']/(TRAINER+'__'+PLANS+'__3d_fullres')
    folds = args.folds.split(',')
    checkpoints = {f: model/('fold_'+f)/'checkpoint_final.pth' for f in folds}
    # Fail before preparing inputs if any requested fold is missing or foreign.
    for f, checkpoint in checkpoints.items():
        identity = json.loads((checkpoint.parent/'refinement_identity.json').read_text())
        if identity['training_manifest_sha256'] != sha256(root/'training.json') or identity['arm'] != args.arm:
            raise ValueError('Checkpoint provenance mismatch: '+str(checkpoint))
        if not checkpoint.is_file():
            raise ValueError('Missing checkpoint: '+str(checkpoint))
    inputs = {c: {'image': sha256(images/(c+'_0000.nii.gz')), 'reference': sha256(references/(c+'.nii.gz'))} for c in cases}
    repo = Path(__file__).absolute().parents[1]
    code = {n: sha256(repo/'tools'/n) for n in ('resize_to_imagechd_grid.py', 'backproject_predictions_to_native.py', 'evaluate_refinement_models.py')}
    # Per-arm cache avoids concurrent writers while applying the identical resize helper/defaults.
    resized = root/'evaluation_inputs'/args.arm
    pin(resized/'identity.json', {'inputs': inputs, 'code': code})
    if not is_complete(resized):
        subprocess.run([sys.executable, str(repo/'tools/resize_to_imagechd_grid.py'), '--input', str(images),
                        '--output', str(resized), '--overwrite'], check=True)
        require_cases(resized, cases, images=True)
        complete(resized, [resized/(c+'_0000.nii.gz') for c in cases])
    jobs = [('fold'+f, [f]) for f in folds]
    if len(folds) == 5:
        jobs.append(('ensemble', folds))
    for suffix, selected in jobs:
        tag = 'ds%03d_%s' % (entry['id'], suffix)
        final = d080/'predictions'/tag; grid = d080/'predictions'/'_grid512'/tag
        identity = {'run': str(root), 'arm': args.arm, 'training': sha256(root/'training.json'),
                    'inputs': inputs, 'code': code, 'checkpoints': {f: sha256(checkpoints[f]) for f in selected},
                    'route': 'imagechd_grid -> predict -> native --no-lcc'}
        if final.exists() and any(final.glob('*.nii.gz')) and not (final/'refinement_evaluation_identity.json').exists():
            raise ValueError('Refusing existing predictions with unknown provenance: '+str(final))
        pin(final/'refinement_evaluation_identity.json', identity)
        if is_complete(final):
            require_cases(final, cases); print('Verified completed evaluation: '+tag); continue
        grid.mkdir(parents=True, exist_ok=True)
        subprocess.run(['nnUNetv2_predict', '-i', str(resized), '-o', str(grid), '-d', str(entry['id']),
                        '-c', '3d_fullres', '-tr', TRAINER, '-p', PLANS, '-f', *selected,
                        '-chk', 'checkpoint_final.pth'], check=True)
        require_cases(grid, cases)
        subprocess.run([sys.executable, str(repo/'tools/backproject_predictions_to_native.py'),
                        '--pred-dir', str(grid), '--native-dir', str(images), '--output-dir', str(final),
                        '--no-lcc', '--overwrite'], check=True)
        require_cases(final, cases)
        for c in cases:
            load_label(final/(c+'.nii.gz'), nib.load(images/(c+'_0000.nii.gz')))
        score(final, references, cases)
        complete(final, [final/(c+'.nii.gz') for c in cases] + [final/'metrics.csv', final/'metrics_summary.json'])
        print('Complete: '+str(final), flush=True)


if __name__ == '__main__':
    main()
