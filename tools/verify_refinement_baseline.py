#!/usr/bin/env python3
"""Read-only checks for reusing existing D090 models/predictions, never retrain them."""
import argparse
import json
import os
from pathlib import Path

from build_refinement_ablation import PLANS, TRAINER
from refinement_common import sha256
from run_refinement_ablation import pin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--folds', choices=['0', '0,1,2,3,4'], default='0,1,2,3,4')
    args = ap.parse_args()
    name = 'Dataset090_ImageCHDPseudoCombined'
    raw = Path(os.environ['nnUNet_raw']); pre = Path(os.environ['nnUNet_preprocessed'])
    model = Path(os.environ['nnUNet_results'])/name/(TRAINER+'__'+PLANS+'__3d_fullres')
    source_plan = pre/name/(PLANS+'.json')
    if json.loads((model/'plans.json').read_text()) != json.loads(source_plan.read_text()):
        raise ValueError('Existing D090 model plans differ from the source plans')
    model_dataset = json.loads((model/'dataset.json').read_text())
    raw_dataset = json.loads((raw/name/'dataset.json').read_text())
    for key in ('labels', 'channel_names', 'numTraining', 'file_ending'):
        if model_dataset[key] != raw_dataset[key]:
            raise ValueError('D090 model/source dataset metadata mismatch: '+key)
    folds = args.folds.split(',')
    paths = [source_plan, model/'plans.json', model/'dataset.json', pre/name/'splits_final.json']
    paths += [model/('fold_'+f)/'checkpoint_final.pth' for f in folds]
    d080 = raw/'Dataset080_ClinicalCaseSanjibDetailed'
    cases = sorted(p.name[:-7] for p in (d080/'labelsTr').glob('*.nii.gz'))
    if not cases:
        raise ValueError('No D080 reference cases')
    tags = ['ds090_fold'+f for f in folds] + (['ds090_ensemble'] if len(folds)==5 else [])
    for tag in tags:
        folder = d080/'predictions'/tag
        found = {p.name[:-7] for p in folder.glob('*.nii.gz')}
        if found != set(cases):
            raise ValueError('Missing/incomplete existing baseline predictions: '+str(folder)+
                             '; generate D090 predictions with the established evaluation route before launch')
        paths.extend(folder/(c+'.nii.gz') for c in cases)
    identity = {'dataset': name, 'folds': folds, 'cases': cases,
                'files': {str(p): sha256(p) for p in paths},
                'limitation': 'Matches current source/model metadata and records existing outputs; does not reconstruct historical training bytes or prediction provenance.'}
    # Separate markers permit expanding a completed fold-0 comparison to all folds.
    pin(Path(args.run)/('baseline_reuse_'+('fivefold' if len(folds)==5 else 'fold0')+'.json'), identity)
    print('Reusing D090 models and predictions; no baseline training or preprocessing')


if __name__ == '__main__':
    main()
