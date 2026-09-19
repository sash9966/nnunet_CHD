#!/usr/bin/env python3
"""Build matched four-arm datasets; freeze plans, normalization, cohort and splits.

No case additions/removals per arm. Existing datasets are never overwritten. The unchanged baseline reuses the source D090 dataset; only the three refinement datasets
are constructed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

from refinement_common import ARMS, sha256, write_json
from run_refinement_ablation import is_complete, pin, read_run

EXCLUDED = {'BAF004', 'BAF005', 'CHIPS001', 'CHIPS002', 'CHIPS005', 'CHIPS006',
            'CHIPS007', 'CHIPS010', 'CHIPS016'}
PLANS = 'nnUNetResEncUNetMPlans'
TRAINER = 'nnUNetTrainerDA5_200epochs'


def fixed_plans(original, dataset):
    plans = copy.deepcopy(original)
    plans['dataset_name'] = dataset
    plans['plans_name'] = PLANS
    for name, config in plans['configurations'].items():
        if 'data_identifier' in config:
            config['data_identifier'] = PLANS + '_' + name
    return plans


def validate_splits(splits, cohort, selected):
    if len(splits) != 5:
        raise ValueError('Expected five existing matched folds')
    for fold in splits:
        tr, val = set(fold['train']), set(fold['val'])
        if tr & val or tr | val != cohort or not selected <= tr or (tr | val) & EXCLUDED:
            raise ValueError('Invalid splits: require full cohort, selected cases train-only and no held-out cases')


def build(args, root):
    cfg = read_run(root)
    raw, pre = Path(args.raw).absolute(), Path(args.preprocessed).absolute()
    source = raw / args.source_dataset
    cohort = {p.name[:-7] for p in (source/'labelsTr').glob('*.nii.gz')}
    selected = set(cfg['cases'])
    if not cohort or not selected <= cohort or cohort & EXCLUDED:
        raise ValueError('Source cohort missing cases or contains clinical holdouts')
    splits_path = pre/args.source_dataset/'splits_final.json'
    splits = json.loads(splits_path.read_text()); validate_splits(splits, cohort, selected)
    plan_path = Path(args.reference_plans).absolute(); reference = json.loads(plan_path.read_text())
    source_plan = pre/args.source_dataset/(PLANS+'.json')
    if json.loads(source_plan.read_text()) != reference:
        raise ValueError('Baseline reuse requires the unchanged source plans')
    dj = json.loads((source/'dataset.json').read_text())
    if dj.get('file_ending') != '.nii.gz' or len(dj['channel_names']) != 1 or set(dj['labels'].values()) != set(range(8)):
        raise ValueError('Expected single-channel CHD dataset with labels 0..7')
    if len(set(args.dataset_ids)) != 3:
        raise ValueError('Three unique refinement dataset IDs required')
    for arm in ARMS:
        if not is_complete(root/'arms'/arm):
            raise ValueError('Incomplete arm: ' + arm)
    sources = {}
    for case in sorted(cohort):
        image = source/'imagesTr'/(case+'_0000.nii.gz'); label = source/'labelsTr'/(case+'.nii.gz')
        if not image.is_file() or not label.is_file():
            raise ValueError('Missing source image/label: ' + case)
        if case in selected and sha256(image) != cfg['cases'][case]['image_sha256']:
            raise ValueError('Refinement used a different image: ' + case)
        if case in selected and sha256(label) != cfg['cases'][case]['seed_sha256']:
            raise ValueError('Cannot reuse D090: refinement seeds differ from source labels: '+case)
        sources[case] = {'image': str(image.absolute()), 'image_sha256': sha256(image),
                         'label': str(label.absolute()), 'label_sha256': sha256(label)}
    identity = {'source_dataset': args.source_dataset, 'cohort': sources,
                'split_sha256': sha256(splits_path), 'reference_plan_sha256': sha256(plan_path),
                'arms': {a: {'id': i, 'dataset': 'Dataset%03d_%s' % (i, {'baseline': 'ImageCHDPseudoBaseline', 'chambers': 'ImageCHDRefinedChambers', 'seqseg': 'ImageCHDRefinedSeqSeg', 'combined': 'ImageCHDRefinedCombined'}[a])} for a, i in zip(ARMS[1:], args.dataset_ids)}}
    if (root/'training.json').exists():
        raise ValueError('Training datasets already built; use preprocess/train stages')
    # Check every destination before any mutation; never clobber or recycle an existing dataset ID.
    for arm, entry in identity['arms'].items():
        i = entry['id']
        if not 1 <= i <= 999 or list(raw.glob('Dataset%03d_*' % i)) or list(pre.glob('Dataset%03d_*' % i)) or list(Path(args.results).absolute().glob('Dataset%03d_*' % i)):
            raise ValueError('Dataset ID unavailable: %d' % i)
    for arm, entry in identity['arms'].items():
        name = entry['dataset']; dst = raw/name; pp = pre/name
        (dst/'imagesTr').mkdir(parents=True); (dst/'labelsTr').mkdir(); pp.mkdir(parents=True)
        for case, item in sources.items():
            label = root/'arms'/arm/(case+'.nii.gz') if case in selected else Path(item['label'])
            (dst/'imagesTr'/(case+'_0000.nii.gz')).symlink_to(item['image'])
            (dst/'labelsTr'/(case+'.nii.gz')).symlink_to(label.absolute())
        dj_arm = dict(dj, numTraining=len(cohort))
        write_json(dst/'dataset.json', dj_arm); write_json(pp/'dataset.json', dj_arm)
        write_json(pp/'splits_final.json', splits)
        write_json(pp/(PLANS+'.json'), fixed_plans(reference, name))
        entry['raw'] = str(dst); entry['preprocessed'] = str(pp)
        entry['labels'] = {c: sha256(dst/'labelsTr'/(c+'.nii.gz')) for c in sorted(cohort)}
        entry['plans_sha256'] = sha256(pp/(PLANS+'.json'))
        write_json(dst/'refinement_manifest.json', {'run': str(root), 'arm': arm, 'selected': sorted(selected),
                                                   'label_sha256': entry['labels'], 'policy': cfg['policy']})
    identity['arms']['baseline'] = {'id': int(args.source_dataset[7:10]), 'dataset': args.source_dataset,
        'raw': str(source), 'preprocessed': str(pre/args.source_dataset), 'reused': True,
        'labels': {c: item['label_sha256'] for c, item in sources.items()}, 'plans_sha256': sha256(source_plan)}
    identity['plans'] = PLANS; identity['trainer'] = TRAINER
    identity['splits'] = splits
    write_json(root/'training.json', identity)
    print('Reused D090 baseline; built three datasets with %d matched cases' % len(cohort))


def validate_training(root, arm):
    study = json.loads((root/'training.json').read_text()); entry = study['arms'][arm]
    pp = Path(entry['preprocessed']); raw = Path(entry['raw'])
    if sha256(pp/(PLANS+'.json')) != entry['plans_sha256'] or json.loads((pp/'splits_final.json').read_text()) != study['splits']:
        raise ValueError('Frozen plans or splits changed')
    if {p.name[:-7] for p in (raw/'labelsTr').glob('*.nii.gz')} != set(entry['labels']):
        raise ValueError('Cohort changed')
    for case, digest in entry['labels'].items():
        if sha256(raw/'labelsTr'/(case+'.nii.gz')) != digest:
            raise ValueError('Training label changed: '+case)
        if sha256(raw/'imagesTr'/(case+'_0000.nii.gz')) != study['cohort'][case]['image_sha256']:
            raise ValueError('Training image changed: '+case)
    os.environ['nnUNet_raw'] = str(raw.parent); os.environ['nnUNet_preprocessed'] = str(pp.parent)
    return study, entry


def preprocess(args, root):
    study, entry = validate_training(root, args.arm)
    if entry.get('reused'):
        raise ValueError('Baseline is reused; do not preprocess it')
    pp = Path(entry['preprocessed']); marker = pp/'refinement_preprocessing_complete.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['plans_sha256'] != entry['plans_sha256']:
            raise ValueError('Preprocessing plan mismatch')
        for p, digest in record['files'].items():
            if not (pp/p).is_file() or sha256(pp/p) != digest:
                raise ValueError('Incomplete/changed preprocessing: '+p)
        print('Verified completed preprocessing: '+args.arm); return
    subprocess.run(['nnUNetv2_extract_fingerprint', '-d', str(entry['id']), '-np', str(args.workers),
                    '--verify_dataset_integrity'], check=True)
    # Do not plan again: all arms retain the exact copied architecture/spacing/normalization.
    subprocess.run(['nnUNetv2_preprocess', '-d', str(entry['id']), '-plans_name', PLANS,
                    '-c', '3d_fullres', '-np', str(args.workers)], check=True)
    plans = json.loads((pp/(PLANS+'.json')).read_text())
    folder = pp/plans['configurations']['3d_fullres']['data_identifier']
    for case in entry['labels']:
        if not (folder/(case+'.pkl')).is_file() or not any((folder/(case+ext)).is_file() for ext in ('.npz', '.b2nd', '.npy')):
            raise ValueError('Missing preprocessed case: '+case)
    files = {str(p.relative_to(pp)): sha256(p) for p in folder.rglob('*') if p.is_file()}
    if not files:
        raise ValueError('Empty preprocessing output')
    write_json(marker, {'plans_sha256': entry['plans_sha256'], 'files': files})


def train(args, root):
    study, entry = validate_training(root, args.arm)
    if entry.get('reused'):
        raise ValueError('Baseline is reused; do not retrain it')
    if not (Path(entry['preprocessed'])/'refinement_preprocessing_complete.json').is_file():
        raise ValueError('Preprocess this arm before training')
    results = Path(args.results).absolute(); os.environ['nnUNet_results'] = str(results)
    os.environ['nnUNet_compile'] = 'f'
    out = results/entry['dataset']/(TRAINER+'__'+PLANS+'__3d_fullres')/('fold_'+str(args.fold))
    identity = {'run': str(root), 'arm': args.arm, 'fold': args.fold, 'plans_sha256': entry['plans_sha256'],
                'training_manifest_sha256': sha256(root/'training.json'), 'trainer': TRAINER,
                'optimization_seed': 'not fixed; paired folds are not matched random-seed replicates'}
    if out.exists() and not (out/'refinement_identity.json').exists():
        raise ValueError('Existing training output has unknown provenance: '+str(out))
    pin(out/'refinement_identity.json', identity)
    if (out/'checkpoint_final.pth').exists():
        print('Completed fold exists: '+str(out)); return
    command = ['nnUNetv2_train', str(entry['id']), '3d_fullres', str(args.fold), '-tr', TRAINER, '-p', PLANS]
    if (out/'checkpoint_latest.pth').exists():
        command += ['--c']
    subprocess.run(command, check=True)
    if not (out/'checkpoint_final.pth').is_file():
        raise ValueError('Training returned without final checkpoint')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', required=True)
    sub = ap.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('build'); p.add_argument('--raw', required=True); p.add_argument('--preprocessed', required=True)
    p.add_argument('--source-dataset', default='Dataset090_ImageCHDPseudoCombined')
    p.add_argument('--results', default=os.environ.get('nnUNet_results', 'nnUNet_results')); p.add_argument('--reference-plans', required=True); p.add_argument('--dataset-ids', type=int, nargs=3, default=[95,96,97])
    p = sub.add_parser('preprocess'); p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--workers', type=int, default=4)
    p = sub.add_parser('train'); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True); p.add_argument('--results', required=True)
    args = ap.parse_args(); root = Path(args.run).absolute()
    {'build': build, 'preprocess': preprocess, 'train': train}[args.stage](args, root)


if __name__ == '__main__':
    main()
