#!/usr/bin/env python3
"""Four-arm native-grid refinement. Run --help; see docs/refinement_four_arm.md.

Stages are explicit; technical failures abort rather than silently become successful arms.
Model inference runs only in nni/seqseg. All artifacts are under a new experiment directory.
"""
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from refinement_common import (ARMS, NAMES, chamber_prompts, compose, directed_seeds,
                               load_label, save_label, sha256, write_json)


def checkpoint_identity(folder):
    folder = Path(folder).absolute()
    files = sorted(folder.glob('fold_*/*.pth')) + sorted(folder.glob('*.json'))
    if not files or not any(p.suffix == '.pth' for p in files):
        raise ValueError('No checkpoints in explicit model folder: ' + str(folder))
    return {'folder': str(folder), 'files': {str(p.relative_to(folder)): sha256(p) for p in files}}


def pin(path, identity):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Run identity changed; use a new run directory: ' + str(path))
    write_json(path, identity)


def complete(folder, files):
    write_json(folder / 'complete.json', {str(p.relative_to(folder)): sha256(p) for p in files})


def is_complete(folder):
    marker = folder / 'complete.json'
    if not marker.exists():
        return False
    records = json.loads(marker.read_text())
    if not records or any(not (folder/k).is_file() or sha256(folder/k) != v for k, v in records.items()):
        raise ValueError('Invalid completed output: ' + str(folder))
    return True


def init(args):
    root = Path(args.run).absolute()
    if (root / 'run.json').exists():
        raise ValueError('Run already initialized; use its existing stages or choose a new directory')
    selection = None
    if args.accepted_csv:
        import csv
        with open(args.accepted_csv) as f:
            cases = [r['case_id'] for r in csv.DictReader(f) if r['intended_use'] == 'pseudo_label_train']
        selection = {'file': str(Path(args.accepted_csv).absolute()), 'sha256': sha256(args.accepted_csv),
                     'filter': 'intended_use == pseudo_label_train'}
    elif args.case_report:
        report = json.loads(Path(args.case_report).read_text())
        cases = report['relabelled'] + report.get('added', [])
    elif args.all_cases:
        cases = [p.name[:-7] for p in Path(args.seeds_dir).glob('*.nii.gz')]
    else:
        cases = args.cases.split(',') if args.cases else []
    if not cases or len(cases) != len(set(cases)):
        raise ValueError('Provide a nonempty unique case list/report')
    if args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError('Accepted cohort count differs from expected %d' % args.expected_cases)
    inputs = {}
    for case in sorted(cases):
        if Path(case).name != case or case in ('.', '..'):
            raise ValueError('Invalid case ID')
        image = Path(args.images_dir).absolute() / (case + '_0000.nii.gz')
        label = Path(args.seeds_dir).absolute() / (case + '.nii.gz')
        im = nib.load(str(image)); _, lab = load_label(label, im)
        # Require actual classwise LCC inputs; do not silently alter the baseline.
        from refinement_common import largest_component
        for sid in range(1, 8):
            if not np.array_equal(largest_component(lab == sid), lab == sid):
                raise ValueError('%s label %d is not an LCC seed' % (case, sid))
        inputs[case] = {'image': str(image), 'seed': str(label),
                        'image_sha256': sha256(image), 'seed_sha256': sha256(label)}
    root.mkdir(parents=True, exist_ok=True)
    policy = {'min_retention': args.min_retention, 'max_growth': args.max_growth,
              'vessel_max_growth': args.vessel_max_growth,
              'cleanup': not args.no_cleanup, 'prompt_margin_mm': args.prompt_margin_mm,
              'seed_inset_mm': args.seed_inset_mm,
              'arbitration': 'own_seed_or_background; conflicting_background_abstains',
              'vessel_mode': args.vessel_mode, 'chamber_mode': args.chamber_mode}
    if not 0 <= args.min_retention <= 1 or args.max_growth < 1 or args.vessel_max_growth < 1 or args.prompt_margin_mm <= 0 or args.seed_inset_mm <= 0:
        raise ValueError('Invalid policy thresholds')
    cfg = {'schema': 1, 'cases': inputs, 'policy': policy, 'selection': selection}
    write_json(root / 'run.json', cfg)
    print('Initialized %d cases in %s' % (len(cases), root))


def read_run(root):
    cfg = json.loads((root/'run.json').read_text())
    for case, item in cfg['cases'].items():
        for key in ('image', 'seed'):
            if sha256(item[key]) != item[key+'_sha256']:
                raise ValueError('Input changed: %s %s' % (case, key))
    sources = {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('refinement_common.py'),
                                         Path(__file__).with_name('label_to_prompts.py'),
                                         Path(__file__).with_name('seqseg_bounded.py')]}
    pin(root/'implementation.json', sources)
    pin(root/'run_identity.json', {'run_sha256': sha256(root/'run.json')})
    return cfg


def run_nni(args, root, cfg):
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    identity = checkpoint_identity(args.model)
    identity.update({'package': importlib.metadata.version('nnInteractive'), 'torch': torch.__version__,
                     'checkpoint': args.checkpoint, 'fold': args.fold})
    pin(root/'nni_model.json', identity)
    sess = nnInteractiveInferenceSession(device=torch.device(args.device))
    sess.initialize_from_trained_model_folder(args.model, use_fold=args.fold, checkpoint_name=args.checkpoint)
    for case, item in cfg['cases'].items():
        folder = root/'nni'/case
        if is_complete(folder):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        im = nib.load(item['image']); _, seed = load_label(item['seed'], im)
        arr = np.asarray(im.dataobj, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite CT: ' + case)
        sess.set_image(arr[None]); spacing = im.header.get_zooms()[:3]
        files, records = [], {}
        for sid in range(1, 5):
            target = torch.zeros(seed.shape, dtype=torch.uint8)
            sess.set_target_buffer(target); sess.reset_interactions()
            lassos, negatives = chamber_prompts(seed, sid, spacing, cfg['policy']['prompt_margin_mm'])
            interactions = []
            for k, (crop, bbox) in enumerate(lassos):
                prompt = folder / ('%d_lasso_%d.npz' % (sid, k))
                np.savez_compressed(prompt, mask=crop, bbox=np.asarray(bbox)); files.append(prompt)
                sess.add_lasso_interaction(crop, include_interaction=True, interaction_bbox=bbox)
                stage = folder / ('%d_positive_%d.nii.gz' % (sid, k))
                save_label(stage, target.cpu().numpy(), im); files.append(stage)
                interactions.append({'type': 'lasso', 'include': True, 'file': prompt.name, 'bbox': bbox})
            for k, neg in enumerate(negatives):
                sess.add_point_interaction(tuple(neg['point']), include_interaction=False)
                stage = folder / ('%d_negative_%d.nii.gz' % (sid, k))
                save_label(stage, target.cpu().numpy(), im); files.append(stage)
                interactions.append({'type': 'point', 'include': False, **neg})
            candidate = folder / ('%d.nii.gz' % sid)
            save_label(candidate, target.cpu().numpy(), im); files.append(candidate)
            records[str(sid)] = {'absent_seed': not bool(lassos), 'ordered_interactions': interactions}
        write_json(folder/'prompts.json', records); files.append(folder/'prompts.json')
        complete(folder, files)
        print('nnInteractive candidates: ' + case, flush=True)


def run_seqseg(args, root, cfg):
    import SimpleITK as sitk
    import inspect
    from seqseg.modules import tracing
    from seqseg.pipeline import classic
    from seqseg import cli
    identity = checkpoint_identity(args.model)
    version = importlib.metadata.version('seqseg')
    identity.update({'version': version, 'fold': args.fold, 'config': args.seqseg_config,
                     'scale': args.scale, 'max_branches': args.max_branches,
                     'max_steps': args.max_steps, 'steps_per_branch': args.steps_per_branch,
                     'max_bifurcations': 2})
    identity['runtime_source_sha256'] = {m.__name__: sha256(inspect.getsourcefile(m)) for m in (tracing, classic, cli)}
    from seqseg.config_models import AlgorithmConfig
    identity['resolved_config'] = dict(AlgorithmConfig.from_name(args.seqseg_config))
    pin(root/'seqseg_model.json', identity)
    for case, item in cfg['cases'].items():
        im = nib.load(item['image']); _, seed = load_label(item['seed'], im)
        spacing = im.header.get_zooms()[:3]
        for sid in (6, 7):
            folder = root/'seqseg'/case/str(sid)
            if is_complete(folder):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            candidate = folder/'candidate.nii.gz'
            if not np.any(seed == sid):
                save_label(candidate, np.zeros(seed.shape, np.uint8), im)
                write_json(folder/'status.json', {'status': 'absent_seed'})
                complete(folder, [candidate, folder/'status.json']); continue
            try:
                rows, details = directed_seeds(seed == sid, im.affine, spacing, cfg['policy']['seed_inset_mm'])
            except ValueError as error:
                save_label(candidate, np.zeros(seed.shape, np.uint8), im)
                write_json(folder/'status.json', {'status': 'degenerate_seed', 'reason': str(error)})
                complete(folder, [candidate, folder/'status.json']); continue
            seeds = [{'name': case+'_0000', 'seeds': rows, 'cardiac_mesh': False}]
            write_json(folder/'seeds.json', seeds); write_json(folder/'seed_details.json', details)
            # Each retry has a fresh output directory; never accept debris from a failed trace.
            attempt = 0
            while (folder/('attempt_%d' % attempt)).exists():
                attempt += 1
            trace = folder/('attempt_%d' % attempt)
            audit = folder/('depth_audit_%d' % attempt)
            command = [sys.executable, str(Path(__file__).with_name('seqseg_bounded.py')),
                       '--audit-dir', str(audit), '--max-bifurcations', '2', '--',
                       'run', 'single', '--image', item['image'], '--outdir', str(trace),
                       '--model-folder', str(Path(args.model).absolute()), '--nnunet-type', '3d_fullres',
                       '--train-dataset', args.train_dataset, '--fold', str(args.fold),
                       '--config-name', args.seqseg_config, '--scale', str(args.scale), '--unit', 'mm',
                       '--seeds-json', str(folder/'seeds.json'), '--max-n-branches', str(args.max_branches),
                       '--max-n-steps', str(args.max_steps), '--max-n-steps-per-branch', str(args.steps_per_branch)]
            write_json(folder/'command.json', command)
            with (folder/('attempt_%d.log' % attempt)).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            matches = list(trace.glob('*_segmentation_*.mha'))
            if len(matches) != 1:
                raise ValueError('Expected exactly one final MHA, found %d in %s' % (len(matches), trace))
            volume = sitk.ReadImage(str(matches[0])); a = sitk.GetArrayFromImage(volume)
            if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
                raise ValueError('Expected binary final MHA, not a probability/other volume')
            ref = sitk.ReadImage(item['image'])
            mapped = sitk.Resample(volume, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(mapped).transpose(2, 1, 0)
            save_label(candidate, mask, im)
            status = {'status': 'completed_empty' if not mask.any() else 'completed',
                      'source': str(matches[0]), 'source_sha256': sha256(matches[0]),
                      'native_voxels': int(mask.sum()), 'mha_voxels': int(np.count_nonzero(a)),
                      'conversion': 'binary_MHA_to_CT_nearest_neighbor; no_surface_roundtrip'}
            write_json(folder/'status.json', status)
            complete(folder, [candidate, folder/'status.json', folder/'seeds.json', folder/'seed_details.json',
                              folder/'command.json', matches[0], folder/('attempt_%d.log' % attempt)] + list(audit.glob('*')))
            print('SeqSeg candidate: %s %s' % (case, NAMES[sid]), flush=True)


def assemble(args, root, cfg):
    summary = {}
    for case, item in cfg['cases'].items():
        im, seed = load_label(item['seed']); spacing = im.header.get_zooms()[:3]
        for arm in args.arms:
            candidates = {}
            vessel_raw_counts = {}
            chamber_raw_counts = {}
            if arm in ('chambers', 'combined'):
                if not is_complete(root/'nni'/case):
                    raise ValueError('Missing completed nnInteractive case: ' + case)
                for sid in range(1, 5):
                    _, a = load_label(root/'nni'/case/('%d.nii.gz' % sid), im)
                    chamber_raw_counts[sid] = int(np.count_nonzero(a))
                    candidates[sid] = (a > 0) | (seed == sid) if cfg['policy']['chamber_mode'] == 'union' else a > 0
            if arm in ('seqseg', 'combined'):
                for sid in (6, 7):
                    folder = root/'seqseg'/case/str(sid)
                    if not is_complete(folder):
                        raise ValueError('Missing completed SeqSeg case/vessel: %s/%d' % (case, sid))
                    _, a = load_label(folder/'candidate.nii.gz', im)
                    vessel_raw_counts[sid] = int(np.count_nonzero(a))
                    candidates[sid] = (a > 0) | (seed == sid) if cfg['policy']['vessel_mode'] == 'union' else a > 0
            out, qc, conflicts = compose(seed, candidates, spacing, cfg['policy']['min_retention'],
                                          cfg['policy']['max_growth'], cfg['policy']['cleanup'], cfg['policy']['vessel_max_growth'])
            if arm in ('chambers', 'combined'):
                for sid in range(1, 5):
                    qc['structures'][str(sid)]['proposal_mode'] = cfg['policy']['chamber_mode']
                    qc['structures'][str(sid)]['nni_raw_candidate_voxels'] = chamber_raw_counts[sid]
                    if cfg['policy']['chamber_mode'] == 'union' and not np.all(out[seed == sid] == sid):
                        raise AssertionError('Conservative chamber mode removed seed voxels')
            if arm in ('seqseg', 'combined'):
                qc['seqseg_status'] = {str(sid): json.loads((root/'seqseg'/case/str(sid)/'status.json').read_text()) for sid in (6, 7)}
                for sid in (6, 7):
                    rec = qc['structures'][str(sid)]
                    rec['seqseg_raw_candidate_voxels'] = vessel_raw_counts[sid]
                    rec['proposal_mode'] = cfg['policy']['vessel_mode']
            folder = root/'arms'/arm
            save_label(folder/(case+'.nii.gz'), out, im)
            save_label(root/'qc'/arm/(case+'_conflicts.nii.gz'), conflicts, im)
            write_json(root/'qc'/arm/(case+'.json'), qc)
            summary[case+':'+arm] = qc
    for arm in args.arms:
        complete(root/'arms'/arm, [root/'arms'/arm/(c+'.nii.gz') for c in cfg['cases']])
    write_json(root/'assembly_summary.json', summary)


def evaluate(args, root, cfg):
    import csv
    rows = []
    for arm in args.arms:
        if not is_complete(root/'arms'/arm):
            raise ValueError('Incomplete arm: ' + arm)
    for case, item in cfg['cases'].items():
        im = nib.load(item['image']); _, gt = load_label(Path(args.gt_dir)/(case+'.nii.gz'), im)
        for arm in args.arms:
            _, a = load_label(root/'arms'/arm/(case+'.nii.gz'), im)
            for sid, name in list(NAMES.items()) + [(0, 'WholeHeart')]:
                g = gt == sid if sid else gt > 0; p = a == sid if sid else a > 0
                count = int(g.sum()+p.sum()); tp = int(np.count_nonzero(g & p))
                rows.append({'case': case, 'arm': arm, 'structure': name, 'reference_present': bool(g.any()),
                             'dice': 2*tp/count if count else 1.,
                             'prediction_voxels': int(p.sum()), 'reference_voxels': int(g.sum())})
    with (root/'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    sub = ap.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('init'); p.add_argument('--images-dir', required=True); p.add_argument('--seeds-dir', required=True)
    group = p.add_mutually_exclusive_group(required=True); group.add_argument('--case-report'); group.add_argument('--cases'); group.add_argument('--all-cases', action='store_true'); group.add_argument('--accepted-csv')
    p.add_argument('--expected-cases', type=int)
    p.add_argument('--min-retention', type=float, default=.7); p.add_argument('--max-growth', type=float, default=4.)
    p.add_argument('--vessel-max-growth', type=float, default=20.)
    p.add_argument('--no-cleanup', action='store_true'); p.add_argument('--prompt-margin-mm', type=float, default=1.)
    p.add_argument('--chamber-mode', choices=['union', 'replace'], default='union')
    p.add_argument('--seed-inset-mm', type=float, default=5.); p.add_argument('--vessel-mode', choices=['replace', 'union'], default='union')
    p = sub.add_parser('nni'); p.add_argument('--model', required=True); p.add_argument('--fold', default='0')
    p.add_argument('--checkpoint', default='checkpoint_final.pth'); p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('seqseg'); p.add_argument('--model', required=True); p.add_argument('--fold', default='all')
    p.add_argument('--train-dataset', default='Dataset006_SEQAORTANDFEMOCT')
    p.add_argument('--seqseg-config', default='aorta_tutorial'); p.add_argument('--scale', type=float, default=.1)
    p.add_argument('--max-branches', type=int, default=14); p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--steps-per-branch', type=int, default=60)
    p = sub.add_parser('assemble'); p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p = sub.add_parser('evaluate'); p.add_argument('--gt-dir', required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    args = ap.parse_args()
    if args.stage == 'init':
        init(args); return
    root = Path(args.run).absolute(); cfg = read_run(root)
    {'nni': run_nni, 'seqseg': run_seqseg, 'assemble': assemble, 'evaluate': evaluate}[args.stage](args, root, cfg)


if __name__ == '__main__':
    main()
