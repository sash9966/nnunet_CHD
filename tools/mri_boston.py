#!/usr/bin/env python3
"""Build the reviewed MRI cohorts, preserve clinical splits, and review pseudo myocardium.

Data stay outside git. Outputs are immutable: choose a new output directory to rebuild.
Requires numpy/nibabel; splits additionally requires scikit-learn (nnU-Net dependency).
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

LABELS = {'background': 0, 'LV-BP': 1, 'RV-BP': 2, 'LA': 3, 'RA': 4,
          'Myo': 5, 'Aorta': 6, 'Pulmonary': 7}
BLOOD = (1, 2, 3, 4, 6, 7)
MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 4, 8: 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(data, affine, path):
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)


def pair(image_path, label_path, allowed=range(8), full=False):
    im, lab = nib.load(image_path), nib.load(label_path)
    x, y = np.asanyarray(im.dataobj), np.asanyarray(lab.dataobj)
    require(x.ndim == 3 and x.shape == y.shape, 'Image/label shape mismatch')
    require(np.allclose(im.affine, lab.affine, atol=1e-5), 'Image/label affine mismatch')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite voxels')
    values = set(np.unique(y).tolist())
    require(values <= set(allowed), 'Unexpected or noninteger labels: ' + str(values))
    if full:
        require(values == set(range(8)), 'Missing structure in ' + str(label_path))
    return im, y.astype(np.uint8)


def ids(root, folder='labelsTr'):
    return sorted(p.name[:-7] for p in (Path(root) / folder).glob('*.nii.gz'))


def start(root):
    require(not root.exists(), 'Output already exists: ' + str(root))
    root.mkdir(parents=True)
    for folder in ('imagesTr', 'labelsTr'):
        (root / folder).mkdir()


def copy_case(src, dst, case, image_folder='imagesTr', label_folder='labelsTr'):
    image = src / image_folder / (case + '_0000.nii.gz')
    label = src / label_folder / (case + '.nii.gz')
    pair(image, label, full=True)
    shutil.copyfile(image, dst / 'imagesTr' / image.name)
    shutil.copyfile(label, dst / 'labelsTr' / label.name)


def finish(root, groups, provenance):
    cases = ids(root)
    require(set(cases) == {c for group in groups.values() for c in group}, 'Group mismatch')
    require(len(cases) == sum(map(len, groups.values())), 'Duplicate case in groups')
    write(root / 'dataset.json', {'channel_names': {'0': 'MRI'}, 'labels': LABELS,
                                 'numTraining': len(cases), 'file_ending': '.nii.gz'})
    write(root / 'case_groups.json', groups)
    write(root / 'provenance.json', provenance)
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + p.relative_to(root).as_posix() + '\n' for p in paths))
    print(json.dumps({'dataset': str(root), 'numTraining': len(cases),
                      'numInference': len(list((root / 'imagesTs').glob('*.nii.gz')))}))


def build201(a):
    clinical, review, boston, out = map(Path, (a.clinical, a.review, a.boston, a.output))
    clinical_ids = ids(clinical)
    require(len(clinical_ids) == 17, 'Expected 17 clinical cases')
    reviewed = ['HVSMR%03d' % i for i in range(10)]
    with (review / 'review_decisions.csv').open(newline='') as f:
        decisions = list(csv.DictReader(f))
    require(len(decisions) == 10 and {r['case'] for r in decisions} == set(reviewed), 'Review roster mismatch')
    require(all(r['decision'] == 'APPROVED' for r in decisions), 'All ten cases must be reviewed')
    start(out)
    for case in clinical_ids:
        copy_case(clinical, out, case)
    for case in reviewed:
        copy_case(review, out, case, 'images', 'candidate_labels')
    for folder in ('imagesTs', 'labelsTs'):
        (out / folder).mkdir()
    for i in range(10, 60):
        case = 'HVSMR%03d' % i
        im, original = pair(boston / ('pat%d_cropped.nii' % i),
                            boston / ('pat%d_cropped_seg.nii' % i), allowed=range(9))
        mapped = np.zeros(original.shape, np.uint8)
        for old, new in MAPPING.items():
            mapped[original == old] = new
        save(np.asanyarray(im.dataobj), im.affine, out / 'imagesTs' / (case + '_0000.nii.gz'))
        save(mapped, im.affine, out / 'labelsTs' / (case + '.nii.gz'))
        pair(out / 'imagesTs' / (case + '_0000.nii.gz'), out / 'labelsTs' / (case + '.nii.gz'), allowed=(0,) + BLOOD)
    shutil.copyfile(review / 'review_decisions.csv', out / 'review_decisions.csv')
    shutil.copyfile(review / 'review_manifest.json', out / 'legacy_transfer_audit.json')
    (out / 'README.md').write_text(
        '# Dataset201: clinical plus reviewed Boston MRI\n\n'
        'Training: 17 clinical + 10 Boston (HVSMR000–009). Native MRI geometry.\n'
        'Boston RA includes SVC/IVC; myocardium transferred from matched legacy label A,\n'
        'only outside revised blood/vessel labels. User approved sampled overlays; this\n'
        'records visual acceptance, not exhaustive voxel-by-voxel adjudication.\n'
        'imagesTs: remaining 50 Boston cases. labelsTs: six human structures ONLY;\n'
        'myocardium unannotated, not a verified negative. Never score myocardium here.\n'
        'This pool is reserved for subsequent annotation/training, not a permanent test set.\n'
        'Labels: 0 background, 1 LV-BP, 2 RV-BP, 3 LA, 4 RA, 5 Myo, 6 Aorta, 7 Pulmonary.\n')
    finish(out, {'clinical': clinical_ids, 'boston_reviewed': reviewed},
           {'source': 'Dataset200_MRI + HVSMR-2.0 + matched legacy HVSMR label A',
            'review': 'User approved all ten sampled overlay reviews on 2026-09-18',
            'boston_label_mapping': MAPPING, 'unannotated_test_labels': [5],
            'clinical_manifest_sha256': digest(clinical / 'SHA256SUMS')})


def splits(a):
    from sklearn.model_selection import KFold
    groups = read(Path(a.dataset) / 'case_groups.json')
    clinical = sorted(groups['clinical'])
    boston = sorted(groups['boston_reviewed'])
    baseline = Path(a.baseline)
    if baseline.is_file():
        base = read(baseline)
        source = 'existing Dataset200 splits: ' + digest(baseline)
    else:
        base = [{'train': [clinical[i] for i in tr], 'val': [clinical[i] for i in va]}
                for tr, va in KFold(5, shuffle=True, random_state=12345).split(clinical)]
        source = 'nnU-Net default: sorted clinical IDs, KFold 5, seed 12345'
    require(len(base) == 5, 'Expected five clinical folds')
    val_all = []
    for fold in base:
        tr, va = fold['train'], fold['val']
        require(len(tr) == len(set(tr)) and len(va) == len(set(va)), 'Duplicate split IDs')
        require(not set(tr) & set(va) and set(tr) | set(va) == set(clinical), 'Invalid clinical split')
        val_all.extend(va)
    require(sorted(val_all) == clinical, 'Each clinical patient must validate exactly once')
    combined = [{'train': sorted(f['train'] + boston), 'val': f['val']} for f in base]
    target = Path(a.output)
    if target.exists():
        require(read(target) == combined, 'Existing splits differ: refusing overwrite')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, combined)
    print('Clinical validation preserved; Boston train-only. Source: ' + source)


def merge_myo(gt, prediction):
    result = gt.copy()
    result[(gt == 0) & (prediction == 5)] = 5
    return result


def pseudo(a):
    dataset, predictions, out = map(Path, (a.dataset, a.predictions, a.output))
    cases = ids(dataset, 'labelsTs')
    require(len(cases) == 50, 'Expected 50 untrained Boston cases')
    require(Path(a.checkpoint).is_file(), 'Teacher checkpoint missing')
    require(not out.exists(), 'Review output exists; refusing to overwrite edits')
    out.mkdir(parents=True)
    (out / 'candidate_labels').mkdir()
    (out / 'conflict_masks').mkdir()
    rows, decisions = [], []
    for case in cases:
        image = dataset / 'imagesTs' / (case + '_0000.nii.gz')
        im, gt = pair(image, dataset / 'labelsTs' / (case + '.nii.gz'), allowed=(0,) + BLOOD)
        _, pred = pair(image, predictions / (case + '.nii.gz'))
        candidate = merge_myo(gt, pred)
        path = out / 'candidate_labels' / (case + '.nii.gz')
        save(candidate, im.affine, path)
        conflicts = (gt != 0) & (pred == 5)
        save(conflicts.astype(np.uint8), im.affine, out / 'conflict_masks' / (case + '.nii.gz'))
        decisions.append({'case': case, 'decision': 'PENDING', 'reviewer': '',
                          'reviewed_sha256': '', 'notes': ''})
        row = {'case': case, 'myocardium_voxels': int((candidate == 5).sum()),
               'conflict_voxels': int(conflicts.sum()), 'initial_sha256': digest(path)}
        for label in BLOOD:
            p, g = pred == label, gt == label
            denom = int(p.sum() + g.sum())
            row['dice_' + str(label)] = 2 * int((p & g).sum()) / denom if denom else None
        rows.append(row)
    write(out / 'six_structure_metrics.json', {'cases': rows, 'evaluated_labels': list(BLOOD),
          'myocardium': 'NOT EVALUATED: no ground truth',
          'macro_dice': float(np.mean([r['dice_' + str(k)] for r in rows for k in BLOOD if r['dice_' + str(k)] is not None]))})
    with (out / 'review_decisions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    write(out / 'teacher_provenance.json', {'checkpoint_sha256': digest(a.checkpoint),
          'dataset_manifest_sha256': digest(dataset / 'SHA256SUMS'),
          'teacher': 'Dataset201 fold_all, DA5 100 epochs',
          'predictions': {case: digest(predictions / (case + '.nii.gz')) for case in cases}})
    (out / 'README.md').write_text(
        'Review candidate_labels against Dataset201/imagesTs in 3D Slicer. Only myocardium\n'
        'may be edited here; the six existing human labels must remain unchanged.\n'
        'Inspect complete volumes, correct missing/excess myocardium, and save. Then set\n'
        'decision=APPROVED, reviewer=<name>, reviewed_sha256=<sha256sum of edited file>\n'
        'in review_decisions.csv. Leave rejected/unreviewed cases unapproved.\n'
        'Metrics use ORIGINAL predictions, before GT merging, and exclude myocardium.\n'
        'Once promoted to Dataset202, these cases are training data, not held-out tests.\n')
    print('Prepared 50 review candidates and six-structure scores at ' + str(out))


def build202(a):
    base, review, out = map(Path, (a.dataset, a.review, a.output))
    groups = read(base / 'case_groups.json')
    require(sum(map(len, groups.values())) == 27, 'Expected Dataset201 with 27 cases')
    with (review / 'review_decisions.csv').open(newline='') as f:
        decisions = list(csv.DictReader(f))
    require(len({r['case'] for r in decisions}) == len(decisions), 'Duplicate review rows')
    approved = [r for r in decisions if r['decision'] == 'APPROVED']
    require(approved, 'No reviewed pseudo myocardium yet; Dataset202 cannot be built')
    teacher = read(review / 'teacher_provenance.json')
    require(teacher['dataset_manifest_sha256'] == digest(base / 'SHA256SUMS'), 'Teacher dataset changed')
    for row in approved:
        case = row['case']
        require(case in ids(base, 'labelsTs'), 'Unknown or training case in pseudo pool')
        path = review / 'candidate_labels' / (case + '.nii.gz')
        require(row['reviewer'].strip() and row['reviewed_sha256'] == digest(path),
                'Missing reviewer or edited-file hash for ' + case)
        _, gt = pair(base / 'imagesTs' / (case + '_0000.nii.gz'), base / 'labelsTs' / (case + '.nii.gz'), allowed=(0,) + BLOOD)
        _, candidate = pair(base / 'imagesTs' / (case + '_0000.nii.gz'), path, full=True)
        require(np.array_equal(np.where(candidate == 5, 0, candidate), gt), 'Human blood/vessel labels changed: ' + case)
    require(np.isfinite(a.clinical_weight) and a.clinical_weight > 0, 'Invalid clinical weight')
    start(out)
    for case in ids(base):
        copy_case(base, out, case)
    for row in approved:
        case = row['case']
        shutil.copyfile(base / 'imagesTs' / (case + '_0000.nii.gz'), out / 'imagesTr' / (case + '_0000.nii.gz'))
        shutil.copyfile(review / 'candidate_labels' / (case + '.nii.gz'), out / 'labelsTr' / (case + '.nii.gz'))
    groups['boston_reviewed_pseudo'] = sorted(r['case'] for r in approved)
    weights = {case: (a.clinical_weight if group == 'clinical' else 1.0)
               for group, cases in groups.items() for case in cases}
    write(out / 'case_weights.json', weights)
    shutil.copyfile(review / 'review_decisions.csv', out / 'review_decisions.csv')
    (out / 'README.md').write_text(
        'Dataset202 uses reviewed pseudo myocardium and six human Boston annotations.\n'
        'Run fold all only. Teacher used every clinical case, so ordinary clinical CV\n'
        'would leak through pseudo labels. Evaluate on new clinical patients.\n'
        'Default case sampling is uniform. Clinical weighting is an optional intervention.\n')
    finish(out, groups, {'teacher': teacher, 'clinical_sampling_weight': a.clinical_weight,
           'approved_cases': approved, 'evaluation': 'New independent clinical patients only'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('build201')
    for key in ('clinical', 'review', 'boston', 'output'):
        q.add_argument('--' + key, required=True)
    q.set_defaults(func=build201)
    q = sub.add_parser('splits')
    for key in ('dataset', 'baseline', 'output'):
        q.add_argument('--' + key, required=True)
    q.set_defaults(func=splits)
    q = sub.add_parser('pseudo')
    for key in ('dataset', 'predictions', 'checkpoint', 'output'):
        q.add_argument('--' + key, required=True)
    q.set_defaults(func=pseudo)
    q = sub.add_parser('build202')
    for key in ('dataset', 'review', 'output'):
        q.add_argument('--' + key, required=True)
    q.add_argument('--clinical-weight', type=float, default=1.0)
    q.set_defaults(func=build202)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
