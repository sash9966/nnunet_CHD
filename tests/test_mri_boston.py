"""Small end-to-end safety tests: geometry, human labels, review gating and splits."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

spec = importlib.util.spec_from_file_location('mri_boston', Path(__file__).parents[1] / 'tools/mri_boston.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MRIWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / 'Dataset201_Test'
        for folder in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
            (self.base / folder).mkdir(parents=True)
        self.full = np.arange(27, dtype=np.uint8).reshape(3, 3, 3) % 8
        self.gt = np.where(self.full == 5, 0, self.full).astype(np.uint8)
        self.groups = {'clinical': ['CLIN%03d' % i for i in range(17)],
                       'boston_reviewed': ['HVSMR%03d' % i for i in range(10)]}
        for case in sum(self.groups.values(), []):
            m.save(self.full.astype(np.float32), np.eye(4), self.base / 'imagesTr' / (case + '_0000.nii.gz'))
            m.save(self.full, np.eye(4), self.base / 'labelsTr' / (case + '.nii.gz'))
        self.pred = self.root / 'predictions'
        self.pred.mkdir()
        for i in range(10, 60):
            case = 'HVSMR%03d' % i
            m.save(self.full.astype(np.float32), np.eye(4), self.base / 'imagesTs' / (case + '_0000.nii.gz'))
            m.save(self.gt, np.eye(4), self.base / 'labelsTs' / (case + '.nii.gz'))
            m.save(self.full, np.eye(4), self.pred / (case + '.nii.gz'))
        m.finish(self.base, self.groups, {'test': True})
        self.checkpoint = self.root / 'teacher.pth'
        self.checkpoint.write_bytes(b'test teacher')
        self.review = self.root / 'review'

    def tearDown(self):
        self.tmp.cleanup()

    def make_review(self):
        m.pseudo(SimpleNamespace(dataset=self.base, predictions=self.pred,
                                output=self.review, checkpoint=self.checkpoint))

    def approve(self):
        p = self.review / 'review_decisions.csv'
        with p.open(newline='') as f:
            rows = list(csv.DictReader(f))
        rows[0].update(decision='APPROVED', reviewer='test reviewer',
                       reviewed_sha256=m.digest(self.review / 'candidate_labels/HVSMR010.nii.gz'))
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def test_review_promotion_and_original_scores(self):
        # A deliberately wrong blood label must be scored as wrong even though
        # the candidate merge later restores the human annotation.
        wrong = self.full.copy()
        wrong[wrong == 1] = 2
        m.save(wrong, np.eye(4), self.pred / 'HVSMR010.nii.gz')
        self.make_review()
        scores = m.read(self.review / 'six_structure_metrics.json')
        self.assertEqual(scores['cases'][0]['dice_1'], 0)
        self.assertNotIn(5, scores['evaluated_labels'])
        self.assertNotIn('dice_5', scores['cases'][0])
        args = SimpleNamespace(dataset=self.base, review=self.review,
                               output=self.root / 'Dataset202_Test', clinical_weight=10)
        with self.assertRaisesRegex(ValueError, 'No reviewed'):
            m.build202(args)
        self.approve()
        m.build202(args)
        self.assertEqual(m.read(args.output / 'dataset.json')['numTraining'], 28)
        weights = m.read(args.output / 'case_weights.json')
        self.assertEqual(weights['CLIN000'], 10)
        self.assertEqual(weights['HVSMR010'], 1)
        _, promoted = m.pair(args.output / 'imagesTr/HVSMR010_0000.nii.gz', args.output / 'labelsTr/HVSMR010.nii.gz', full=True)
        np.testing.assert_array_equal(np.where(promoted == 5, 0, promoted), self.gt)

    def test_reject_changed_review_hash_and_human_labels(self):
        self.make_review()
        self.approve()
        mask = self.full.copy()
        mask[0, 0, 0] = 5
        path = self.review / 'candidate_labels/HVSMR010.nii.gz'
        m.save(mask, np.eye(4), path)
        args = SimpleNamespace(dataset=self.base, review=self.review,
                               output=self.root / 'Dataset202_Test', clinical_weight=1)
        with self.assertRaisesRegex(ValueError, 'hash'):
            m.build202(args)
        mask[0, 0, 1] = 2
        m.save(mask, np.eye(4), path)
        self.approve()
        with self.assertRaisesRegex(ValueError, 'Human blood'):
            m.build202(args)
        self.assertFalse(args.output.exists())

    def test_clinical_splits_remain_identical(self):
        from sklearn.model_selection import KFold
        clinical = self.groups['clinical']
        baseline = [{'train': [clinical[i] for i in tr], 'val': [clinical[i] for i in va]}
                    for tr, va in KFold(5, shuffle=True, random_state=12345).split(clinical)]
        args = SimpleNamespace(dataset=self.base, baseline=self.root / 'baseline.json', output=self.root / 'splits.json')
        m.splits(args)  # absence reproduces stock nnU-Net
        first = m.read(args.output)
        for old, new in zip(baseline, first):
            self.assertEqual(old['val'], new['val'])
            self.assertEqual(set(new['train']), set(old['train'] + self.groups['boston_reviewed']))
        m.write(args.baseline, baseline)
        m.splits(args)  # existing baseline produces identical output
        baseline.reverse()
        m.write(args.baseline, baseline)
        with self.assertRaisesRegex(ValueError, 'Existing splits differ'):
            m.splits(args)

    def test_geometry_mismatch_rejected(self):
        path = self.base / 'labelsTr/CLIN000.nii.gz'
        affine = np.eye(4)
        affine[0, 3] = 1
        m.save(self.full, affine, path)
        with self.assertRaisesRegex(ValueError, 'affine mismatch'):
            m.pair(self.base / 'imagesTr/CLIN000_0000.nii.gz', path)


if __name__ == '__main__':
    unittest.main()
