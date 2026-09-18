"""Launcher dependencies and final-mask evaluation; no scheduler or model execution."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
sys.path.insert(0, str(Path(__file__).absolute().parents[1]/'tools'))
from evaluate_refinement_models import require_cases, score
from refinement_common import save_label


class EvaluationTests(unittest.TestCase):
    def test_scores_actual_final_masks_and_requires_complete_cases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); gt = root/'gt'; pred = root/'pred'
            a = np.zeros((5,5,5), np.uint8); a[1:3,1:3,1:3] = 6
            b = a.copy(); b[1,1,1] = 0
            ref = nib.Nifti1Image(a, np.eye(4))
            save_label(gt/'case.nii.gz', a, ref); save_label(pred/'case.nii.gz', b, ref)
            require_cases(pred, ['case']); score(pred, gt, ['case'])
            with (pred/'metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            ao = next(r for r in rows if r['structure']=='Aorta')
            self.assertAlmostEqual(float(ao['dice']), 14/15)
            self.assertAlmostEqual(float(ao['recall']), 7/8)
            with self.assertRaises(ValueError): require_cases(pred, ['case','missing'])

    def test_launcher_dependencies_fold0_and_duplicate_guard(self):
        launcher = Path(__file__).absolute().parents[1]/'scripts/CHD_refinement_submit.sh'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); binary = root/'bin'; binary.mkdir()
            fake = binary/'sbatch'
            fake.write_text('#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\np=Path(os.environ["FAKE_LOG"])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a));print(100+len(a))\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], REPO=str(root),
                       RUN=str(root/'run'), FOLDS='0', FAKE_LOG=str(root/'jobs.json'), RESUBMIT='0')
            subprocess.run(['bash',str(launcher)],env=env,check=True,capture_output=True)
            jobs = json.loads((root/'jobs.json').read_text())
            self.assertEqual(len(jobs),4)
            self.assertIn('--dependency=afterok:101',jobs[1])
            self.assertIn('--dependency=afterok:102',jobs[2])
            self.assertIn('--array=0,5,10,15%2',jobs[2])
            self.assertIn('--dependency=afterok:103',jobs[3])
            self.assertEqual(jobs[3][-1],'predict')
            repeat = subprocess.run(['bash',str(launcher)],env=env,capture_output=True)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual(len(json.loads((root/'jobs.json').read_text())),4)


if __name__ == '__main__': unittest.main()
