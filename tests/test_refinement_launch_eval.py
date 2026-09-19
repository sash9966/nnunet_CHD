"""Launcher dependencies and final-mask evaluation; no scheduler or model execution."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
sys.path.insert(0, str(Path(__file__).absolute().parents[1]/'tools'))
from evaluate_refinement_models import require_cases, score
from refinement_common import save_label, write_json
from verify_refinement_baseline import main as verify_baseline
from build_refinement_ablation import PLANS, TRAINER


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

    def test_baseline_reuse_checks_models_and_complete_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); name = 'Dataset090_ImageCHDPseudoCombined'
            raw, pre, results = root/'raw', root/'pre', root/'results'
            model = results/name/(TRAINER+'__'+PLANS+'__3d_fullres')
            plans = {'dataset_name': name, 'configurations': {}}
            dataset = {'labels': {'background': 0}, 'channel_names': {'0': 'CT'}, 'numTraining': 147, 'file_ending': '.nii.gz'}
            write_json(model/'plans.json', plans); write_json(pre/name/(PLANS+'.json'), plans)
            write_json(model/'dataset.json', dataset); write_json(raw/name/'dataset.json', dataset)
            write_json(pre/name/'splits_final.json', [])
            checkpoint = model/'fold_0'/'checkpoint_final.pth'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'fixture')
            d080 = raw/'Dataset080_ClinicalCaseSanjibDetailed'
            (d080/'labelsTr').mkdir(parents=True); (d080/'labelsTr'/'case.nii.gz').write_bytes(b'fixture')
            prediction = d080/'predictions'/'ds090_fold0'/'case.nii.gz'; prediction.parent.mkdir(parents=True)
            env = {'nnUNet_raw': str(raw), 'nnUNet_preprocessed': str(pre), 'nnUNet_results': str(results)}
            with patch.dict(os.environ, env), patch.object(sys, 'argv', ['verify','--run',str(root/'run'),'--folds','0']):
                with self.assertRaisesRegex(ValueError, 'Missing/incomplete'):
                    verify_baseline()
                prediction.write_bytes(b'fixture')
                verify_baseline()
                self.assertTrue((root/'run'/'baseline_reuse_fold0.json').is_file())
                write_json(model/'plans.json', {'different': True})
                with self.assertRaisesRegex(ValueError, 'plans differ'):
                    verify_baseline()

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
            self.assertIn('--array=1-3%2',jobs[1])
            self.assertIn('--array=1-3%2',jobs[3])
            self.assertIn('--dependency=afterok:102',jobs[2])
            self.assertIn('--array=5,10,15%2',jobs[2])
            self.assertIn('--dependency=afterok:103',jobs[3])
            self.assertEqual(jobs[3][-1],'predict')
            repeat = subprocess.run(['bash',str(launcher)],env=env,capture_output=True)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual(len(json.loads((root/'jobs.json').read_text())),4)


if __name__ == '__main__': unittest.main()
