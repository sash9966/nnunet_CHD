"""Offline workflow test: python3 scripts/dataset200_mri/tests/test_workflow.py."""
import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parents[1]
class WorkflowTest(unittest.TestCase):
 def test_submission_snapshot_and_dependencies(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);repo=p/'repo';repo.mkdir();(repo/'nnunetv2').mkdir();(repo/'nnunetv2'/'__init__.py').write_text('# original code\n')
   shutil.copytree(SCRIPTS,repo/'scripts')
   subprocess.run(['git','init','-q',str(repo)],check=True)
   subprocess.run(['git','-C',str(repo),'add','.'],check=True)
   subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
   raw=p/'raw'/'Dataset200_MRI';raw.mkdir(parents=True)
   (raw/'dataset.json').write_text('{}');(raw/'SHA256SUMS').write_text('fixture\n')
   b=p/'env'/'bin';b.mkdir(parents=True);(b/'python').symlink_to(sys.executable)
   (b/'squeue').write_text('#!/bin/bash\nexit 0\n')
   (b/'sbatch').write_text('#!'+sys.executable+'\nimport os,json,pathlib,sys\np=pathlib.Path(os.environ["MOCK_LOG"])\na=json.loads(p.read_text());a.append(sys.argv[1:]);p.write_text(json.dumps(a));print(100+len(a))\n')
   for name in ('squeue','sbatch'):(b/name).chmod(0o755)
   env=dict(os.environ,PATH=str(b)+':'+os.environ['PATH'],MOCK_LOG=str(p/'jobs.json'),CHD_REPO=str(repo),NNUNET_ENV=str(p/'env'),nnUNet_raw=str(p/'raw'),nnUNet_preprocessed=str(p/'pre'),nnUNet_results=str(p/'results'),USER='test')
   for mode,count in [('all',2),('cv',3),('both',4)]:
    (p/'jobs.json').write_text('[]')
    subprocess.run(['bash',str(repo/'scripts'/'submit.sh'),mode],env=env,check=True)
    jobs=json.loads((p/'jobs.json').read_text());self.assertEqual(len(jobs),count)
    for i,job in enumerate(jobs[1:],1):self.assertIn(f'--dependency=afterok:{100+i}',job)
    if mode!='all':self.assertIn('--array=1-4%2',jobs[-1])
   runs=list((p/'results'/'_run_records'/'Dataset200_MRI').iterdir());self.assertEqual(len(runs),3)
   (repo/'nnunetv2'/'__init__.py').write_text('# changed after submission\n')
   for run in runs:
    self.assertEqual((run/'code'/'nnunetv2'/'__init__.py').read_text(),'# original code\n')
    self.assertTrue(json.loads((run/'submission.json').read_text())['git_commit'])
    self.assertEqual((run/'SHA256SUMS').read_text(),'fixture\n')
 def test_summary(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for fold in range(5):
    d=root/f'fold_{fold}'/'validation';d.mkdir(parents=True)
    cases=[{'reference_file':f'/labels/case{i}.nii.gz','metrics':{str(k):{'Dice':.8} for k in range(1,8)}} for i in range(fold,17,5)]
    (d/'summary.json').write_text(json.dumps({'metric_per_case':cases}))
   subprocess.run([sys.executable,str(SCRIPTS/'summarize_cv.py'),str(root)],check=True,capture_output=True)
   result=json.loads((root/'cv_summary.json').read_text());self.assertEqual(result['held_out_cases'],17);self.assertAlmostEqual(result['macro_mean_dice'],.8)
if __name__=='__main__':unittest.main()
