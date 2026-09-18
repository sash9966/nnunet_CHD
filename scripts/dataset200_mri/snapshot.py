#!/usr/bin/env python3
"""Capture submission inputs locally; no image data or records are sent to GitHub."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
scripts,repo,run=map(Path,sys.argv[1:])
run.mkdir(parents=True,exist_ok=False)
def git(*args):
 return subprocess.check_output(['git','-C',str(repo),*args],text=True)
shutil.copytree(scripts,run/'scripts',ignore=shutil.ignore_patterns('logs','__pycache__'))
shutil.copytree(repo/'nnunetv2',run/'code'/'nnunetv2',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
raw=Path(os.environ['nnUNet_raw'])/os.environ['DATASET']
for name in ('SHA256SUMS','dataset.json'):
 shutil.copyfile(raw/name,run/name)
(run/'working_tree.patch').write_text(git('diff','HEAD','--'))
(run/'git_status.log').write_text(git('status','--porcelain'))
keys=['TRAINER','PLANS','PLANNER','CONFIGURATION','DATASET','NNUNET_ENV','nnUNet_raw','nnUNet_preprocessed','nnUNet_results','CPU_PARTITION','GPU_PARTITION','GPU_GRES','TRAIN_TIME','PREP_TIME','CV_CONCURRENCY','nnUNet_n_proc_DA','OMP_NUM_THREADS','MKL_NUM_THREADS']
record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_commit=git('rev-parse','HEAD').strip(),original_repository=str(repo),settings={k:os.environ[k] for k in keys},mode=os.environ['RUN_MODE'],note='Training imports the captured code/nnunetv2 tree, including local edits and untracked trainers.')
(run/'submission.json').write_text(json.dumps(record,indent=2)+'\n')

files=sorted(f for folder in ('scripts','code') for f in (run/folder).rglob('*') if f.is_file())
(run/'snapshot_SHA256SUMS').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(run)}\n' for f in files))
