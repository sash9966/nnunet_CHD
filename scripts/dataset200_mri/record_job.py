#!/usr/bin/env python3
import datetime,json,os,platform,subprocess,sys
from pathlib import Path
import torch
run=Path(os.environ['RUN_DIR'])
job=os.environ.get('SLURM_JOB_ID','interactive')
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),python=sys.version,host=platform.node(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
(run/f'job-{job}.json').write_text(json.dumps(record,indent=2)+'\n')
with (run/f'packages-{job}.log').open('w') as f:
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
