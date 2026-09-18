#!/usr/bin/env python3
"""Aggregate held-out per-case Dice from five nnU-Net validation summaries."""
import csv,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[];seen=set()
for fold in range(5):
 p=root/f'fold_{fold}'/'validation'/'summary.json'
 data=json.loads(p.read_text())
 for case in data['metric_per_case']:
  name=Path(case['reference_file']).name
  if name in seen: raise RuntimeError(f'Duplicate held-out case: {name}')
  seen.add(name)
  row={'case':name,'fold':fold}
  for label in range(1,8):row[str(label)]=case['metrics'][str(label)]['Dice']
  rows.append(row)
if len(rows)!=17:raise RuntimeError(f'Expected 17 held-out cases, found {len(rows)}')
with (root/'cv_per_case_dice.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['case','fold']+[str(i) for i in range(1,8)]);w.writeheader();w.writerows(rows)
means={}
for label in range(1,8):
 vals=[float(r[str(label)]) for r in rows if r[str(label)] is not None and math.isfinite(float(r[str(label)]))]
 means[str(label)]=sum(vals)/len(vals) if vals else None
result={'held_out_cases':len(rows),'mean_dice_per_label':means,'macro_mean_dice':sum(v for v in means.values() if v is not None)/sum(v is not None for v in means.values())}
(root/'cv_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
