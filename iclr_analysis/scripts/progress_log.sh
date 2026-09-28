#!/bin/bash
# One line every ten minutes: how many papers each reviewer subset has a score for, and how many workers are up.
R=${SCISLOP_ROOT}/paper/draft_v6/review; cd $R
while true; do
  python3 - <<'PY' >> logs/progress.log
import json,glob,os,subprocess,datetime
def n(sub,ids=None):
    c=0
    for f in glob.glob(f'results/reviews/{sub}/*.json'):
        i=os.path.splitext(os.path.basename(f))[0]
        if ids is not None and i not in ids: continue
        try: fin=(json.load(open(f)) or {}).get('final') or {}
        except Exception: continue
        if isinstance(fin,dict) and isinstance(fin.get('Overall',fin.get('Rating')),(int,float)): c+=1
    return c
yf=set(json.load(open('records/reviewer_subset_years_fill.json'))['ids'])
run=subprocess.run(['squeue','-u','user','-h','-o','%T'],capture_output=True,text=True).stdout.split()
print(f"{datetime.datetime.now():%m-%d %H:%M} b3a2026 {n('b3a_2026')}/616  yf_b2h {n('b2h',yf)}/{len(yf)}  yf_b3a {n('b3a',yf)}/{len(yf)}  running {run.count('RUNNING')} pending {run.count('PENDING')}",flush=True)
PY
  sleep 600
done
