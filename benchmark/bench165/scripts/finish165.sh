#!/bin/bash
# Session-independent supervisor: waits for b3a / b2h / citation, resubmits each at most once
# on silent death, then runs the final aggregation. Runs as a CPU SLURM job.
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
SUP=$B/results/supervisor; mkdir -p $SUP
PYCNT() { python3 -c "
import json,glob,sys
ok=0
for f in glob.glob('$B/results/reviews/$1/*.json'):
    try:
        r=json.load(open(f)); fin=r.get('final') or {}
        if isinstance(fin,dict) and (fin.get('Overall') is not None or (fin.get('n_items') or 0)>0): ok+=1
    except Exception: pass
print(ok)"; }
for i in $(seq 1 200); do   # up to ~33h
  B3A=$(PYCNT b3a); B2H=$(PYCNT b2h)
  CIT=0; [ -s "$B/results/slop/citation/summary.json" ] && CIT=1
  echo "$(date +%H:%M) b3a=$B3A b2h=$B2H cit=$CIT" >> $SUP/status.log
  # resubmit qwen job if b3a stalled with no 4-GPU job running
  if [ "$B3A" -lt 143 ] && ! squeue -u $USER -h -o "%b %T" | grep -q "gpu:4 RUNNING" && [ ! -f $SUP/qwen_resubmitted ]; then
    touch $SUP/qwen_resubmitted
    sbatch --parsable --qos=${SLURM_QOS} -p vram48 --gres=gpu:4 --cpus-per-task=16 --mem=180G \
      --wrap "bash $B/scripts/run165_qwen_high.sh" >> $SUP/status.log 2>&1
  fi
  # b2h fallback already queued as a q-high job; if BOTH gone and incomplete, resubmit once
  if [ "$B2H" -lt 143 ] && [ "$(squeue -u $USER -h | wc -l)" -eq 0 ] && [ ! -f $SUP/b2h_resubmitted ]; then
    touch $SUP/b2h_resubmitted
    sbatch --parsable --qos=${SLURM_QOS} -p vram48 --gres=gpu:1 --cpus-per-task=8 --mem=90G \
      --wrap "bash $B/scripts/run165_b2h_high.sh" >> $SUP/status.log 2>&1
  fi
  # citation fallback: server gone, summary missing, no citation process anywhere
  if [ "$CIT" -eq 0 ] && ! curl -s http://vermeer:8765/v1/models >/dev/null 2>&1 \
     && [ "$B3A" -ge 143 ] && [ ! -f $SUP/cit_resubmitted ]; then
    touch $SUP/cit_resubmitted
    sbatch --parsable --qos=${SLURM_QOS} -p vram48 --gres=gpu:4 --cpus-per-task=16 --mem=180G \
      --wrap "bash $B/scripts/run165_citation_high.sh" >> $SUP/status.log 2>&1
  fi
  # cancel the redundant pending b2h-high once mid finishes
  if [ "$B2H" -ge 143 ]; then scancel 1400190 2>/dev/null; fi
  if [ "$B3A" -ge 143 ] && [ "$B2H" -ge 143 ] && [ "$CIT" -eq 1 ]; then
    python3 $B/scripts/aggregate165.py > $B/results/final_aggregate.log 2>&1
    date > $B/results/ALL_DONE_165
    echo FINISHED >> $SUP/status.log
    exit 0
  fi
  sleep 600
done
echo TIMEOUT >> $SUP/status.log
