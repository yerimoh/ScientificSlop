#!/bin/bash
# Session-independent supervisor for the faithful detector rerun.
B=${SCISLOP_ROOT}/paper/draft_v6/scislopbench/bench165
SUP=$B/results/supervisor; mkdir -p $SUP
CNT() { cat $B/results/$1 2>/dev/null | wc -l; }
for i in $(seq 1 400); do   # up to ~33h
  BN=$(CNT binoculars_faithful.jsonl)
  D0=$(CNT detectgpt_faithful.s0.jsonl); D1=$(CNT detectgpt_faithful.s1.jsonl); D2=$(CNT detectgpt_faithful.s2.jsonl)
  DG=$((D0+D1+D2))
  RUN=$(squeue -u $USER -h -o "%j" | grep -c faith)
  echo "$(date +%m%d-%H:%M) bino=$BN dgpt=$DG($D0/$D1/$D2) jobs=$RUN" >> $SUP/faithful_status.log
  # done when nothing is running and counts stopped short of nothing (texts=286; short docs may skip)
  if [ "$RUN" -eq 0 ]; then
    if [ "$BN" -ge 280 ] && [ "$DG" -ge 280 ]; then break; fi
    # silent death: resubmit each missing piece at most once
    if [ "$BN" -lt 280 ] && [ ! -f $SUP/bino_faith_resub ]; then touch $SUP/bino_faith_resub
      sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:1 --cpus-per-task=8 --mem=64G -J bino_faith \
        -o $SUP/bino_faith_r2.log --wrap "bash $B/scripts/run165_bino_faithful.sh" >> $SUP/faithful_status.log 2>&1; fi
    for s in 0 1 2; do
      C=$(CNT detectgpt_faithful.s$s.jsonl)
      if [ "$C" -lt 90 ] && [ ! -f $SUP/dgpt_faith_resub_$s ]; then touch $SUP/dgpt_faith_resub_$s
        sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:1 --cpus-per-task=8 --mem=64G -J dgpt_faith$s \
          -o $SUP/dgpt_faith${s}_r2.log --wrap "bash $B/scripts/run165_dgpt_faithful.sh $s 3" >> $SUP/faithful_status.log 2>&1; fi
    done
  fi
  sleep 300
done
python3 $B/scripts/analyze_faithful.py > $SUP/faithful_final.log 2>&1
touch $B/results/FAITHFUL_DONE
