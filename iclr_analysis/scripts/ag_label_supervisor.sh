#!/bin/bash
# 0916. Keep the Argument_Graph label prefetch alive until the ICLR label cache is complete: the prefix-caching
# vLLM servers (0.6.3) have twice shut themselves down after a 500, so (1) resubmit the q-mid server job when it is
# gone, (2) restart the forward and reverse prefetchers whenever they are not running and a server answers,
# (3) stop when cache_status says complete. Log: logs/ag_label_supervisor.log
D6=${SCISLOP_ROOT}/paper/draft_v6; LLM=${SCISLOP_ROOT}/artifact-ai2science/_llm
cd $D6; LOG=review/logs/ag_label_supervisor.log; exec >> $LOG 2>&1
up() { curl -s --max-time 5 "$(cat $1)/models" 2>/dev/null | grep -q qwen; }
running() { ps -eo args | grep -v grep | grep -q "prefetch_labels.py --papers review/results/ag/papers_iclr.json $1"; }
while true; do
  if python3 slop/Argument/Argument_Graph/code/cache_status.py review/results/ag/papers_iclr.json; then echo "[$(date '+%m-%d %H:%M')] ICLR label cache complete"; break; fi
  if ! squeue -u $USER -h -o "%j" | grep -q "^qwen32_pc$"; then echo "[$(date '+%m-%d %H:%M')] q-mid server job gone, resubmitting"; sbatch $LLM/serve_qwen_pc.sbatch; fi
  EP1=$LLM/llm_endpoint_qwen_pc.txt; EP2=$LLM/llm_endpoint_qwen_pc2.txt
  up $EP1 && E1=$(cat $EP1) || E1=""; up $EP2 && E2=$(cat $EP2) || E2=""
  [ -z "$E1" ] && E1=$E2; [ -z "$E2" ] && E2=$E1
  if [ -n "$E1" ] && ! running "--workers 40"; then echo "[$(date '+%m-%d %H:%M')] start forward prefetch on $E1"; LLM_ENDPOINT=$E1 setsid nohup python3 slop/Argument/Argument_Graph/code/prefetch_labels.py --papers review/results/ag/papers_iclr.json --workers 40 --log review/logs/ag_prefetch_iclr.log > /dev/null 2>&1 & fi
  if [ -n "$E2" ] && ! running "--workers 64 --reverse"; then echo "[$(date '+%m-%d %H:%M')] start reverse prefetch on $E2"; LLM_ENDPOINT=$E2 setsid nohup python3 slop/Argument/Argument_Graph/code/prefetch_labels.py --papers review/results/ag/papers_iclr.json --workers 64 --reverse --log review/logs/ag_prefetch_iclr_rev.log > /dev/null 2>&1 & fi
  sleep 120
done
