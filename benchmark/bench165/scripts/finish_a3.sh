#!/bin/bash
# A3 judge-ablation supervisor (session-independent). Waits for the primary Argument_Graph
# run (AG165_DONE), then submits E1 judge cells and E2 PMI cells, then aggregates.
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
SUP=$B/results/supervisor
for i in $(seq 1 288); do [ -f $B/results/AG165_DONE ] && break; sleep 300; done
[ -f $B/results/AG165_DONE ] || { echo A3_TIMEOUT_WAITING_AG165 >> $SUP/a3_status.log; exit 1; }
if [ ! -f $SUP/a3_submitted ]; then
  touch $SUP/a3_submitted
  sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:2 --cpus-per-task=12 --mem=120G --time=0-8:00:00 \
    -J ag_j_mistral -o $SUP/ag_judge_mistral24b.log \
    --wrap "bash $B/scripts/run165_ag_judge.sh mistral24b mistralai/Mistral-Small-24B-Instruct-2501 mistral 2 8766" >> $SUP/a3_status.log 2>&1
  sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:2 --cpus-per-task=12 --mem=120G --time=0-8:00:00 \
    -J ag_j_gemma -o $SUP/ag_judge_gemma27b.log \
    --wrap "bash $B/scripts/run165_ag_judge.sh gemma27b google/gemma-2-27b-it gemma 2 8767 8192" >> $SUP/a3_status.log 2>&1
  sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:1 --cpus-per-task=8 --mem=90G --time=0-8:00:00 \
    -J ag_j_qwen14 -o $SUP/ag_judge_qwen14b.log \
    --wrap "bash $B/scripts/run165_ag_judge.sh qwen14b Qwen/Qwen2.5-14B-Instruct qwen14 1 8768" >> $SUP/a3_status.log 2>&1
  for cell in "mistral7b mistralai/Mistral-7B-Instruct-v0.1" "gptj6b EleutherAI/gpt-j-6b" \
              "qwen1.5b Qwen/Qwen2.5-1.5B-Instruct" "qwen14b Qwen/Qwen2.5-14B-Instruct"; do
    set -- $cell
    sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=0-4:00:00 \
      -J ag_pmi_$1 -o $SUP/ag_pmi_$1.log \
      --wrap "bash $B/scripts/run165_ag_pmi.sh $1 $2" >> $SUP/a3_status.log 2>&1
  done
fi
need="AG_JUDGE_mistral24b_DONE AG_JUDGE_gemma27b_DONE AG_JUDGE_qwen14b_DONE AG_PMI_mistral7b_DONE AG_PMI_gptj6b_DONE AG_PMI_qwen1.5b_DONE AG_PMI_qwen14b_DONE"
for i in $(seq 1 288); do
  miss=0; for m in $need; do [ -f $B/results/$m ] || miss=$((miss+1)); done
  echo "$(date +%m%d-%H:%M) a3 missing=$miss" >> $SUP/a3_status.log
  [ "$miss" -eq 0 ] && break
  # if nothing of ours is running or queued and cells are missing, stop looping (a later session will look)
  if [ "$(squeue -u $USER -h -o %j | grep -cE '^ag_(j|pmi)_')" -eq 0 ] && [ "$i" -gt 3 ]; then break; fi
  sleep 300
done
python3 $B/scripts/ag_ablation_summary.py > $SUP/a3_summary.log 2>&1
touch $B/results/A3_DONE
