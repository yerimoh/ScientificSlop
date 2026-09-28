#!/bin/bash
# 0916 open-model pilot, post-processing. Waits until no run_arm shard of a1_base.qwen32b is alive, then:
# deterministic 4 items (measure_rounds, TAG), Argument graph caches (labels on the editor server = same Qwen2.5-32B,
# PMI on one q-mid GPU), AG measurement, aggregate (effects.qwen32b.*), results doc. Log: logs/finish_open_pilot.log
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); D6=$(dirname "$ROOT"); ARM=${1:-a1_base}; LLM=${SCISLOP_ROOT}/artifact-ai2science/_llm
LOG=$ROOT/logs/finish_open_pilot.$ARM.log; exec >> $LOG 2>&1
export EOR_BACKEND=vllm EOR_MODEL=Qwen2.5-32B-Instruct EOR_TAG=.qwen32b
while ps -eo args | grep -v grep | grep -q "run_arm.py --arm $ARM --rounds 1 --shard"; do echo "[$(date '+%m-%d %H:%M')] shards running: $(ps -eo args | grep -v grep | grep -c "run_arm.py --arm $ARM --rounds 1 --shard")"; sleep 300; done
echo "[$(date '+%m-%d %H:%M')] shards done"; python3 $HERE/status.py | grep $ARM
cd $HERE; python3 measure_rounds.py --arm $ARM --round 1
cd $D6; ARM=$ARM python3 - <<'PY'
import json, sys
sys.path.insert(0, 'Effects_of_revision/code')
import os; os.environ['EOR_TAG'] = '.qwen32b'; ARM = os.environ['ARM']
from common import ai_codes, round_dir, ai_record, load_progress
out = []
prog = load_progress(ARM)
for c in ai_codes():
    r = prog.get(c, {}).get(1)
    if r and r['exec'] in ('ok', 'nothing_to_fix') and round_dir(ARM, c, 1).exists():
        rec = ai_record(c, round_dir(ARM, c, 1)); rec['id'] = f'{ARM}.qwen32b/R1/{c}'; out.append(rec)
json.dump(out, open(f'Effects_of_revision/results/ag/papers_trees_qwen32b_{ARM}.json', 'w')); print('trees', len(out))
PY
LLM_ENDPOINT=$(cat $LLM/llm_endpoint_qwen_edit.txt) python3 slop/Argument/Argument_Graph/code/prefetch_labels.py --papers Effects_of_revision/results/ag/papers_trees_qwen32b_$ARM.json --workers 32 --log Effects_of_revision/logs/ag_prefetch_qwen32b.$ARM.log
J=$(sbatch --parsable --partition=vram48 --qos=${SLURM_QOS} -N1 -n1 -c4 --gres=gpu:1 --time=03:00:00 --job-name=ag_pmi_q32 -o Effects_of_revision/logs/ag_pmi_qwen32b.$ARM.log --wrap="cd $D6 && python3 slop/Argument/Argument_Graph/code/pmi_worker.py --papers Effects_of_revision/results/ag/papers_trees_qwen32b_$ARM.json --shard 0 --nshards 1")
echo "pmi job $J"; while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 120; done
until python3 slop/Argument/Argument_Graph/code/cache_status.py Effects_of_revision/results/ag/papers_trees_qwen32b_$ARM.json; do echo "[$(date '+%m-%d %H:%M')] AG caches incomplete, retrying labels"; LLM_ENDPOINT=$(cat $LLM/llm_endpoint_qwen_edit.txt) python3 slop/Argument/Argument_Graph/code/prefetch_labels.py --papers Effects_of_revision/results/ag/papers_trees_qwen32b_$ARM.json --workers 32 --log Effects_of_revision/logs/ag_prefetch_qwen32b.$ARM.log; sleep 60; done
cd $HERE; python3 measure_rounds.py --arm $ARM --round 1 --items argument_graph
python3 aggregate.py > $ROOT/logs/aggregate_qwen32b.log 2>&1
EOR_ITEMS=argument_graph EOR_OUT=_ag python3 aggregate.py > $ROOT/logs/aggregate_qwen32b_ag.log 2>&1
echo "[$(date '+%m-%d %H:%M')] OPEN_PILOT_DONE"
