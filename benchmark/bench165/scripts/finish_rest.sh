#!/bin/bash
# Session-independent supervisor for fig_graph vision + Argument_Graph bench165.
# Resubmits each at most once, then produces machine summaries so a later session only writes prose.
ROOT=${SCISLOP_ROOT}
B=$ROOT/paper/draft_v6/scislopbench/bench165
FG=$ROOT/paper/draft_v6/slop/Artifacts/fig_graph
SUP=$B/results/supervisor
for i in $(seq 1 288); do   # up to 24h
  C=$(ls $FG/roi_extract/vis_*.json 2>/dev/null | wc -l)
  FR=$(squeue -u $USER -h -o "%j" | grep -c "^figgraph_vis$")
  AR=$(squeue -u $USER -h -o "%j" | grep -c "^ag165$")
  echo "$(date +%m%d-%H:%M) figcache=$C/96 figjob=$FR agjob=$AR agdone=$([ -f $B/results/AG165_DONE ] && echo 1 || echo 0)" >> $SUP/rest_status.log
  # fig_graph: vision cache complete or job dead
  if [ ! -f $SUP/fig_summary_done ]; then
    if [ "$C" -ge 96 ] || { [ "$FR" -eq 0 ] && [ -f $SUP/fig_resub ]; }; then
      cd $FG/code && python3 measure.py --dry-run > $SUP/fig_dryrun.log 2>&1
      python3 $B/scripts/summarize_pairs.py $FG/results/papers.jsonl \
        $B/results/slop/fig_graph_summary.json slop_score,undescribed_box_rate >> $SUP/fig_dryrun.log 2>&1 \
        && touch $SUP/fig_summary_done
    elif [ "$FR" -eq 0 ] && [ ! -f $SUP/fig_resub ]; then
      touch $SUP/fig_resub
      cd $FG && sbatch -J figgraph_vis roi_extract/run_vision.sbatch >> $SUP/rest_status.log 2>&1
    fi
  fi
  # Argument_Graph
  if [ ! -f $SUP/ag_summary_done ]; then
    if [ -f $B/results/AG165_DONE ]; then
      python3 $B/scripts/summarize_pairs.py $B/results/slop/argument_graph/papers.jsonl \
        $B/results/slop/argument_graph_summary.json slop_score,ABU >> $SUP/rest_status.log 2>&1 \
        && touch $SUP/ag_summary_done
    elif [ "$AR" -eq 0 ] && [ ! -f $SUP/ag_resub ]; then
      touch $SUP/ag_resub
      sbatch --qos=${SLURM_QOS} -p vram48 --gres=gpu:4 --cpus-per-task=16 --mem=180G --time=0-16:00:00 \
        -J ag165 -o $SUP/ag165_r2.log --wrap "bash $B/scripts/run165_ag.sh" >> $SUP/rest_status.log 2>&1
    fi
  fi
  [ -f $SUP/fig_summary_done ] && [ -f $SUP/ag_summary_done ] && break
  sleep 300
done
touch $B/results/REST_DONE
