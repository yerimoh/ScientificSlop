#!/bin/bash
# supervise.sh ARM NSHARD [ROUNDS] [SUBSET]
# Runs every shard in parallel and keeps re-launching until no paper is left. A shard that
# hits the account's Claude session limit exits 17 without recording a no-op; the supervisor
# then waits out the quota window instead of burning it (0914 incident, see README).
ARM=$1; K=$2; ROUNDS=${3:-3}; SUBSET=${4:-}; ORDER=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
LA="${ARM}${EOR_TAG:-}"
LOG="$ROOT/logs/${LA}.supervise.log"
mkdir -p "$ROOT/logs"
WAIT_LIMIT=900       # quota exhausted: poll every 15 min
WAIT_NORMAL=300      # a shard finished its slice: restart promptly
for pass in $(seq 1 200); do
  pids=(); limited=0
  for s in $(seq 0 $((K-1))); do
    python3 "$HERE/run_arm.py" --arm "$ARM" --rounds "$ROUNDS" --shard "$s" --nshard "$K" ${SUBSET:+--subset "$SUBSET"} ${ORDER:+--order-by-arms "$ORDER"} \
      >> "$ROOT/logs/${LA}.shard${s}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  for s in $(seq 0 $((K-1))); do
    tail -n 5 "$ROOT/logs/${LA}.shard${s}.log" 2>/dev/null | grep -q "STOP: session limit" && limited=1
  done
  left=$(python3 "$HERE/remaining.py" "$ARM" "$ROUNDS" "$SUBSET" 2>/dev/null || echo "?")
  echo "[$(date +%m-%d\ %H:%M)] pass $pass left=$left limited=$limited" >> "$LOG"
  if [ "$left" = "0" ]; then echo "[$(date +%m-%d\ %H:%M)] COMPLETE" >> "$LOG"; exit 0; fi
  if [ "$limited" = "1" ]; then
    W=$(python3 "$HERE/wait_secs.py" "$ARM" 2>/dev/null || echo $WAIT_LIMIT)
    [ "$W" -gt 2700 ] && W=2700            # probe every 45 min: capacity often returns early
    echo "[$(date +%m-%d\ %H:%M)] quota exhausted, sleeping ${W}s" >> "$LOG"
    sleep "$W"
  else sleep $WAIT_NORMAL; fi
done
echo "[$(date +%m-%d\ %H:%M)] gave up after 200 passes, left=$left" >> "$LOG"
