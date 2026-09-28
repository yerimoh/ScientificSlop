#!/bin/bash
# Measure every finished round of every arm (CPU only, idempotent, safe to re-run).
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
mkdir -p "$ROOT/logs"
for a in a1_base a2_code a3_review a4_slop; do
  python3 "$HERE/measure_rounds.py" --arm $a --round 1 --round 2 --round 3 >> "$ROOT/logs/measure.log" 2>&1
done
if [ -z "$EOR_TAG" ]; then for it in macro_redund xsec_ref citation evidence_gap; do
  python3 "$HERE/measure_rounds.py" --arm a4s_$it --round 1 >> "$ROOT/logs/measure.log" 2>&1
done; fi
python3 "$HERE/aggregate.py" > "$ROOT/logs/aggregate.log" 2>&1
echo "MEASURE_ALL_DONE $(date)" >> "$ROOT/logs/measure.log"
