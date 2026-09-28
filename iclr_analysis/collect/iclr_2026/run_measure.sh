#!/bin/sh
# Drive mold_vs_pangram.py to completion.
#
# The measurement process gets killed part-way through on this host (no traceback -- a handful
# of very large PDFs, most likely the OOM killer). The script itself is resumable and records
# attempts, so the fix is simply to keep restarting it: each pass measures more papers, and any
# paper that has taken the process down twice is abandoned. Ten passes is far more than enough.
cd "$(dirname "$0")" || exit 1
for i in $(seq 1 10); do
    echo "=== pass $i ==="
    python3 -u mold_vs_pangram.py && break
    echo "=== pass $i died, resuming ==="
done
