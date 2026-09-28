"""Reclassify rounds that were recorded as a no-op but whose editor never ran because the
account hit its Claude session limit. Appends a corrected record (blocked, retryable) with a
current timestamp; load_progress takes the newest record per (code, round), so a later real
run overrides the correction. Nothing is deleted.

  python3 reclassify.py [--apply]
"""
import argparse, glob, json, sys, time, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, load_progress, progress_path, runs_dir  # noqa: E402
from run_arm import ARMS  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--apply', action='store_true')
ap.add_argument('--cascade', action='store_true', help='also invalidate later rounds built on a reclassified round and delete their trees')
a = ap.parse_args()
MARKS = ('session limit', 'disabled Claude subscription access', 'Use an Anthropic API key', 'overloaded', 'Too Many Requests')
total = {}
for arm in ARMS:
    prog = load_progress(arm)
    fix = []
    for code, rounds in prog.items():
        for n, r in rounds.items():
            if r.get('exec') != 'failed':
                continue
            logs = glob.glob(str(runs_dir(arm) / code / 'logs' / f'{code}_{arm}_R{n}_a*_exec.txt'))
            if any(any(m in open(p, errors='ignore').read() for m in MARKS) for p in logs):
                fix.append((code, n))
                if a.cascade:
                    for n2 in range(n + 1, 4):
                        if n2 in rounds:
                            fix.append((code, n2))
    total[arm] = len(fix)
    if a.apply and fix:
        import shutil
        with open(progress_path(arm), 'a') as f:
            for code, n in sorted(set(fix)):
                if a.cascade:
                    shutil.rmtree(runs_dir(arm) / code / f'R{n}', ignore_errors=True)
                f.write(json.dumps({'code': code, 'round': n, 'arm': arm, 'exec': 'blocked',
                                    'reason': 'cli_unavailable_reclassified' + ('_cascade' if a.cascade else ''),
                                    'ts': time.strftime('%Y-%m-%d %H:%M:%S')}) + '\n')
print(json.dumps(total, indent=1))
print('applied' if a.apply else 'dry run; pass --apply')
