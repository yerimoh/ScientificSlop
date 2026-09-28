"""Progress summary per arm / round / exec state.  python3 status.py [arm ...]"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_progress, ai_codes, EOR
from run_arm import ARMS

arms = sys.argv[1:] or ARMS
N = len(ai_codes())
for arm in arms:
    prog = load_progress(arm)
    if not prog:
        continue
    cnt = collections.Counter()
    for code, rounds in prog.items():
        for n, r in rounds.items():
            cnt[(n, r['exec'])] += 1
    line = ' | '.join(f"R{n}:{ex}={c}" for (n, ex), c in sorted(cnt.items()))
    print(f'{arm:18s} papers={len(prog):3d}/{N}  {line}')
