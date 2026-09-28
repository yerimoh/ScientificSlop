"""How many papers still need work in an arm.  python3 remaining.py ARM ROUNDS [SUBSET]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_progress, ai_codes
arm, rounds = sys.argv[1], int(sys.argv[2])
subset = sys.argv[3] if len(sys.argv) > 3 else ''
codes = ai_codes()
if subset:
    codes = [c for c in codes if c in set(json.load(open(subset))['codes'])]
prog = load_progress(arm)
left = 0
for c in codes:
    for n in range(1, rounds + 1):
        r = prog.get(c, {}).get(n)
        if not r or r['exec'] not in ('ok', 'failed', 'nothing_to_fix'):
            left += 1
            break
print(left)
