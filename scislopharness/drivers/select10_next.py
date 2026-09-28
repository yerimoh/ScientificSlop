"""Next ten papers. Drawn from the same 60-paper subset as the first ten, with the same quartile stratification and the same randomness, excluding papers already run.
The seed is fixed and the list written down first so the pick cannot depend on results."""
import os
import json, random
from pathlib import Path
SLOP = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6")
API = Path(__file__).resolve().parents[1]
sub = json.load(open(SLOP / 'Effects_of_revision/results/subset60.json'))
codes, r0, cuts = sub['codes'], sub['r0_score'], sub['quartile_cuts']
done = set(json.load(open(API / 'papers10.json'))['codes'])


def q(c):
    return sum(r0[c] > x for x in cuts)


rng = random.Random(920)
by_q = {}
for c in codes:
    if c not in done:
        by_q.setdefault(q(c), []).append(c)
for k in sorted(by_q):
    by_q[k].sort(); rng.shuffle(by_q[k])
chosen, i = [], 0
while len(chosen) < 10 and any(by_q.values()):
    b = sorted(by_q)[i % len(by_q)]
    if by_q[b]:
        chosen.append(by_q[b].pop())
    i += 1
chosen = sorted(chosen)
rec = {'seed': 920, 'n': len(chosen), 'codes': chosen, 'excludes': sorted(done),
       'quartile': {c: q(c) for c in chosen}, 'r0_mean_slop': {c: r0[c] for c in chosen}}
(API / 'papers10_next.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
