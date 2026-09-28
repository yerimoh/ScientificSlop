"""Pick the 10 manuscripts for this run. Stratified over the R0 slop quartiles of the 60-paper subset,
fixed seed, chosen before any result was seen. The three pilot manuscripts are forced in so that this
run has a reference point against the pilot."""
import os
import json, random, sys
from pathlib import Path
SLOP = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6")
OUT = Path(__file__).resolve().parents[1] / 'papers10.json'
PILOT = ['FA0002', 'FA0005', 'FA0198']
sub = json.load(open(SLOP / 'Effects_of_revision/results/subset60.json'))
codes, r0 = sub['codes'], sub['r0_score']
cuts = sub['quartile_cuts']


def q(c):
    v = r0[c]
    return sum(v > x for x in cuts)


rng = random.Random(919)
chosen = list(PILOT)
by_q = {}
for c in codes:
    if c not in chosen:
        by_q.setdefault(q(c), []).append(c)
for k in sorted(by_q):
    by_q[k].sort(); rng.shuffle(by_q[k])
i = 0
while len(chosen) < 10:
    bucket = sorted(by_q)[i % len(by_q)]
    if by_q[bucket]:
        chosen.append(by_q[bucket].pop())
    i += 1
chosen = PILOT + sorted(c for c in chosen if c not in PILOT)
rec = {'seed': 919, 'n': len(chosen), 'codes': chosen, 'pilot_reference': PILOT,
       'quartile': {c: q(c) for c in chosen}, 'r0_mean_slop': {c: r0[c] for c in chosen}}
OUT.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
