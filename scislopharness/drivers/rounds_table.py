"""Per-round table. For every manuscript and item: R0, R1, R2, R3, and the paired human paper.
Human scores come from the benchmark's own measurement of the matched human manuscript."""
import os
import json, sys
from pathlib import Path
API = Path(__file__).resolve().parent.parent
BENCH = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/bench165")
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H

ITEMS = H.ITEMS
items165 = json.load(open(BENCH / 'items165.json'))['items']
hu_of = {i['pair']: i['arxiv'] for i in items165 if i['label'] == 0}

human = {}
for it in ITEMS:
    rows = {(r['corpus'], r['id']): r for r in
            (json.loads(l) for l in open(BENCH / 'results/slop' / it / 'papers.jsonl') if l.strip())}
    for pair, arx in hu_of.items():
        r = rows.get(('HU', arx))
        if r:
            human.setdefault(pair, {})[it] = r.get('slop_score')

codes = json.load(open(API / 'papers10.json'))['codes']
data = {}
for c in codes:
    f = API / 'runs/A-loc' / c / 'trajectory.json'
    if not f.exists():
        continue
    t = json.load(open(f))
    per = {'R0': t['r0']['scores']}
    for r in t['rounds']:
        per[f"R{r['round']}"] = r['scores_after']
    data[c] = per


def fmt(v):
    return '   .   ' if v is None else f'{v:.3f}'


out = []
for it in ITEMS:
    out.append(f'\n### {it}\n')
    out.append('| Paper | R0 | R1 | R2 | R3 | Human pair |')
    out.append('|---|---|---|---|---|---|')
    cols = {k: [] for k in ('R0', 'R1', 'R2', 'R3')}
    hv = []
    for c in codes:
        if c not in data:
            continue
        row = [fmt(data[c].get(k, {}).get(it)) for k in ('R0', 'R1', 'R2', 'R3')]
        h = (human.get(c) or {}).get(it)
        for k in cols:
            v = data[c].get(k, {}).get(it)
            if v is not None:
                cols[k].append(v)
        if h is not None:
            hv.append(h)
        out.append(f'| {c} | ' + ' | '.join(row) + f' | {fmt(h)} |')
    mean = lambda v: fmt(sum(v) / len(v)) if v else '   .   '
    out.append('| **Mean** | ' + ' | '.join(f'**{mean(cols[k])}**' for k in ('R0', 'R1', 'R2', 'R3'))
               + f' | **{mean(hv)}** |')
TEXT = '\n'.join(out)


def render():
    return '## Scores per round (R0, R1, R2, R3, human pair)\n' + TEXT


(API / 'rounds_table.md').write_text('# Scores per round (R0, R1, R2, R3, human pair)\n' + TEXT + '\n')
if __name__ == '__main__':
    print(TEXT)
