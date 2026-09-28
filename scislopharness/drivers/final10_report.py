#!/usr/bin/env python3
"""final_10 results. Per item, R0 R1 R2 R3 for the ten papers, each paper's human pair, and the mean."""
import os
import json, sys
from pathlib import Path
API = Path(__file__).resolve().parent.parent
OUT = API / 'final_10'
DRAFT = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6")
BENCH = DRAFT / 'scislopbench/bench165'
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(API / 'code'))
import harness as H

ITEMS = list(H.ITEMS) + ['argument_graph', 'fig_exposition']
SRC = {'argument_graph': BENCH / 'results/slop/argument_graph/papers.jsonl',
       'fig_exposition': DRAFT / 'slop/Artifacts/fig_exposition/results/papers.jsonl'}


def human_all(codes):
    items = json.load(open(BENCH / 'items165.json'))['items']
    arx = {i['pair']: i['arxiv'] for i in items if i['label'] == 0}
    out = {c: {} for c in codes}
    for it in ITEMS:
        p = SRC.get(it, BENCH / 'results/slop' / it / 'papers.jsonl')
        if not p.exists():
            continue
        rows = {r['id']: r for r in (json.loads(l) for l in open(p) if l.strip()) if r.get('corpus') == 'HU'}
        for c in codes:
            r = rows.get(arx.get(c))
            if r:
                out[c][it] = r.get('slop_score')
    return out


def fmt(v):
    return '  .  ' if v is None else f'{v:.3f}'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    codes = json.load(open(API / 'papers10.json'))['codes']
    recs = {c: json.loads((OUT / c / 'result.json').read_text()) for c in codes
            if (OUT / c / 'result.json').exists()}
    hu = human_all(codes)
    L = ['# final_10 — ten papers, up to round 3, six items', '',
         'One Haiku editor call; the Sonnet gate runs in parallel on chunks of three changes. The human pair is the benchmark-paired human '
         'paper measured with the same measurers; the harness never sees these values.', '']
    for it in ITEMS:
        L += [f'### {it}', '', '| Paper | R0 | R1 | R2 | R3 | Human pair |', '|---|---|---|---|---|---|']
        cols = {n: [] for n in range(4)}
        hus = []
        for c in codes:
            r = recs.get(c)
            if not r:
                L.append(f'| {c} | (none) | | | | {fmt(hu[c].get(it))} |'); continue
            vs = [(r['per_round'].get(f'R{n}') or {}).get(it) for n in range(4)]
            for n in range(4):
                cols[n].append(vs[n])
            hus.append(hu[c].get(it))
            L.append(f'| {c} | ' + ' | '.join(fmt(v) for v in vs) + f' | {fmt(hu[c].get(it))} |')
        L.append('| **Mean** | ' + ' | '.join(f'**{fmt(mean(cols[n]))}**' for n in range(4)) +
                 f' | **{fmt(mean(hus))}** |')
        L.append('')
    L += ['### Time and cost per paper', '', '| Paper | Total | Editor $ | Gate $ | Total $ | Editor s per round | Gate s per round |',
          '|---|---|---|---|---|---|---|']
    for c in codes:
        r = recs.get(c)
        if not r:
            continue
        rs = r.get('round_seconds') or []
        L.append(f"| {c} | {r['wall_s']:.0f}s | {r['editor_usd']:.2f} | {r['gate_usd']:.2f} | "
                 f"{r['editor_usd'] + r['gate_usd']:.2f} | " +
                 ' / '.join(f'{x[1]:.0f}' for x in rs) + ' | ' +
                 ' / '.join(('.' if x[2] is None else f'{x[2]:.0f}') for x in rs) + ' |')
    tot = sum(r['editor_usd'] + r['gate_usd'] for r in recs.values())
    L.append(f"| **Total** | | | | **{tot:.2f}** | | |")
    L += ['', '### Gate and guards', '', '| Paper | Kept/reverted/retired per round | Hard violations | Soft | Diagram edits |', '|---|---|---|---|---|']
    for c in codes:
        r = recs.get(c)
        if not r:
            continue
        L.append(f'| {c} | ' + ' / '.join(f'{a}:{b}k{c2}r{d}e' for a, b, c2, d in r['gate_counts']) + ' | ' +
                 ' / '.join(f'{a}:{len(b)}' for a, b, _ in r['guards']) + ' | ' +
                 ' / '.join(f'{a}:{",".join(s) if s else "-"}' for a, _, s in r['guards']) + ' | ' +
                 ' / '.join(f'{a}:{s}' for a, s, ch in r['figure_edits']) + ' |')
    (OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
