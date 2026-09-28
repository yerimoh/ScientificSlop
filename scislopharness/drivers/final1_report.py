#!/usr/bin/env python3
"""final_1 report. One manuscript, three configurations, three rounds, six items.

Per item the table reads R0 R1 R2 R3 for each configuration and, in the last column, the same
measurer's score on the human paper this manuscript is paired with in the benchmark. The human
column is a reference the harness never sees; nothing in the loop targets it.
"""
import os
import json, sys
from pathlib import Path

API = Path(__file__).resolve().parent.parent
OUT = API / 'final_1'
DRAFT = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6")
BENCH = DRAFT / 'scislopbench/bench165'
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(API / 'code'))
import harness as H

ITEMS = list(H.ITEMS) + ['argument_graph', 'fig_exposition']
SRC = {'argument_graph': BENCH / 'results/slop/argument_graph/papers.jsonl',
       'fig_exposition': DRAFT / 'slop/Artifacts/fig_exposition/results/papers.jsonl'}
CONFIGS = ['haiku_haikugate', 'haiku_sonnetgate']
LABEL = {'haiku_haikugate': 'Haiku editor / Haiku gate',
         'haiku_sonnetgate': 'Haiku editor / Sonnet gate',
}


def human_scores(code: str) -> dict:
    items = json.load(open(BENCH / 'items165.json'))['items']
    arx = next((i['arxiv'] for i in items if i['label'] == 0 and i['pair'] == code), None)
    out = {}
    for it in ITEMS:
        p = SRC.get(it, BENCH / 'results/slop' / it / 'papers.jsonl')
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('corpus') == 'HU' and r.get('id') == arx:
                out[it] = r.get('slop_score'); break
    return arx, out


def fmt(v):
    return '  .  ' if v is None else f'{v:.3f}'



# Time and score move together with how far the serial stretch inside a round is split. Three settings side by side.
VARIANTS = [('_sequential', 'sequential (1 editor, 1 gate)'),
            ('_parallel_both', 'editor per file + gate chunks'),
            ('', 'gate chunks only (adopted)')]


def variants_section(code: str, hu: dict) -> list:
    rows = {}
    for d, name in VARIANTS:
        for c in CONFIGS:
            f = (OUT / d / c / 'result.json') if d else (OUT / c / 'result.json')
            if f.exists():
                rows[(d, c)] = json.loads(f.read_text())
    L = ['### Time and final score per harness setting', '',
         'Round cap, skill and measurers are the same for all three. The only difference is how many ways the editor '
         'and gate calls inside one round are split.', '',
         '| Setting | Config | Total s | Total $ | ' + ' | '.join(ITEMS) + ' |',
         '|---|---|---|---|' + '---|' * len(ITEMS)]
    for d, name in VARIANTS:
        for c in CONFIGS:
            r = rows.get((d, c))
            if not r:
                continue
            last = (r['per_round'].get('R3') or r['per_round'].get('R2') or {})
            L.append(f"| {name} | {LABEL[c].replace(' editor', '').replace(' gate', '')} | {r['wall_s']:.0f} | "
                     f"{r['editor_usd'] + r['gate_usd']:.2f} | " + ' | '.join(fmt(last.get(i)) for i in ITEMS) + ' |')
    L.append('| Human pair | | | | ' + ' | '.join(fmt(hu.get(i)) for i in ITEMS) + ' |')
    return L + ['']


def main(code='FA0002'):
    recs = {}
    for c in CONFIGS:
        f = OUT / c / 'result.json'
        if f.exists():
            recs[c] = json.loads(f.read_text())
    arx, hu = human_scores(code)
    L = [f'# final_1 — {code}, three configs, up to round 3', '',
         f'Human pair is {arx}: the benchmark-paired human paper measured with the same measurers; the harness never sees these values.',
         'Round cap is 3. Configs differ only in the editor and gate models; skill, definitions, guards and measurers are the same.', '']
    for it in ITEMS:
        L += [f'### {it}', '', '| Config | R0 | R1 | R2 | R3 | Human pair |', '|---|---|---|---|---|---|']
        for c in CONFIGS:
            r = recs.get(c)
            if not r:
                L.append(f'| {LABEL[c]} | (incomplete) | | | | {fmt(hu.get(it))} |'); continue
            pr = r['per_round']
            L.append('| ' + LABEL[c] + ' | ' + ' | '.join(
                fmt((pr.get(f'R{n}') or {}).get(it)) for n in range(4)) + f' | {fmt(hu.get(it))} |')
        L.append('')
    L += ['### Time and cost per config', '',
          '| Config | Total | Editor $ | Gate $ | Total $ | Editor s per round | Gate s per round | Measure s |',
          '|---|---|---|---|---|---|---|---|']
    for c in CONFIGS:
        r = recs.get(c)
        if not r:
            L.append(f'| {LABEL[c]} | (incomplete) | | | | | | |'); continue
        rs = r.get('round_seconds') or []
        L.append(f"| {LABEL[c]} | {r['wall_s']:.0f}s | {r['editor_usd']:.3f} | {r['gate_usd']:.3f} | "
                 f"{r['editor_usd'] + r['gate_usd']:.3f} | " +
                 ' / '.join(f'{x[1]:.0f}' for x in rs) + ' | ' +
                 ' / '.join(('.' if x[2] is None else f'{x[2]:.0f}') for x in rs) + ' | ' +
                 ' / '.join(('.' if len(x) < 4 or x[3] is None else f'{x[3]:.0f}') for x in rs) + ' |')
    L += ['', '### Gate and guards', '', '| Config | Kept/reverted per round | Hard violations | Diagram edits | Saved diagrams |', '|---|---|---|---|---|']
    for c in CONFIGS:
        r = recs.get(c)
        if not r:
            L.append(f'| {LABEL[c]} | (incomplete) | | | |'); continue
        L.append(f"| {LABEL[c]} | " + ' / '.join(f'{a}:{b}k {c2}r' for a, b, c2 in r['gate_counts']) + ' | ' +
                 ' / '.join(f'{a}:{len(b)}' for a, b in r['guards']) + ' | ' +
                 ' / '.join(f'{a}:{s}{"" if ch else " (unchanged)"}' for a, s, ch in r['figure_edits']) + ' | ' +
                 ' '.join(r['figures_saved']) + ' |')
    L += variants_section(code, hu)
    (OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main(*sys.argv[1:])
