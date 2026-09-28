"""Write the 6-item results to RESULTS_6ITEMS.md. Per item, R0..R5 and the human pair."""
import os
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
BENCH = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/bench165")
FIGEXP = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/fig_exposition/results")
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
import harness as H          # noqa: E402
import extra_items as X      # noqa: E402

ALL = H.ITEMS + X.EXTRA
SRC = {**{it: BENCH / 'results/slop' / it / 'papers.jsonl' for it in H.ITEMS},
       'argument_graph': BENCH / 'results/slop/argument_graph/papers.jsonl',
       'fig_exposition': FIGEXP / 'papers.jsonl'}


def human_scores():
    items = json.load(open(BENCH / 'items165.json'))['items']
    hu = {i['pair']: i['arxiv'] for i in items if i['label'] == 0}
    out = {}
    for it, p in SRC.items():
        if not p.exists():
            continue
        rows = {(r.get('corpus'), r.get('id')): r for r in
                (json.loads(l) for l in open(p) if l.strip())}
        for pair, arx in hu.items():
            r = rows.get(('HU', arx))
            if r:
                out.setdefault(pair, {})[it] = r.get('slop_score')
    return out


def main():
    rows = [json.loads(l) for l in open(API / 'results_6items.jsonl') if l.strip()]
    seen, uniq = set(), []
    for r in reversed(rows):
        if r['code'] not in seen:
            seen.add(r['code']); uniq.append(r)
    uniq.reverse()
    hu = human_scores()
    ROUND_LIMIT = 3      # design value. The cap set by LAYERS_AND_STEPS_0919.md; reporting stops here too.
    nmax = min(ROUND_LIMIT, max((r['rounds'] for r in uniq), default=0))
    cols = ['R0'] + [f'R{i}' for i in range(1, nmax + 1)]
    md = [f'# 6-item run results ({len(uniq)} papers, updated {time.strftime("%Y-%m-%d %H:%M")})\n',
          f'Editor {uniq[0]["editor_model"] if uniq else "-"}, reviewer {uniq[0]["reviewer_model"] if uniq else "-"}, '
          f'skill {uniq[0]["skill"] if uniq else "-"}, round cap {nmax} (design value).',
          'Argument graph and Figure exposition were added to the 4 deterministic items. Figure exposition is an image item, so its value',
          'is constant across rounds and written as NA if the figure is dropped from the manuscript.\n']
    fmt = lambda v: '   .   ' if v is None else f'{v:.3f}'
    for it in ALL:
        md += [f'\n## {it}\n', '| Paper | ' + ' | '.join(cols) + ' | Human pair |',
               '|---' * (len(cols) + 2) + '|']
        for r in uniq:
            vals = [fmt(r['per_round'].get(c, {}).get(it)) for c in cols]
            md.append(f"| {r['code']} | " + ' | '.join(vals) + f" | {fmt((hu.get(r['code']) or {}).get(it))} |")
    md += ['\n## Run summary\n', '| Paper | Stop | Rounds | Kept/reverted/retired | Hard | Soft | Wall clock | Editor $ | Review $ |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in uniq:
        md.append(f"| {r['code']} | {r['stop']} | {min(r['rounds'], nmax)} | {r['kept']}/{r['reverted']}/{r['retired']} | "
                  f"{r['guards']['hard'] or 'none'} | {r['guards']['soft'] or 'none'} | {r['wall_s']/60:.0f} min | "
                  f"{r['editor_usd']} | {r['reviewer_usd']} |")
    (API / 'RESULTS_6ITEMS.md').write_text('\n'.join(md) + '\n')
    print(f'RESULTS_6ITEMS.md updated, {len(uniq)} papers')


if __name__ == '__main__':
    main()
