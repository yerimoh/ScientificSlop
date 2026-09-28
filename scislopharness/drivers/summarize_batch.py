"""Write batch results (final_<suffix>/results.jsonl) as per-item R0..R3 + human pair tables with means, plus per-paper time.

  python3 summarize_batch.py --dir ../final_v14
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import summarize6 as S6

ITEMS = ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph', 'fig_exposition']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(API / 'final_v14'))
    ap.add_argument('--include-fa0002', action='store_true')
    ap.add_argument('--results', default='results.jsonl')
    ap.add_argument('--out', default='RESULTS_BATCH.md')
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    d = Path(a.dir)
    recs = {}
    for line in open(d / a.results):
        if line.strip():
            r = json.loads(line); recs[r['code']] = r
    if a.include_fa0002:
        for p in d.glob('*/result.json'):
            r = json.load(open(p))
            if r['code'] not in recs:
                recs[r['code']] = r
    hu = S6.human_scores()
    codes = sorted(recs)
    fmt = lambda v: '   .   ' if v is None else f'{v:.3f}'
    bsp = d / ('batch_summary.json' if a.results == 'results.jsonl' else a.results.replace('.jsonl', '_summary.json'))
    bs = json.load(open(bsp)) if bsp.exists() else {}
    md = [f'# Batch results ({a.title or d.name}, {len(codes)} papers, updated {time.strftime("%Y-%m-%d %H:%M")})', '',
          f'Editor Haiku 4.5 (per-file edit blocks, thinking 1024) + reviewer Sonnet 5 (text gate), skill {next(iter(recs.values()))["skill"] if recs else "-"}, rounds 3.',
          f'Batch wall clock {bs.get("wall_s", 0)/60:.1f} min (concurrent runs). evidence_gap 0.5 = disclosed gap, a value absent on the human axis. argument_graph missing (.) means the label server was unavailable.', '']
    for it in ITEMS:
        md += [f'## {it}', '', '| Paper | R0 | R1 | R2 | R3 | Human pair |', '|---|---|---|---|---|---|']
        cols = {c: [] for c in ('R0', 'R1', 'R2', 'R3')}
        for c in codes:
            pr = recs[c]['per_round']
            vals = [pr.get(k, {}).get(it) for k in ('R0', 'R1', 'R2', 'R3')]
            for k, v in zip(('R0', 'R1', 'R2', 'R3'), vals):
                if v is not None: cols[k].append(v)
            md.append(f'| {c} | ' + ' | '.join(fmt(v) for v in vals) + f" | {fmt((hu.get(c) or {}).get(it))} |")
        hv = [(hu.get(c) or {}).get(it) for c in codes]; hv = [v for v in hv if v is not None]
        md.append('| **Mean** | ' + ' | '.join(f'**{sum(cols[k])/len(cols[k]):.3f}**' if cols[k] else '.' for k in ('R0', 'R1', 'R2', 'R3')) + f" | **{sum(hv)/len(hv):.3f}** |" if hv else ' | . |')
        md.append('')
    md += ['## Time and gate per paper', '', '| Paper | Wall clock | Per-round (editor, gate, measure) s | Kept/reverted | Diagram edits | Hard guards | Editor $ | Review $ |', '|---|---|---|---|---|---|---|---|']
    for c in codes:
        r = recs[c]
        md.append(f"| {c} | {r['wall_s']/60:.1f} min | {r['round_seconds']} | {[(g[1], g[2]) for g in r['gate_counts']]} | {[f[1] for f in r['figure_edits']]} | "
                  f"{sum(len(g[1]) for g in r['guards'])} | {r['editor_usd']} | {r['gate_usd']} |")
    (d / a.out).write_text('\n'.join(md) + '\n')
    print(d / a.out)


if __name__ == '__main__':
    main()
