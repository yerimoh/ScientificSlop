"""Speed-optimization runs in one table. Per version: round time (editor, gate, measure), per-item R0..R3, human pair.

  python3 compare_versions.py v3 v5 v6 v8 [--code FA0002] [--out ../final_compare/COMPARE.md]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import summarize6 as S6   # noqa: E402

NOTES = {
    'v3': 'agent editor (unlimited thinking), gate 3 chunks in parallel (tool-based), location cap 8',
    'v4': 'edits_parallel editor thinking 0, gate chunks + separate retire (tool-based)',
    'v5': 'edits_parallel thinking 1024, gate chunks 12 turns + inline retire 6 turns',
    'v6': 'v5 + other-file context for the editor, gate 6 turns (missing verdict on turn overflow = treated as revert)',
    'v8': 'agent editor thinking 1024, text-only gate (no tools, JSON response)',
    'v9': 'agent_parallel editor (one agent per file) thinking 1024, text-only gate',
    'v10': 'v8 + retire decision by Haiku (run_v2 default)',
    'v11': 'v10 + editor thinking 512',
    'v12': 'v10 + location cap 6 + gate rule 8 (merges reverted)',
    'v13': 'edits_parallel editor + other-file context + thinking 1024, text gate + rule 8, cap 8',
    'v14': 'v13 + skill v0.5 (evidence gap repair (b) explicit gap, disclosed gap 0.5) + gate rule 9',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('versions', nargs='+')
    ap.add_argument('--code', default='FA0002')
    ap.add_argument('--out', default=str(API / 'final_compare' / 'COMPARE.md'))
    a = ap.parse_args()
    hu = S6.human_scores().get(a.code, {})
    md = [f'# Speed-optimization comparison ({a.code}, Haiku editor + Sonnet gate, updated {time.strftime("%Y-%m-%d %H:%M")})', '',
          'Round time is editor + gate + measure wall-clock seconds. Gate time is the longest parallel chunk (retire call included).', '']
    md += ['| Version | Config | R1 | R2 | R3 | Total (min) | Editor $ | Gate $ | Missing verdicts |', '|---|---|---|---|---|---|---|---|---|']
    rows = {}
    for v in a.versions:
        p = API / f'runs_haiku_sonnetgate_{v}' / 'A-loc' / a.code / 'trajectory.json'
        if not p.exists():
            md.append(f'| {v} | {NOTES.get(v, "")} | (none) | | | | | | |'); continue
        t = json.load(open(p)); rows[v] = t
        secs, miss = [], 0
        for r in t['rounds']:
            g = r.get('gate') or {}; c = g.get('call') or {}
            secs.append(round((r['call'].get('wall_s') or 0) + (c.get('wall_s') or 0) + (r.get('measure_s') or 0)))
            miss += sum(1 for x in g.get('verdicts', []) if x['verdict'] == 'missing')
        gate_usd = sum(((r.get('gate') or {}).get('call') or {}).get('cost_usd') or 0 for r in t['rounds'])
        md.append(f"| {v} | {NOTES.get(v, '')} | " + ' | '.join(f'{s}s' for s in secs) + ' | ' * (3 - len(secs)) +
                  f" | {t['wall_total_s']/60:.1f} | {t['cost_total_usd']:.2f} | {gate_usd:.2f} | {miss} |")
    items = ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph', 'fig_exposition']
    fmt = lambda x: '.' if x is None else f'{x:.3f}'
    for it in items:
        md += ['', f'## {it} (human pair {fmt(hu.get(it))})', '', '| Version | R0 | R1 | R2 | R3 |', '|---|---|---|---|---|']
        for v, t in rows.items():
            sc = [t['r0']['scores'].get(it)] + [r['scores_after'].get(it) for r in t['rounds']]
            sc = sc + [None] * (4 - len(sc))
            md.append(f'| {v} | ' + ' | '.join(fmt(x) for x in sc) + ' |')
    md += ['', '## Gate kept/reverted/missing', '', '| Version | R1 | R2 | R3 |', '|---|---|---|---|']
    for v, t in rows.items():
        cells = []
        for r in t['rounds']:
            g = r.get('gate') or {}
            from collections import Counter
            c = Counter(x['verdict'] for x in g.get('verdicts', []))
            cells.append(f"{c.get('keep', 0)}/{c.get('revert', 0)}/{c.get('missing', 0)}")
        md.append(f'| {v} | ' + ' | '.join(cells) + ' |')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(md) + '\n'); print(out)


if __name__ == '__main__':
    main()
