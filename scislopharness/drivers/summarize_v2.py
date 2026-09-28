"""Write the final_v2/<config>/result.json files as per-item R0..R3 + human pair tables. The human pair is attached only in the report.

  python3 summarize_v2.py [--out ../final_v2/RESULTS.md]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import summarize6 as S6   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(API / 'final_v2'))
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    d = Path(a.dir)
    recs = [json.load(open(p)) for p in sorted(d.glob('*/result.json'))]
    hu = S6.human_scores()
    items = list(recs[0]['per_round']['R0'].keys()) if recs else []
    md = [f'# Single-paper run results ({d.name}, updated {time.strftime("%Y-%m-%d %H:%M")})', '',
          'The human pair column is the value of the same-pair human paper in bench165 and enters the harness nowhere. Per-round outputs are in',
          '`runs_<config>/A-loc/<code>/rounds/R<n>/` and `ROUNDS.md`; diagrams in `final_v2/<config>/figures/`.', '']
    fmt = lambda v: '   .   ' if v is None else f'{v:.3f}'
    for r in recs:
        code = r['code']; pr = r['per_round']; cols = [c for c in ('R0', 'R1', 'R2', 'R3') if c in pr]
        md += [f"## {r['config']}. Editor {r['editor']}, reviewer {r['gate']}, {code}", '',
               '| Item | ' + ' | '.join(cols) + ' | Human pair |', '|---' * (len(cols) + 2) + '|']
        for it in items:
            md.append(f'| {it} | ' + ' | '.join(fmt(pr[c].get(it)) for c in cols) + f" | {fmt((hu.get(code) or {}).get(it))} |")
        md += ['', f"Stop {r['stop']}, rounds {r['rounds']}, wall clock {r['wall_s']/60:.0f} min, editor ${r['editor_usd']}, reviewer ${r['gate_usd']}.",
               f"Gate kept/reverted {r['gate_counts']}. Diagram edits {r['figure_edits']}. Hard guards {r['guards']}.",
               f"Round time (editor, gate, measure) {r['round_seconds']}.", '']
        spec = API / 'specimen_cache' / f'{code}.json'
        if spec.exists():
            sp = json.load(open(spec))
            md += [f"Specimen retrieval. found={sp.get('found')}, search {sp.get('wall_s')}s ${sp.get('cost_usd') or 0:.2f}. {str(sp.get('reason') or '')[:300]}", '']
    out = Path(a.out) if a.out else d / 'RESULTS.md'
    out.write_text('\n'.join(md) + '\n')
    print(out)


if __name__ == '__main__':
    main()
