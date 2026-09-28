#!/usr/bin/env python3
"""Pick case candidates per item. Puts what the two arms did on the same paper and item side by side.

The two arms share the editor and the definitions and differ only in the gate.
    a4_slop        given only the slop definitions. No gate (Effects_of_revision/runs/a4_slop)
    ours           same definitions plus an evidence-checking gate (final_10)

The selection criterion is not the score gap but where the gate actually issued a verdict. Kept and reverted changes are laid out
next to the original text so a person can read them and judge the quality difference.
"""
import os
import json, re, sys
from pathlib import Path

API = Path(__file__).resolve().parent.parent
EOR = Path(os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/Effects_of_revision")
BASE = EOR / 'runs' / 'a4_slop'
OURS = API / 'final_10'
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H

ITEMS = list(H.ITEMS) + ['argument_graph', 'fig_exposition']


def base_scores(code: str) -> dict:
    """Per-round scores of the baseline arm, read from the measurement records that arm left behind."""
    out = {}
    for r in ('R1', 'R2', 'R3'):
        f = BASE / code / r / 'measure' / 'summary.json'
        if not f.exists():
            f = BASE / code / r / 'summary.json'
        if f.exists():
            try:
                out[r] = json.loads(f.read_text()).get('scores') or {}
            except Exception:
                pass
    return out


def gate_calls(code: str) -> list:
    """All gate verdicts from our arm: round, verdict, reason, original and revised text."""
    out = []
    t = OURS / code / 'run_tree' / 'trajectory.json'
    if not t.exists():
        return out
    tj = json.loads(t.read_text())
    for rd in tj.get('rounds', []):
        g = rd.get('gate') or {}
        for v in g.get('verdicts', []):
            out.append({'round': rd['round'], **v})
    return out


def main():
    codes = json.load(open(API / 'papers10.json'))['codes']
    rows = []
    for c in codes:
        f = OURS / c / 'result.json'
        if not f.exists():
            continue
        ours = json.loads(f.read_text())['per_round']
        base = base_scores(c)
        g = gate_calls(c)
        rows.append({'code': c, 'ours': ours, 'base': base,
                     'n_verdicts': len(g), 'n_revert': sum(1 for x in g if x['verdict'] == 'revert'),
                     'categories': sorted({x.get('category') for x in g if x.get('category')})})
    print(f'{"paper":8s} {"item":16s} {"R0":>6s} {"baseR3":>8s} {"oursR3":>7s}  gate verdicts')
    for r in rows:
        for it in ITEMS:
            o0 = (r['ours'].get('R0') or {}).get(it)
            o3 = (r['ours'].get('R3') or {}).get(it)
            b3 = (r['base'].get('R3') or {}).get(it)
            if o0 is None:
                continue
            f = lambda v: '  .   ' if v is None else f'{v:.3f}'
            print(f"{r['code']:8s} {it:16s} {f(o0)} {f(b3):>8s} {f(o3):>7s}  "
                  f"{r['n_verdicts']} verdicts, {r['n_revert']} reverted")
            break
    json.dump(rows, open(API / 'final_10' / 'casepick.json', 'w'), ensure_ascii=False, indent=1)
    print('\nR3 mean of both arms per item')
    for it in ITEMS:
        o = [(r['ours'].get('R3') or {}).get(it) for r in rows]
        b = [(r['base'].get('R3') or {}).get(it) for r in rows]
        o = [x for x in o if x is not None]; b = [x for x in b if x is not None]
        print(f'  {it:16s} ours {sum(o)/len(o):.3f} (n={len(o)})   base ' +
              (f'{sum(b)/len(b):.3f} (n={len(b)})' if b else 'no record'))


if __name__ == '__main__':
    main()
