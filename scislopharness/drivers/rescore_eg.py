#!/usr/bin/env python3
"""Re-score only evidence_gap per round.

The main run finished before the standard quote environment was added to the measurer. This item is a deterministic measurer that reads only tex, so if the round trees
are still intact, re-measuring needs neither a model nor a GPU. Only the score is swapped in; other items are untouched.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
import harness as H

OUT = API / 'final_10'


def score(code: str, tree: Path) -> float | None:
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, str(H.EOR_CODE / 'measure_tree.py'), '--item', 'evidence_gap',
                            '--out', td, '--ai', f'{code}={tree}'], capture_output=True, text=True, timeout=900)
        p = Path(td) / 'papers.jsonl'   # measure_tree writes directly to --out
        if not p.exists():
            print(code, tree.name, 'no output', (r.stderr or '')[-300:])
            return None
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                if d.get('corpus') == 'AI':
                    return d.get('slop_score')
    return None


def main():
    codes = json.load(open(API / 'papers10.json'))['codes']
    for c in codes:
        rf = OUT / c / 'result.json'
        if not rf.exists():
            continue
        rec = json.loads(rf.read_text())
        trees = {'R0': H.fars_paper_dir(c)}
        for n in (1, 2, 3):
            t = OUT / c / 'run_tree' / f'R{n}'
            if t.is_dir():
                trees[f'R{n}'] = t
        old = {k: (rec['per_round'].get(k) or {}).get('evidence_gap') for k in trees}
        new = {k: score(c, t) for k, t in trees.items()}
        for k, v in new.items():
            if k in rec['per_round']:
                rec['per_round'][k]['evidence_gap'] = v
        rec.setdefault('rescored', {})['evidence_gap'] = {'reason': 're-measured after adding the quote environment to the measurer (0920)',
                                                          'before': old, 'after': new}
        rf.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(c, ' '.join(f'{k}:{old[k]}->{new[k]}' for k in sorted(trees)))


if __name__ == '__main__':
    main()
