#!/usr/bin/env python3
"""T2. Deterministically picks one instance from the project records and turns it into an exhibit. No model call.

evidence_gap counts whether the paper actually displays one specimen of the evidence it aggregates. T1 has a Sonnet search agent
read the records and point to a specimen, but it filters out `optimize_trace` wholesale by path name, so the prediction dumps
inside it are thrown away too. FA0059's 100-row question/gold_answer/prediction file was discarded that way.
T2 looks at content, not paths. It finds lists holding several dicts with the same keys and decides from the key names
whether they are instances or an aggregate table.

Two grades.
    instance   rows with both an input-like key and an output- or gold-like key. Displayed as is, they make an example
    row        rows with an identifier and values but no input and output. Instance-level, but weak to call an example
The grade at which a paper opened is recorded in the result. If there is nothing it is T0, and that too is a result.

  python3 specimen_t2.py [CODE ...] [--write]        -> ../specimen_t2/<CODE>.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
sys.path.insert(0, str(API.parent / 'temp' / 'code'))
sys.path.insert(0, str(HERE))
OUT = API / 'specimen_t2'

IN_KEYS = ('question', 'prompt', 'input', 'query', 'sentence', 'sent', 'source', 'context',
           'instruction', 'problem', 'text', 'passage', 'utterance')
OUT_KEYS = ('prediction', 'pred', 'output', 'response', 'generation', 'completion', 'answer')
GOLD_KEYS = ('gold_answer', 'gold', 'ground_truth', 'reference', 'label', 'target', 'expected')
ID_KEYS = ('instance_id', 'example_id', 'sample_id', 'question_idx', 'idx', 'index', 'id', 'name', 'task')
# Names that count sizes or counts. Even if the name looks similar, they are not an instance's input or output (input_tokens, question_idx)
COUNTER = re.compile(r'_(tokens?|count|num|n|size|len|length|idx|index|id|pct|rate|time|sec|steps?|score)$', re.I)
# Notes of the pipeline itself, not experiment instances of this paper. Filtered by path
PROCESS = re.compile(r'(^|/)(traces?|task_plan|FARS_MEMO|\.venv)(/|$)'
                     r'|(^|/)(REPORT|README|task_plan|effectiveness_evaluation_report)\.(md|json)$', re.I)
MIN_TEXT = 20           # to be called an input, the text must be at least this long


def _rows(obj, path='', depth=0):
    """Finds every list holding 3 or more dicts with the same keys."""
    found = []
    if isinstance(obj, list) and len(obj) >= 3 and all(isinstance(x, dict) for x in obj[:3]):
        found.append((path, obj))
    elif isinstance(obj, dict) and depth < 5:
        for k, v in obj.items():
            found += _rows(v, f'{path}.{k}' if path else k, depth + 1)
    elif isinstance(obj, list) and depth < 5:
        for i, v in enumerate(obj[:5]):
            found += _rows(v, f'{path}[{i}]', depth + 1)
    return found


def _pick(keys, names, skip_counters: bool = True) -> str | None:
    """One key matching the role, following the priority of the name list. In the input and output slots, names that count sizes
    are skipped (input_tokens is not an input), but in the identifier slot such a name is the identifier itself (instance_id)."""
    low = {k: k.lower() for k in keys}
    for n in names:
        for k, kl in low.items():
            if skip_counters and COUNTER.search(kl):
                continue
            if kl == n or kl.endswith('_' + n) or kl.startswith(n + '_'):
                return k
    return None


def _textlen(v) -> int:
    return len(v) if isinstance(v, str) else (len(json.dumps(v, ensure_ascii=False)) if isinstance(v, (dict, list)) else 0)


def _grade(rows: list) -> tuple[str, dict]:
    """Is a row an example or one line of an aggregate table? Candidates are picked by key name and judged by value."""
    keys = list(rows[0].keys())
    i, o, g = _pick(keys, IN_KEYS), _pick(keys, OUT_KEYS), _pick(keys, GOLD_KEYS)
    if o and g and o == g:                      # when gold_answer is caught as both
        o = _pick([k for k in keys if k != g], OUT_KEYS)
    if i and (o or g):
        # Confirm by value. The input slot must hold text for this to be an example
        n = sum(1 for r in rows[:10] if _textlen(r.get(i)) >= MIN_TEXT)
        if n >= max(2, len(rows[:10]) // 2):
            return 'instance', {'input': i, 'output': o, 'gold': g}
    idk = _pick(keys, ID_KEYS, skip_counters=False)
    if idk and len(keys) >= 3:
        return 'row', {'id': idk}
    return 'note', {}


def _load(p: Path):
    try:
        if p.suffix.lower() == '.jsonl':
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return json.loads(p.read_text())
    except Exception:
        return None


def find(code: str) -> dict:
    import harness as H
    root = H.fars_paper_dir(code).parent.parent.parent
    exp = root / 'code' / 'exp'
    cands = []
    for p in sorted(exp.rglob('*')) if exp.is_dir() else []:
        if not p.is_file() or p.suffix.lower() not in ('.json', '.jsonl'):
            continue
        if '/.venv/' in str(p) or '/node_modules/' in str(p):
            continue
        d = _load(p)
        if d is None:
            continue
        rel = str(p.relative_to(root))
        for where, rows in _rows(d):
            grade, keys = _grade(rows)
            if grade == 'note':
                continue
            # Process records are not experiment instances of this paper. But rows confirmed as examples by their values are kept
            # regardless of the folder name. FA0059's prediction dump used to be discarded just for sitting under optimize_trace
            if PROCESS.search(rel) and grade != 'instance':
                continue
            cands.append({'file': rel, 'where': where, 'n_rows': len(rows),
                          'grade': grade, 'keys': keys, 'rows': rows})
    if not cands:
        return {'code': code, 'tier': 'T0', 'found': False,
                'reason': 'no list of instance rows in code/exp; the records hold aggregates only'}
    # example rows first, and among them files with more rows first
    cands.sort(key=lambda c: (0 if c['grade'] == 'instance' else 1, -c['n_rows']))
    best = cands[0]
    rows = best['rows']
    k = best['keys']
    if best['grade'] == 'instance':
        # Pick a row short enough to display. Length is the only selection rule and values are never touched
        def size(r):
            return sum(len(str(r.get(x) or '')) for x in (k.get('input'), k.get('output'), k.get('gold')) if x)
        idx = min(range(len(rows)), key=lambda i: (size(rows[i]) < 40, size(rows[i])))
    else:
        idx = 0
    return {'code': code, 'tier': 'T2', 'found': True, 'grade': best['grade'],
            'file': best['file'], 'where': best['where'], 'n_rows': best['n_rows'],
            'row_index': idx, 'keys': k, 'row': rows[idx],
            'other_candidates': [{kk: c[kk] for kk in ('file', 'grade', 'n_rows')} for c in cands[1:6]]}


def render(rec: dict) -> str:
    """An exhibit showing one row as is. Values are written character for character from the record and nothing new is written."""
    r, k = rec['row'], rec.get('keys') or {}
    lines = []
    order = [k.get('input'), k.get('output'), k.get('gold')]
    order = [x for x in order if x] + [x for x in r if x not in order]
    for key in order:
        v = r.get(key)
        v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        lines.append(f'{key}: {v}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('codes', nargs='*')
    a = ap.parse_args()
    codes = a.codes or json.load(open(API / 'papers10.json'))['codes']
    OUT.mkdir(parents=True, exist_ok=True)
    for c in codes:
        rec = find(c)
        if rec.get('found'):
            rec['rendered'] = render(rec)
        (OUT / f'{c}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        if rec.get('found'):
            print(f"{c}: {rec['tier']} {rec['grade']:8s} {rec['file']} row {rec['row_index']}/{rec['n_rows']} "
                  f"keys {rec['keys']}")
        else:
            print(f"{c}: T0  {rec['reason']}")


if __name__ == '__main__':
    main()
