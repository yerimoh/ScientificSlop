"""Benchmark re-scoring of revised papers (Fig 4, right panel).

For every arm and round, the revised AI paper is paired again with its original human
partner and scored exactly as in section 5.2: PairAcc, AUROC and TPR at 5% FPR, with the
same score directions as bench165 (Binoculars lower = AI; Fast-DetectGPT and NTS higher =
AI; SciSlop higher = AI). Human-side scores are the bench165 ones (unchanged papers).
SciSlop here is the four-item deterministic aggregate (plane mean of item means, N/A
skipped, macro via slop_score_agg), the same rule as slop/SLOP_SCORE.md.

  python3 detect_aggregate.py  -> results/summary/detectors.{json,md}
Rounds without detector output are reported as missing, never as a value.
"""
import itertools, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, BENCH, ITEMS, ai_codes, hu_record, load_progress  # noqa: E402

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
ROUNDS = [1, 3]
DET = {'binoculars': ('binoculars', -1),            # Table 2 configuration (Qwen2.5-7B pair, 6 windows)
       'binoculars_faithful': ('binoculars', -1),   # Hans et al. setting (falcon-7b pair, first 512 tokens)
       'fast_detectgpt': ('fast_detectgpt', +1), 'nts': ('nts', +1)}


def rows(path, key=None):
    out = {}
    if not Path(path).exists():
        return out
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        out[r['id']] = r if key is None else r.get(key)
    return out


def metrics(ai, hu, direction):
    """ai, hu: {code: score}; pairs by code."""
    pairs = [c for c in ai if c in hu and ai[c] is not None and hu[c] is not None]
    if not pairs:
        return None
    corr = 0.0
    for c in pairs:
        d = (ai[c] - hu[c]) * direction
        corr += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    A = [ai[c] * direction for c in pairs]; H = [hu[c] * direction for c in pairs]
    auroc = sum(1.0 if a > h else (0.5 if a == h else 0.0) for a, h in itertools.product(A, H)) / (len(A) * len(H))
    hs = sorted(H, reverse=True); t5 = hs[int(len(H) * 0.05)]
    tpr5 = sum(a > t5 for a in A) / len(A)
    return dict(n_pairs=len(pairs), pair_acc=round(corr / len(pairs), 3), auroc=round(auroc, 3), tpr_at_fpr5=round(tpr5, 3))


def slop_paper_score(item_rows: dict, code: str):
    """Four-item aggregate for one paper: planes Structure(macro agg, xsec), Argument(citation),
    Artifacts(evidence_gap); plane = mean of available items; paper = mean of available planes."""
    def s(it, key='slop_score'):
        r = item_rows[it].get(code)
        return None if r is None else r.get(key)
    planes = []
    st = [v for v in (s('macro_redund', 'slop_score_agg'), s('xsec_ref')) if v is not None]
    if st:
        planes.append(sum(st) / len(st))
    for it in ('citation', 'evidence_gap'):
        v = s(it)
        if v is not None:
            planes.append(v)
    return sum(planes) / len(planes) if planes else None


def main():
    codes = ai_codes()
    hu_id = {c: hu_record(c)['id'] for c in codes}
    # human-side detector scores and slop rows from bench165
    hu_det = {}
    for fname, (key, _) in DET.items():
        R = rows(BENCH / 'results' / f'{fname}.jsonl', key)
        hu_det[fname] = {c: R.get(f'HU_{hu_id[c]}') for c in codes}
    hu_slop_rows = {it: rows(BENCH / 'results/slop' / it / 'papers.jsonl') for it in ITEMS}
    hu_slop = {c: slop_paper_score(hu_slop_rows, hu_id[c]) for c in codes}
    ai0_slop = {c: slop_paper_score(hu_slop_rows, c) for c in codes}

    out = {'systems': list(DET) + ['sciSlop4'], 'R0': {}, 'arms': {}}
    for fname, (key, d) in DET.items():
        R = rows(BENCH / 'results' / f'{fname}.jsonl', key)
        out['R0'][fname] = metrics({c: R.get(f'AI_{c}') for c in codes}, hu_det[fname], d)
    out['R0']['sciSlop4'] = metrics(ai0_slop, hu_slop, +1)

    for arm in ARMS:
        prog = load_progress(arm)
        out['arms'][arm] = {}
        for n in ROUNDS:
            ok = {c for c in codes if (prog.get(c, {}).get(n) or {}).get('exec') in ('ok', 'nothing_to_fix')}
            rec = {}
            for fname, (key, d) in DET.items():
                p = EOR / 'results/detectors' / arm / f'R{n}' / f'{fname}.jsonl'
                R = rows(p, key)
                ai = {c: R.get(f'AI_{c}') for c in codes if c in ok and f'AI_{c}' in R}
                rec[fname] = metrics(ai, hu_det[fname], d) if ai else {'missing': True}
            item_rows = {it: rows(EOR / 'results/slop' / arm / f'R{n}' / it / 'papers.jsonl') for it in ITEMS}
            ai = {c: slop_paper_score(item_rows, c) for c in codes if c in ok and item_rows['xsec_ref'].get(c)}
            rec['sciSlop4'] = metrics(ai, hu_slop, +1) if ai else {'missing': True}
            out['arms'][arm][n] = rec

    o = EOR / 'results/summary'; o.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o / 'detectors.json', 'w'), indent=1)
    L = ['# Benchmark re-scoring of revised papers (PairAcc / AUROC / TPR@5%FPR vs the original human partner)\n',
         '| arm | round | ' + ' | '.join(out['systems']) + ' |', '|---|---|' + '---|' * len(out['systems'])]
    def cell(m):
        if not m or m.get('missing'):
            return 'missing'
        return f"{m['pair_acc']:.3f} / {m['auroc']:.3f} / {m['tpr_at_fpr5']:.3f} (n={m['n_pairs']})"
    L.append('| original | R0 | ' + ' | '.join(cell(out['R0'][s]) for s in out['systems']) + ' |')
    for arm in ARMS:
        for n in ROUNDS:
            L.append(f'| {arm} | R{n} | ' + ' | '.join(cell(out['arms'][arm][n].get(s)) for s in out['systems']) + ' |')
    (o / 'detectors.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
