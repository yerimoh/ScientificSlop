"""fig_exposition across revision rounds. The item reads the method figure IMAGE (framework_overview),
which no arm can edit: round trees carry tex/bib/sty only and the editors have no image tool. The
item is therefore constant across rounds by construction, and the only thing a revision can change is
whether the figure is still included. This script counts that: for every arm and round, papers whose
tex still \\includegraphics the framework_overview file, papers where the include was dropped, and
the R0 fig_exposition values carried unchanged. -> results/summary/fig_exposition_rounds.{json,md}
"""
import json, os, re, sys, glob, statistics
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, SLOP, ai_codes, load_progress, round_dir, fars_paper_dir, tex_files

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop'] + [f'a4s_{i}' for i in ('macro_redund', 'xsec_ref', 'citation', 'evidence_gap')]
INC = re.compile(r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}')


def includes(d):
    out = set()
    for f in tex_files(d):
        t = re.sub(r'(?<!\\)%.*', '', f.read_text(errors='ignore'))
        out |= {m.group(1).strip() for m in INC.finditer(t)}
    return out


def has_overview(incs):
    return any('framework_overview' in x or 'method_diagrams' in x for x in incs)


def main():
    fe = {json.loads(l)['id']: json.loads(l) for l in open(f'{SLOP}/Artifacts/fig_exposition/results/papers.jsonl')}
    codes = ai_codes()
    r0 = {c: has_overview(includes(fars_paper_dir(c))) for c in codes}
    out = {'note': 'figure image unchanged across rounds by construction; counts are of the \\includegraphics of the method figure',
           'r0': {'n': len(codes), 'n_with_overview_include': sum(r0.values()),
                  'fig_exposition_mean': round(statistics.mean(fe[c]['slop_score'] for c in codes if c in fe and fe[c].get('slop_score') is not None), 4),
                  'n_scored': sum(1 for c in codes if c in fe and fe[c].get('slop_score') is not None)},
           'arms': {}}
    for arm in ARMS:
        prog = load_progress(arm)
        A = {}
        for n in (1, 2, 3):
            rows = []
            for c in codes:
                r = prog.get(c, {}).get(n)
                if not r or r['exec'] not in ('ok', 'nothing_to_fix') or not round_dir(arm, c, n).exists():
                    continue
                rows.append((c, has_overview(includes(round_dir(arm, c, n)))))
            if not rows:
                continue
            kept = [c for c, h in rows if h]; dropped = [c for c, h in rows if not h and r0[c]]
            A[f'R{n}'] = {'n': len(rows), 'kept_include': len(kept), 'dropped_include': len(dropped), 'dropped_codes': dropped,
                          'fig_exposition_mean_kept': round(statistics.mean(fe[c]['slop_score'] for c in kept if c in fe), 4) if kept else None}
        out['arms'][arm] = A
    os.makedirs(EOR / 'results/summary', exist_ok=True)
    json.dump(out, open(EOR / 'results/summary/fig_exposition_rounds.json', 'w'), indent=1)
    L = ['# fig_exposition across rounds (auto-generated)\n', out['note'] + '\n',
         f"R0: {out['r0']}\n\n| arm | round | n | include kept | include dropped | fig_exposition mean (kept, = R0 value) |\n|---|---|---|---|---|---|\n"]
    for arm, A in out['arms'].items():
        for n, v in A.items():
            L.append(f"| {arm} | {n} | {v['n']} | {v['kept_include']} | {v['dropped_include']} | {v['fig_exposition_mean_kept']} |\n")
    (EOR / 'results/summary/fig_exposition_rounds.md').write_text(''.join(L))
    print(''.join(L))


if __name__ == '__main__':
    main()
