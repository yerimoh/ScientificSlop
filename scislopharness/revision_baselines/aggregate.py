"""Aggregate results for 5.3.1 (arm x round x item) and 5.3.2 (single-item heatmap).

  python3 aggregate.py            -> results/summary/{effects.json, effects.md, interactions.json, interactions.md}
Reads results/slop/<arm>/R<n>/<item>/papers.jsonl, bench165 R0 + human scores, progress
records (no-op, dangling, words). Paper is the unit; paired bootstrap CIs over papers.
Nothing here decides anything; it prints what was measured.
"""
import json, os, random, statistics, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, BENCH, ITEMS, ai_codes, load_progress, hu_record, cite_keys, fars_paper_dir, round_dir, key, TAG  # noqa: E402

if os.environ.get('EOR_ITEMS'):                 # 0916: e.g. EOR_ITEMS=argument_graph EOR_OUT=_ag -> effects_ag.{json,md}
    ITEMS = os.environ['EOR_ITEMS'].split(',')
OUT_SUFFIX = os.environ.get('EOR_OUT', '')
MAIN_ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
SINGLE_ARMS = [f'a4s_{it}' for it in ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap']]
ROUNDS = [1, 2, 3]
random.seed(914)


def score_of(item, r):
    if r is None:
        return None
    if item == 'macro_redund':
        return r.get('slop_score_agg')
    return r.get('slop_score')


def rows(path):
    if not Path(path).exists():
        return {}
    return {json.loads(l)['id']: json.loads(l) for l in open(path) if l.strip()}


def r0_rows(item):
    return rows(BENCH / 'results/slop' / item / 'papers.jsonl')


def arm_rows(arm, n, item):
    return rows(EOR / 'results/slop' / key(arm) / f'R{n}' / item / 'papers.jsonl')


def boot_ci(diffs, B=2000):
    if len(diffs) < 3:
        return (None, None)
    ms = []
    for _ in range(B):
        s = [random.choice(diffs) for _ in diffs]
        ms.append(statistics.mean(s))
    ms.sort()
    return (round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B)], 4))


def wilcoxon_p(diffs):
    try:
        from scipy.stats import wilcoxon
        d = [x for x in diffs if x != 0]
        if len(d) < 5:
            return None
        return float(wilcoxon(d).pvalue)
    except Exception:
        return None


def complete_codes(arms, rounds=(1, 2, 3)):
    """Papers finished in every one of `arms` at every round. Fig 4 compares arms on this
    set, so an arm that is further along cannot look different merely by covering other
    papers."""
    progs = {a: load_progress(a) for a in arms}
    out = []
    for c in ai_codes():
        if all((progs[a].get(c, {}).get(n) or {}).get('exec') in ('ok', 'nothing_to_fix')
               for a in arms for n in rounds):
            out.append(c)
    return out


def effects(only_codes=None):
    codes = [c for c in ai_codes() if only_codes is None or c in set(only_codes)]
    R0 = {it: r0_rows(it) for it in ITEMS}
    HU = {it: {} for it in ITEMS}
    for c in codes:
        h = hu_record(c)
        for it in ITEMS:
            HU[it][c] = score_of(it, R0[it].get(h['id'])) if h else None
    out = {'items': ITEMS, 'arms': {}, 'human_mean': {}, 'r0_mean': {}}
    for it in ITEMS:
        hv = [v for v in HU[it].values() if v is not None]
        av = [score_of(it, R0[it].get(c)) for c in codes]; av = [v for v in av if v is not None]
        out['human_mean'][it] = round(statistics.mean(hv), 4) if hv else None
        out['r0_mean'][it] = round(statistics.mean(av), 4) if av else None
    for arm in MAIN_ARMS + SINGLE_ARMS:
        prog = load_progress(arm)
        if not prog:
            continue
        A = {'rounds': {}, 'guards': {}}
        for n in ROUNDS:
            execs = {c: prog.get(c, {}).get(n, {}).get('exec') for c in codes}
            n_ok = sum(1 for v in execs.values() if v in ('ok', 'nothing_to_fix'))
            n_failed = sum(1 for v in execs.values() if v == 'failed')
            if n_ok == 0:
                continue
            per_item = {}
            for it in ITEMS:
                RN = arm_rows(arm, n, it)
                if not RN:
                    continue
                pairs = []
                for c in codes:
                    if execs.get(c) not in ('ok', 'nothing_to_fix'):
                        continue
                    a0, a1 = score_of(it, R0[it].get(c)), score_of(it, RN.get(c))
                    if a0 is None or a1 is None:
                        continue
                    d0, d1 = R0[it][c].get('slop_denominator'), RN[c].get('slop_denominator')
                    pairs.append((c, a0, a1, d0, d1))
                if not pairs:
                    continue
                diffs = [a1 - a0 for _, a0, a1, _, _ in pairs]
                masked = sum(1 for _, a0, a1, d0, d1 in pairs if a1 < a0 and d0 and d1 is not None and d1 < d0)
                reached_h = sum(1 for c, a0, a1, _, _ in pairs if HU[it].get(c) is not None and a1 <= HU[it][c])
                reached_h0 = sum(1 for c, a0, a1, _, _ in pairs if HU[it].get(c) is not None and a0 <= HU[it][c])
                per_item[it] = {
                    'n': len(pairs), 'mean_r0': round(statistics.mean(a0 for _, a0, _, _, _ in pairs), 4),
                    'mean_rn': round(statistics.mean(a1 for _, _, a1, _, _ in pairs), 4),
                    'mean_diff': round(statistics.mean(diffs), 4), 'ci95': boot_ci(diffs), 'wilcoxon_p': wilcoxon_p(diffs),
                    'n_decreased': sum(1 for d in diffs if d < 0), 'n_increased': sum(1 for d in diffs if d > 0),
                    'n_masked_by_denominator_drop': masked,
                    'mean_denominator_ratio': round(statistics.mean(d1 / d0 for _, _, _, d0, d1 in pairs if d0 and d1 is not None), 3) if any(d0 for _, _, _, d0, _ in pairs) else None,
                    'n_at_or_below_human_partner': reached_h, 'n_at_or_below_human_partner_r0': reached_h0,
                }
            words = [(prog[c][n]['words_before'], prog[c][n]['words_after']) for c in codes
                     if prog.get(c, {}).get(n, {}).get('exec') == 'ok' and prog[c][n].get('words_before')]
            # citation-specific guard: bundling two citing sentences into one is the intended fix and
            # lowers the citing-sentence denominator, so count cited WORKS that disappeared instead.
            lost = []
            for c in codes:
                if execs.get(c) not in ('ok', 'nothing_to_fix'):
                    continue
                d = round_dir(arm, c, n)
                if d.exists():
                    k0, kn = cite_keys(fars_paper_dir(c)), cite_keys(d)
                    lost.append(len(k0 - kn))
            dang = [len(prog[c][n].get('dangling_new', [])) for c in codes if prog.get(c, {}).get(n, {}).get('exec') == 'ok']
            A['rounds'][n] = {'n_ok': n_ok, 'n_failed_noop': n_failed,
                              'n_blocked': sum(1 for v in execs.values() if v == 'blocked'),
                              'n_missing': sum(1 for v in execs.values() if v is None),
                              'mean_word_ratio': round(statistics.mean(b / a for a, b in words), 3) if words else None,
                              'papers_with_new_dangling_cites': sum(1 for d in dang if d > 0),
                              'papers_that_dropped_cited_works': sum(1 for x in lost if x > 0),
                              'mean_cited_works_dropped': round(statistics.mean(lost), 2) if lost else None,
                              'items': per_item}
        out['arms'][arm] = A
    return out


def interactions():
    sub = json.load(open(EOR / 'results/subset60.json'))['codes'] if (EOR / 'results/subset60.json').exists() else []
    R0 = {it: r0_rows(it) for it in set(ITEMS) | {a[4:] for a in SINGLE_ARMS}}   # target items too (EOR_ITEMS may differ)
    out = {'subset_n': len(sub), 'rows': {}}
    for arm in SINGLE_ARMS + ['a4_slop']:
        prog = load_progress(arm)
        if not prog:
            continue
        target = arm[4:] if arm.startswith('a4s_') else 'joint'
        row = {'target': target, 'cells': {}, 'n_ok': 0}
        oks = [c for c in sub if prog.get(c, {}).get(1, {}).get('exec') in ('ok', 'nothing_to_fix')]
        row['n_ok'] = len(oks)
        cell_diffs = {}
        for it in ITEMS:
            RN = arm_rows(arm, 1, it)
            diffs, diffs_t = [], []
            for c in oks:
                a0, a1 = score_of(it, R0[it].get(c)), score_of(it, RN.get(c))
                if a0 is None or a1 is None:
                    continue
                diffs.append(a1 - a0)
                if target != 'joint':
                    t0 = score_of(target, R0[target].get(c))
                    if t0 is not None and t0 > 0:
                        diffs_t.append(a1 - a0)
            cell_diffs[it] = diffs
            row['cells'][it] = {'n': len(diffs), 'mean_diff': round(statistics.mean(diffs), 4) if diffs else None,
                                'ci95': boot_ci(diffs), 'n_increased': sum(1 for d in diffs if d > 0),
                                'n_decreased': sum(1 for d in diffs if d < 0),
                                'mean_diff_when_target_defect_present': round(statistics.mean(diffs_t), 4) if diffs_t else None,
                                'n_target_defect_present': len(diffs_t)}
        if target != 'joint' and target in cell_diffs:
            RT = arm_rows(arm, 1, target)
            both = 0; n_t = 0
            for c in oks:
                t0, t1 = score_of(target, R0[target].get(c)), score_of(target, RT.get(c))
                if t0 is None or t1 is None or t1 >= t0:
                    continue
                n_t += 1
                worse = False
                for it in ITEMS:
                    if it == target:
                        continue
                    o0, o1 = score_of(it, R0[it].get(c)), score_of(it, arm_rows(arm, 1, it).get(c))
                    if o0 is not None and o1 is not None and o1 > o0:
                        worse = True
                both += worse
            row['n_target_decreased'] = n_t
            row['n_target_decreased_and_other_increased'] = both
        out['rows'][arm] = row
    return out


def md_effects(E):
    L = ['# 5.3.1 Effects of revision (auto-generated, numbers only)\n',
         f"R0 mean: {E['r0_mean']}\nhuman partner mean: {E['human_mean']}\n"]
    for arm, A in E['arms'].items():
        L.append(f'\n## {arm}\n')
        for n, R in A['rounds'].items():
            L.append(f"R{n}: ok={R['n_ok']} noop={R['n_failed_noop']} blocked={R['n_blocked']} missing={R['n_missing']} "
                     f"word_ratio={R['mean_word_ratio']} dangling_papers={R['papers_with_new_dangling_cites']} "
                     f"papers_dropping_cited_works={R.get('papers_that_dropped_cited_works')} mean_works_dropped={R.get('mean_cited_works_dropped')}\n")
            L.append('| item | n | R0 | Rn | diff | CI95 | p | dec/inc | masked(den drop) | den ratio | <=human R0->Rn |\n|---|---|---|---|---|---|---|---|---|---|---|\n')
            for it, v in R['items'].items():
                L.append(f"| {it} | {v['n']} | {v['mean_r0']} | {v['mean_rn']} | {v['mean_diff']} | {v['ci95']} | "
                         f"{None if v['wilcoxon_p'] is None else round(v['wilcoxon_p'], 4)} | {v['n_decreased']}/{v['n_increased']} | "
                         f"{v['n_masked_by_denominator_drop']} | {v['mean_denominator_ratio']} | {v['n_at_or_below_human_partner_r0']}->{v['n_at_or_below_human_partner']} |\n")
    return ''.join(L)


def md_inter(I):
    L = [f"# 5.3.2 Interactions (auto-generated), subset n={I['subset_n']}\n\n| target \\ measured | " + ' | '.join(ITEMS) + ' | n_ok | target dec | dec & other inc |\n|---|' + '---|' * (len(ITEMS) + 3) + '\n']
    for arm, row in I['rows'].items():
        cells = ' | '.join(f"{row['cells'].get(it, {}).get('mean_diff')} {row['cells'].get(it, {}).get('ci95')}" for it in ITEMS)
        L.append(f"| {row['target']} | {cells} | {row['n_ok']} | {row.get('n_target_decreased')} | {row.get('n_target_decreased_and_other_increased')} |\n")
    return ''.join(L)


if __name__ == '__main__':
    out = EOR / 'results/summary'; out.mkdir(parents=True, exist_ok=True)
    SFX = TAG + OUT_SUFFIX
    E = effects(); json.dump(E, open(out / f'effects{SFX}.json', 'w'), indent=1); (out / f'effects{SFX}.md').write_text(md_effects(E))
    common = complete_codes(MAIN_ARMS)
    C = effects(common); C['common_codes'] = common; C['n_common'] = len(common)
    json.dump(C, open(out / f'effects_common{SFX}.json', 'w'), indent=1)
    (out / f'effects_common{SFX}.md').write_text(f'# Same papers in all four arms, n={len(common)}\n\n' + md_effects(C))
    print(f'common set: {len(common)} papers')
    if not TAG:
        I = interactions(); json.dump(I, open(out / f'interactions{OUT_SUFFIX}.json', 'w'), indent=1); (out / f'interactions{OUT_SUFFIX}.md').write_text(md_inter(I))
    print((out / f'effects{SFX}.md').read_text()[:3000]); print((out / f'interactions{OUT_SUFFIX}.md').read_text()[:1500])
