"""Tables of results/RESULTS_OPEN_PILOT_0916.md: Qwen2.5-32B-Instruct (a1_base.qwen32b, R1) beside Claude Haiku 4.5
(a1_base, R1), same prompt, same acceptance guards, 143 papers. Reads results/summary/{effects,effects.qwen32b,
effects_ag,effects.qwen32b_ag}.json and the progress records. Numbers are never retyped; the reading is hand-written
below the marker."""
import json, glob, statistics, collections
from pathlib import Path
EOR = Path(__file__).resolve().parent.parent
S = EOR / 'results/summary'
E = {'haiku': json.load(open(S / 'effects.json')), 'qwen': json.load(open(S / 'effects.qwen32b.json'))}
A = {'haiku': json.load(open(S / 'effects_ag.json')), 'qwen': json.load(open(S / 'effects.qwen32b_ag.json'))}
NAME = {'macro_redund': 'Recycled sentences', 'xsec_ref': 'Unused objects', 'citation': 'Isolated citations', 'evidence_gap': 'No instance shown', 'argument_graph': 'Argument graph'}
def ci(c): return f"({c[0]:+.3f}, {c[1]:+.3f})" if c and c[0] is not None else 'n/a'
def pv(p): return 'n/a' if p is None else ('<1e-4' if p < 1e-4 else f"{p:.4f}")
def row(it, ed, arm='a1_base'):
    src = A[ed] if it == 'argument_graph' else E[ed]
    return ((src['arms'].get(arm) or {}).get('rounds') or {}).get('1', {}).get('items', {}).get(it)
L = ['# Open-source editor pilot. Qwen2.5-32B-Instruct vs Claude Haiku 4.5, a1 base prompting, R1, 143 papers (0916)\n',
     'Auto-generated tables (`code/write_results_open_pilot.py`). Design in DESIGN.md §10. Both editors ran one round with the same prompt (`prompts.a1_prompt`) and the same acceptance guards; only the editor and the decoding backend differ. Sources `results/summary/effects.qwen32b.json`, `effects.qwen32b_ag.json`, and on the Haiku side `effects.json`, `effects_ag.json`.\n',
     '## 1. Mean slop score per item, R0 → R1 (143 papers)\n',
     '| Item | Editor | n | R0 | R1 | R1−R0 | 95% CI | p (Wilcoxon) | Decreased/increased | Drop from denominator shrinkage | Denominator ratio | At or below human pair R0→R1 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for it in ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph']:
    for ed, lab in (('haiku', 'Claude Haiku 4.5'), ('qwen', 'Qwen2.5-32B-Instruct')):
        v = row(it, ed)
        if not v: L.append(f"| {NAME[it]} | {lab} | – | | | | | | | | | |"); continue
        L.append(f"| {NAME[it]} | {lab} | {v['n']} | {v['mean_r0']:.3f} | {v['mean_rn']:.3f} | {v['mean_diff']:+.3f} | {ci(v['ci95'])} | {pv(v['wilcoxon_p'])} | {v['n_decreased']}/{v['n_increased']} | {v['n_masked_by_denominator_drop']} | {v['mean_denominator_ratio']} | {v['n_at_or_below_human_partner_r0']}→{v['n_at_or_below_human_partner']} |")
if 'a4_slop' in E['qwen']['arms']:
    L += ['\n## 1b. Slop-aware condition (a4_slop), R0 → R1 (143 papers). Haiku edits files via tools; Qwen receives the same feedback as text and returns the files\n',
          '| Item | Editor | n | R0 | R1 | R1−R0 | 95% CI | p (Wilcoxon) | Decreased/increased | Drop from denominator shrinkage | Denominator ratio | At or below human pair R0→R1 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for it in ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph']:
        for ed, lab in (('haiku', 'Claude Haiku 4.5 (agent)'), ('qwen', 'Qwen2.5-32B-Instruct (text)')):
            v = row(it, ed, 'a4_slop')
            if not v: L.append(f"| {NAME[it]} | {lab} | – | | | | | | | | | |"); continue
            L.append(f"| {NAME[it]} | {lab} | {v['n']} | {v['mean_r0']:.3f} | {v['mean_rn']:.3f} | {v['mean_diff']:+.3f} | {ci(v['ci95'])} | {pv(v['wilcoxon_p'])} | {v['n_decreased']}/{v['n_increased']} | {v['n_masked_by_denominator_drop']} | {v['mean_denominator_ratio']} | {v['n_at_or_below_human_partner_r0']}→{v['n_at_or_below_human_partner']} |")
L += ['\n## 2. Guards (R1)\n', '| Editor | ok / no-op | Word-count ratio | Papers with dangling cites | Papers that dropped cited works (mean) | Time per paper (s) | Mean files accepted | Not returned / identical / length ratio / macro loss (file counts) |', '|---|---|---|---|---|---|---|---|']
for ed, lab, tag, arm in [('haiku', 'Claude Haiku 4.5, a1', '', 'a1_base'), ('qwen', 'Qwen2.5-32B-Instruct, a1', '.qwen32b', 'a1_base')] + \
                         ([('haiku', 'Claude Haiku 4.5, a4 (agent)', '', 'a4_slop'), ('qwen', 'Qwen2.5-32B-Instruct, a4 (text)', '.qwen32b', 'a4_slop')] if 'a4_slop' in E['qwen']['arms'] else []):
    R = E[ed]['arms'][arm]['rounds']['1']
    recs = [json.loads(l) for f in glob.glob(str(EOR / f'progress/{arm}{tag}.shard*.jsonl')) + [str(EOR / f'progress/{arm}{tag}.jsonl')] if Path(f).exists() for l in open(f)]
    latest = {}
    for r in recs:
        if r.get('round') == 1 and (r['code'] not in latest or r.get('ts', '') >= latest[r['code']].get('ts', '')): latest[r['code']] = r
    recs = [r for r in latest.values() if r['exec'] == 'ok']
    dts = [r['dt'] for r in recs]; nrev = [r['info'].get('n_revised', 0) for r in recs if isinstance(r.get('info'), dict)]
    kept = collections.Counter()
    for r in recs:
        for v in (r['info'].get('files') or {}).values():
            if v != 'revised': kept[v.split(':')[1].split('_')[0] if ':' in v else v] += 1
    L.append(f"| {lab} | {R['n_ok']} / {R['n_failed_noop']} (incl. nothing_to_fix) | {R['mean_word_ratio']} | {R['papers_with_new_dangling_cites']} | {R.get('papers_that_dropped_cited_works')} ({R.get('mean_cited_works_dropped')}) | {round(statistics.mean(dts)) if dts else '–'} | {round(statistics.mean(nrev), 2) if nrev else '–'} | {kept.get('not', 0)} / {kept.get('identical', 0)} / {kept.get('length', 0)} / {kept.get('lost', 0)} |")
L.append('\n<!-- reading below is written by hand -->\n')
out = EOR / 'results/RESULTS_OPEN_PILOT_0916.md'
prev = out.read_text() if out.exists() else ''
hand = prev.split('<!-- reading below is written by hand -->\n', 1)[1] if '<!-- reading below is written by hand -->' in prev else ''
out.write_text('\n'.join(L) + hand); print('\n'.join(L))
