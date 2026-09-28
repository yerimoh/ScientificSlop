"""Tables of results/RESULTS_AG_0916.md from results/summary/{effects_ag,interactions_ag,fig_exposition_rounds}.json.
Numbers are never retyped; the Korean reading is appended by hand below the marker line."""
import json, os
from pathlib import Path
EOR = Path(__file__).resolve().parent.parent
E = json.load(open(EOR / 'results/summary/effects_ag.json')); I = json.load(open(EOR / 'results/summary/interactions_ag.json'))
F = json.load(open(EOR / 'results/summary/fig_exposition_rounds.json'))
ARM = {'a1_base': 'A1 Base prompting', 'a2_code': 'A2 Claude Code', 'a3_review': 'A3 Reviewer refinement', 'a4_slop': 'A4 Slop-aware',
       'a4s_macro_redund': 'single: Recycled sentences', 'a4s_xsec_ref': 'single: Unused objects', 'a4s_citation': 'single: Isolated citations', 'a4s_evidence_gap': 'single: No instance shown'}
def ci(c): return f"({c[0]:+.3f}, {c[1]:+.3f})" if c and c[0] is not None else 'n/a'
def pv(p): return 'n/a' if p is None else ('<1e-4' if p < 1e-4 else f"{p:.4f}")
L = ['# Effects of revision. Additional aggregation for Argument graph and Figure exposition (0916)\n',
     'Auto-generated tables (`code/write_results_ag.py`). Sources `results/summary/effects_ag.json`, `effects_common_ag.json`, `interactions_ag.json`, `fig_exposition_rounds.json`. Design in DESIGN.md §9. Haiku conditions only.\n',
     f"Argument graph R0 mean {E['r0_mean']}, human-pair mean {E['human_mean']}\n",
     '## 1. Argument graph (declared / key claims), mean per round (R0 → R1 → R2 → R3)\n',
     '| Condition | n(R3) | R0 | R1 | R2 | R3 | R3−R0 | 95% CI | p (Wilcoxon) | Decreased/increased papers (R3) | Drop from denominator shrinkage (R3) | Key-claim count ratio R3/R0 | At or below human pair R0→R3 |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for arm in ['a1_base', 'a2_code', 'a3_review', 'a4_slop']:
    A = E['arms'].get(arm, {}).get('rounds', {})
    if not A: continue
    it = {int(n): R['items'].get('argument_graph') for n, R in A.items()}
    r3 = it.get(3) or it.get(max(it))
    L.append(f"| {ARM[arm]} | {r3['n']} | {r3['mean_r0']:.3f} | " + ' | '.join(f"{it[n]['mean_rn']:.3f}" if it.get(n) else '–' for n in (1, 2, 3)) +
             f" | {r3['mean_diff']:+.3f} | {ci(r3['ci95'])} | {pv(r3['wilcoxon_p'])} | {r3['n_decreased']}/{r3['n_increased']} | {r3['n_masked_by_denominator_drop']} | {r3['mean_denominator_ratio']} | {r3['n_at_or_below_human_partner_r0']}→{r3['n_at_or_below_human_partner']} |")
L += ['\n## 2. Argument graph change under single-item targeting conditions (60 papers, R1). Cell = mean R1−R0 (95% CI), decreased/increased papers\n',
      '| Instructed item | n | Argument graph R1−R0 | Decreased/increased |', '|---|---|---|---|']
for arm, row in I['rows'].items():
    c = row['cells'].get('argument_graph', {})
    if c.get('mean_diff') is None: continue
    L.append(f"| {ARM.get(arm, row['target'])} | {c['n']} | {c['mean_diff']:+.3f} {ci(c['ci95'])} | {c['n_decreased']}/{c['n_increased']} |")
L += ['\n## 3. Figure exposition. Value is constant across rounds (image item); only whether the figure include is kept\n',
      f"R0: all 143 papers include framework_overview, fig_exposition mean {F['r0']['fig_exposition_mean']} (number of internalized kinds out of 6 / 6)\n",
      '| Condition | Round | n | \\includegraphics kept | Missing | Papers missing it |', '|---|---|---|---|---|---|']
for arm, A in F['arms'].items():
    for n, v in A.items():
        L.append(f"| {ARM.get(arm, arm)} | {n} | {v['n']} | {v['kept_include']} | {v['dropped_include']} | {', '.join(v['dropped_codes']) or '–'} |")
L.append('\n<!-- reading below is written by hand -->\n')
out = EOR / 'results/RESULTS_AG_0916.md'
prev = out.read_text() if out.exists() else ''
hand = prev.split('<!-- reading below is written by hand -->\n', 1)[1] if '<!-- reading below is written by hand -->' in prev else ''
out.write_text('\n'.join(L) + hand); print('\n'.join(L))
