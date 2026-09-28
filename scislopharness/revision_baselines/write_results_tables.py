"""Rewrite the numeric sections (1-3) of results/RESULTS_0915.md from results/summary/*.json,
keeping everything from '## 4.' on. Run after aggregate.py."""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR
E=json.load(open(EOR/'results/summary/effects.json')); I=json.load(open(EOR/'results/summary/interactions.json'))
IT={'macro_redund':'Recycled sentences','xsec_ref':'Unused objects','citation':'Isolated citations','evidence_gap':'No instance shown'}
AR={'a1_base':'A1 Base prompting','a2_code':'A2 Claude Code','a3_review':'A3 Reviewer refinement','a4_slop':'A4 Slop-aware'}
L=['# Effects of revision. Final aggregation (0915)\n',
   'All four conditions × 143 papers × R1–R3 complete; single-item 4 conditions × 60 papers R1 complete. Zero no-op edits. Sources `results/summary/{effects,effects_common,interactions,detectors}.json`, figures `results/figures/`.\n',
   f"R0 mean {E['r0_mean']}\nHuman-pair mean {E['human_mean']}\n",
   '## 1. Mean slop score per item (R0 → R1 → R3), 143 papers\n',
   '| Item | Condition | R0 | R1 | R3 | R3−R0 | 95% CI | p (Wilcoxon) | Decreased/increased papers | At or below human pair R0→R3 |\n|---|---|---|---|---|---|---|---|---|---|']
for it in IT:
    for a in AR:
        r1=E['arms'][a]['rounds']['1']['items'].get(it); r3=E['arms'][a]['rounds']['3']['items'].get(it)
        if not r3: continue
        p=r3['wilcoxon_p']; p='n/a' if p is None else (f'{p:.4f}' if p>=1e-4 else '<1e-4')
        L.append(f"| {IT[it]} | {AR[a]} | {r3['mean_r0']:.3f} | {r1['mean_rn']:.3f} | {r3['mean_rn']:.3f} | {r3['mean_diff']:+.3f} | ({r3['ci95'][0]:+.3f}, {r3['ci95'][1]:+.3f}) | {p} | {r3['n_decreased']}/{r3['n_increased']} | {r3['n_at_or_below_human_partner_r0']}→{r3['n_at_or_below_human_partner']} (n={r3['n']}) |")
L+=['\n## 2. Guards (R1 / R3)\n','| Condition | No-op edits | Word-count ratio | Papers with dangling cites | Papers that dropped cited works (mean dropped) | Score drop from denominator shrinkage: macro / xsec / citation |\n|---|---|---|---|---|---|']
for a in AR:
    row=[]
    for n in ('1','3'):
        R=E['arms'][a]['rounds'][n]; m=R['items']
        row.append(f"R{n}: noop {R['n_failed_noop']}, words {R['mean_word_ratio']}, dangling {R['papers_with_new_dangling_cites']}, works dropped {R['papers_that_dropped_cited_works']} ({R['mean_cited_works_dropped']}), denominator drop {m['macro_redund']['n_masked_by_denominator_drop']}/{m['xsec_ref']['n_masked_by_denominator_drop']}/{m['citation']['n_masked_by_denominator_drop']} (citation denominator ratio {m['citation']['mean_denominator_ratio']})")
    L.append(f"| {AR[a]} | "+' <br> '.join(row)+' |')
L+=['\n## 3. Single-item targeting (60 papers, R1). Cell = mean change (95% CI)\n','| Instructed item \\ measured | '+' | '.join(IT.values())+' | Papers with target decreased | Target decreased & another item increased |\n|---|'+'---|'*6]
for k,row in I['rows'].items():
    name='all four (joint)' if row['target']=='joint' else IT[row['target']]
    cells=' | '.join(f"{row['cells'][it]['mean_diff']:+.3f} ({row['cells'][it]['ci95'][0]:+.3f}, {row['cells'][it]['ci95'][1]:+.3f})" for it in IT)
    L.append(f"| {name} | {cells} | {row.get('n_target_decreased','')} | {row.get('n_target_decreased_and_other_increased','')} |")
p=EOR/'results/RESULTS_0915.md'; old=open(p).read() if p.exists() else ''
tail=old[old.index('## 4.'):] if '## 4.' in old else ''
open(p,'w').write('\n'.join(L)+'\n\n'+tail); print('tables rewritten; tail kept:', bool(tail))
