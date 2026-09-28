"""Summarize the component ablation: name -> +definition -> +location -> +fix direction (= shipped method, final_v14).

Reads  final_abl_{name,def,defloc}/results.jsonl (this directory) and ver1/api/final_v14/results.jsonl (full arm).
Writes RESULTS.md (tables), results_table.json (numbers), tab_component_ablation.tex (LaTeX rows only),
       fig_component_ablation.{pdf,png} (six panels, slop minus human mean per round, 95% bootstrap bands).

Conventions follow fig_harness_result.py: macro_redund on the summed scale min(1, s/0.10); a missing later round carries
the last measured value forward; human = matched human paper of each AI paper (bench165); only papers present in all
four arms enter the comparison so that every arm is measured on the same set.

  python3 summarize_ablation.py [--out ../]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ABL = HERE.parent
VER1 = ABL.parent / 'ver1'
API = VER1 / 'api'
sys.path.insert(0, str(VER1 / 'temp' / 'code')); sys.path.insert(0, str(API / 'code'))
import summarize6 as S6   # noqa: E402  human_scores()

ITEMS = ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph', 'fig_exposition']
SHORT = {'macro_redund': 'Macro\nredundancy', 'xsec_ref': 'Cross-section\nreferences', 'citation': 'Citation\nisolation',
         'evidence_gap': 'Evidence\ngap', 'argument_graph': 'Argument\ngraph', 'fig_exposition': 'Figure\nexposition'}
LONG = {'macro_redund': 'Macro redundancy', 'xsec_ref': 'Cross-section references', 'citation': 'Citation isolation',
        'evidence_gap': 'Evidence gap', 'argument_graph': 'Argument graph', 'fig_exposition': 'Figure exposition'}
import os
SET = os.environ.get('ABL_SET', 'components')
if SET == 'components':      # EXPERIMENTS_0919 §5.3 paragraph 3: remove one part of the method at a time, gate on
    ARMS = ['gateonly', 'noloc', 'nogate', 'full']
    ARM_LABELS = {'gateonly': 'Gate only (no definitions, no locations)', 'noloc': 'Definitions + gate (no locations)',
                  'nogate': 'Definitions + locations (no gate)', 'full': 'SciSlopHarness (full)'}
    ARM_GIVEN = {'gateonly': 'generic improvement instruction; reviewer on', 'noloc': 'SciSlop.md (definitions + fix); reviewer on',
                 'nogate': 'SciSlop.md + SLOP_FINDINGS.md; reviewer off', 'full': 'SciSlop.md + SLOP_FINDINGS.md; reviewer on'}
    RESULT_DIRS = {'gateonly': 'final_abl_gateonly', 'noloc': 'final_abl_noloc', 'nogate': 'final_abl_nogate'}
else:                        # skill-wording ablation (appendix), archived
    ARMS = ['name', 'def', 'defloc', 'full']
    ARM_LABELS = {'name': 'Name only', 'def': '+ Definition', 'defloc': '+ Location', 'full': '+ Direction (full)'}
    ARM_GIVEN = {'name': 'names', 'def': 'names, definitions', 'defloc': 'names, definitions, located instances',
                 'full': 'names, definitions, located instances, repair direction'}
    RESULT_DIRS = {k: f'_abandoned_skillwording/final_abl_{k}' for k in ('name', 'def', 'defloc')}


def agg(item, v):
    if v is None:
        return None
    return min(1.0, v / 0.10) if item == 'macro_redund' else v


def load(p: Path) -> dict:
    last = {}
    if not p.exists():
        return last
    for l in open(p):
        if l.strip():
            r = json.loads(l); last[r['code']] = r
    return last


def value_at(pr: dict, item: str, n: int):
    """score at round n, carrying the last measured value forward (a stopped run keeps its final manuscript)."""
    for k in range(n, -1, -1):
        v = pr.get(f'R{k}', {}).get(item)
        if v is not None:
            return agg(item, v)
    return None


def boot(vals, rng, B=4000):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return (np.nan, np.nan, np.nan, 0)
    idx = rng.integers(0, len(a), (B, len(a)))
    m = a[idx].mean(1)
    return (float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(a)))


def paired_boot(x, y, rng, B=4000):
    """mean(x - y) with bootstrap CI over papers; x, y aligned lists (None dropped pairwise)."""
    pairs = [(p, q) for p, q in zip(x, y) if p is not None and q is not None]
    if not pairs:
        return (np.nan, np.nan, np.nan, 0)
    d = np.array([p - q for p, q in pairs], float)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(d)))


def fmt(v, d=3):
    return '.' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ABL))
    ap.add_argument('--fig-name', default='fig_component_ablation')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    FINISHED = ('round_limit', 'no_op', 'gated_noop_twice', 'no_remaining_units')
    runs = {arm: {c: r for c, r in load(ABL / d / 'results.jsonl').items() if r.get('stop') in FINISHED} for arm, d in RESULT_DIRS.items()}
    runs['full'] = load(API / 'final_v14' / 'results.jsonl')
    hu = S6.human_scores()
    common = sorted(set.intersection(*[set(v) for v in runs.values()])) if all(runs.values()) else []
    present = {arm: sorted(v) for arm, v in runs.items()}
    rng = np.random.default_rng(922)
    data = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'n_common': len(common), 'common_codes': common,
            'n_per_arm': {k: len(v) for k, v in present.items()}, 'items': {}}
    md = [f'# Component ablation. Which component of the editor information carries the gain? ({time.strftime("%Y-%m-%d %H:%M")})', '',
          'Same harness as the shipped method (Haiku 4.5 per-file edit-block editor, thinking 1024; Sonnet 5 text gate reading the full '
          'SciSlop_v0.5.md and the full located-instance list in every arm; 3 rounds; location cap 8; no per-file targets). '
          'Only what the editor reads changes.', '',
          '| arm | editor receives | papers done |', '|---|---|---|']
    for arm in ARMS:
        md.append(f'| {ARM_LABELS[arm]} | {ARM_GIVEN[arm]} | {len(present[arm])} |')
    md += ['', f'Comparison set = papers finished in all four arms: **{len(common)}**. Human = matched human paper (bench165). '
           'macro_redund on the summed scale min(1, s/0.10). A stopped run carries its last manuscript forward.', '']
    # ---- per item tables
    series = {}
    for item in ITEMS:
        codes = [c for c in common if runs['full'][c]['per_round'].get('R0', {}).get(item) is not None]
        human_vals = [agg(item, (hu.get(c) or {}).get(item)) for c in codes]
        human = float(np.nanmean([v for v in human_vals if v is not None])) if any(v is not None for v in human_vals) else np.nan
        rows = {}
        per_code = {}
        for arm in ARMS:
            pts = []
            per_code[arm] = {}
            for n in range(4):
                vals = [value_at(runs[arm][c]['per_round'], item, n) for c in codes]
                per_code[arm][n] = vals
                pts.append(boot(vals, rng))
            rows[arm] = pts
        series[item] = {'human': human, 'rows': rows, 'codes': codes}
        # deltas R3 - R0 per arm, and marginal gains between consecutive arms at R3
        d30 = {arm: paired_boot(per_code[arm][3], per_code[arm][0], rng) for arm in ARMS}
        marg = {}
        for i in range(1, len(ARMS)):
            marg[ARMS[i]] = paired_boot(per_code[ARMS[i]][3], per_code[ARMS[i - 1]][3], rng)
        dist = {arm: [abs(rows[arm][n][0] - human) if not np.isnan(human) else np.nan for n in range(4)] for arm in ARMS}
        data['items'][item] = {'human_mean': human, 'n': len(codes),
                               'rounds': {arm: [list(p) for p in rows[arm]] for arm in ARMS},
                               'delta_R3_R0': {arm: list(d30[arm]) for arm in ARMS},
                               'marginal_R3_vs_previous_arm': {arm: list(v) for arm, v in marg.items()},
                               'abs_distance_to_human': dist}
        md += [f'## {LONG[item]}  (n = {len(codes)}, human mean {fmt(human)})', '',
               '| arm | R0 | R1 | R2 | R3 | R3 − R0 [95% CI] | |R3 − human| | R3 − previous arm [95% CI] |', '|---|---|---|---|---|---|---|---|']
        for arm in ARMS:
            m = [rows[arm][n][0] for n in range(4)]
            dd = d30[arm]
            mg = marg.get(arm)
            md.append(f'| {ARM_LABELS[arm]} | ' + ' | '.join(fmt(x) for x in m) +
                      f' | {fmt(dd[0], 3)} [{fmt(dd[1], 3)}, {fmt(dd[2], 3)}] | {fmt(dist[arm][3])} | ' +
                      (f'{fmt(mg[0], 3)} [{fmt(mg[1], 3)}, {fmt(mg[2], 3)}]' if mg else '') + ' |')
        md.append('')
    # ---- gate, guards, cost, time
    md += ['## Gate, guards, cost and time (per paper, comparison set)', '',
           '| arm | rounds executed | gate kept | gate reverted | kept share | hard-guard rounds | editor $ | gate $ | wall min |', '|---|---|---|---|---|---|---|---|---|']
    ops = {}
    for arm in ARMS:
        rs = [runs[arm][c] for c in common]
        kept = sum((g[1] or 0) for r in rs for g in r['gate_counts']); rev = sum((g[2] or 0) for r in rs for g in r['gate_counts'])
        hard = sum(1 for r in rs for g in r['guards'] if g[1])
        nr = np.mean([r['rounds'] for r in rs]) if rs else np.nan
        e_usd = np.mean([r['editor_usd'] for r in rs]) if rs else np.nan; g_usd = np.mean([r['gate_usd'] for r in rs]) if rs else np.nan
        wall = np.mean([r['wall_s'] for r in rs]) / 60 if rs else np.nan
        ops[arm] = {'rounds_mean': float(nr), 'kept': kept, 'reverted': rev, 'kept_share': (kept / (kept + rev)) if kept + rev else None,
                    'hard_guard_rounds': hard, 'editor_usd_mean': float(e_usd), 'gate_usd_mean': float(g_usd), 'wall_min_mean': float(wall)}
        md.append(f'| {ARM_LABELS[arm]} | {fmt(nr, 2)} | {kept} | {rev} | {fmt(ops[arm]["kept_share"], 2)} | {hard} | {fmt(e_usd, 2)} | {fmt(g_usd, 2)} | {fmt(wall, 1)} |')
    data['ops'] = ops
    md += ['', 'Notes. Figure exposition can only move when the editor returns an ERASE block whose phrases match the located list; '
           'arms without locations therefore cannot move it by construction. Argument graph needs the label server and the PMI daemon; '
           'a missing value carries the previous round forward. evidence_gap 0.5 = acknowledged gap (harness verdict), a value the human axis does not have.']
    (out / 'RESULTS.md').write_text('\n'.join(md) + '\n')
    (out / 'results_table.json').write_text(json.dumps(data, indent=1))
    # ---- LaTeX rows (numbers only; caption and prose are the author's)
    tex = ['% Generated by ablation_components/code/summarize_ablation.py on ' + time.strftime('%Y-%m-%d %H:%M') + f'. n = {len(common)} papers. Values = mean slop at R3 (R0 in the first row), macro on min(1,s/0.10).',
           '\\begin{tabular}{l' + 'c' * len(ITEMS) + '}', '\\toprule',
           'Editor receives & ' + ' & '.join(LONG[i] for i in ITEMS) + ' \\\\', '\\midrule',
           'Original (R0) & ' + ' & '.join(fmt(series[i]['rows']['full'][0][0], 3) for i in ITEMS) + ' \\\\', '\\midrule']
    for arm in ARMS:
        tex.append(f'{ARM_LABELS[arm]} & ' + ' & '.join(fmt(series[i]['rows'][arm][3][0], 3) for i in ITEMS) + ' \\\\')
    tex += ['\\midrule', 'Human mean & ' + ' & '.join(fmt(series[i]['human'], 3) for i in ITEMS) + ' \\\\', '\\bottomrule', '\\end{tabular}']
    (out / 'tab_component_ablation.tex').write_text('\n'.join(tex) + '\n')
    # ---- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, RULE, GRID, HUMAN_RULE, BELOW = '#1A1A1A', '#C8C8C8', '#E8E8E8', '#3E7A54', '#EDF3EE'
    COLORS = {'name': '#B8B8B8', 'def': '#8FB3D9', 'defloc': '#3F7FCB', 'full': '#1747C9',
              'gateonly': '#B8B8B8', 'noloc': '#8FB3D9', 'nogate': '#C4A24A'}
    MARKERS = {'name': 'o', 'def': 's', 'defloc': '^', 'full': 'P', 'gateonly': 'o', 'noloc': 's', 'nogate': 'D'}
    LW = {'name': 0.9, 'def': 1.0, 'defloc': 1.15, 'full': 1.5, 'gateonly': 0.9, 'noloc': 1.0, 'nogate': 1.05}
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8,
                         'axes.linewidth': 0.55, 'axes.edgecolor': RULE, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(5.5, 1.75))
    left, right, bottom, top = 0.075, 0.998, 0.30, 0.80
    cw = (right - left) / 6
    axes = [fig.add_axes([left + i * cw + 0.008, bottom, cw - 0.018, top - bottom]) for i in range(6)]
    lo, hi = -0.25, 0.48
    for ax, item in zip(axes, ITEMS):
        s = series[item]; human = s['human']
        ax.set_ylim(lo, hi); ax.set_xlim(-0.16, 3.4)
        ax.axhspan(lo, 0, color=BELOW, linewidth=0, zorder=0); ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.45, zorder=1); ax.axhline(0, color=HUMAN_RULE, lw=1.15, zorder=2)
        for arm in ARMS:
            arr = np.array(s['rows'][arm])[:, :3] - human
            ax.fill_between(range(4), arr[:, 1], arr[:, 2], color=COLORS[arm], alpha=0.13, linewidth=0)
            ax.plot(range(4), arr[:, 0], color=COLORS[arm], marker=MARKERS[arm], markersize=3.2 if arm != 'full' else 4.0, markevery=[1, 2, 3],
                    linewidth=LW[arm], markeredgecolor='white', markeredgewidth=0.4, zorder=3 if arm != 'full' else 6)
        ax.plot([0], [s['rows']['full'][0][0] - human], marker='o', markersize=3.2, color=INK, markerfacecolor='white', markeredgewidth=0.8, zorder=5)
        ax.set_title(SHORT[item], fontsize=7.0, pad=2.6, color=INK, linespacing=1.05)
        ax.set_xticks(range(4)); ax.tick_params(labelsize=6.4, length=1.8, pad=1.4)
        ticks = [t for t in (-0.2, 0.0, 0.2, 0.4) if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([('−%.1f' % -t) if t < 0 else ('0' if t == 0 else '+%.1f' % t) for t in ticks])
        if item != ITEMS[0]:
            ax.tick_params(labelleft=False)
        ax.spines[['top', 'right']].set_visible(False); ax.spines['bottom'].set_bounds(0, 3)
    fig.text(0.012, 0.5 * (bottom + top), 'Slop − human', rotation=90, ha='center', va='center', fontsize=6.8)
    fig.text(0.5 * (left + right), 0.135, 'Revision round', ha='center', va='center', fontsize=6.8)
    handles = [Line2D([], [], color=INK, marker='o', markersize=3.2, lw=0, markerfacecolor='white', markeredgewidth=0.8, label='Original')]
    handles += [Line2D([], [], color=COLORS[arm], marker=MARKERS[arm], markersize=3.2, lw=LW[arm], markeredgecolor='white', markeredgewidth=0.4,
                       label=ARM_LABELS[arm]) for arm in ARMS]
    handles.append(Line2D([], [], color=HUMAN_RULE, lw=1.15, label='Human mean'))
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=5.9, handlelength=1.5, columnspacing=0.9,
               handletextpad=0.5, bbox_to_anchor=(0.5 * (left + right), -0.02))
    fig.savefig(out / f'{a.fig_name}.pdf'); fig.savefig(out / f'{a.fig_name}.png', dpi=220)
    print(out / 'RESULTS.md'); print(out / f'{a.fig_name}.pdf')
    print('n common', len(common), '| per arm', data['n_per_arm'])
    for item in ITEMS:
        print(item, 'R3:', {arm: round(series[item]['rows'][arm][3][0], 3) for arm in ARMS}, 'human', round(series[item]['human'], 3))


if __name__ == '__main__':
    main()
