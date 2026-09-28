"""Figure. Share of the original-to-human gap that each condition closes after round 3, per measure.

ratio(arm, item) = (mean R0 - mean R3_arm) / (mean R0 - mean human), over the papers finished in every arm.
0 = the original manuscripts, 100 = the human mean, beyond 100 = past the human mean. The 95% band is a
bootstrap over papers (the ratio is recomputed on each resample). --norm full instead divides by the gain of the
full method, so 100 = SciSlopHarness.

  python3 fig_component_ratio.py [--norm human|full] [--out ..]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize_ablation as SA   # noqa: E402  (components set: gateonly / noloc / nogate / full)

ITEMS = SA.ITEMS
ORDER = ['macro_redund', 'citation', 'xsec_ref', 'evidence_gap', 'argument_graph', 'fig_exposition']
LABEL = {'macro_redund': 'Macro redundancy', 'citation': 'Citation isolation', 'xsec_ref': 'Cross-section references',
         'evidence_gap': 'Evidence gap', 'argument_graph': 'Argument graph', 'fig_exposition': 'Figure exposition'}
ARMS = ['gateonly', 'noloc', 'nogate', 'full']
NAME = {'gateonly': 'Review only', 'noloc': 'Definitions + review', 'nogate': 'Definitions + locations, no review',
        'full': 'SciSlopHarness (definitions + locations + review)'}
# Okabe-Ito hues for the three ablation arms (CVD-safe set), the paper's accent blue for the method, distinct marker
# shapes as the second channel. Human rule green as in Figure 3.
COLOR = {'gateonly': '#6E6E6E', 'noloc': '#56B4E9', 'nogate': '#E69F00', 'full': '#1747C9'}   # min OKLab dE x100 = 10.3 under deutan/protan/tritan simulation (checked 0922)
SHORT = {'gateonly': 'Review only', 'noloc': 'Definitions + review', 'nogate': 'Definitions + locations, no review', 'full': 'SciSlopHarness'}
MARKER = {'gateonly': 'o', 'noloc': 's', 'nogate': 'D', 'full': 'P'}
SIZE = {'gateonly': 5.2, 'noloc': 5.2, 'nogate': 5.2, 'full': 7.0}
INK, MUTED, RULE, GRID, HUMAN, PAST = '#1A1A1A', '#6B6B6B', '#C8C8C8', '#ECECEC', '#3E7A54', '#EDF3EE'
TEXTWIDTH = 5.5


def per_paper(runs, hu, codes, item):
    out = {}
    for arm in ARMS:
        out[arm] = np.array([[SA.value_at(runs[arm][c]['per_round'], item, 0), SA.value_at(runs[arm][c]['per_round'], item, 3)] for c in codes], float)
    h = np.array([SA.agg(item, (hu.get(c) or {}).get(item)) if (hu.get(c) or {}).get(item) is not None else np.nan for c in codes], float)
    return out, h


def ratio(vals, h, arm, idx, norm):
    r0 = np.nanmean(vals['full'][idx, 0]); r3 = np.nanmean(vals[arm][idx, 1])
    if norm == 'human':
        den = r0 - np.nanmean(h[idx])
    else:
        den = r0 - np.nanmean(vals['full'][idx, 1])
    return np.nan if abs(den) < 1e-9 else 100.0 * (r0 - r3) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--norm', default='human', choices=['human', 'full'])
    ap.add_argument('--out', default=str(SA.ABL))
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    FINISHED = ('round_limit', 'no_op', 'gated_noop_twice', 'no_remaining_units')
    runs = {arm: {c: r for c, r in SA.load(SA.ABL / d / 'results.jsonl').items() if r.get('stop') in FINISHED} for arm, d in SA.RESULT_DIRS.items()}
    runs['full'] = SA.load(SA.API / 'final_v14' / 'results.jsonl')
    hu = SA.S6.human_scores()
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    ops = json.load(open(SA.ABL / 'results_table.json'))['ops']
    rng = np.random.default_rng(922)
    B = 4000
    table = {}
    for item in ORDER:
        codes = [c for c in common if runs['full'][c]['per_round'].get('R0', {}).get(item) is not None]
        vals, h = per_paper(runs, hu, codes, item)
        ok = ~np.isnan(h)
        idx_all = np.where(ok)[0]
        table[item] = {}
        for arm in ARMS:
            m = ratio(vals, h, arm, idx_all, a.norm)
            bs = [ratio(vals, h, arm, rng.choice(idx_all, len(idx_all)), a.norm) for _ in range(B)]
            table[item][arm] = (m, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), int(len(idx_all)))
    # ---- draw
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8,
                         'axes.linewidth': 0.55, 'axes.edgecolor': RULE, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(TEXTWIDTH, 2.25))
    ax = fig.add_axes([0.30, 0.245, 0.68, 0.685])
    xmin, xmax = (-45, 165) if a.norm == 'human' else (-60, 320)
    ax.set_xlim(xmin, xmax)
    n = len(ORDER)
    ys = np.arange(n)[::-1]
    ax.set_ylim(-0.6, n - 0.4)
    # regions and rules
    ax.axvspan(100, xmax, color=PAST, lw=0, zorder=0)
    ax.axvline(0, color=RULE, lw=0.7, zorder=1)
    ax.axvline(100, color=HUMAN if a.norm == 'human' else COLOR['full'], lw=1.15, zorder=2)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.6, zorder=1)
    # marks with 95% whiskers, slight vertical offsets so four markers on a row do not sit on top of each other
    off = {'gateonly': 0.21, 'noloc': 0.07, 'nogate': -0.07, 'full': -0.21}
    for item, y in zip(ORDER, ys):
        for arm in ARMS:
            m, lo, hi, _ = table[item][arm]
            if np.isnan(m):
                continue
            yy = y + off[arm]
            mc, hc = min(max(m, xmin + 4), xmax - 4), min(hi, xmax - 2)
            # a 95% band wider than the axis (Argument graph: the gap to the human mean is 0.07, so the share is unstable)
            # is drawn dotted and clipped, so it reads as off-scale rather than as a measured span
            wide = (hi - lo) > (xmax - xmin)
            ax.plot([max(lo, xmin + 2), hc], [yy, yy], color=COLOR[arm], lw=0.9, alpha=0.45 if not wide else 0.35,
                    linestyle='-' if not wide else (0, (1.2, 2.2)), solid_capstyle='butt', zorder=3)
            ax.plot([mc], [yy], marker=MARKER[arm], markersize=SIZE[arm], color=COLOR[arm], markeredgecolor='white', markeredgewidth=0.7,
                    lw=0, zorder=5 if arm != 'full' else 6)
            if m > xmax - 4:      # off-scale to the right: arrow and the value
                ax.annotate('', xy=(xmax - 1, yy), xytext=(xmax - 9, yy), arrowprops=dict(arrowstyle='-|>', color=COLOR[arm], lw=0.9, shrinkA=0, shrinkB=0), zorder=6)
                ax.text(xmax - 11, yy + 0.02, f'{m:.0f}%', fontsize=6.2, color=INK, ha='right', va='center', zorder=7)
    # selective direct labels: the method's value on each row
    for item, y in zip(ORDER, ys):
        m = table[item]['full'][0]
        ax.text(m, y + off['full'] - 0.26, f'{m:.0f}%', fontsize=6.0, color=INK, ha='center', va='top', zorder=7)
    ax.set_yticks(ys); ax.set_yticklabels([LABEL[i] for i in ORDER], fontsize=7.6, color=INK)
    ax.tick_params(axis='y', length=0, pad=6)
    xt = [t for t in (-25, 0, 25, 50, 75, 100, 125, 150) if xmin <= t <= xmax] if a.norm == 'human' else [t for t in (0, 50, 100, 150, 200, 250, 300) if xmin <= t <= xmax]
    ax.set_xticks(xt); ax.set_xticklabels([f'{t:d}%' for t in xt], fontsize=6.8)
    ax.tick_params(axis='x', length=2, pad=2, color=RULE)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_bounds(xmin, xmax)
    ax.set_xlabel('Share of the gap between the original manuscripts and the human mean closed after round 3' if a.norm == 'human'
                  else 'Reduction relative to SciSlopHarness after round 3', fontsize=7.2, color=INK, labelpad=3)
    ax.text(0, n - 0.42, 'original', fontsize=6.4, color=MUTED, ha='center', va='bottom')
    ref_name, past_name = ('human mean', 'past the human mean') if a.norm == 'human' else ('SciSlopHarness', 'more than SciSlopHarness')
    ax.text(98 if a.norm == 'human' else 97, n - 0.42, ref_name, fontsize=6.4, color=HUMAN if a.norm == 'human' else COLOR['full'], ha='right', va='bottom')
    ax.text(xmax - 2, n - 0.42, past_name, fontsize=6.4, color=MUTED, ha='right', va='bottom')
    handles = [Line2D([], [], color=COLOR[arm], marker=MARKER[arm], markersize=SIZE[arm] * 0.85, lw=0, markeredgecolor='white', markeredgewidth=0.6,
                      label=f'{SHORT[arm]} ({ops[arm]["hard_guard_rounds"]})') for arm in ARMS]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=6.3, handletextpad=0.45, columnspacing=1.1,
               bbox_to_anchor=(0.5 * (0.30 + 0.98), -0.01))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or ('fig_component_ratio' if a.norm == 'human' else 'fig_component_ratio_vsfull')
    fig.savefig(out / f'{name}.pdf'); fig.savefig(out / f'{name}.png', dpi=220)
    (out / f'{name}_data.json').write_text(json.dumps({'norm': a.norm, 'n_common': len(common), 'ratio_pct': {i: {arm: list(v) for arm, v in table[i].items()} for i in ORDER}}, indent=1))
    print(out / f'{name}.pdf', '| n', len(common))
    for i in ORDER:
        print(f'{LABEL[i]:26s}', ' | '.join(f'{arm} {table[i][arm][0]:6.0f}% [{table[i][arm][1]:.0f},{table[i][arm][2]:.0f}]' for arm in ARMS))


if __name__ == '__main__':
    main()
