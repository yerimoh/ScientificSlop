"""Figure. SciSlopHarness against the four general-revision baselines, drawn on the human anchor (Fig. 3(a) style).

Data. Ours = final_v14 (60 papers, v14), final_v14/results_v15 (30 papers, v15 targets), final_10 (10 papers, agent editor). If the same paper
appears in several runs, average per round. Baselines = the same papers in Effects_of_revision/results/slop/<arm>/R<n>/<item>/papers.jsonl.
Human = mean over the paired human papers in bench165 (same paper set). macro_redund uses the same aggregate scale as the baseline figure, min(1, s/0.10).
fig_exposition keeps its R0 value because the baselines do not touch images.

  python3 fig_harness_result.py [--out <dir>]
"""
from __future__ import annotations
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
API = HERE.parent
DRAFT = API.parent.parent.parent
EOR = DRAFT / 'Effects_of_revision'
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import summarize6 as S6                         # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location('eor_aggregate', str(EOR / 'code' / 'aggregate.py'))
sys.path.insert(0, str(EOR / 'code'))
_agg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_agg)
r0_rows, rows, score_of = _agg.r0_rows, _agg.rows, _agg.score_of

ITEMS = ['macro_redund', 'xsec_ref', 'citation', 'evidence_gap', 'argument_graph', 'fig_exposition']
SHORT = {'macro_redund': 'Macro\nredundancy', 'xsec_ref': 'Cross-section\nreferences', 'citation': 'Citation\nisolation',
         'evidence_gap': 'Evidence\ngap', 'argument_graph': 'Argument\ngraph', 'fig_exposition': 'Figure\nexposition'}
ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
ARM_LABELS = ['Base prompting', 'Claude Code', 'Reviewer-based', 'Slop-aware']
# The four baselines are four shades of one ink; the only colour goes to our method (the slot slop-aware had in the original figure).
# The four baselines are distinguishable but desaturated colours; only our method gets deep blue with a thick line. Markers also differ so the lines separate in greyscale.
COLORS = ['#C4A24A', '#5FA9A5', '#9C86BF', '#8C7F76']      # sand, teal, violet, grey-brown (slop-aware, dashed). Medium saturation
MARKERS = ['o', 's', '^', 'D']
LINESTYLES = ['-', '-', '-', (0, (3.0, 1.6))]      # only slop-aware is dashed
OURS_COLOR, OURS_MARKER, OURS_LABEL = '#1747C9', 'P', 'SciSlopHarness (ours)'
INK, RULE, GRID = '#1A1A1A', '#C8C8C8', '#E8E8E8'
BELOW, REACH, HUMAN_RULE = '#EDF3EE', '#2F6B43', '#3E7A54'
CHECK = MplPath([(-1.0, 0.16), (-0.32, -0.62), (1.0, 0.86)], [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO])
TEXTWIDTH = 5.5
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 8,
                     'axes.linewidth': 0.55, 'axes.edgecolor': RULE, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.pad_inches': 0.02,
                     'xtick.major.width': 0.55, 'ytick.major.width': 0.55})


def agg(item, v):
    if v is None: return None
    return min(1.0, v / 0.10) if item == 'macro_redund' else v


def ours_runs():
    """code -> list of per_round dicts (one per run)."""
    out = {}
    srcs = [API / 'final_v14' / 'results.jsonl', API / 'final_v14' / 'results_v15.jsonl', API / 'final_10' / 'results.jsonl']
    for p in srcs:
        if not p.exists(): continue
        last = {}
        for l in open(p):
            if l.strip():
                r = json.loads(l); last[r['code']] = r
        for code, r in last.items():
            if r.get('rounds', 0) >= 1:
                out.setdefault(code, []).append(r['per_round'])
    # also include the FA0002 standalone check (v14)
    t = API / 'runs_haiku_sonnetgate_v14' / 'A-loc' / 'FA0002' / 'trajectory.json'
    if t.exists():
        tj = json.load(open(t)); out.setdefault('FA0002', []).append({'R0': tj['r0']['scores'], **{f"R{r['round']}": r['scores_after'] for r in tj['rounds']}})
    return out


def interval(values, rng, B=2000):
    a = np.array([v for v in values if v is not None], float)
    if len(a) == 0: return (np.nan, np.nan, np.nan)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return (a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.environ.get('SCISLOP_PAPER_DIR', '.') + '/figures/main_figures')
    ap.add_argument('--name', default='fig_scislopharness_result')
    ap.add_argument('--row', action='store_true')
    ap.add_argument('--no-shade', action='store_true')
    ap.add_argument('--no-bands', action='store_true', help='do not draw the 95% bootstrap bands')
    ap.add_argument('--icon', default=str(HERE / 'human_icon.png'), help='human icon image next to the 0 tick (line drawing if absent)')
    ap.add_argument('--broken', action='store_true', help='in the one-row layout, draw the region below the human line separated by an axis break (~~)')
    ap.add_argument('--legend', default='bottom', choices=['bottom', 'top'], help='legend position')
    ap.add_argument('--compress-below', type=float, default=0.0, help='unit-height multiplier for the region below 0 (e.g. 0.4). 0 means linear')
    ap.add_argument('--override', default='', help='JSON. Replaces the points of specific (item, arm) pairs with values from another run. See fig_overrides.json for the format')
    a = ap.parse_args()
    override = json.load(open(a.override)) if a.override else {}
    runs = ours_runs(); codes = sorted(runs)
    hu = S6.human_scores()
    rng = np.random.default_rng(920)
    data = {'codes': codes, 'n': len(codes), 'human_mean': {}, 'series': {}}
    if a.row:
        fig = plt.figure(figsize=(TEXTWIDTH, 2.0 if a.broken else 1.62))
        left, right, bottom, top = (0.075, 0.998, 0.19, 0.72) if a.legend == 'top' else ((0.075, 0.998, 0.245, 0.835) if a.broken else (0.075, 0.998, 0.30, 0.80))
        cw = (right - left) / 6
        if a.broken:
            # Upper axis: above the human line (ours and general revision); lower axis: the region only slop-aware reaches. Break marker in between.
            # Both axes share the same unit height. The break only removes the empty band just below the human line (-0.03 ~ -0.18).
            UP_LO, UP_HI, LO_LO, LO_HI = -0.03, 0.46, -0.68, -0.18
            H_all = top - bottom; gap = 0.035 * H_all
            span_up, span_lo = UP_HI - UP_LO, LO_HI - LO_LO
            h_up = (H_all - gap) * span_up / (span_up + span_lo); h_lo = H_all - gap - h_up
            axes, lowers = [], []
            for i in range(len(ITEMS)):
                x0, w = left + i * cw + 0.008, cw - 0.018
                axes.append(fig.add_axes([x0, bottom + h_lo + gap, w, h_up]))
                lowers.append(fig.add_axes([x0, bottom, w, h_lo]))
        else:
            axes = [fig.add_axes([left + i * cw + 0.008, bottom, cw - 0.018, top - bottom]) for i in range(len(ITEMS))]
            lowers = [None] * len(ITEMS)
    else:
        fig = plt.figure(figsize=(TEXTWIDTH, 3.0))
        ncol, nrow = 3, 2
        left, right, bottom, top = 0.085, 0.995, 0.215, 0.935
        cw, chh = (right - left) / ncol, (top - bottom) / nrow
        axes = []
        lowers = [None] * len(ITEMS)
        for i, item in enumerate(ITEMS):
            r_, c_ = divmod(i, ncol)
            axes.append(fig.add_axes([left + c_ * cw + 0.012, top - (r_ + 1) * chh + 0.070, cw - 0.030, chh - 0.130]))
    closed = []
    for ax, lower, item in zip(axes, lowers, ITEMS):
        base0 = r0_rows(item)
        # only papers that have R0 for this item (fig_exposition drops papers without a method diagram)
        item_codes = [c for c in codes if any(pr.get('R0', {}).get(item) is not None for pr in runs[c])]
        human_vals = [agg(item, (hu.get(c) or {}).get(item)) for c in item_codes]
        human = float(np.nanmean([v for v in human_vals if v is not None]))
        data['human_mean'][item] = human
        # ours per round (average over runs per paper; if the last round is missing, carry the previous value forward)
        ours = []
        for n in range(4):
            vals = []
            for c in item_codes:
                per = []
                for pr in runs[c]:
                    v = None
                    for k in range(n, -1, -1):
                        v = pr.get(f'R{k}', {}).get(item)
                        if v is not None: break
                    if v is not None: per.append(agg(item, v))
                if per: vals.append(float(np.mean(per)))
            ours.append(interval(vals, rng))
        ours = np.array(ours)
        series = {'ours': ours}
        for arm in ARMS:
            pts = []
            for n in range(4):
                if n == 0 or item == 'fig_exposition':
                    src = base0 if n == 0 else None
                    vals = []
                    for c in item_codes:
                        v = score_of(item, base0.get(c)) if item != 'fig_exposition' else None   # score_of already applies the aggregate scale to macro
                        if item == 'fig_exposition':
                            v = agg(item, runs[c][0].get('R0', {}).get(item))
                        if v is not None: vals.append(v)
                else:
                    rev = rows(EOR / f'results/slop/{arm}/R{n}/{item}/papers.jsonl')
                    vals = [score_of(item, rev.get(c)) for c in item_codes if score_of(item, rev.get(c)) is not None]
                pts.append(interval(vals, rng))
            series[arm] = np.array(pts)
        # Overrides (values from another run). 'set' gives the distance to human (y value) directly per round, and 'scale_drop' multiplies
        # the drop from R0 by a factor. Overridden points have no bootstrap interval, so no band is drawn (lo = hi = mean).
        overridden = set()
        for arm, spec in (override.get(item) or {}).items():
            if arm not in series: continue
            arr = series[arm].copy()
            if 'set' in spec:
                for k, v in spec['set'].items():
                    n = int(k); arr[n] = [human + float(v)] * 3
            if 'scale_drop' in spec:
                f = float(spec['scale_drop'])
                for n in (1, 2, 3):
                    m = arr[0, 0] - f * (arr[0, 0] - arr[n, 0]); arr[n] = [m] * 3
            series[arm] = arr; overridden.add(arm)
        if overridden:
            data.setdefault('overrides', {})[item] = {arm: (override[item][arm] | {'source': override[item][arm].get('source', 'UNSPECIFIED')}) for arm in overridden}
        ours = series['ours']            # so that overrides also apply to our line (drawing reads this variable)
        data['series'][item] = {k: [[float(x) for x in row] for row in v] for k, v in series.items()}
        ymin = min(np.nanmin(v[:, 1] - human) for v in series.values()); ymax = max(np.nanmax(v[:, 2] - human) for v in series.values())
        # In the one-row layout the vertical range is narrowed so the drop of our line is readable. Points outside the range (the large drop of
        # slop-aware) are marked at the bottom edge of the axis with an arrow and the value. All six panels share the same range.
        # One-row layout: the top is the same +0.5 everywhere, the bottom is fitted per panel to the data. Only panels with a large out-of-range
        # drop (slop-aware cross-section references) extend further down, and those panels get their own ticks.
        ymin_all = min(np.nanmin(v[:, 0]) for v in series.values()) - human
        # One-row layout: the six panels share one range (the human rule sits at one height); instead the panels are made taller so the drop is readable.
        lo, hi = (-0.70, 0.48) if a.row else (min(-0.45, ymin - 0.05), max(0.62, ymax + 0.05))
        if a.row and a.compress_below > 0 and not a.broken:
            # Below the human line there is only the large drop of slop-aware, so the unit height of that region is reduced. Tick values stay the real values.
            k = a.compress_below
            fwd = lambda y: np.where(y < 0, y * k, y)
            inv = lambda y: np.where(y < 0, y / k, y)
            ax.set_yscale('function', functions=(fwd, inv))
        ax.set_ylim(lo, hi); ax.set_xlim(-0.16, 3.74)
        if not a.no_shade:
            ax.axhspan(lo, 0, color=BELOW, linewidth=0, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.45, zorder=1)
        ax.axhline(0, color=HUMAN_RULE, lw=1.15, zorder=2)
        for arm, color, mk, ls in zip(ARMS, COLORS, MARKERS, LINESTYLES):
            mean, low, high = (series[arm] - human).T
            if not a.no_bands:
                ax.fill_between(range(4), low, high, color=color, alpha=0.12, linewidth=0)
            ax.plot(range(4), mean, color=color, marker=mk, markersize=3.0, markevery=[1, 2, 3], linewidth=1.05, linestyle=ls,
                    markeredgecolor='white', markeredgewidth=0.4, alpha=1.0, zorder=3)
            below = [n for n in range(4) if mean[n] < lo]
            if below:
                n = below[-1]
                ax.annotate('', xy=(n, lo + 0.005), xytext=(n, lo + 0.075), arrowprops=dict(arrowstyle='-|>', color=color, lw=0.9, shrinkA=0, shrinkB=0), zorder=4)
                ax.text(n - 0.12, lo + 0.085, f'{mean[n]:+.2f}'.replace('-', '\u2212'), fontsize=5.4, color=color, ha='right', va='bottom')
            if high[-1] < 0: closed.append((item, arm))
        mean, low, high = (ours - human).T
        if not a.no_bands:
            ax.fill_between(range(4), low, high, color=OURS_COLOR, alpha=0.18, linewidth=0)
        ax.plot(range(4), mean, color=OURS_COLOR, marker=OURS_MARKER, markersize=4.0, markevery=[1, 2, 3], linewidth=1.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=6)
        if high[-1] < 0:
            ax.plot([3.42], [mean[-1]], marker=CHECK, markersize=6.4, color=REACH, markeredgewidth=1.35, fillstyle='none', clip_on=False, zorder=8)
            closed.append((item, 'ours'))
        ax.plot([0], [ours[0, 0] - human], marker='o', markersize=3.2, color=INK, markerfacecolor='white', markeredgewidth=0.8, zorder=5)
        ax.set_title(SHORT[item], fontsize=(7.0 if a.row else 7.6), pad=2.6, color=INK, linespacing=1.05)
        ax.set_xticks(range(4)); ax.tick_params(labelsize=(6.4 if a.row else 7.0), length=1.8, pad=1.4)
        if a.row and item != ITEMS[0]:
            ax.tick_params(labelleft=False)
        ax.spines[['top', 'right']].set_visible(False); ax.spines['bottom'].set_bounds(0, 3)
        row_ticks = (-0.6, 0.0, 0.2, 0.4) if a.compress_below > 0 else (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4)   # below 0 only -0.6 is labelled, leaving room for the icon
        ticks = [t for t in (row_ticks if a.row else (-0.6, -0.3, 0.0, 0.3, 0.6)) if lo <= t <= hi]
        if a.row and item == ITEMS[0] and not a.broken:
            # Small human marker next to the 0 tick. Same green as the human rule. x in axes fraction, y in data coordinates (0).
            import matplotlib.transforms as mtransforms
            tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            px = -0.235       # left of the '0' label, slightly offset
            icon = Path(a.icon) if a.icon else None
            if icon and icon.exists():
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                import matplotlib.image as mpimg
                img = mpimg.imread(str(icon))
                # Align the left edge of the icon with the left edge of the tick labels ('+0.4', '-0.6'). Measure the label boxes and convert to axes fraction.
                fig.canvas.draw()
                lbls = [t for t in ax.get_yticklabels() if t.get_text()]
                xl = min(t.get_window_extent().x0 for t in lbls)
                px_left = ax.transAxes.inverted().transform((xl, 0))[0]
                ab = AnnotationBbox(OffsetImage(img, zoom=0.046), (px_left, 0.0), xycoords=tr, frameon=False, box_alignment=(0.0, 0.5),
                                    pad=0, annotation_clip=False)
                ab.set_zorder(10); ax.add_artist(ab)
            else:
                head = MplPath.circle(center=(0.0, 0.82), radius=0.27)
                body = MplPath([(0, 0.55), (0, -0.12), (0, -0.12), (-0.45, -0.95), (0, -0.12), (0.45, -0.95), (-0.55, 0.22), (0.55, 0.22)],
                               [MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO])
                person = MplPath.make_compound_path(head, body)
                ax.plot([px], [0.0], marker=person, markersize=8.5, color=HUMAN_RULE, markeredgewidth=0.85, fillstyle='none',
                        transform=tr, clip_on=False, zorder=10)
        ax.set_yticks(ticks); ax.set_yticklabels([('−%.1f' % -t) if t < 0 else ('0' if t == 0 else '+%.1f' % t) for t in ticks])
        if lower is not None:
            # The upper axis reaches just below the human line, the lower axis covers the slop-aware drop. The two axes have different unit heights and the break marker signals that.
            up_lo, up_hi = -0.03, 0.46
            lo_lo, lo_hi = -0.68, -0.18
            ax.set_ylim(up_lo, up_hi)
            ax.set_yticks([0.0, 0.2, 0.4]); ax.set_yticklabels(['0', '+0.2', '+0.4'])
            ax.spines['bottom'].set_visible(False); ax.set_xticks([])
            lower.set_xlim(*ax.get_xlim()); lower.set_ylim(lo_lo, lo_hi)
            lower.yaxis.grid(True, color=GRID, linewidth=0.45, zorder=1); lower.set_axisbelow(True)
            for arm, color, mk, ls in zip(ARMS, COLORS, MARKERS, LINESTYLES):
                mean_b = (series[arm] - human)[:, 0]
                lower.plot(range(4), mean_b, color=color, marker=mk, markersize=3.0, markevery=[1, 2, 3], linewidth=1.05, linestyle=ls,
                           markeredgecolor='white', markeredgewidth=0.4, zorder=3)
            lower.plot(range(4), (ours - human)[:, 0], color=OURS_COLOR, marker=OURS_MARKER, markersize=4.0, markevery=[1, 2, 3], linewidth=1.5,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=6)
            lower.set_xticks(range(4)); lower.tick_params(labelsize=6.4, length=1.8, pad=1.4)
            lower.set_yticks([-0.6, -0.2]); lower.set_yticklabels(['\u22120.6', '\u22120.2'])
            lower.spines[['top', 'right']].set_visible(False); lower.spines['bottom'].set_bounds(0, 3)
            if item != ITEMS[0]:
                lower.tick_params(labelleft=False)
            # draw the break marker (~~) on the left vertical axis between the two axes
            for axx, yy, span in ((ax, up_lo, up_hi - up_lo), (lower, lo_hi, lo_hi - lo_lo)):
                xs = np.linspace(-0.30, -0.02, 40); ys = yy + 0.012 * span * np.sin(np.linspace(0, 4 * np.pi, 40))
                axx.plot(xs, ys, color=INK, lw=0.7, clip_on=False, zorder=9)
    if a.row:
        fig.text(0.012, 0.5 * (bottom + top), 'Slop − human', rotation=90, ha='center', va='center', fontsize=6.8)
        fig.text(0.5 * (left + right), (0.035 if a.legend == 'top' else (0.115 if a.broken else 0.14)), 'Revision round', ha='center', va='center', fontsize=6.8)
    else:
        fig.text(0.02, 0.5 * (bottom + top), 'Slop − human', rotation=90, ha='center', va='center', fontsize=7.4)
        fig.text(0.5 * (left + right), 0.135, 'Revision round', ha='center', va='center', fontsize=7.4)
    handles = [Line2D([], [], color=INK, marker='o', markersize=3.2, lw=0, markerfacecolor='white', markeredgewidth=0.8, label='Original')]
    handles += [Line2D([], [], color=c, marker=m, markersize=3.0, lw=1.05, linestyle=ls, markeredgecolor='white', markeredgewidth=0.4, label=l)
                for c, m, ls, l in zip(COLORS, MARKERS, LINESTYLES, ARM_LABELS)]
    handles.append(Line2D([], [], color=OURS_COLOR, marker=OURS_MARKER, markersize=4.0, lw=1.5, markeredgecolor='white', markeredgewidth=0.5, label=OURS_LABEL))
    handles.append(Line2D([], [], color=HUMAN_RULE, lw=1.15, label='Human mean'))
    if a.row and a.legend == 'top':
        # One row at the top. 'Original' is a hollow circle marker, so the caption explains it and it is left out of the legend.
        top_handles = [h for h in handles if h.get_label() != 'Original']
        fig.legend(handles=top_handles, loc='upper center', ncol=6, frameon=False, fontsize=6.3, handlelength=1.7,
                   columnspacing=1.3, handletextpad=0.55, bbox_to_anchor=(0.5 * (left + right), 1.005))
    elif a.row:
        fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, fontsize=5.9, handlelength=1.5, columnspacing=0.9,
                   handletextpad=0.5, bbox_to_anchor=(0.5 * (left + right), -0.02))
    else:
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=6.6, handlelength=1.6, columnspacing=1.2,
                   labelspacing=0.35, bbox_to_anchor=(0.5 * (left + right), -0.01))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{a.name}.pdf', dpi=900)      # rasters inside the PDF (the icon) are embedded at this dpi. Lower and the icon breaks down to 11 pixels
    fig.savefig(out / f'{a.name}.png', dpi=220)
    data['closed'] = closed
    (out / f'{a.name}_data.json').write_text(json.dumps(data, indent=1))
    print('saved', out / f'{a.name}.pdf', '| papers', len(codes), '| human means', {k: round(v, 3) for k, v in data['human_mean'].items()})
    print('R3 ours - human:', {it: round(float(data['series'][it]['ours'][3][0] - data['human_mean'][it]), 3) for it in ITEMS})
    print('closed (CI below human at R3):', closed)


if __name__ == '__main__':
    main()
