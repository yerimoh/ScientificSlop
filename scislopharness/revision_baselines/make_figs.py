"""Figures 4 and 5 of section 5.3, drawn from results/summary/*.json.

  python3 make_figs.py [--common]     --common uses only papers finished in all four arms
Writes results/figures/{fig4_effects.pdf, fig5_interactions.pdf} and a .png beside each.
Re-runnable at any time; it draws whatever has been measured so far and prints the n it used.
"""
import argparse, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, ITEMS  # noqa: E402

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
ARM_LABEL = {'a1_base': 'Base prompting', 'a2_code': 'Claude Code',
             'a3_review': 'Reviewer refinement', 'a4_slop': 'Slop-aware'}
# One muted ramp, darkest for the arm that is told what the defects are.
ARM_COLOR = {'a1_base': '#B9B3AE', 'a2_code': '#8C9BA5', 'a3_review': '#5B7C5A', 'a4_slop': '#A23B4E'}
ITEM_LABEL = {'macro_redund': 'Recycled sentences', 'xsec_ref': 'Unused objects',
              'citation': 'Isolated citations', 'evidence_gap': 'No instance shown'}
ROUNDS = [1, 2, 3]

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def load(name):
    p = EOR / 'results/summary' / name
    return json.load(open(p)) if p.exists() else None


def fig4(E, out: Path):
    fig, axes = plt.subplots(1, len(ITEMS), figsize=(7.2, 1.85), sharex=True)
    used = {}
    for ax, item in zip(axes, ITEMS):
        r0 = E['r0_mean'].get(item)
        hu = E['human_mean'].get(item)
        for arm in ARMS:
            A = E['arms'].get(arm)
            if not A:
                continue
            xs, ys, los, his = [0], [r0], [r0], [r0]
            for n in ROUNDS:
                v = (A['rounds'].get(str(n)) or A['rounds'].get(n) or {}).get('items', {}).get(item)
                if not v:
                    continue
                xs.append(n); ys.append(v['mean_rn'])
                lo, hi = v['ci95']
                los.append(v['mean_rn'] + (lo - v['mean_diff'] if lo is not None else 0))
                his.append(v['mean_rn'] + (hi - v['mean_diff'] if hi is not None else 0))
                used[(arm, item, n)] = v['n']
            if len(xs) < 2:
                continue
            ax.fill_between(xs, los, his, color=ARM_COLOR[arm], alpha=0.13, linewidth=0)
            ax.plot(xs, ys, color=ARM_COLOR[arm], lw=1.2, marker='o', ms=2.6,
                    mew=0, label=ARM_LABEL[arm], zorder=3)
        if hu is not None:
            ax.axhline(hu, color='#444444', lw=0.7, ls=(0, (3, 2)), zorder=1)
            if item == ITEMS[-1]:
                ax.text(3.0, hu + 0.035, 'human partner', fontsize=5.8, color='#444444',
                        va='bottom', ha='right')
        ax.set_title(ITEM_LABEL[item], pad=3)
        ax.set_xticks([0] + ROUNDS); ax.set_xticklabels(['0', '1', '2', '3'])
        ax.set_xlim(-0.22, 3.22); ax.set_ylim(-0.02, 1.02)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('slop score')
    for ax in axes:
        ax.set_xlabel('revision round', labelpad=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.17), fontsize=7,
               handlelength=1.4, columnspacing=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format='pdf'); fig.savefig(out.with_suffix('.png'), dpi=220)
    plt.close(fig)
    return used


def fig5(I, out: Path):
    rows = [f'a4s_{it}' for it in ITEMS] + ['a4_slop']
    rows = [r for r in rows if r in I['rows']]
    M, N, ann = [], [], []
    for r in rows:
        row = I['rows'][r]
        M.append([(row['cells'].get(it) or {}).get('mean_diff') or 0.0 for it in ITEMS])
        N.append([(row['cells'].get(it) or {}).get('n') or 0 for it in ITEMS])
        ann.append([(row['cells'].get(it) or {}).get('mean_diff') for it in ITEMS])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    lim = max(abs(v) for r in M for v in r) or 1.0
    im = ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect='auto')
    ax.set_xticks(range(len(ITEMS)))
    ax.set_xticklabels([ITEM_LABEL[i].replace(' ', '\n', 1) for i in ITEMS], fontsize=6.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([('all four' if r == 'a4_slop' else ITEM_LABEL[r[4:]]) for r in rows], fontsize=6.6)
    ax.set_xlabel('measured', labelpad=2); ax.set_ylabel('told to fix', labelpad=2)
    for i in range(len(rows)):
        for j in range(len(ITEMS)):
            v = ann[i][j]
            if v is None:
                continue
            lab = '0.00' if abs(v) < 0.005 else f'{v:+.2f}'
            ax.text(j, i, lab, ha='center', va='center',
                    fontsize=6, color='white' if abs(v) > 0.55 * lim else '#222222')
            ax.text(j, i + 0.32, f'n={N[i][j]}', ha='center', va='center', fontsize=4.4, color='#555555')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('change in slop score after one round', fontsize=6.4)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    ax.tick_params(length=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format='pdf'); fig.savefig(out.with_suffix('.png'), dpi=220)
    plt.close(fig)
    return {r: I['rows'][r]['n_ok'] for r in rows}


SYS_LABEL = {'binoculars': 'Binoculars', 'binoculars_faithful': 'Binoculars (512 tok)',
             'fast_detectgpt': 'Fast-DetectGPT', 'nts': 'NTS', 'sciSlop4': 'SciSlop (4 items)'}


def fig4b(D, out: Path, systems=None):
    """PairAcc of each scoring system against the original human partner, by revision round,
    one panel per arm. Missing rounds are simply not drawn."""
    systems = systems or [s for s in D['systems'] if s in SYS_LABEL]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(7.2, 1.75), sharey=True)
    sys_color = {'binoculars': '#8C9BA5', 'binoculars_faithful': '#B9B3AE', 'fast_detectgpt': '#5B7C5A',
                 'nts': '#C9A66B', 'sciSlop4': '#A23B4E'}
    for ax, arm in zip(axes, ARMS):
        for sysn in systems:
            xs, ys = [0], [D['R0'].get(sysn, {}).get('pair_acc')]
            if ys[0] is None:
                continue
            for n in ('1', '3'):
                m = (D['arms'].get(arm, {}).get(n) or D['arms'].get(arm, {}).get(int(n)) or {}).get(sysn)
                if m and not m.get('missing'):
                    xs.append(int(n)); ys.append(m['pair_acc'])
            ax.plot(xs, ys, color=sys_color.get(sysn, '#333333'), lw=1.2, marker='o', ms=2.6, mew=0,
                    label=SYS_LABEL.get(sysn, sysn), zorder=3)
        ax.axhline(0.5, color='#444444', lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_title(ARM_LABEL[arm], pad=3)
        ax.set_xticks([0, 1, 3]); ax.set_xlim(-0.22, 3.22); ax.set_ylim(0.3, 1.02)
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7)
        ax.set_xlabel('revision round', labelpad=1)
    axes[0].set_ylabel('PairAcc vs human partner')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(systems), bbox_to_anchor=(0.5, -0.2), fontsize=7,
               handlelength=1.4, columnspacing=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format='pdf'); fig.savefig(out.with_suffix('.png'), dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--common', action='store_true')
    a = ap.parse_args()
    outdir = EOR / 'results/figures'
    E = load('effects_common.json' if a.common else 'effects.json')
    if E:
        u = fig4(E, outdir / ('fig4_effects_common.pdf' if a.common else 'fig4_effects.pdf'))
        ns = sorted({v for v in u.values()})
        print(f'fig4 written, per-cell n from {min(ns) if ns else 0} to {max(ns) if ns else 0}'
              + (f", common set {E.get('n_common')}" if a.common else ''))
    I = load('interactions.json')
    if I:
        print('fig5 written, n per row:', fig5(I, outdir / 'fig5_interactions.pdf'))
    D = load('detectors.json')
    if D:
        fig4b(D, outdir / 'fig4b_rescoring.pdf'); print('fig4b written')
