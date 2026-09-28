"""Render the revision figures inserted in the ICLR manuscript.

Run: MPLCONFIGDIR=/tmp/scislop-mpl python3 make_paper_figures.py
Uses the completed Haiku experiment, excluding mitigate and Sonnet runs.
Preserves the saved score convention (macro_redund uses slop_score_agg).
Effect bands are bootstrap intervals for means, computed from paper scores;
interaction intervals are the saved paired-bootstrap intervals for changes.

Argument graph joins both figures (DESIGN.md section 9). It is measured on the
same round trees but was never named in any feedback. The compact manuscript pair
shows revision trajectories, then compares single-versus-joint reductions with bars
and other-feedback reductions with individual points, all on the same scale.
Figures are authored at the 5.5in ICLR text width, so declared point sizes are the
printed point sizes. Fixed canvas widths preserve the intended type size in LaTeX.

Outputs
  fig_revision_effects        rounds x item, one float
  fig_revision_interactions   single-item vs joint matrix, one float
  fig_revision_combined       the two side by side in a single row, one float
  fig_revision_*_compact      the compact manuscript pair (PDF and PNG)
  fig_revision_effects_anchored  the effects figure drawn on the human mean, full width
  fig_revision_*_row          the manuscript pair, one row, (a) effects and (b) interactions
  fig_revision_pair_compact   a PNG preview of the pair, without captions
"""
import os
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
import numpy as np

from aggregate import r0_rows, rows, score_of

EOR = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("SCISLOP_PAPER_DIR", "paper"))   # the paper's LaTeX folder (figures/, tables/)
OUT = PAPER / "figures/main_figures"
TEXTWIDTH = 5.5                      # ICLR \textwidth, in inches

DET_ITEMS = ["macro_redund", "xsec_ref", "citation", "evidence_gap"]
AG_ITEM = "argument_graph"
ITEMS = DET_ITEMS + [AG_ITEM]
LABEL = {"macro_redund": "Macro redundancy", "xsec_ref": "Cross-section references",
         "citation": "Citation isolation", "evidence_gap": "Evidence gap",
         "argument_graph": "Argument graph"}
SHORT = {"macro_redund": "Macro\nredundancy", "xsec_ref": "Cross-section\nreferences",
         "citation": "Citation\nisolation", "evidence_gap": "Evidence\ngap",
         "argument_graph": "Argument\ngraph"}

ARMS = ["a1_base", "a2_code", "a3_review", "a4_slop"]
ARM_LABELS = ["Base prompting", "Claude Code", "Reviewer-based refinement", "Slop-aware"]
# The three general conditions are one family, so they are one ink at three lightnesses; the
# condition that is told what the items are gets the only hue. Keeps the plane colours free to
# mean planes, which is what they mean in Table 1.
COLORS = ["#B3ACA3", "#8D857C", "#4F4943", "#8E3B4E"]
MARKERS = ["o", "s", "^", "D"]
INK = "#1A1A1A"
RULE = "#C8C8C8"
GRID = "#E8E8E8"
HUMAN = "#2A2A2A"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.55, "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "savefig.pad_inches": 0.02,
})

# 0917: one open-source editor (Qwen2.5-32B-Instruct, same base-prompting protocol, R1 only).
OPEN = {"arm": "a1_base.qwen32b", "label": "Qwen2.5-32B (open)",
        "color": "#D09030", "marker": "v", "rounds": (1,)}


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _expected():
    """Saved per-round means, keyed by item, from the deterministic and the AG aggregates."""
    det = json.loads((EOR / "results/summary/effects_common.json").read_text())
    ag = json.loads((EOR / "results/summary/effects_common_ag.json").read_text())
    assert det["common_codes"] == ag["common_codes"]
    assert det["n_common"] == ag["n_common"] == 143
    merged = {"codes": det["common_codes"], "r0_mean": {}, "human_mean": {}, "arms": {}}
    for src in (det, ag):
        merged["r0_mean"].update(src["r0_mean"])
        merged["human_mean"].update(src["human_mean"])
        for arm, body in src["arms"].items():
            slot = merged["arms"].setdefault(arm, {})
            for rnd, payload in body["rounds"].items():
                slot.setdefault(rnd, {}).update(payload["items"])
    return merged


def _interaction_cells():
    det = json.loads((EOR / "results/summary/interactions.json").read_text())
    ag = json.loads((EOR / "results/summary/interactions_ag.json").read_text())
    arms = [f"a4s_{item}" for item in DET_ITEMS] + ["a4_slop"]
    cells = {}
    for arm in arms:
        merged = dict(det["rows"][arm]["cells"])
        merged.update(ag["rows"][arm]["cells"])
        cells[arm] = merged
    assert all(cells[a][i]["n"] == 60 for a in arms for i in ITEMS)
    return arms, cells


# ----------------------------------------------------------------------------- effects panels

def _draw_effects(fig, axes, legend_ax, extra=None, tag_text=LABEL, tag_fs=6.0,
                  tick_fs=6.6, label_fs=7.2, legend_fs=6.6, marker=3.0, lw=1.2,
                  xlabel_gap=0.115, ylabel_at=(0, 3), xlabel_span=(3, 4),
                  legend_ncol=1, show_xlabel=True):
    data = _expected()
    codes = data["codes"]
    rng = np.random.default_rng(914)
    for ax, item in zip(axes, ITEMS):
        baseline = r0_rows(item)
        original = [score_of(item, baseline.get(c)) for c in codes]
        original = [v for v in original if v is not None]
        first = interval(original, rng)
        assert abs(first[0] - data["r0_mean"][item]) < 0.00006
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which="both", color=GRID, linewidth=0.45)
        for arm, color, mk in zip(ARMS, COLORS, MARKERS):
            points = [first]
            for n in (1, 2, 3):
                revised = rows(EOR / f"results/slop/{arm}/R{n}/{item}/papers.jsonl")
                values = [score_of(item, revised.get(c)) for c in codes
                          if score_of(item, baseline.get(c)) is not None
                          and score_of(item, revised.get(c)) is not None]
                expected = data["arms"][arm][str(n)][item]
                point = interval(values, rng)
                assert len(values) == expected["n"]
                assert abs(point[0] - expected["mean_rn"]) < 0.00006
                points.append(point)
            mean, low, high = np.array(points).T
            ax.fill_between(range(4), low, high, color=color, alpha=0.13, linewidth=0)
            ax.plot(range(4), mean, color=color, marker=mk, markersize=marker,
                    linewidth=lw, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        if extra:
            pts = [first]
            for n in extra["rounds"]:
                revised = rows(EOR / f"results/slop/{extra['arm']}/R{n}/{item}/papers.jsonl")
                values = [score_of(item, revised.get(c)) for c in codes
                          if score_of(item, baseline.get(c)) is not None
                          and score_of(item, revised.get(c)) is not None]
                pts.append(interval(values, rng))
            mean, low, high = np.array(pts).T
            xs = list(range(len(pts)))
            ax.fill_between(xs, low, high, color=extra["color"], alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=extra["color"], marker=extra["marker"], markersize=marker + 0.2,
                    linewidth=lw, linestyle=(0, (2.4, 1.6)), markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)
        ax.axhline(data["human_mean"][item], color=HUMAN, linestyle=(0, (3.2, 2)), lw=0.85, zorder=2)
        ax.set_title(tag_text[item], fontsize=tag_fs, pad=2.4, color=INK, linespacing=1.05)
        ax.set_xticks(range(4))
        ax.set_xlim(-0.14, 3.14)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticks([0.25, 0.75], minor=True)
        ax.tick_params(labelsize=tick_fs, length=1.8, pad=1.2)
        ax.tick_params(which="minor", length=0)
        ax.spines[["top", "right"]].set_visible(False)
    for i, ax in enumerate(axes):
        if i not in ylabel_at:
            ax.tick_params(labelleft=False)
    for i in ylabel_at:
        axes[i].set_yticklabels(["0", ".5", "1"])
        axes[i].set_ylabel("Slop score", fontsize=label_fs, labelpad=1.5)
    box, last = axes[xlabel_span[0]].get_position(), axes[xlabel_span[1]].get_position()
    if show_xlabel:
        fig.text(0.5 * (box.x0 + last.x1), box.y0 - xlabel_gap,
                 "Revision round", ha="center", fontsize=label_fs)

    handles = [Line2D([], [], color=c, marker=m, markersize=marker, lw=lw,
                      markeredgecolor="white", markeredgewidth=0.4, label=l)
               for c, m, l in zip(COLORS, MARKERS, ARM_LABELS)]
    if extra:
        handles.append(Line2D([], [], color=extra["color"], marker=extra["marker"],
                              markersize=marker + 0.2, lw=lw, linestyle=(0, (2.4, 1.6)),
                              markeredgecolor="white", markeredgewidth=0.4, label=extra["label"]))
    handles.append(Line2D([], [], color=HUMAN, linestyle=(0, (3.2, 2)), lw=0.85, label="Human mean"))
    legend_ax.legend(handles=handles, loc="center" if legend_ncol > 1 else "center left",
                     bbox_to_anchor=(0.5, 0.5) if legend_ncol > 1 else (-0.02, 0.52),
                     ncol=legend_ncol, frameon=False, fontsize=legend_fs, columnspacing=1.0,
                     labelspacing=0.42 if not extra else 0.26,
                     handlelength=1.6, handletextpad=0.45, borderpad=0)


def effects(extra=None, name="fig_revision_effects"):
    # Two rows of three. Five panels and the legend fill all six cells, which keeps the panels
    # near square without the wasted half-row that a 4 + 1 split leaves behind.
    fig = plt.figure(figsize=(TEXTWIDTH, 2.05))
    grid = fig.add_gridspec(2, 3, left=0.076, right=0.997, top=0.895, bottom=0.155,
                            wspace=0.11, hspace=0.44)
    axes = [fig.add_subplot(grid[r, c]) for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    legend_ax = fig.add_subplot(grid[1, 2])
    legend_ax.axis("off")
    _draw_effects(fig, axes, legend_ax, extra=extra)
    save(fig, name)


# ------------------------------------------------------------------------ interaction matrix

DIVERGING = LinearSegmentedColormap.from_list(   # artifacts slate down, argument rose up
    "slopdiv", ["#2E4A55", "#6D8A96", "#B6C8CF", "#FFFFFF", "#F4CFD3", "#E4959E", "#A8515C"])


def _draw_interactions(fig, ax, cax=None, val_fs=7.4, ci_fs=5.5, tick_fs=6.2, label_fs=7.2,
                       show_ci=True, ci_short=False):
    arms, cells = _interaction_cells()
    matrix = np.array([[cells[a][i]["mean_diff"] for i in ITEMS] for a in arms])
    ax.set_facecolor("#F2F2F0")
    im = ax.imshow(matrix, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8),
                   aspect="auto")
    for row in range(5):                                # keep near-white cells visible as cells
        for col in range(5):
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor="#DCDAD6", linewidth=0.7, zorder=4))
    ax.set_xticks(range(5), [SHORT[i] for i in ITEMS], fontsize=tick_fs, linespacing=1.05)
    ax.set_yticks(range(5), [SHORT[i] for i in DET_ITEMS] + ["All four\nitems"],
                  fontsize=tick_fs, linespacing=1.05)
    ax.tick_params(length=0, pad=3.0)
    ax.set_xlabel("Measured item", fontsize=label_fs, labelpad=2)
    ax.set_ylabel("Feedback provided", fontsize=label_fs, labelpad=2)

    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.axhline(3.5, color=INK, linewidth=1.0, zorder=6)     # joint row below the single-item rows
    ax.axvline(3.5, color=INK, linewidth=1.0, zorder=6)     # item no feedback ever named, at right

    for row, arm in enumerate(arms):
        for col, item in enumerate(ITEMS):
            cell = cells[arm][item]
            v, (lo, hi) = cell["mean_diff"], cell["ci95"]
            rgb = np.asarray(im.cmap(im.norm(v))[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            light = np.dot(linear, [0.2126, 0.7152, 0.0722]) < 0.179
            solid = lo < 0 and hi < 0 or lo > 0 and hi > 0   # interval clear of no change
            ax.text(col, row - (0.15 if show_ci else 0.0),
                    f"{v:+.3f}".replace("+0.000", "0.000"),
                    ha="center", va="center", fontsize=val_fs,
                    fontweight="bold" if (solid and not show_ci) else "normal",
                    color="white" if light else INK)
            if show_ci:
                txt = (f"[{lo:+.2f}, {hi:+.2f}]".replace("0.", ".").replace("+", "")
                       if ci_short else f"[{lo:.3f}, {hi:.3f}]")
                ax.text(col, row + 0.24, txt, ha="center", va="center",
                        fontsize=ci_fs, color="#EDEDED" if light else "#5A5A5A")
    for i in range(4):                                  # the cell whose item the row named
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=INK, linewidth=1.1, zorder=5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    if cax is not None:
        cb = fig.colorbar(im, cax=cax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
        cb.set_label("Change in slop score", fontsize=6.4, labelpad=2.5)
        cb.ax.tick_params(labelsize=6.2, length=1.5, width=0.5)
        cb.ax.set_yticklabels(["-0.8", "-0.4", "0", "+0.4", "+0.8"])
        cb.outline.set_linewidth(0.4)
        cb.outline.set_edgecolor(RULE)


def interactions():
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.10))
    fig.subplots_adjust(left=0.165, right=0.885, top=0.99, bottom=0.205)
    cax = fig.add_axes([0.902, 0.26, 0.013, 0.58])
    _draw_interactions(fig, ax, cax)
    save(fig, "fig_revision_interactions")


# ----------------------------------------------------------------------------- the two in a row

def combined(extra=None, name="fig_revision_combined"):
    """Both panels of the revision result in a single float, side by side.

    The matrix loses its printed intervals at this width. It keeps them as weight, a value in
    bold being one whose 95% interval is clear of no change, which the caption states.
    """
    fig = plt.figure(figsize=(TEXTWIDTH, 2.42))
    left = fig.add_gridspec(2, 3, left=0.048, right=0.472, top=0.885, bottom=0.165,
                            wspace=0.13, hspace=0.50)
    axes = [fig.add_subplot(left[r, c]) for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    legend_ax = fig.add_subplot(left[1, 2])
    legend_ax.axis("off")
    _draw_effects(fig, axes, legend_ax, extra=extra, tag_text=SHORT, tag_fs=4.9,
                  tick_fs=5.8, label_fs=6.6, legend_fs=5.5, marker=2.4, lw=1.0,
                  xlabel_gap=0.125)

    ax = fig.add_axes([0.632, 0.165, 0.318, 0.72])
    cax = fig.add_axes([0.962, 0.22, 0.011, 0.58])
    _draw_interactions(fig, ax, cax, val_fs=6.0, tick_fs=5.2, label_fs=6.6, show_ci=False)

    fig.text(0.255, 0.975, "(a) Effects of revision", ha="center", fontsize=7.0)
    fig.text(0.760, 0.975, "(b) Single-item vs joint feedback", ha="center", fontsize=7.0)
    save(fig, name)


# ------------------------------------------------- the two floats sharing one row of the page

HALF_L, HALF_R, HALF_H = 2.42, 2.97, 2.70      # 0.44 and 0.54 of \textwidth, same height


def effects_half(extra=None, name="fig_revision_effects_half"):
    """Figure 4 authored for the left slot of a side-by-side row. Three rows of two, with the
    legend in the sixth cell, since two columns is what 2.42in can carry."""
    fig = plt.figure(figsize=(HALF_L, HALF_H))
    grid = fig.add_gridspec(3, 2, left=0.155, right=0.995, top=0.945, bottom=0.115,
                            wspace=0.13, hspace=0.62)
    axes = [fig.add_subplot(grid[r, c]) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    legend_ax = fig.add_subplot(grid[2, 1])
    legend_ax.axis("off")
    _draw_effects(fig, axes, legend_ax, extra=extra, tag_fs=6.2, tick_fs=6.0, label_fs=6.8,
                  legend_fs=5.8, marker=2.6, lw=1.1, xlabel_gap=0.085,
                  ylabel_at=(0, 2, 4), xlabel_span=(4, 4))
    save(fig, name)


def interactions_half(name="fig_revision_interactions_half"):
    """Figure 5 authored for the right slot. A cell is too narrow here for a printed interval,
    and two decimals would render the small ones as point intervals, so the interval becomes
    weight instead. A value in bold is one whose 95% interval is clear of no change."""
    fig, ax = plt.subplots(figsize=(HALF_R, HALF_H))
    fig.subplots_adjust(left=0.205, right=0.905, top=0.99, bottom=0.145)
    cax = fig.add_axes([0.925, 0.20, 0.019, 0.60])
    _draw_interactions(fig, ax, cax, val_fs=6.8, tick_fs=5.8, label_fs=6.8, show_ci=False)
    save(fig, name)


# --------------------------------------------- compact, full-width manuscript figures

COMPACT_LEFT, COMPACT_RIGHT = 0.075, 0.99
GAP_LEFT, GAP_RIGHT = 0.083, 0.996      # shared by both revision figures, so items line up


def _save_fixed(fig, name):
    """Keep identical canvas widths, so LaTeX preserves the intended point sizes."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def effects_compact():
    """Compare revision methods in five equal-size trajectory panels."""
    fig = plt.figure(figsize=(TEXTWIDTH, 1.16))
    cell_width = (COMPACT_RIGHT - COMPACT_LEFT) / len(ITEMS)
    axes = [fig.add_axes([COMPACT_LEFT + i * cell_width + 0.011, 0.265,
                          cell_width - 0.022, 0.455]) for i in range(len(ITEMS))]
    legend_ax = fig.add_axes([0.01, 0.005, 0.98, 0.13])
    legend_ax.axis("off")
    _draw_effects(fig, axes, legend_ax, tag_text=SHORT, tag_fs=8.0,
                  tick_fs=7.2, label_fs=7.7, legend_fs=7.0, marker=3.0, lw=1.15,
                  ylabel_at=(0,), xlabel_span=(0, 4), legend_ncol=5,
                  show_xlabel=False)
    # A shared left-hand axis key avoids repeating a label below every small panel.
    fig.text(COMPACT_LEFT - 0.018, 0.195, "Round", ha="right", va="center", fontsize=7.2)
    _save_fixed(fig, "fig_revision_effects_compact")


def interactions_compact():
    """Pair direct and joint reductions; show every other-feedback effect as a dot.

    One shared scale and the same item order as Figure 4. All 25 saved
    effects are retained (9 bars, 16 points). Reduction is R0 minus R1, so the
    saved mean differences and both confidence-interval endpoints are negated.
    Other-feedback points are unaggregated; no source identity is encoded by
    their small horizontal offsets, which only keep coincident points visible.
    """
    _, cells = _interaction_cells()
    # Full text width, with the axes on the same left and right edges as the effects row, so an
    # item's bars sit directly under that item's trajectory panel.
    fig = plt.figure(figsize=(TEXTWIDTH, 1.20))
    ax = fig.add_axes([GAP_LEFT, 0.285, GAP_RIGHT - GAP_LEFT, 0.545])
    # Both bars are the condition that is told the items, so they are one hue at two strengths,
    # and the darker one is the same ink the slop-aware line carries in the effects figure.
    single_color, joint_color, other_color = "#C79AA4", COLORS[3], "#7C8288"
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.075, 0.84)
    ax.set_yticks([0, 0.4, 0.8], ["0", ".4", ".8"])
    ax.set_ylabel("Slop removed in\none round", fontsize=7.4, labelpad=2, linespacing=1.15)
    ax.set_xticks(range(5), [SHORT[item] for item in ITEMS], fontsize=7.6, linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=7.0, length=2, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#E5E5E5", linewidth=0.45)
    ax.axhline(0, color="#909090", linewidth=0.7, zorder=2)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    def plot_cell(arm, item, x, color, as_bar):
        cell = cells[arm][item]
        mean = -cell["mean_diff"]
        low, high = -cell["ci95"][1], -cell["ci95"][0]
        error = [[mean - low], [high - mean]]
        assert low <= mean <= high
        if as_bar:
            bars = ax.bar(x, mean, width=0.22, color=color, linewidth=0,
                          yerr=error, zorder=3,
                          error_kw={"ecolor": "#353B3E", "elinewidth": 0.8,
                                    "capsize": 2, "capthick": 0.8})
            bars[0].set_gid(f"{arm}:{item}")
        else:
            points = ax.errorbar(x, mean, yerr=error, fmt="o", color=color,
                                markersize=3, markeredgecolor="white",
                                markeredgewidth=0.35, elinewidth=0.65,
                                capsize=1.0, capthick=0.65, zorder=4)
            points.lines[0].set_gid(f"{arm}:{item}")

    bar_count = point_count = 0
    for col, item in enumerate(ITEMS):
        if item in DET_ITEMS:
            plot_cell(f"a4s_{item}", item, col - 0.25, single_color, True)
            bar_count += 1
        plot_cell("a4_slop", item, col, joint_color, True)
        bar_count += 1
        others = [source for source in DET_ITEMS if source != item]
        for source, offset in zip(others, np.linspace(-0.09, 0.09, len(others))):
            plot_cell(f"a4s_{source}", item, col + 0.29 + offset, other_color, False)
            point_count += 1
    assert (bar_count, point_count) == (9, 16)
    # Missing direct feedback is not a zero-height result.
    ax.text(3.875, 0.10, "n/a", ha="center", va="center", fontsize=6.8, color="#68737B")
    handles = [Rectangle((0, 0), 1, 1, facecolor=single_color, label="Named this item only"),
               Rectangle((0, 0), 1, 1, facecolor=joint_color, label="Told all four (slop-aware)"),
               Line2D([], [], color=other_color, marker="o", lw=0, markersize=3.5,
                      label="Named a different item")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.54, 0.945),
               ncol=3, fontsize=7.0, frameon=False, handlelength=1.15,
               handletextpad=0.45, columnspacing=1.3, borderpad=0)
    _save_fixed(fig, "fig_revision_interactions_compact")


# ------------------------------------------- 0917: the effects figure drawn on the human anchor

# Every claim in 5.3.1 has the form "this condition ends above / below the matched human mean",
# so the human mean is the origin and the plotted quantity is the distance from it. On a raw 0-1
# axis that origin sits at a different height in each panel, and the two crossings the section
# turns on (macro redundancy .006 against .041, citation .368 against .413) shrink to a hairline.
# Anchored, the origin is one rule across the row, the region under it is a place a line can
# enter, and the verdict is marked beside the endpoint it judges rather than left to be measured.
GAP_YLIM = (-0.75, 0.62)
GAP_TICKS = [-0.6, -0.3, 0.0, 0.3, 0.6]
GAP_TICKLAB = ["\u22120.6", "\u22120.3", "0", "+0.3", "+0.6"]
# The human level used to be near-black, which is the darkest arm's ink as well, so the two
# read as one line. It now carries the same green as the check that judges against it, and no
# arm is ever drawn in that hue.
BELOW = "#EDF3EE"        # at or below the matched human mean
REACH = "#2F6B43"
HUMAN_RULE = "#3E7A54"
NOTE = "#6E6E6E"
CHECK = MplPath([(-1.0, 0.16), (-0.32, -0.62), (1.0, 0.86)],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO])


def _human_anchor_series(item, codes, rng, data):
    """Per arm, rounds 0-3 as (mean, lo, hi) distance from the matched human mean."""
    human = data["human_mean"][item]
    baseline = r0_rows(item)
    original = [score_of(item, baseline.get(c)) for c in codes]
    original = [v for v in original if v is not None]
    first = interval(original, rng)
    assert abs(first[0] - data["r0_mean"][item]) < 0.00006
    out = {}
    for arm in ARMS:
        points = [first]
        for n in (1, 2, 3):
            revised = rows(EOR / f"results/slop/{arm}/R{n}/{item}/papers.jsonl")
            values = [score_of(item, revised.get(c)) for c in codes
                      if score_of(item, baseline.get(c)) is not None
                      and score_of(item, revised.get(c)) is not None]
            expected = data["arms"][arm][str(n)][item]
            point = interval(values, rng)
            assert len(values) == expected["n"]
            assert abs(point[0] - expected["mean_rn"]) < 0.00006
            points.append(point)
        out[arm] = np.array(points) - human
    return out, human


def effects_anchored(name="fig_revision_effects_anchored"):
    fig = plt.figure(figsize=(TEXTWIDTH, 1.50))
    left, right = GAP_LEFT, GAP_RIGHT
    cell = (right - left) / len(ITEMS)
    axes = [fig.add_axes([left + i * cell + 0.010, 0.275, cell - 0.020, 0.455])
            for i in range(len(ITEMS))]

    data = _expected()
    codes = data["codes"]
    rng = np.random.default_rng(914)
    closed, humans = [], []
    for ax, item in zip(axes, ITEMS):
        series, human = _human_anchor_series(item, codes, rng, data)
        humans.append(human)
        ax.set_ylim(*GAP_YLIM)
        ax.set_xlim(-0.16, 3.74)
        ax.axhspan(GAP_YLIM[0], 0, color=BELOW, linewidth=0, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.45, zorder=1)
        ax.axhline(0, color=HUMAN_RULE, lw=1.15, zorder=2)
        for arm, color, mk in zip(ARMS, COLORS, MARKERS):
            mean, low, high = series[arm].T
            ax.fill_between(range(4), low, high, color=color, alpha=0.16, linewidth=0)
            # Round 0 is the same manuscript for every arm, so it carries one shared marker
            # instead of four coincident ones.
            ax.plot(range(4), mean, color=color, marker=mk, markersize=2.9, markevery=[1, 2, 3],
                    linewidth=1.15, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
            if high[-1] < 0:              # 95% interval clear of the matched human mean
                ax.plot([3.42], [mean[-1]], marker=CHECK, markersize=6.4, color=REACH,
                        markeredgewidth=1.35, fillstyle="none", clip_on=False, zorder=8)
                closed.append((item, arm))
        ax.plot([0], [series[ARMS[0]][0, 0]], marker="o", markersize=3.2, color=INK,
                markerfacecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_title(SHORT[item], fontsize=7.6, pad=2.8, color=INK, linespacing=1.05)
        ax.set_xticks(range(4))
        ax.set_yticks(GAP_TICKS)
        ax.tick_params(labelsize=7.0, length=1.8, pad=1.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 3)
        if item != ITEMS[0]:
            ax.tick_params(labelleft=False)
    assert closed == [("macro_redund", "a4_slop"), ("xsec_ref", "a4_slop"),
                      ("citation", "a4_slop")], closed
    assert [round(h, 2) for h in humans] == [0.04, 0.77, 0.41, 0.37, 0.39], humans

    axes[0].set_yticklabels(GAP_TICKLAB)
    axes[0].set_ylabel("Slop \u2212 human", fontsize=7.4, labelpad=1.6)
    # Nothing is lettered inside the field. The rule, the shading and the axis label carry the
    # reading, and the caption carries the reference values and the untargeted control.
    fig.text(0.5 * (left + right), 0.108, "Revision round", ha="center", va="center",
             fontsize=7.4)

    handles = [Line2D([], [], color=INK, marker="o", markersize=3.2, lw=0,
                      markerfacecolor="white", markeredgewidth=0.8, label="Original")]
    handles += [Line2D([], [], color=c, marker=m, markersize=2.9, lw=1.15,
                       markeredgecolor="white", markeredgewidth=0.4, label=l)
                for c, m, l in zip(COLORS, MARKERS, ARM_LABELS)]
    handles.append(Line2D([], [], color=HUMAN_RULE, lw=1.15, label="Human mean"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.030), ncol=6,
               frameon=False, fontsize=7.0, handlelength=1.5, handletextpad=0.40,
               columnspacing=0.85, borderpad=0)
    _save_fixed(fig, name)


# ------------------------------- 0917b: the pair as one row, effects left and interactions right

ROW_H = 1.95
ROW_L, ROW_R = 0.600, 0.375       # fractions of \\textwidth for the two subfigures


def effects_anchored_left(name="fig_revision_effects_row"):
    """The anchored effects figure narrowed for the left slot of a single-row pair.

    Three panels over two, which is also the partition the section argues. The top row holds the
    items explicit feedback closes, the bottom row the one it does not and the item no feedback
    ever named, and the sixth cell takes the key. The rows are set far apart because a two-line
    panel title otherwise lands on the tick labels of the row above.
    """
    fig = plt.figure(figsize=(ROW_L * TEXTWIDTH, ROW_H))
    grid = fig.add_gridspec(2, 3, left=0.104, right=0.995, top=0.905, bottom=0.150,
                            wspace=0.17, hspace=0.90)
    axes = [fig.add_subplot(grid[r, c]) for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))]
    legend_ax = fig.add_subplot(grid[1, 2])
    legend_ax.axis("off")

    data = _expected()
    codes = data["codes"]
    rng = np.random.default_rng(914)
    closed = []
    for ax, item in zip(axes, ITEMS):
        series, human = _human_anchor_series(item, codes, rng, data)
        ax.set_ylim(*GAP_YLIM)
        ax.set_xlim(-0.18, 4.00)
        ax.axhspan(GAP_YLIM[0], 0, xmax=0.795, color=BELOW, linewidth=0, zorder=0)
        ax.set_axisbelow(True)
        for tick in GAP_TICKS:               # drawn by hand so they stop at the verdict gutter
            if tick:
                ax.axhline(tick, xmax=0.795, color=GRID, linewidth=0.45, zorder=1)
        ax.axhline(0, color=HUMAN_RULE, lw=1.05, zorder=2, xmax=0.795)
        for arm, color, mk in zip(ARMS, COLORS, MARKERS):
            mean, low, high = series[arm].T
            ax.fill_between(range(4), low, high, color=color, alpha=0.16, linewidth=0)
            ax.plot(range(4), mean, color=color, marker=mk, markersize=2.5, markevery=[1, 2, 3],
                    linewidth=1.0, markeredgecolor="white", markeredgewidth=0.35, zorder=3)
            if high[-1] < 0:               # 95% interval clear of the matched human mean
                ax.plot([3.62], [mean[-1]], marker=CHECK, markersize=5.8, color=REACH,
                        markeredgewidth=1.3, fillstyle="none", zorder=8)
                closed.append((item, arm))
        ax.plot([0], [series[ARMS[0]][0, 0]], marker="o", markersize=2.8, color=INK,
                markerfacecolor="white", markeredgewidth=0.7, zorder=5)
        ax.set_title(SHORT[item], fontsize=6.6, pad=2.4, color=INK, linespacing=1.05)
        ax.set_xticks(range(4))
        ax.set_yticks(GAP_TICKS)
        ax.tick_params(labelsize=6.1, length=1.6, pad=1.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 3)
        if ax not in (axes[0], axes[3]):
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticklabels(GAP_TICKLAB)
    assert closed == [("macro_redund", "a4_slop"), ("xsec_ref", "a4_slop"),
                      ("citation", "a4_slop")], closed
    for i in (0, 3):
        axes[i].set_ylabel("Slop \u2212 human", fontsize=6.4, labelpad=2.6)
    box = axes[3].get_position()
    fig.text(0.5 * (box.x0 + axes[4].get_position().x1), box.y0 - 0.108,
             "Revision round", ha="center", va="center", fontsize=6.4)

    handles = [Line2D([], [], color=INK, marker="o", markersize=2.8, lw=0,
                      markerfacecolor="white", markeredgewidth=0.7, label="Original")]
    short_arms = ["Base prompting", "Claude Code", "Reviewer-based", "Slop-aware"]
    handles += [Line2D([], [], color=c, marker=m, markersize=2.5, lw=1.0,
                       markeredgecolor="white", markeredgewidth=0.35, label=l)
                for c, m, l in zip(COLORS, MARKERS, short_arms)]
    handles.append(Line2D([], [], color=HUMAN_RULE, lw=1.05, label="Human mean"))
    legend_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.06, 0.52),
                     ncol=1, frameon=False, fontsize=6.1, labelspacing=0.42,
                     handlelength=1.5, handletextpad=0.40, borderpad=0)
    _save_fixed(fig, name)


def _paired_single_vs_joint():
    """Within-paper difference between the two doses, on the prespecified 60-paper subset.

    The bars in the panel are two independent means, so the question of whether a pair really
    differs is not answered by whether their intervals overlap. It is answered here, by a
    paired bootstrap of (joint score minus single score) over the same papers.
    """
    subset = set(json.loads((EOR / "results/subset60.json").read_text())["codes"])
    rng = np.random.default_rng(914)
    out = {}
    for item in DET_ITEMS:
        base = r0_rows(item)
        single = rows(EOR / f"results/slop/a4s_{item}/R1/{item}/papers.jsonl")
        joint = rows(EOR / f"results/slop/a4_slop/R1/{item}/papers.jsonl")
        codes = [c for c in subset
                 if all(score_of(item, src.get(c)) is not None for src in (base, single, joint))]
        diff = np.array([score_of(item, joint[c]) - score_of(item, single[c]) for c in codes])
        assert len(diff) == 60, (item, len(diff))
        draws = rng.choice(diff, size=(4000, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out[item] = (float(diff.mean()), float(lo), float(hi), lo > 0 or hi < 0)
    assert [item for item in DET_ITEMS if out[item][3]] == ["xsec_ref"], out
    return out


def interactions_dumbbell(name="fig_revision_interactions_row"):
    """Single against joint feedback, one stem pair per item, for the right slot of the row.

    Each item gets two stems from no change, the upper for feedback that names it alone and the
    lower for feedback that names all four, so two equal doses are two equal lengths and the
    reader is not asked to compare two heights across a gap. Nothing marks which pair really
    differs, so the paired test stays in the caption and `_paired_single_vs_joint` asserts that
    the item named there is still the only one.
    """
    _, cells = _interaction_cells()
    assert [i for i in DET_ITEMS if _paired_single_vs_joint()[i][3]] == ["xsec_ref"]
    fig = plt.figure(figsize=(ROW_R * TEXTWIDTH, ROW_H))
    ax = fig.add_axes([0.300, 0.140, 0.682, 0.680])
    single_color, joint_color = "#C28A96", COLORS[3]
    offset = 0.19

    def reduction(arm, item):
        cell = cells[arm][item]
        return -cell["mean_diff"], -cell["ci95"][1], -cell["ci95"][0]

    def stem(x, lo, hi, y, color):
        ax.plot([0, x], [y, y], color=color, lw=2.2, alpha=0.42, solid_capstyle="butt", zorder=3)
        ax.plot([lo, hi], [y, y], color=color, lw=0.7, zorder=5)
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.072, y + 0.072], color=color, lw=0.7, zorder=5)
        ax.plot([x], [y], marker="o", markersize=3.4, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    pairs = 0
    for row, item in enumerate(ITEMS):
        y = -row
        if row:                                  # one item from the next, as a hairline
            ax.axhline(y + 0.5, color="#E2E2DE", lw=0.5, zorder=0)
        joint = reduction("a4_slop", item)
        if item in DET_ITEMS:
            single = reduction(f"a4s_{item}", item)
            stem(*single, y + offset, single_color)
            pairs += 1
        stem(*joint, y - offset, joint_color)
    assert pairs == 4

    ax.set_ylim(-4.62, 0.62)
    ax.set_xlim(-0.055, 0.855)
    ax.set_yticks([-i for i in range(5)], [SHORT[i] for i in ITEMS], fontsize=6.6,
                  linespacing=1.05)
    ax.set_xticks([0, 0.4, 0.8], ["0", ".4", ".8"])
    ax.tick_params(axis="y", length=0, pad=3.4)
    ax.tick_params(axis="x", labelsize=6.1, length=1.6, pad=1.4)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=GRID, linewidth=0.45)
    ax.axvline(0, color="#C2C2C2", lw=0.6, zorder=2)
    ax.set_xlabel("Slop removed in one round", fontsize=6.4, labelpad=1.8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    handles = [Line2D([], [], color=single_color, marker="o", lw=2.2, markersize=3.4,
                      alpha=0.62, markeredgecolor="white", markeredgewidth=0.5,
                      label="Named this item only"),
               Line2D([], [], color=joint_color, marker="o", lw=2.2, markersize=3.4,
                      alpha=0.62, markeredgecolor="white", markeredgewidth=0.5,
                      label="Named all four items")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.300, 1.005), ncol=1,
               frameon=False, fontsize=6.1, labelspacing=0.34, handlelength=1.5,
               handletextpad=0.40, borderpad=0)
    _save_fixed(fig, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-only", action="store_true",
                        help="Render only the compact full-width manuscript figures.")
    args = parser.parse_args()
    if not args.compact_only:
        effects()
        interactions()
        combined()
        effects_half()
        interactions_half()
    effects_compact()
    interactions_compact()
    effects_anchored()
    effects_anchored_left()
    interactions_dumbbell()
    panels = [plt.imread(OUT / f"fig_revision_{part}_compact.png")
              for part in ("effects", "interactions")]
    preview_width = max(panel.shape[1] for panel in panels)
    panels = [np.pad(panel, ((0, 0),
                            ((preview_width - panel.shape[1]) // 2,
                             (preview_width - panel.shape[1] + 1) // 2),
                            (0, 0)), constant_values=1.0) for panel in panels]
    plt.imsave(OUT / "fig_revision_pair_compact.png", np.concatenate(panels, axis=0))
    print(f"Wrote revision PDFs and PNG previews to {OUT}")
