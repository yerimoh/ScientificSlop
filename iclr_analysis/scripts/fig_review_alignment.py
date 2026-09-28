r"""Figure 2 of the paper, redrawn as vector graphics at final on-page size.

The version that shipped before this script was a JPEG raster: a matplotlib figure that had been
passed through an image-generation edit, then cropped into four PDFs with MuPDF. That is why the
panels looked soft in print. It also invented a Pangram line in panel (a); scale_points.json carries
no Pangram, because the commercial judgment exists only for ICLR 2026 and panel (a) pools all years.
This script draws the same three panels straight from the data files, so every glyph stays vector.

Four files are written, one per subfigure plus the shared legend strip, because the paper references
the panels separately (fig:review-scores-rating, -dimensions, -acceptance). Each page is exactly the
size it occupies in the ICLR layout, so \includegraphics[width=\linewidth] scales it by one and a
point of type here is a point of type on the page. Panel titles are left out; the subcaptions
supply them.

(a) Calibrated AI probability against the review score the paper received, each paper first expressed
    against the mean of its own year.
(b) Absolute Spearman correlation with the four scores an ICLR 2026 reviewer gives. Blue and red fill
    marks p < 0.05 with the sign, gray marks an estimate that is not significant.
(c) Rejected against accepted inside each year, as the rate at which the system calls the rejected
    paper more AI-like.

Reads results/scale_points.json, results/dimensions.json, results/scores_years.csv.
Writes results/alignment/fig_review_alignment_{a,b,c,legend}.pdf, which are copied into the paper at
figures/main_figures/ with the same names.
"""
import csv
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{HERE}/results/alignment"
SP = json.load(open(f"{HERE}/results/scale_points.json"))
DM = json.load(open(f"{HERE}/results/dimensions.json"))

SYS = ["binoculars", "detectgpt", "nts", "pangram_fraction_ai", "rev_b2h", "rev_b3a", "agg3"]
FAMILY = [("AI detection:", ["binoculars", "detectgpt", "nts", "pangram_fraction_ai"]),
          ("Reviewers:", ["rev_b2h", "rev_b3a"]),
          ("Our method:", ["agg3"])]
COL = {"binoculars": "#183B70", "detectgpt": "#7252B5", "nts": "#008875", "pangram_fraction_ai": "#42A8DB",
       "rev_b2h": "#B74921", "rev_b3a": "#E69B34", "agg3": "#202329"}
FAMCOL = {"AI detection:": "#285B8C", "Reviewers:": "#B96829", "Our method:": "#202329"}
LS = {"binoculars": "-", "detectgpt": (0, (5, 2.4)), "nts": (0, (1, 1.8)),
      "pangram_fraction_ai": (0, (5, 1.8, 1, 1.8)), "rev_b2h": "-", "rev_b3a": (0, (4, 1.8)), "agg3": "-"}
MK = {"binoculars": "o", "detectgpt": "s", "nts": "D", "pangram_fraction_ai": "^", "rev_b2h": "v",
      "rev_b3a": "P", "agg3": "o"}
MS = {"binoculars": 2.5, "detectgpt": 2.9, "nts": 3.2, "pangram_fraction_ai": 3.5,
      "rev_b2h": 2.7, "rev_b3a": 3.0, "agg3": 3.2}
LAB = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "pangram_fraction_ai": "Pangram",
       "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "agg3": "SciSlop (ours)"}
DIR = {"binoculars": -1, "detectgpt": +1, "nts": +1, "rev_b2h": -1, "rev_b3a": -1, "agg3": +1}

NEG, POS, NS_FACE, NS_TEXT = "#337F9D", "#C25A59", "#F1F2F4", "#8A929B"
DIMS = ["rating", "soundness", "presentation", "contribution"]
DIMLAB = ["Overall", "Soundness", "Presentation", "Contribution"]
YEARS = [str(y) for y in range(2017, 2026)]
MIN_SIDE = 15
RHO_FULL = 0.15  # correlation that saturates a cell, as in the ver2 panel design

# On-page geometry. ICLR \textwidth is 5.5in and the figure sits in a 0.98\linewidth minipage.
STRIP = 5.5 * 0.98
# Panel (b) has to seat seven row names, four dimension names and a colour bar, so it takes the
# width that (a) and (c) can spare. These three fractions are mirrored in 04_experiment.tex.
FRAC_A, FRAC_B, FRAC_C = 0.2730, 0.4440, 0.2730
W_A, W_B, W_C = STRIP * FRAC_A, STRIP * FRAC_B, STRIP * FRAC_C
H_PANEL = 1.3078

# Type is set at final size, so these numbers are the points that reach the page.
FS_TICK, FS_LABEL, FS_CELL, FS_ROW, FS_NOTE = 5.8, 6.6, 5.8, 5.2, 4.6
FS_DIM = 5.2        # the four dimension names have to clear 27pt columns
FS_LEGEND_MAX = 6.6  # the legend never outgrows the axis labels
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": FS_TICK,
                     "axes.linewidth": 0.5, "pdf.fonttype": 42, "text.color": "#252B33",
                     "axes.labelcolor": "#333B45", "xtick.color": "#4D5661", "ytick.color": "#4D5661",
                     "savefig.transparent": False, "figure.facecolor": "white"})


def marker_style(k):
    """Hollow diamonds stay distinct from filled squares even in grayscale."""
    return {"marker": MK[k], "ms": MS[k], "mfc": "white" if k == "nts" else COL[k],
            "mec": COL[k] if k == "nts" else "white", "mew": 0.7 if k == "nts" else 0.35}


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#929AA3")
    ax.spines["bottom"].set_color("#929AA3")
    ax.tick_params(length=2, width=0.5, labelsize=FS_TICK, color="#929AA3", pad=1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EDF1", linewidth=0.45)


def draw(ax, k, xs, ys):
    ours = k == "agg3"
    ax.plot(xs, ys, color=COL[k], ls=LS[k], lw=1.5 if ours else 0.95,
            **marker_style(k), zorder=6 if ours else 3, solid_capstyle="round")


def tint(color, amount):
    return tuple(1 - amount * (1 - v) for v in to_rgb(color))


# ---------------------------------------------------------------- (a) review score
def panel_a(path):
    fig, ax = plt.subplots(figsize=(W_A, H_PANEL))
    for k in SYS:
        if k not in SP["levels"]:
            continue  # Pangram has no per-score history; it is an ICLR 2026 only judgment.
        pts = sorted((int(b), v["mean_percentile"]) for b, v in SP["levels"][k].items() if not b.startswith("_"))
        if len(pts) >= 3:
            draw(ax, k, [a for a, _ in pts], [b for _, b in pts])
    xs = sorted({int(b) for k in SP["levels"] for b in SP["levels"][k] if not b.startswith("_")})
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(0, 70)
    ax.set_yticks(range(0, 71, 10))
    ax.set_xlabel("ICLR review score", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel("Calibrated AI probability (%)", fontsize=FS_LABEL, labelpad=1.5)
    tidy(ax)
    fig.subplots_adjust(left=0.205, right=0.995, top=0.975, bottom=0.175)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- (b) review dimensions
def panel_b(path):
    fig = plt.figure(figsize=(W_B, H_PANEL))
    # The row labels, the four columns and the colour bar all compete for 150pt of width, so the
    # heatmap keeps the middle 59% and the bar's own captions sit inside its vertical margin.
    ax = fig.add_axes([0.222, 0.200, 0.645, 0.755])
    cax = fig.add_axes([0.893, 0.270, 0.024, 0.620])
    ys = list(range(len(SYS)))[::-1]
    for yi, k in zip(ys, SYS):
        for xi, dim in enumerate(DIMS):
            c = DM["cells"].get(k, {}).get(dim)
            if not c:
                continue
            significant = c["p"] < 0.05
            color = NEG if c["rho"] < 0 else POS
            strength = min(abs(c["rho"]) / RHO_FULL, 1.0)
            face = tint(color, 0.16 + 0.64 * strength) if significant else NS_FACE
            ax.add_patch(Rectangle((xi - 0.49, yi - 0.46), 0.98, 0.92, facecolor=face,
                                   edgecolor="white", linewidth=0.5, zorder=2))
            ax.text(xi, yi, f"{abs(c['rho']):.2f}", ha="center", va="center", fontsize=FS_CELL,
                    zorder=3, color=("#1C2B33" if significant else NS_TEXT))
        # A colour stub ties each row to the shared legend below the figure.
        ax.plot([-0.60, -0.60], [yi - 0.28, yi + 0.28], color=COL[k], lw=1.6, clip_on=False, zorder=4)
    ax.set_xlim(-0.5, len(DIMS) - 0.5)
    ax.set_ylim(-0.5, len(SYS) - 0.5)
    ax.set_xticks(range(len(DIMS)), DIMLAB)
    ax.set_yticks(ys, [LAB[k] for k in SYS])
    for t, k in zip(ax.get_yticklabels(), SYS):
        t.set_color("#30353D")
        if k == "agg3":
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, labelsize=FS_ROW, pad=4)
    ax.tick_params(axis="x", length=0, labelsize=FS_DIM, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("ICLR 2026 review dimensions", fontsize=FS_LABEL, labelpad=1.5)

    # Drawn as stacked bands rather than imshow, so the bar stays vector like the rest of the page.
    cmap = LinearSegmentedColormap.from_list("likeness", [tint(NEG, 0.80), "white", tint(POS, 0.80)])
    bands = 192
    for i in range(bands):
        cax.add_patch(Rectangle((0, i / bands), 1, 1.0 / bands + 1e-3,
                                facecolor=cmap(i / (bands - 1)), edgecolor="none", linewidth=0))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.text(0.5, 0.5, "AI likeness", rotation=90, ha="center", va="center",
             fontsize=FS_NOTE, color="#2F3740", transform=cax.transAxes)
    cax.text(0.5, 1.045, "More AI", ha="center", va="bottom", fontsize=FS_NOTE,
             color="#7B4A48", transform=cax.transAxes, clip_on=False)
    cax.text(0.5, -0.045, "Less AI", ha="center", va="top", fontsize=FS_NOTE,
             color="#3C6274", transform=cax.transAxes, clip_on=False)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- (c) decisions by year
def panel_c(path):
    rows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv"))
            if r["group"] in ("reject", "accept", "oral")]
    num = lambda r, c: None if r.get(c) in (None, "", "None") else float(r[c])
    auroc = lambda p, n: sum(1.0 if a > b else 0.5 if a == b else 0.0
                             for a, b in itertools.product(p, n)) / (len(p) * len(n))
    fig, ax = plt.subplots(figsize=(W_C, H_PANEL))
    dumped = {}
    for k, d in DIR.items():
        xs, ys = [], []
        for i, y in enumerate(YEARS):
            rej = [num(r, k) * d for r in rows if r["year"] == y and r["group"] == "reject" and num(r, k) is not None]
            acc = [num(r, k) * d for r in rows if r["year"] == y and r["group"] in ("accept", "oral") and num(r, k) is not None]
            if len(rej) >= MIN_SIDE and len(acc) >= MIN_SIDE:
                xs.append(i)
                ys.append(auroc(rej, acc))
        if len(xs) >= 5:
            draw(ax, k, xs, ys)
            dumped[k] = {YEARS[i]: round(v, 3) for i, v in zip(xs, ys)}
    ax.axhline(0.5, color="#9AA1AA", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.98, 0.97, "Chance = 0.50", transform=ax.transAxes, fontsize=FS_NOTE,
            color="#707985", ha="right", va="top")
    ax.set_xticks(range(len(YEARS)), [y[2:] for y in YEARS])
    ax.set_xlim(-0.35, len(YEARS) - 0.65)
    ax.set_ylim(0.36, 0.82)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("ICLR year", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel("Rejection AUROC", fontsize=FS_LABEL, labelpad=1.5)
    tidy(ax)
    fig.subplots_adjust(left=0.205, right=0.995, top=0.975, bottom=0.175)
    fig.savefig(path)
    plt.close(fig)
    return dumped


# ---------------------------------------------------------------- shared legend
def legend_row(families, size, swatch_w, sep):
    items = []
    for name, keys in families:
        items.append(TextArea(name, textprops={"size": size, "weight": "bold", "color": FAMCOL[name]}))
        for k in keys:
            swatch = DrawingArea(swatch_w, 7, 0, 0)
            swatch.add_artist(Line2D([0, swatch_w / 2, swatch_w], [3.5] * 3, color=COL[k], ls=LS[k],
                                     lw=1.5 if k == "agg3" else 0.95, markevery=[1], **marker_style(k)))
            items.append(HPacker(children=[swatch, TextArea(LAB[k], textprops={
                "size": size, "weight": "bold" if k == "agg3" else "normal"})],
                align="center", pad=0, sep=2.0))
    return HPacker(children=items, align="center", pad=0, sep=sep)


def legend(path):
    """Ten entries on one line only fit below 5pt, which is not readable in print, so the three
    families take two lines and the type is grown to the largest size both lines still clear."""
    rows = [FAMILY[:1], FAMILY[1:]]
    limit = (STRIP - 0.06) * 72  # points of usable width, keeping a hairline margin
    size, swatch_w, sep = 5.0, 13.0, 6.0
    best = None
    for _ in range(60):
        fig = plt.figure(figsize=(STRIP, 1.0))
        packers = [legend_row(r, size, swatch_w, sep) for r in rows]
        box = AnchoredOffsetbox(loc="center", child=VPacker(children=packers, align="center", pad=0, sep=size * 0.62),
                                pad=0, frameon=False, bbox_to_anchor=(0.5, 0.5),
                                bbox_transform=fig.transFigure, borderpad=0)
        fig.add_artist(box)
        renderer = fig.canvas.get_renderer()
        ext = box.get_window_extent(renderer)
        width = ext.width * 72 / fig.dpi
        if width > limit:
            plt.close(fig)
            break
        best = (size, width, ext.height * 72 / fig.dpi)
        plt.close(fig)
        if size >= FS_LEGEND_MAX:
            break
        size, swatch_w, sep = min(size * 1.03, FS_LEGEND_MAX), swatch_w * 1.03, sep * 1.03
    if best is None:
        raise RuntimeError("legend never fit the strip")
    size = best[0]
    swatch_w, sep = 13.0 * size / 5.0, 6.0 * size / 5.0
    height = (best[2] + 4.0) / 72  # a little air above and below the two lines
    fig = plt.figure(figsize=(STRIP, height))
    packers = [legend_row(r, size, swatch_w, sep) for r in rows]
    fig.add_artist(AnchoredOffsetbox(loc="center", child=VPacker(children=packers, align="center",
                                                                 pad=0, sep=size * 0.62),
                                     pad=0, frameon=False, bbox_to_anchor=(0.5, 0.5),
                                     bbox_transform=fig.transFigure, borderpad=0))
    fig.savefig(path)
    plt.close(fig)
    print(f"  legend set on two lines at {size:.2f}pt (widest {best[1]:.1f}/{limit:.1f}pt)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel_a(f"{OUT}/fig_review_alignment_a.pdf")
    panel_b(f"{OUT}/fig_review_alignment_b.pdf")
    c = panel_c(f"{OUT}/fig_review_alignment_c.pdf")
    legend(f"{OUT}/fig_review_alignment_legend.pdf")
    json.dump({"panel_c": c}, open(f"{OUT}/panel_c.json", "w"), indent=1)
    print(f"wrote 4 vector PDFs to {OUT}")
    for k, v in c.items():
        print(f"  (c) {LAB[k]:16s} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))
