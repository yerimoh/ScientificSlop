r"""The review-score figure for Section 5.2. One row, three panels, one shared legend.

Every panel uses the same colour and marker for a system, and no panel carries a legend of its own.

(a) Against the score the paper received. Each system is first turned into the classifier it claims to be on
    SciSlopBench, where authorship is known, so its own raw number becomes the probability it assigns to AI
    authorship. The panel averages that probability at each review score, after expressing each paper against the
    mean of its own year, since the corpus drifts by year.
(b) Against the four scores a reviewer actually gives. ICLR 2026 is the one year that carries an overall rating and
    a separate soundness, presentation and contribution score for every paper. Circle area is proportional to
    absolute Spearman correlation. Significant associations (p < 0.05) are filled blue (higher review scores,
    less AI-like) or red (higher review scores, more AI-like); non-significant estimates are hollow grey.
(c) Against the decision, year by year. Inside each year the rejected papers are set against the accepted ones and
    the value is how often the system calls the rejected one more AI-like. One half means it cannot tell them
    apart. A system appears only in the years where both sides hold at least fifteen papers it scored.

Reads results/scale_points.json, results/dimensions.json, results/scores_years.csv.
Outputs results/fig_final_review.{pdf,png}
"""
import csv, itertools, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = json.load(open(f"{HERE}/results/scale_points.json"))
DM = json.load(open(f"{HERE}/results/dimensions.json"))
SYS = ["binoculars", "detectgpt", "nts", "pangram_fraction_ai", "rev_b2h", "rev_b3a", "agg3"]
# one hue family per kind of system, so the reader sees the grouping before reading a single name.
# Detectors use distinct cool hues; automated reviewers use warm hues, ours charcoal.
FAMILY = [("AI text detectors", ["binoculars", "detectgpt", "nts", "pangram_fraction_ai"]),
          ("Automated reviewers", ["rev_b2h", "rev_b3a"]),
          ("Our method", ["agg3"])]
COL = {"binoculars": "#183B70", "detectgpt": "#7252B5", "nts": "#008875", "pangram_fraction_ai": "#42A8DB",
       "rev_b2h": "#B74921", "rev_b3a": "#E69B34", "agg3": "#202329"}
FAMCOL = {"AI text detectors": "#285B8C", "Automated reviewers": "#B96829", "Our method": "#202329"}
LS = {"binoculars": "-", "detectgpt": (0, (5, 2.4)), "nts": (0, (1, 1.8)),
      "pangram_fraction_ai": (0, (5, 1.8, 1, 1.8)), "rev_b2h": "-", "rev_b3a": (0, (4, 1.8)), "agg3": "-"}
MK = {"binoculars": "o", "detectgpt": "s", "nts": "D", "pangram_fraction_ai": "^", "rev_b2h": "v",
      "rev_b3a": "P", "agg3": "o"}
MS = {"binoculars": 3.8, "detectgpt": 4.5, "nts": 5.0, "pangram_fraction_ai": 5.5,
      "rev_b2h": 4.1, "rev_b3a": 4.5, "agg3": 5.2}


def marker_style(k):
    # Hollow diamonds remain distinct from filled squares even in grayscale.
    return {"marker": MK[k], "ms": MS[k], "mfc": "white" if k == "nts" else COL[k],
            "mec": COL[k] if k == "nts" else "white", "mew": 1.1 if k == "nts" else 0.5}


LAB = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "pangram_fraction_ai": "Pangram",
       "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist", "agg3": "SciSlop (ours)"}
DIR = {"binoculars": -1, "detectgpt": +1, "nts": +1, "rev_b2h": -1, "rev_b3a": -1, "agg3": +1}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.65, "pdf.fonttype": 42, "text.color": "#252B33",
                     "axes.labelcolor": "#333B45", "xtick.color": "#4D5661", "ytick.color": "#4D5661"})


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#929AA3"); ax.spines["bottom"].set_color("#929AA3")
    ax.tick_params(length=3, labelsize=8, color="#929AA3")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EDF1", linewidth=0.6)


def draw(ax, k, xs, ys, **kw):
    ours = k == "agg3"
    ax.plot(xs, ys, color=COL[k], ls=LS[k], lw=2.3 if ours else 1.45,
            **marker_style(k), zorder=6 if ours else 3, **kw)


fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55),
                         gridspec_kw={"wspace": 0.62, "width_ratios": [1.05, 1.13, 1.03]})

# ---------------------------------------------------------------- (a)
ax = axes[0]
for k in SYS:
    if k not in SP["levels"]:
        continue
    pts = sorted((int(b), v["mean_percentile"]) for b, v in SP["levels"][k].items() if not b.startswith("_"))
    if len(pts) >= 3:
        draw(ax, k, [a for a, _ in pts], [b for _, b in pts])
xs = sorted({int(b) for k in SP["levels"] for b in SP["levels"][k] if not b.startswith("_")})
ax.set_xticks(xs); ax.set_xlabel("ICLR review score", fontsize=9, labelpad=5)
ax.set_ylabel("Calibrated AI probability (%)", fontsize=9, labelpad=5)
ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3); ax.set_ylim(0, 70)
tidy(ax)
ax.set_title("(a)  AI probability vs. review score", fontsize=10, pad=11, loc="left")

# ---------------------------------------------------------------- (b)
ax = axes[1]
DIMS = [("rating", "overall"), ("soundness", "soundness"), ("presentation", "presentation"), ("contribution", "contribution")]
ROWS = [k for k in SYS]
ys = list(range(len(ROWS)))[::-1]
NEG, POS, NS = "#337F9D", "#C25A59", "#B9C0C8"
for lo, hi, color in [(2.5, 6.5, "#F5F8FB"), (0.5, 2.5, "#FCF8F1"), (-0.5, 0.5, "#F1F3F5")]:
    ax.axhspan(lo, hi, color=color, zorder=0, linewidth=0)
for yi, k in zip(ys, ROWS):
    for xi, (dim, _) in enumerate(DIMS):
        c = DM["cells"].get(k, {}).get(dim)
        if not c:
            continue
        significant = c["p"] < 0.05
        color = NEG if c["rho"] < 0 else POS
        ax.scatter(xi, yi, s=1500 * abs(c["rho"]),
                   facecolor=color if significant else "white",
                   edgecolor=color if significant else NS,
                   linewidth=0.8 if significant else 1.0, zorder=3)
ax.set_xticks(range(len(DIMS))); ax.set_xticklabels(["Overall", "Soundness", "Presentation", "Contribution"])
ax.set_yticks(ys); ax.set_yticklabels([LAB[k] for k in ROWS])
for t, k in zip(ax.get_yticklabels(), ROWS):
    t.set_color("#30353D")
    if k == "agg3":
        t.set_fontweight("bold")
ax.set_xlim(-0.5, len(DIMS) - 0.5); ax.set_ylim(-0.5, len(ROWS) - 0.5)
ax.axhline(2.5, color="white", lw=3); ax.axhline(0.5, color="white", lw=3)
# A narrow colour strip connects the row labels to the shared system legend.
for yi, k in zip(ys, ROWS):
    ax.plot([-0.65, -0.65], [yi - 0.29, yi + 0.29], color=COL[k], lw=2.5, clip_on=False)
ax.set_xlabel("ICLR 2026 review dimensions", fontsize=9, labelpad=5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(length=0, axis="y", labelsize=8, pad=12)
ax.tick_params(length=0, axis="x", labelsize=7, pad=5)
ax.set_title("(b)  What higher review scores imply", fontsize=10, pad=11, loc="left")

# ---------------------------------------------------------------- (c)
ax = axes[2]
rows = [r for r in csv.DictReader(open(f"{HERE}/results/scores_years.csv")) if r["group"] in ("reject", "accept", "oral")]
num = lambda r, c: None if r.get(c) in (None, "", "None") else float(r[c])
YEARS = [str(y) for y in range(2017, 2026)]
auroc = lambda p, n: sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(p, n)) / (len(p) * len(n))
MIN_SIDE = 15
OUT = {}
for k, d in DIR.items():
    xs2, ys2 = [], []
    for i, y in enumerate(YEARS):
        rej = [num(r, k) * d for r in rows if r["year"] == y and r["group"] == "reject" and num(r, k) is not None]
        acc = [num(r, k) * d for r in rows if r["year"] == y and r["group"] in ("accept", "oral") and num(r, k) is not None]
        if len(rej) >= MIN_SIDE and len(acc) >= MIN_SIDE:
            xs2.append(i); ys2.append(auroc(rej, acc))
    if len(xs2) >= 5:
        draw(ax, k, xs2, ys2)
        OUT[k] = {YEARS[i]: round(v, 3) for i, v in zip(xs2, ys2)}
ax.axhline(0.5, color="#9AA1AA", lw=0.9, ls=(0, (3, 2)), zorder=1)
ax.text(0.975, 0.965, "Chance = 0.50", transform=ax.transAxes,
        fontsize=7, color="#707985", ha="right", va="top")
ax.set_xticks(range(len(YEARS))); ax.set_xticklabels([y[2:] for y in YEARS], fontsize=6.4)
ax.set_xlabel("ICLR year", fontsize=9, labelpad=5)
ax.set_ylabel("Rejection AUROC", fontsize=9, labelpad=5)
ax.set_xlim(-0.3, len(YEARS) - 0.7); ax.set_ylim(0.36, 0.82)
tidy(ax)
ax.set_title("(c)  Rejected vs. accepted papers", fontsize=10, pad=11, loc="left")

# Pack legends by their actual text widths: model names occupy one shared row.
groups = []
for name, keys in FAMILY:
    entries = []
    for k in keys:
        swatch = DrawingArea(24, 10, 0, 0)
        swatch.add_artist(Line2D([0, 12, 24], [5, 5, 5], color=COL[k], ls=LS[k],
                                lw=2.2 if k == "agg3" else 1.45, markevery=[1], **marker_style(k)))
        label = TextArea(LAB[k], textprops={"size": 8.5, "weight": "bold" if k == "agg3" else "normal"})
        entries.append(HPacker(children=[swatch, label], align="center", pad=0, sep=4))
    title = TextArea(name, textprops={"size": 8, "weight": "bold", "color": FAMCOL[name]})
    groups.append(VPacker(children=[title, HPacker(children=entries, align="center", pad=0, sep=13)],
                          align="left", pad=0, sep=6))
legend = AnchoredOffsetbox(loc="lower center", child=HPacker(children=groups, align="top", pad=0, sep=26),
                           pad=0, frameon=False, bbox_to_anchor=(0.52, 0.025),
                           bbox_transform=fig.transFigure, borderpad=0)
fig.add_artist(legend)

# Explain the reading direction in words, without requiring statistical notation.
key_items = [TextArea("(b)  Higher review score:", textprops={"size": 8})]
for color, filled, label in [(NEG, True, "Less AI-like"), (POS, True, "More AI-like"),
                             (NS, False, "Not significant")]:
    swatch = DrawingArea(11, 10, 0, 0)
    swatch.add_artist(Line2D([5.5], [5], marker="o", ms=6.5, linestyle="none",
                            mec=color, mfc=color if filled else "white", mew=1))
    key_items.append(HPacker(children=[swatch, TextArea(label, textprops={"size": 8})],
                             align="center", pad=0, sep=4))
key = AnchoredOffsetbox(loc="center", child=HPacker(children=key_items, align="center", pad=0, sep=15),
                       pad=0, frameon=False, bbox_to_anchor=(0.52, 0.224),
                       bbox_transform=fig.transFigure, borderpad=0)
fig.add_artist(key)
fig.text(0.52, 0.174, "Filled: significant association (p < 0.05)     |     Larger circle: stronger association",
         ha="center", va="center", fontsize=7.5, color="#626D78")
fig.subplots_adjust(left=0.063, right=0.987, top=0.88, bottom=0.36)
if __name__ == "__main__":
    fig.savefig(f"{HERE}/results/fig_final_review.pdf"); fig.savefig(f"{HERE}/results/fig_final_review.png", dpi=300)
    json.dump({"panel_c": OUT}, open(f"{HERE}/results/final_review_panel_c.json", "w"), indent=1)
    print("saved fig_final_review")
    for k, v in OUT.items():
        print(f"  (c) {LAB[k]:16s} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))
