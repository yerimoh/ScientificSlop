r"""The figure for the review-score paragraph of Section 5.2. One row, two panels.

(a) How AI-written each system judges a paper to be, against the review score the paper actually received. Every
    system is first turned into the classifier it claims to be on SciSlopBench, where authorship is known, so its
    own raw number becomes a probability, and the panel then averages that probability at each review score.
(b) The two things a system can do. The horizontal axis is how far it moves across the review scale in panel (a),
    the vertical axis is how well it separates the AI papers of the benchmark from their human counterparts.

Reads results/scale_points.json and results/calibration.json, both written by fig_scale_points.py and
calibrate_scores.py. Outputs results/fig_main_review.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = json.load(open(f"{HERE}/results/scale_points.json"))
CAL = json.load(open(f"{HERE}/results/calibration.json"))["calibration"]
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]
LAB = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "rev_b2h": "CycleReviewer",
       "rev_b3a": "AI Scientist", "agg3": "SciSlop (ours)"}
COL = {"binoculars": "#2a78d6", "detectgpt": "#1baf7a", "nts": "#eda100", "rev_b2h": "#eb6834",
       "rev_b3a": "#e87ba4", "agg3": "#111111"}
MK = {"binoculars": "o", "detectgpt": "s", "nts": "D", "rev_b2h": "v", "rev_b3a": "P", "agg3": "o"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, labelsize=7, color="#555555")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.30, "width_ratios": [1.25, 1]})

# ---- (a) review score against judged AI-likeness
END = {"rev_b2h": (5, 13), "binoculars": (5, 5), "rev_b3a": (5, -3), "nts": (5, -11), "detectgpt": (5, -19), "agg3": (5, 0)}
span = {}
for k in ORDER:
    pts = sorted((int(b), v["mean_percentile"]) for b, v in SP["levels"][k].items() if not b.startswith("_"))
    if len(pts) < 3:
        continue
    ours = k == "agg3"
    ax1.plot([a for a, _ in pts], [b for _, b in pts], color=COL[k], lw=2.4 if ours else 1.1, marker=MK[k],
             ms=5 if ours else 3.2, mec="white" if ours else COL[k], mew=0.9 if ours else 0.5,
             zorder=6 if ours else 3, clip_on=False)
    off = END[k]
    ax1.annotate(LAB[k], xy=pts[-1], xytext=off, textcoords="offset points", fontsize=6.4, va="center", ha="left",
                 color="#111111" if ours else COL[k], fontweight="bold" if ours else "normal",
                 arrowprops=None if abs(off[1]) < 4 else dict(arrowstyle="-", color=COL[k], lw=0.4, shrinkA=1, shrinkB=1))
    span[k] = pts[0][1] - pts[-1][1]          # how far it falls from the lowest to the highest review score
xs = [a for a, _ in pts]
ax1.axhline(50, color="#d5d5d5", lw=0.6, ls=":", zorder=1)
ax1.text(min(xs) - 0.2, 50.8, "coin flip", fontsize=5.6, color="#999999", va="bottom", ha="left")
ax1.set_xticks(xs)
ax1.set_xlabel("ICLR review score", fontsize=8)
ax1.set_ylabel("how AI-written the system says the paper is (%)", fontsize=7.6)
ax1.set_xlim(min(xs) - 0.3, max(xs) + 3.0); ax1.set_ylim(0, 70)
tidy(ax1)

# ---- (b) the two axes
OFF2 = {"binoculars": (0, 8), "detectgpt": (0, -13), "nts": (0, -13), "rev_b2h": (0, 8), "rev_b3a": (0, -13), "agg3": (0, 10)}
for k in ORDER:
    if k not in span:
        continue
    x, y = span[k], CAL[k]["auroc"]
    ours = k == "agg3"
    ax2.scatter([x], [y], color=COL[k], marker=MK[k], s=75 if ours else 28, edgecolor="white" if ours else COL[k],
                linewidth=0.9 if ours else 0.5, zorder=6 if ours else 3)
    ax2.annotate(LAB[k], xy=(x, y), xytext=OFF2[k], textcoords="offset points", fontsize=6.2, ha="center",
                 color="#111111" if ours else "#555555", fontweight="bold" if ours else "normal")
ax2.axhline(0.5, color="#d5d5d5", lw=0.6, ls=":"); ax2.axvline(0, color="#d5d5d5", lw=0.6, ls=":")
ax2.set_xlabel("falls across the review scale (points)", fontsize=7.6)
ax2.set_ylabel("separates AI papers from human ones", fontsize=7.6)
ax2.set_xlim(-6, 20); ax2.set_ylim(0.45, 1.08); ax2.set_yticks([0.5, 0.7, 0.9, 1.0])
tidy(ax2)
fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.19)
fig.savefig(f"{HERE}/results/fig_main_review.pdf"); fig.savefig(f"{HERE}/results/fig_main_review.png", dpi=300)
print("saved fig_main_review")
for k in ORDER:
    if k in span:
        print(f"  {LAB[k]:16s} falls {span[k]:+.1f} points across the scale, benchmark AUROC {CAL[k]['auroc']:.3f}")
