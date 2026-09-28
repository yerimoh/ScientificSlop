r"""Two figures for the review-score analysis (0917).

fig_review_final  one row, three panels, shared legend
  (a) ICLR 2026 on its own, one year and one review scale, so the score on the x axis is comparable across papers
  (b) every ICLR year from 2017 to 2025, how much more often a poorly rated paper lands in the upper half of a
      system's AI-likeness than a highly rated one
  (c) the two things a system can do, separating AI papers from human ones and following the reviewers

fig_two_axes      one row, two panels
  (a) the same two axes for the individual measures, where the strongest provenance measure is the one reviewers
      do not price
  (b) review score by slop tercile, for all 2026 papers and for the papers Pangram calls fully human

Reads results/papers_2026.jsonl, results/year_rate.json, results/analysis_2026.json, results/analysis_extra.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P26 = [json.loads(l) for l in open(f"{HERE}/results/papers_2026.jsonl")]
YR = json.load(open(f"{HERE}/results/year_rate.json"))
EX = json.load(open(f"{HERE}/results/analysis_extra.json"))
SUMM = YR["summary"]["tercile"]
STYLE = {"pangram": ("#7a6fd0", "^", "Pangram"), "binoculars": ("#2a78d6", "o", "Binoculars"),
         "detectgpt": ("#1baf7a", "s", "DetectGPT"), "nts": ("#eda100", "D", "NTS"),
         "rev_b2h": ("#eb6834", "v", "CycleReviewer"), "rev_b3a": ("#e87ba4", "P", "AI Scientist"),
         "agg3": ("#111111", "o", "SciSlop (ours)")}
PROV = {"binoculars": 0.677, "detectgpt": 0.674, "nts": 0.624, "rev_b2h": 0.572, "rev_b3a": 0.534,
        "agg3": EX["two_axes"]["agg3"]["provenance_auroc"]}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    ax.tick_params(length=2.5, labelsize=7, color="#555555")


# ============================================================ figure 1
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw={"wspace": 0.34})

ax = axes[0]
# ICLR 2026, and only the papers where every system has a score, so the six lines rest on the same documents.
# x is the review score itself, each point drawn at the mean score of its quartile.
A26 = json.load(open(f"{HERE}/results/analysis_2026.json"))
CS = A26.get("common_subset", {})
QU = CS.get("levels", {})
for k in ["pangram", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]:
    cells = QU.get(k) or {}
    pts = sorted((float(b), v["mean_percentile"]) for b, v in cells.items())
    if len(pts) < 3:
        continue
    c, mk, lab = STYLE[k]; ours = k == "agg3"
    ax.plot([a for a, _ in pts], [b for _, b in pts], color=c, lw=2.2 if ours else 1.0, marker=mk,
            ms=4.5 if ours else 2.8, mec="white" if ours else c, mew=0.8 if ours else 0.5,
            zorder=5 if ours else 3, label=lab)
ax.axhline(50, color="#cccccc", lw=0.6, ls=":")
ax.set_xlabel("mean review score, ICLR 2026", fontsize=7.5)
ax.set_xticks([4, 5, 6, 7]); ax.set_xticks([3.5, 4.5, 5.5, 6.5], minor=True)
ax.tick_params(axis="x", which="minor", length=1.5, color="#999999")
ax.set_ylabel("AI-likeness (mean percentile)", fontsize=7.5)
ax.set_ylim(38, 62); ax.set_yticks([40, 45, 50, 55, 60]); tidy(ax)

ax = axes[1]
years = sorted(int(y) for y in YR["years"] if YR["years"][y]["n"] >= 40)
for k in ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]:
    if k not in SUMM:
        continue
    per = SUMM[k]["per_year"]
    xs = [y for y in years if str(y) in per]; ys = [per[str(y)] for y in xs]
    c, mk, lab = STYLE[k]; ours = k == "agg3"
    ax.plot(xs, ys, color=c, lw=2.2 if ours else 1.0, marker=mk, ms=4.5 if ours else 2.6,
            mec="white" if ours else c, mew=0.8 if ours else 0.5, zorder=5 if ours else 3,
            label=lab if k in ("rev_b2h", "rev_b3a") else None)
ax.axhline(0, color="#999999", lw=0.7)
ax.set_xlabel("ICLR year", fontsize=7.5)
ax.set_ylabel("high minus low rated (pp)", fontsize=7.5)
ax.set_xticks(years); ax.set_xticklabels([str(y)[2:] for y in years], fontsize=7); tidy(ax)

ax = axes[2]
OFF = {"binoculars": (0, 8), "detectgpt": (0, -13), "nts": (0, -13), "rev_b2h": (0, 8), "rev_b3a": (0, -13), "agg3": (0, 9)}
for k in ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"]:
    if k not in SUMM:
        continue
    x, y = SUMM[k]["mean_gap"], PROV[k]
    c, mk, lab = STYLE[k]; ours = k == "agg3"
    ax.scatter([x], [y], color=c, marker=mk, s=70 if ours else 26, edgecolor="white" if ours else c,
               linewidth=0.8 if ours else 0.5, zorder=5 if ours else 3)
    ax.annotate(lab, xy=(x, y), xytext=OFF[k], textcoords="offset points", fontsize=5.8, ha="center",
                color="#111111" if ours else "#555555", fontweight="bold" if ours else "normal")
ax.axhline(0.5, color="#cccccc", lw=0.6, ls=":"); ax.axvline(0, color="#cccccc", lw=0.6, ls=":")
ax.set_xlabel("follows the reviewers (pp)", fontsize=7.5)
ax.set_ylabel("separates AI from human (AUROC)", fontsize=7.5)
ax.set_xlim(-42, 22); ax.set_ylim(0.44, 1.10); ax.set_yticks([0.5, 0.7, 0.9, 1.0]); tidy(ax)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=6.6, handlelength=1.8,
           columnspacing=1.0, bbox_to_anchor=(0.5, -0.03))
fig.subplots_adjust(left=0.072, right=0.995, top=0.97, bottom=0.30)
fig.savefig(f"{HERE}/results/fig_review_final.pdf"); fig.savefig(f"{HERE}/results/fig_review_final.png", dpi=300)

# ============================================================ figure 2
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.25), gridspec_kw={"wspace": 0.32, "width_ratios": [1.15, 1]})
ax = axes[0]
IL = {"macro_redund": ("Macro redundancy", "#95627A", "s", (0, 9)), "xsec_ref": ("Cross-section refs.", "#95627A", "^", (0, -14)),
      "citation": ("Citation isolation", "#E4959E", "D", (0, 9)), "evidence_gap": ("Evidence gap", "#6D8A96", "v", (0, 9)),
      "argument_graph": ("Argument graph", "#E4959E", "P", (0, -14)), "agg3": ("SciSlop (ours)", "#111111", "o", (0, 10))}
for k, (lab, c, mk, off) in IL.items():
    v = EX["two_axes"].get(k) or {}
    x, y = v.get("quality_gap_pp"), v.get("provenance_auroc")
    if x is None or y is None:
        continue
    ours = k == "agg3"
    ax.scatter([x], [y], color=c, marker=mk, s=70 if ours else 30, edgecolor="white" if ours else c,
               linewidth=0.8 if ours else 0.5, zorder=5 if ours else 3)
    ax.annotate(lab, xy=(x, y), xytext=off, textcoords="offset points", fontsize=5.8, ha="center",
                color="#111111" if ours else "#555555", fontweight="bold" if ours else "normal")
ax.axhline(0.5, color="#cccccc", lw=0.6, ls=":"); ax.axvline(0, color="#cccccc", lw=0.6, ls=":")
ax.set_xlabel("follows the reviewers (pp)", fontsize=7.5)
ax.set_ylabel("separates AI from human (AUROC)", fontsize=7.5)
ax.set_xlim(-26, 10); ax.set_ylim(0.44, 1.10); ax.set_yticks([0.5, 0.7, 0.9, 1.0]); tidy(ax)

ax = axes[1]
names = ["low slop", "middle", "high slop"]
for tag, c, mk, lab in [("all 2026 papers", "#888888", "o", "all ICLR 2026 papers"),
                        ("Pangram calls fully human", "#111111", "o", "Pangram calls them human")]:
    d = EX["layer"][tag]
    ys = [d[n]["mean_rating"] for n in names]
    ax.plot(range(3), ys, color=c, lw=2.2 if "Pangram" in tag else 1.2, marker=mk, ms=5 if "Pangram" in tag else 3.5,
            mec="white", mew=0.8, label=lab, zorder=4 if "Pangram" in tag else 3)
    for i, n in enumerate(names):
        ax.annotate(f"{d[n]['accept_rate']:.0f}%", xy=(i, ys[i]), xytext=(0, 8 if "Pangram" in tag else -13), textcoords="offset points",
                    fontsize=5.6, ha="center", color=c)
ax.set_xticks(range(3)); ax.set_xticklabels(["low", "middle", "high"], fontsize=7.5)
ax.set_xlabel("SciSlop tercile", fontsize=7.5); ax.set_ylabel("mean review score", fontsize=7.5)
ax.set_xlim(-0.3, 2.3); tidy(ax)
ax.legend(frameon=False, fontsize=6.4, loc="lower left", handlelength=1.8)
fig.subplots_adjust(left=0.095, right=0.995, top=0.97, bottom=0.21)
fig.savefig(f"{HERE}/results/fig_two_axes.pdf"); fig.savefig(f"{HERE}/results/fig_two_axes.png", dpi=300)
print("saved fig_review_final and fig_two_axes")
