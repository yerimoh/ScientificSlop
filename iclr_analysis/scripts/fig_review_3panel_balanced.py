"""Fig 3, year x domain balanced (0916 night). Reads results/balanced.json (analyze_balanced.py) and draws the same
three panels as fig_review_3panel.py with post-stratified means. Ours is drawn twice: solid = SciSlop by the paper's
aggregation rule (plane mean of the five measures that run on ICLR), dashed = the three deterministic measures
(macro, cross-section, citation), the subset that panel (c) shows to carry the within-human gradient.
Output results/fig_review_scores3_balanced.{pdf,png}."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _systems import HERE, GROUPS, BANDS, ORDER, ITEMS, STYLE
BAL = json.load(open(f"{HERE}/results/balanced.json"))
A, Bp, C = BAL["panel_a"], BAL["panel_b"], BAL["panel_c"]
# panel (a) composition (0916 night, user): x = within-year rating quartile from analyze_compositions.py, so every year
# contributes equally to every bin and the year-specific rating scales do not mix. --bands restores the absolute bands.
QUART = "--bands" not in sys.argv
if QUART:
    CQ = json.load(open(f"{HERE}/results/compositions.json"))["compositions"]["Q_within_year_quartile"]
    A = {k: {"balanced_bands": v["means"], "balanced_ci": v["ci"]} for k, v in CQ["systems"].items()}
    XT_A = [f"Q{k+1}\n({m:.1f})" for k, m in enumerate(CQ["bin_mean_rating"])]
else:
    XT_A = [f"{a}–{b}" for a, b in BANDS]
LABELS = {"binoculars": "Binoculars", "detectgpt": "DetectGPT", "nts": "NTS", "rev_b2h": "CycleReviewer", "rev_b3a": "AI Scientist",
          "ours": "SciSlop (ours, 5 measures)", "agg3": "SciSlop (ours, 3 deterministic)"}
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.08], "wspace": 0.34})
XT_G = ["FARS\n(AI)", "ICLR\nreject", "ICLR\naccept", "ICLR\noral"]
LINES = ORDER + ["agg3"]
for ax, data, key, xt in [(ax1, A, "balanced_bands", XT_A), (ax2, Bp, "balanced_groups", XT_G)]:
    for name in LINES:
        c, mk = STYLE["ours" if name == "agg3" else name]; ours = name in ("ours", "agg3")
        ax.plot(range(4), data[name][key], color=c, lw=2.0 if ours else 1.0, ls="--" if name == "agg3" else "-", marker=mk, ms=5 if ours else 3,
                mfc="white" if name == "agg3" else c, mec=c if name == "agg3" else ("white" if ours else c), mew=0.8 if ours else 0.5, zorder=6 if name == "ours" else (5 if ours else 3), label=LABELS[name])
    ax.set_xticks(range(4)); ax.set_xticklabels(xt, fontsize=7.2)
for ax in (ax1, ax2, ax3):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=7.2, color="#555555")
    ax.set_xlim(-0.3, 3.3)
ax1.set_ylim(30, 70); ax1.set_yticks([30, 40, 50, 60, 70]); ax1.set_xlabel("review-score quartile within ICLR year (mean score)" if QUART else "mean review score of ICLR papers", fontsize=7.2)
ax1.set_ylabel("AI-likeness (score percentile)", fontsize=7.2)
ax1.axhline(50, color="#cccccc", lw=0.6, ls=":", zorder=1)
ax2.set_ylim(30, 95); ax2.set_yticks([30, 50, 70, 90]); ax2.set_xlabel("paper group", fontsize=7.2)
ax2.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
NUDGE = {"macro_redund": 5, "fig_exposition": -4, "citation": 3, "argument_graph": -3, "evidence_gap": 2}
for k, lab, col, ls, mk in ITEMS:
    g = C[k]["groups"]; y = [g[gr]["mean"] for gr in GROUPS]
    lo = [y[i] - g[gr]["ci95"][0] for i, gr in enumerate(GROUPS)]; hi = [g[gr]["ci95"][1] - y[i] for i, gr in enumerate(GROUPS)]
    ax3.errorbar(range(4), y, yerr=[lo, hi], color=col, ls=ls, lw=1.0, marker=mk, ms=3.2, mfc="white" if ls == "--" else col, mew=0.7, capsize=1.3, elinewidth=0.5, label=lab, zorder=3)
    ax3.annotate(lab, xy=(3, y[3]), xytext=(3, NUDGE.get(k, 0)), textcoords="offset points", fontsize=5.6, color="#333333", va="center", ha="left")
ax3.set_xticks(range(4)); ax3.set_xticklabels(XT_G, fontsize=7.2)
ax3.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
ax3.set_ylim(0, 1.0); ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax3.set_ylabel("slop score (mean, 95% CI)", fontsize=7.2); ax3.set_xlabel("paper group", fontsize=7.2)
for ax, t in [(ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")]:
    ax.text(-0.22, 1.02, t, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=6.6, handlelength=2.0, columnspacing=0.9, bbox_to_anchor=(0.45, -0.02))
fig.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.30)
suf = "" if QUART else "_bands"
fig.savefig(f"{HERE}/results/fig_review_scores3_balanced{suf}.pdf"); fig.savefig(f"{HERE}/results/fig_review_scores3_balanced{suf}.png", dpi=300)
print("saved fig_review_scores3_balanced")
