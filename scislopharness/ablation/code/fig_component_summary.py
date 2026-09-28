"""Main-text summary of the component ablation: one mean and one cost per condition.

Reads fig_component_tradeoff_data.json (written by fig_component_tradeoff.py from the same 40 papers) so the
two figures cannot disagree. Left, the six-item mean distance from the human mean after round 3 with its 95%
bootstrap interval; right, the rounds that broke a hard guard. The per-item view is the appendix figure.
Also writes tab_component_summary.tex, the same four numbers as a table, for the author to choose from.

  python3 fig_component_summary.py
"""
from __future__ import annotations
import os
import json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent; ABL = HERE.parent
V9 = Path(os.environ.get("SCISLOP_PAPER_DIR", "."))
INK, MUTED, HAIR, GRID = "#1A1A1A", "#6E6E6E", "#C8C8C8", "#E8E8E8"
HUMAN, BAND, BRICK = "#3E7A54", "#F3F1EC", "#B3453A"
ROWS = [("original", "Original", None), ("nogate", "Definitions + locations", (1, 1, 0)), ("gateonly", "Review only", (0, 0, 1)),
        ("noloc", "Definitions + review", (1, 0, 1)), ("full", "SciSlopHarness", (1, 1, 1))]
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})

d = json.loads((ABL / "fig_component_tradeoff_data.json").read_text())
mean = d["mean_distance_ci"]; guards = d["guards"]

W, H = 2.75, 1.10
fig = plt.figure(figsize=(W, H))
top, bottom = 0.86, 0.20
n = len(ROWS); rh = (top - bottom) / n
ys = {a: top - (i + 0.5) * rh for i, (a, _, _) in enumerate(ROWS)}
label_x, kept_x = 0.30, [0.345, 0.395, 0.445]
axA = fig.add_axes([0.50, bottom, 0.19, top - bottom]); axB = fig.add_axes([0.815, bottom, 0.175, top - bottom])
for ax in (axA, axB):
    ax.set_zorder(1); ax.patch.set_alpha(0); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False); ax.spines["bottom"].set_color(HAIR)
    ax.tick_params(axis="x", labelsize=5.8, length=1.6, pad=1.2, color=HAIR)
fig.patches.append(Rectangle((0.01, ys["full"] - rh * 0.47), 0.985, rh * 0.94, transform=fig.transFigure, fc=BAND, ec="none", zorder=-1, figure=fig))
fig.add_artist(plt.Line2D([0.01, 0.995], [ys["original"] - rh * 0.5] * 2, color=HAIR, lw=0.5))
for x, h in zip(kept_x, ["Def.", "Loc.", "Rev."]):
    fig.text(x, top + 0.02, h, fontsize=4.8, color=MUTED, ha="center", va="bottom")
for a, lab, kept in ROWS:
    fig.text(label_x, ys[a], lab, fontsize=5.6, color=MUTED if a == "original" else INK, ha="right", va="center",
             weight="bold" if a == "full" else "normal")
    if kept:
        for x, k in zip(kept_x, kept):
            fig.text(x, ys[a], "✓" if k else "–", fontsize=5.8, color=INK if k else HAIR, ha="center", va="center")
yf = {a: (ys[a] - bottom) / (top - bottom) for a in ys}

axA.set_xlim(0, 0.37); axA.set_xticks([0, 0.1, 0.2, 0.3]); axA.set_xticklabels(["0", ".1", ".2", ".3"])
axA.axvline(0, color=HUMAN, lw=1.05, zorder=2)
for t in (0.1, 0.2, 0.3):
    axA.axvline(t, color=GRID, lw=0.4, zorder=0)
for a, _, _ in ROWS:
    m, lo, hi = mean[a]; c = MUTED if a == "original" else INK
    axA.plot([lo, hi], [yf[a]] * 2, color=c, lw=0.8, zorder=3)
    axA.plot([m], [yf[a]], marker="D", markersize=3.8, color=c, markeredgecolor="white", markeredgewidth=0.5, lw=0, zorder=4)
    fig.text(0.79, ys[a], f"{m:.2f}", fontsize=5.4, color=c, ha="right", va="center", family="DejaVu Sans Mono",
             weight="bold" if a == "full" else "normal")
axA.set_title("Distance from human", fontsize=5.4, loc="left", pad=2.2, color=INK)

axB.set_xlim(0, 36); axB.set_xticks([0, 10, 20, 30]); axB.set_xticklabels(["0", "10", "20", "30%"])
for t in (10, 20, 30):
    axB.axvline(t, color=GRID, lw=0.4, zorder=0)
for a, _, kept in ROWS:
    if not kept:
        continue
    g = guards[a]; share = 100 * g["share_rounds"]
    if g["rounds"]:
        axB.add_patch(Rectangle((0, yf[a] - 0.07), share, 0.14, fc=BRICK, ec="none", zorder=3))
        axB.text(share + 0.8, yf[a], f"{share:.0f}%", fontsize=5.6, color=INK, ha="left", va="center")
    else:
        axB.text(0.8, yf[a], "0%", fontsize=6.0, color=HUMAN, ha="left", va="center", weight="bold")
axB.set_title("Guard breaks", fontsize=5.4, loc="left", pad=2.2, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(ABL / f"fig_component_summary.{ext}", dpi=300 if ext == "png" else None)
plt.close(fig)
shutil.copy2(ABL / "fig_component_summary.pdf", V9 / "figures/main_figures/fig_component_summary.pdf")

# the same numbers as a four-row table, for the author to choose instead of the figure
rows_tex = []
for a, lab, kept in ROWS:
    if not kept:
        continue
    m, lo, hi = mean[a]; g = guards[a]
    k = " & ".join("\\yes" if v else "\\no" for v in kept)
    name = "\\textbf{\\methodName}" if a == "full" else lab
    sh = f"{100 * g['share_rounds']:.0f}\\%"
    rows_tex.append(f"{name} & {k} & {m:.2f} [{lo:.2f}, {hi:.2f}] & {'\\textbf{0\\%}' if g['rounds'] == 0 else sh} \\\\")
tab = r"""% Component ablation, main-text summary (0923). Same 40 papers and numbers as fig_component_summary /
% fig_component_tradeoff_data.json. Distance = mean over the six measures of |round-3 mean - human mean|,
% 95% bootstrap interval over papers. Guard = rounds that removed a cited work, lost a reported number, or
% left a dangling reference. Per-item values are in the appendix (tab_component_ablation, fig_component_tradeoff).
\providecommand{\yes}{$\checkmark$}
\providecommand{\no}{$\times$}
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}l ccc c r@{}}
\toprule
Condition & Def. & Loc. & Rev. & \shortstack[c]{Distance from human mean\\after round 3, six measures} & \shortstack[c]{Rounds breaking\\a guard} \\
\midrule
""" + "\n".join(rows_tex[:-1]) + "\n\\midrule\n" + rows_tex[-1] + r"""
\bottomrule
\end{tabular}
\caption{}
\label{tab:component_summary}
\end{table}
"""
(V9 / "tables/main_tables/tab_component_summary.tex").write_text(tab)
print("means:", {a: [round(v, 3) for v in mean[a]] for a in mean}); print("wrote fig_component_summary + tab_component_summary")
