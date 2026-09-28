"""Component ablation drawn on the two axes that decide it: slop left, and what the manuscript paid.

The six-panel trajectory figure and the share-of-gap dot plot both draw one axis, the score, on which the
unreviewed editor looks best. What makes the full method the right choice is the second axis, the rounds
that broke a hard guard, which those figures carry only as a number in a legend. Here every condition is
one row, ordered by damage, with (a) how far each measure still sits from the human mean after round 3
and (b) the guard-breaking rounds, so the reader sees the trade-off and the one row that is far left in (a)
and empty in (b).

Same data rules as summarize_ablation.py: 40 papers finished in every arm, macro redundancy on min(1, s/0.10),
a stopped run carries its last manuscript forward, human = matched human paper. Overshoot past the human mean
is drawn hollow and counts as distance, not as gain.

  python3 fig_component_tradeoff.py
"""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ABL = HERE.parent
sys.path.insert(0, str(HERE))
import summarize_ablation as SA                      # agg, value_at, load, ARMS, ITEMS, paths
S6 = SA.S6
V9_FIG = ABL.parents[2] / "draft_v9/figures/main_figures"

ITEMS = SA.ITEMS
NAME = SA.LONG
PLANE = {"macro_redund": "#95627A", "xsec_ref": "#95627A", "argument_graph": "#E4959E", "citation": "#E4959E",
         "fig_exposition": "#6D8A96", "evidence_gap": "#6D8A96"}
INK, MUTED, HAIR, GRID = "#1A1A1A", "#6E6E6E", "#C8C8C8", "#E8E8E8"
HUMAN = "#3E7A54"; BAND = "#F3F1EC"; BRICK = "#B3453A"; NUM = "#9A8080"; OTHER = "#CFC6C6"
ROWS = [("nogate", "Definitions + locations", (True, True, False)),
        ("gateonly", "Review only", (False, False, True)),
        ("noloc", "Definitions + review", (True, False, True)),
        ("full", "SciSlopHarness", (True, True, True))]
GUARD_KIND = {"cited_works_removed": ("removed a cited work", BRICK),
              "numbers_lost": ("lost a reported number", NUM),
              "dangling_cites": ("other", OTHER), "refs_undefined": ("other", OTHER), "cited_works_added": ("other", OTHER)}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def main():
    table = json.loads((ABL / "results_table.json").read_text())
    common = table["common_codes"]; assert len(common) == 40
    runs = {arm: SA.load(ABL / d / "results.jsonl") for arm, d in SA.RESULT_DIRS.items()}
    runs["full"] = SA.load(SA.API / "final_v14" / "results.jsonl")
    hu = S6.human_scores()
    rng = np.random.default_rng(922)

    # ---- distances per item, per arm, with a bootstrap over papers of the six-item mean distance
    def item_arrays(arm, item, n):
        codes = [c for c in common if runs["full"][c]["per_round"].get("R0", {}).get(item) is not None]
        x = np.array([SA.value_at(runs[arm][c]["per_round"], item, n) for c in codes], float)
        h = np.array([SA.agg(item, (hu.get(c) or {}).get(item)) if (hu.get(c) or {}).get(item) is not None else np.nan for c in codes], float)
        return x, h

    dist, signed, mean_ci = {}, {}, {}
    for arm in ["original"] + [r[0] for r in ROWS]:
        src, n = ("full", 0) if arm == "original" else (arm, 3)
        per_item_x, per_item_h = {}, {}
        for item in ITEMS:
            x, h = item_arrays(src, item, n); per_item_x[item], per_item_h[item] = x, h
            d = np.nanmean(x) - np.nanmean(h)
            signed[(arm, item)] = float(d); dist[(arm, item)] = abs(float(d))
        point = float(np.mean([dist[(arm, i)] for i in ITEMS]))
        m = len(common); draws = []
        for _ in range(4000):
            idx = rng.integers(0, m, m)
            draws.append(np.mean([abs(np.nanmean(per_item_x[i][idx]) - np.nanmean(per_item_h[i][idx])) for i in ITEMS]))
        mean_ci[arm] = (point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    # cross-check against the numbers the paper table already carries
    for arm in [r[0] for r in ROWS]:
        for item in ITEMS:
            assert abs(dist[(arm, item)] - table["items"][item]["abs_distance_to_human"][arm][3]) < 0.002, (arm, item)

    # ---- guard-breaking rounds per arm, by kind, on the same 40 papers
    # A round is counted once, under its most severe violation, so the kinds partition the rounds and the
    # stacked bar is exactly as long as the share of rounds that broke a guard.
    SEVERITY = ["removed a cited work", "lost a reported number", "other"]
    guards = {}
    for arm in [r[0] for r in ROWS]:
        kinds, papers, rounds, total = {k: 0 for k in SEVERITY}, 0, 0, 0
        for c in common:
            g = runs[arm][c].get("guards") or []
            total += len(g)
            hit = False
            for _, viol in g:
                if viol:
                    rounds += 1; hit = True
                    worst = min((GUARD_KIND[v][0] for v in viol), key=SEVERITY.index)
                    kinds[worst] += 1
            papers += hit
        guards[arm] = {"rounds": rounds, "papers": papers, "total_rounds": total, "kinds": kinds,
                       "share_rounds": rounds / total if total else 0.0, "share_papers": papers / len(common),
                       "share_kinds": {k: v / total if total else 0.0 for k, v in kinds.items()}}
        assert sum(kinds.values()) == rounds
    assert {a: guards[a]["rounds"] for a in guards} == {"nogate": 28, "gateonly": 18, "noloc": 12, "full": 0}, guards

    # ---- draw
    W, H = 5.5, 2.15
    fig = plt.figure(figsize=(W, H))
    top, bottom = 0.80, 0.27
    rows_all = ["original"] + [r[0] for r in ROWS]
    ys = {a: top - (i + 0.5) * (top - bottom) / len(rows_all) for i, a in enumerate(rows_all)}
    label_x, kept_x = 0.235, [0.262, 0.292, 0.322]
    axA = fig.add_axes([0.365, bottom, 0.30, top - bottom]); axB = fig.add_axes([0.760, bottom, 0.232, top - bottom])
    for ax in (axA, axB):
        ax.set_zorder(1); ax.patch.set_alpha(0)
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(HAIR); ax.tick_params(axis="x", labelsize=6.2, length=2, pad=1.5, color=HAIR)
    rh = (top - bottom) / len(rows_all)
    yfrac = {a: (ys[a] - bottom) / (top - bottom) for a in rows_all}
    # the chosen row, banded across the whole width
    fig.patches.append(Rectangle((0.01, ys["full"] - rh * 0.46), 0.985, rh * 0.92, transform=fig.transFigure,
                                 fc=BAND, ec="none", zorder=-1, figure=fig))
    fig.add_artist(Line2D([0.01, 0.995], [ys["original"] - rh * 0.5] * 2, color=HAIR, lw=0.5))

    # row labels and the kept-components glyphs
    fig.text(label_x, ys["original"], "Original", fontsize=6.6, color=MUTED, ha="right", va="center")
    for arm, lab, kept in ROWS:
        fig.text(label_x, ys[arm], lab, fontsize=6.6, color=INK, ha="right", va="center",
                 weight="bold" if arm == "full" else "normal")
        for x, k in zip(kept_x, kept):
            fig.text(x, ys[arm], "✓" if k else "–", fontsize=6.4, color=INK if k else HAIR, ha="center", va="center")
    for x, h in zip(kept_x, ["Def.", "Loc.", "Rev."]):
        fig.text(x, top + 0.035, h, fontsize=5.6, color=MUTED, ha="center", va="bottom")

    # (a) distance from the human mean
    axA.set_xlim(0, 0.56); axA.set_xticks([0, 0.2, 0.4]); axA.set_xticklabels(["0", "0.2", "0.4"])
    axA.axvline(0, color=HUMAN, lw=1.15, zorder=2)
    for t in (0.2, 0.4):
        axA.axvline(t, color=GRID, lw=0.45, zorder=0)
    for arm in rows_all:
        y = yfrac[arm]
        for item in ITEMS:
            d = dist[(arm, item)]; below = signed[(arm, item)] < -1e-9
            axA.plot([d], [y], marker="o", markersize=3.6, markerfacecolor="white" if below else PLANE[item],
                     markeredgecolor=PLANE[item], markeredgewidth=0.9, lw=0, zorder=3, alpha=0.95)
        m, lo, hi = mean_ci[arm]
        axA.plot([lo, hi], [y, y], color=INK if arm != "original" else MUTED, lw=0.8, zorder=4)
        axA.plot([m], [y], marker="D", markersize=4.2, color=INK if arm != "original" else MUTED,
                 markeredgecolor="white", markeredgewidth=0.6, lw=0, zorder=5)
        fig.text(0.735, ys[arm], f"{m:.2f}", fontsize=6.0, color=INK if arm != "original" else MUTED, ha="right", va="center",
                 family="DejaVu Sans Mono", weight="bold" if arm == "full" else "normal")
    axA.set_title("(a) Slop left after round 3, distance from the human mean", fontsize=6.2, loc="left", pad=3, color=INK)
    fig.text(0.735, bottom - 0.04, "mean", fontsize=5.6, color=MUTED, ha="right", va="top")

    # (b) rounds that broke a hard guard, stacked by kind
    axB.set_xlim(0, 68); axB.set_xticks([0, 10, 20, 30]); axB.set_xticklabels(["0", "10", "20", "30%"])
    for t in (10, 20, 30):
        axB.axvline(t, color=GRID, lw=0.45, zorder=0)
    order = SEVERITY; kind_col = {"removed a cited work": BRICK, "lost a reported number": NUM, "other": OTHER}
    for arm, _, _ in ROWS:
        y = yfrac[arm]; x = 0.0; g = guards[arm]
        for k in order:
            w = 100 * g["share_kinds"][k]
            if w:
                axB.add_patch(Rectangle((x, y - 0.055), w, 0.11, fc=kind_col[k], ec="white", lw=0.4, zorder=3)); x += w
        if g["rounds"]:
            axB.text(x + 0.9, y, f"{100 * g['share_rounds']:.0f}%, {g['papers']}/{len(common)} papers",
                     fontsize=5.3, color=INK, ha="left", va="center")
        else:
            axB.text(0.6, y, "0%", fontsize=6.4, color=HUMAN, ha="left", va="center", weight="bold")
    axB.set_title("(b) Rounds that broke a hard guard", fontsize=6.2, loc="left", pad=3, color=INK)

    # keys, outside the frame
    h1 = [Line2D([], [], marker="o", lw=0, markersize=3.6, markerfacecolor=c, markeredgecolor=c, label=l)
          for c, l in (("#95627A", "Structure"), ("#E4959E", "Argument"), ("#6D8A96", "Artifacts"))]
    h2 = [Line2D([], [], marker="o", lw=0, markersize=3.6, markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=0.9,
                 label="crossed below the human mean"),
          Line2D([], [], marker="D", lw=0.8, markersize=4.2, color=INK, label="mean of the six, 95% CI")]
    fig.legend(handles=h1, loc="center left", bbox_to_anchor=(0.365, 0.105), ncol=3, frameon=False, fontsize=5.5,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.0, borderpad=0)
    fig.legend(handles=h2, loc="center left", bbox_to_anchor=(0.365, 0.045), ncol=2, frameon=False, fontsize=5.5,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.0, borderpad=0)
    short = {"removed a cited work": "cited work removed", "lost a reported number": "number lost", "other": "other"}
    hb = [Rectangle((0, 0), 1, 1, fc=kind_col[k], ec="none", label=short[k]) for k in order]
    fig.legend(handles=hb, loc="center left", bbox_to_anchor=(0.655, 0.105), ncol=3, frameon=False, fontsize=5.5,
               handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderpad=0)

    for ext in ("pdf", "png"):
        fig.savefig(ABL / f"fig_component_tradeoff.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    (ABL / "fig_component_tradeoff_data.json").write_text(json.dumps(
        {"n": len(common), "distance": {f"{a}:{i}": dist[(a, i)] for a in rows_all for i in ITEMS},
         "signed": {f"{a}:{i}": signed[(a, i)] for a in rows_all for i in ITEMS},
         "mean_distance_ci": mean_ci, "guards": guards}, indent=1))
    if V9_FIG.is_dir():
        shutil.copy2(ABL / "fig_component_tradeoff.pdf", V9_FIG / "fig_component_tradeoff.pdf")
    print("mean distance (point, lo, hi):", {a: tuple(round(v, 3) for v in mean_ci[a]) for a in rows_all})
    print("guards:", guards)
    print("wrote", ABL / "fig_component_tradeoff.pdf", "and", V9_FIG / "fig_component_tradeoff.pdf")


if __name__ == "__main__":
    main()
