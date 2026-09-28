"""Figures for the ICLR-2026 x Pangram corpus. Light-surface PNGs for the report.

Palette: dataviz reference categorical slots 1/2/3/7 (blue, orange, aqua, violet), fixed order,
validated for the light surface under all-pairs CVD separation. Every series is direct-labeled,
so identity never rests on color alone.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
BANDS = [("= 0", 0.0, 1e-12), ("0–.05", 1e-12, 0.05), (".05–.15", 0.05, 0.15),
         (".15–.30", 0.15, 0.30), (".30–.50", 0.30, 0.50), ("> .50", 0.50, 1.01)]
PRED_ORDER = ["Fully human-written", "Lightly AI-edited", "Moderately AI-edited",
              "Heavily AI-edited", "Fully AI-generated"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name))]


def banded(papers):
    out = []
    for lab, lo, hi in BANDS:
        if lab == "= 0":
            g = [p for p in papers if p["fraction_ai"] == 0]
        else:
            g = [p for p in papers if lo <= p["fraction_ai"] < hi]
        out.append((lab, g))
    return out


def fig_accept_rate(papers):
    groups = banded(papers)
    labs = [l for l, _ in groups]
    vals = [100 * np.mean([p["accept"] for p in g]) for _, g in groups]
    ns = [len(g) for _, g in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labs, vals, color=SERIES[0], width=0.62, zorder=3)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}%", ha="center",
                fontsize=10, color=INK, fontweight="medium")
        ax.text(b.get_x() + b.get_width() / 2, 0.8, f"n={n:,}", ha="center", fontsize=8, color=SURFACE)
    ax.set_ylabel("acceptance rate")
    ax.set_xlabel("Pangram AI fraction of the submission text")
    ax.set_title("More AI text, lower acceptance (ICLR 2026)",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.text(0.01, 0.005, f"n = {sum(ns):,} submissions with a Pangram verdict", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/accept_rate_by_pangram.png", dpi=200)
    plt.close(fig)


def fig_distribution(papers):
    fa = np.array([p["fraction_ai"] for p in papers])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvspan(0.639, 0.851, color=SERIES[1], alpha=0.14, zorder=1)
    ax.axvspan(0.0, 0.07, color=SERIES[2], alpha=0.14, zorder=1)
    ax.hist(fa, bins=50, color=SERIES[0], zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Pangram AI fraction of the submission text")
    ax.set_ylabel("submissions (log scale)")
    ax.annotate("pre-LLM landmark\npapers (0–7%)", xy=(0.07, 700), xytext=(0.22, 2500),
                fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("fully agent-written\nFARS papers (64–85%)", xy=(0.745, 90), xytext=(0.52, 900),
                fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_title("ICLR 2026 submissions against two reference points",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    n_fars = int((fa >= 0.639).sum())
    fig.text(0.01, 0.005, f"n = {len(fa):,}; {n_fars:,} ({100*n_fars/len(fa):.1f}%) reach the FARS band",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/pangram_distribution.png", dpi=200)
    plt.close(fig)


def fig_review_generosity(reviews):
    labs, vals, ns = [], [], []
    for k in PRED_ORDER:
        g = [r["rating"] for r in reviews if r["prediction"] == k and r["rating"] is not None]
        labs.append(k.replace(" ", "\n", 1))
        vals.append(float(np.mean(g)))
        ns.append(len(g))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labs, vals, color=SERIES[1], width=0.62, zorder=3)
    base = vals[0]
    ax.axhline(base, color=INK2, lw=1.2, ls="--", zorder=4)
    ax.text(4.42, base, " human\n baseline", va="center", fontsize=8.5, color=INK2)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 3.62, f"n={n:,}", ha="center", fontsize=8, color=SURFACE)
    ax.set_ylim(3.6, max(vals) * 1.04)
    ax.set_ylabel("mean rating given")
    ax.set_title("AI-written reviews are more generous (+0.30 rating)",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    fig.text(0.01, 0.005, f"n = {sum(ns):,} ICLR 2026 reviews, Pangram authorship verdict",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/review_generosity.png", dpi=200)
    plt.close(fig)


def fig_subscores(papers):
    keys = [("rating_mean", "rating"), ("soundness_mean", "soundness"),
            ("presentation_mean", "presentation"), ("contribution_mean", "contribution")]
    groups = banded(papers)
    labs = [l for l, _ in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ends = []
    for (key, name), c in zip(keys, SERIES):
        ys = []
        for _, g in groups:
            v = [p[key] for p in g if p.get(key) is not None]
            ys.append(float(np.mean(v)))
        ys = [100 * y / ys[0] for y in ys]          # indexed to the fully-human band = 100
        ax.plot(labs, ys, color=c, lw=2, marker="o", ms=8, zorder=3, label=name)
        ends.append((name, c, ys[-1]))
    ends.sort(key=lambda t: -t[2])
    span = max(e[2] for e in ends) - min(e[2] for e in ends)
    step = max(span / 3, 2.2)
    top = max(e[2] for e in ends) + step
    for i, (name, c, y) in enumerate(ends):
        ax.annotate(f"{name} {y:.0f}", xy=(len(labs) - 1, y), xytext=(len(labs) - 0.75, top - i * step),
                    color=c, fontsize=9, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.6))
    ax.axhline(100, color=MUTED, lw=1, ls="--", zorder=2)
    ax.set_ylabel("score, indexed to the fully-human band (= 100)")
    ax.set_xlabel("Pangram AI fraction of the submission text")
    ax.set_title("Every review dimension falls as AI fraction rises",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK2)
    ax.set_xlim(-0.3, len(labs) - 0.55)
    fig.subplots_adjust(right=0.80)
    fig.text(0.01, 0.005, f"n = {len(papers):,} submissions", fontsize=8, color=MUTED)
    fig.savefig(f"{FIGS}/subscores_by_pangram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_same_score(papers):
    """Holding the review score fixed, does AI text still cost acceptance?"""
    bands = [(0, 3.5, "< 3.5"), (3.5, 4.5, "3.5–4.5"), (4.5, 5.5, "4.5–5.5"),
             (5.5, 6.5, "5.5–6.5"), (6.5, 11, "≥ 6.5")]
    labs, v0, v1 = [], [], []
    for lo, hi, lab in bands:
        g = [p for p in papers if p.get("rating_mean") is not None and lo <= p["rating_mean"] < hi]
        a0 = [p["accept"] for p in g if p["fraction_ai"] == 0]
        a1 = [p["accept"] for p in g if p["fraction_ai"] > 0.30]
        if len(a0) < 30 or len(a1) < 30:
            continue
        labs.append(lab)
        v0.append(100 * np.mean(a0))
        v1.append(100 * np.mean(a1))
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, v0, w, color=SERIES[0], zorder=3, label="no AI text (fraction = 0)")
    ax.bar(x + w / 2 + 0.01, v1, w, color=SERIES[1], zorder=3, label="AI fraction > 0.30")
    for xi, (a, b) in enumerate(zip(v0, v1)):
        ax.text(xi - w / 2, a + 1.6, f"{a:.0f}%", ha="center", fontsize=9, color=SERIES[0])
        ax.text(xi + w / 2, b + 1.6, f"{b:.0f}%", ha="center", fontsize=9, color=SERIES[1])
        if b - a < -2:
            ax.annotate(f"{b - a:.0f} pp", xy=(xi, min(max(a, b) + 8, 99)), ha="center",
                        fontsize=9, color=INK)
    ax.set_xticks(x, labs)
    ax.set_ylabel("acceptance rate")
    ax.set_xlabel("mean review rating")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("Same scores, worse odds: the AI penalty is largest at the borderline",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.text(0.01, 0.005, "n = 19,303 ICLR 2026 submissions", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/accept_at_same_score.png", dpi=200)
    plt.close(fig)


def main():
    papers = [p for p in load("papers_2026.jsonl") if p["fraction_ai"] is not None]
    reviews = load("reviews_2026.jsonl")
    fig_accept_rate(papers)
    fig_distribution(papers)
    fig_review_generosity(reviews)
    fig_subscores([p for p in papers if p.get("rating_mean") is not None])
    fig_same_score(papers)
    print("wrote", sorted(os.listdir(FIGS)))


if __name__ == "__main__":
    main()
