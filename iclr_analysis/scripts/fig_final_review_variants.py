"""Render alternate panel (b) designs without replacing the original figure.

Run: python3 scripts/fig_final_review_variants.py
Outputs fig_final_review_ver{2,3,4}.{png,pdf} and a panel comparison.
All versions use the same estimates and the original p < 0.05 criterion.
"""
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.text import Text

BASE = runpy.run_path(str(Path(__file__).with_name("fig_final_review.py")))
RESULTS = Path(BASE["HERE"]) / "results"
ROWS, DIMS, CELLS = BASE["ROWS"], BASE["DIMS"], BASE["DM"]["cells"]
NEG, POS, NS = BASE["NEG"], BASE["POS"], "#AEB6BF"
TITLES = {
    2: "Direction + strength",
    3: "Meaning at a glance",
    4: "Direction + bar length",
}
NOTES = {
    2: "Arrow: change in AI judgment as review score rises     |     Number: association strength (absolute correlation)",
    3: "Read each cell as: higher review score → judged less / more AI-like.     n.s. = not statistically significant.",
    4: "Left: less AI-like     |     Right: more AI-like     |     Length: association strength (column edge = 0.15)",
}


def tint(color, amount):
    return tuple(1 - amount * (1 - value) for value in to_rgb(color))


def panel(ax, version, title):
    ax.clear()
    for yi, k in zip(range(6, -1, -1), ROWS):
        for xi, (dim, _) in enumerate(DIMS):
            cell = CELLS[k][dim]
            rho, significant = cell["rho"], cell["p"] < 0.05
            color = NEG if rho < 0 else POS
            arrow = "↓" if rho < 0 else "↑"
            if version == 2:
                # Grey cells retain non-significant estimates without implying no effect.
                face = tint(color, 0.16 + 0.64 * abs(rho) / 0.15) if significant else "#F3F4F6"
                ax.add_patch(Rectangle((xi - 0.49, yi - 0.47), 0.98, 0.94,
                                       facecolor=face, edgecolor="white", linewidth=0.7))
                ax.text(xi, yi, f"{arrow} {abs(rho):.2f}", ha="center", va="center", fontsize=8.2,
                        color=("white" if abs(rho) >= 0.11 else "#233640") if significant else "#8A929B")
            elif version == 3:
                ax.add_patch(Rectangle((xi - 0.46, yi - 0.41), 0.92, 0.82,
                                       facecolor=tint(color, 0.13) if significant else "#F6F7F8",
                                       edgecolor="none"))
                ax.text(xi, yi, f"{arrow} {'Less' if rho < 0 else 'More'}" if significant else "n.s.",
                        ha="center", va="center", fontsize=8.4,
                        color=color if significant else "#A2A9B1", weight="bold" if significant else "normal")
            else:
                ax.add_patch(Rectangle((xi - 0.46, yi - 0.44), 0.92, 0.88,
                                       facecolor="#F6F7F9", edgecolor="none", zorder=0))
                ax.barh(yi, rho / 0.15 * 0.42, left=xi, height=0.49,
                        color=color if significant else "#CCD1D7", zorder=2)
        ax.plot([-0.65, -0.65], [yi - 0.29, yi + 0.29], color=BASE["COL"][k], lw=2.5, clip_on=False)
    if version == 4:
        for xi in range(4):
            ax.plot([xi, xi], [-0.45, 6.45], color="#8D97A2", lw=0.55, zorder=3)
    for y in (2.5, 0.5):
        ax.axhline(y, color="white", lw=3.5, zorder=4)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_xticks(range(4), ["Overall", "Soundness", "Presentation", "Contribution"])
    ax.set_yticks(range(6, -1, -1), [BASE["LAB"][k] for k in ROWS])
    ax.tick_params(axis="x", length=0, labelsize=7, pad=5)
    ax.tick_params(axis="y", length=0, labelsize=8, pad=12)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("ICLR 2026 review dimensions", fontsize=9, labelpad=5)
    ax.set_title(title, fontsize=10, pad=11, loc="left")


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1:
            raise ValueError(f"Text outside figure: {text.get_text()}")
    for ax in fig.axes:
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            boxes = [t.get_window_extent(renderer) for t in labels if t.get_visible()]
            if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
                raise ValueError("Overlapping tick labels")


def save(fig, stem):
    check_layout(fig)
    fig.savefig(RESULTS / f"{stem}.png", dpi=300)
    fig.savefig(RESULTS / f"{stem}.pdf")
    print(f"Saved {stem}.png and .pdf")


fig = BASE["fig"]
BASE["key"].remove()
for text in list(fig.texts):
    if text.get_text().startswith("Filled:"):
        text.remove()
# The reading direction stays explicit in all alternatives.
key = fig.text(0.52, 0.225,
               "(b)  Higher review score:    ↓ Less AI-like     /     ↑ More AI-like",
               ha="center", va="center", fontsize=8.4)
sig = fig.text(0.52, 0.181,
               "Blue / red: significant association (p < 0.05)     |     Gray: not significant",
               ha="center", va="center", fontsize=7.5, color="#626D78")
note = fig.text(0.52, 0.143, "", ha="center", va="center", fontsize=7.1, color="#626D78")
for version in (2, 3, 4):
    panel(BASE["axes"][1], version, "(b)  What higher review scores imply")
    key.set_text("(b)  Higher review score:    " +
                 ("← Less AI-like     /     More AI-like →" if version == 4 else
                  "↓ Less AI-like     /     ↑ More AI-like"))
    note.set_text(NOTES[version])
    save(fig, f"fig_final_review_ver{version}")

# A separate comparison puts only the changing panel next to its alternatives.
comparison, axs = plt.subplots(1, 3, figsize=(13.8, 3.7))
comparison.subplots_adjust(left=0.09, right=0.99, top=0.85, bottom=0.25, wspace=0.64)
for version, ax in zip((2, 3, 4), axs):
    panel(ax, version, f"ver{version}  ·  {TITLES[version]}")
comparison.text(0.54, 0.115,
                "As review scores rise: blue = less AI-like; red = more AI-like.    Gray = not significant (p ≥ 0.05).",
                ha="center", fontsize=9)
comparison.text(0.54, 0.052,
                "ver2: numbers show absolute correlation     |     ver3: direction of significant associations     |     ver4: bar length shows strength",
                ha="center", fontsize=8, color="#626D78")
save(comparison, "fig_final_review_versions_comparison")
