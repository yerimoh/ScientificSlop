"""Fig 3 for §5.2 paragraph 3. One row, two panels, shared legend below.
Left : x = mean public review score in four bands (2-3, 4-5, 6-7, 8-9), y = each system's score as a within-sample
       percentile oriented toward AI (detectors: AI score up; reviewers: lower overall = up; SciSlop: slop up).
Right: the same percentile (pooled FARS + ICLR) for FARS AI papers and rejected / accepted / oral ICLR papers.
SciSlop in both panels = mean of the three deterministic measures that run on ICLR (macro redundancy, cross-section
references, citation). Data: results/papers_stairs.jsonl (quality-filtered), results/detectors/*.jsonl,
results/reviews/<sys>/*.json, bench165 results for the FARS side.
"""
import json, os, glob, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"; ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
GROUPS = ["FARS", "reject", "accept", "oral"]
group = {k: v["group"] for k, v in P.items() if v["group"] in GROUPS}
rating = {k: v["rating"] for k, v in P.items() if v.get("rating") is not None and v["src"] == "iclr"}
items = json.load(open(f"{B}/items165.json"))["items"]; ai_codes = {i["pair"] for i in items if i["label"] == 1}


def agg3(v):
    if any(v.get(k) is None for k in ("macro_redund", "xsec_ref", "citation")): return None
    return ((v["macro_redund"] + v["xsec_ref"]) / 2 + v["citation"]) / 2


def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)): out[r["id"]] = r[key]
    return out


S, DIR, LABEL = {}, {}, {}
for name, key, d, label in [("binoculars", "binoculars", -1, "Binoculars"), ("detectgpt", "detectgpt", +1, "DetectGPT"), ("nts", "nts", +1, "NTS")]:
    d_f = {k[3:]: v for k, v in jl(f"{B}/results/{key}.jsonl", key).items() if k.startswith("AI_") and k[3:] in ai_codes}
    S[name] = {**d_f, **jl(f"{HERE}/results/detectors/{key}*.jsonl", key)}; DIR[name] = d; LABEL[name] = label
for sysname, dname, label in [("b2h", "B2h_cyclereviewer", "CycleReviewer"), ("b3a", "B3a_ai_scientist", "AI Scientist")]:
    d = {}
    for code in ai_codes:
        fp = f"{ARCH}/{dname}/runs/{code}/logs/{code}_{sysname}_R1_review.json"
        if os.path.exists(fp):
            v = (json.load(open(fp)).get("final") or {}).get("Overall")
            if isinstance(v, (int, float)): d[code] = v
    for f in glob.glob(f"{HERE}/results/reviews/{sysname}/*.json"):
        r = json.load(open(f)); fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)): d[r["item_id"]] = v
    S[f"rev_{sysname}"] = d; DIR[f"rev_{sysname}"] = -1; LABEL[f"rev_{sysname}"] = label
S["scislop"] = {k: agg3(v) for k, v in P.items() if agg3(v) is not None}; DIR["scislop"] = +1; LABEL["scislop"] = "SciSlop (ours)"
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "scislop"]
STYLE = {"binoculars": ("#2a78d6", "o"), "detectgpt": ("#1baf7a", "s"), "nts": ("#eda100", "D"), "rev_b2h": ("#eb6834", "v"), "rev_b3a": ("#e87ba4", "P"), "scislop": ("#111111", "o")}
BANDS = [(2, 3), (4, 5), (6, 7), (8, 9)]


def band_of(r):
    r = int(round(r))
    for k, (a, b) in enumerate(BANDS):
        if a <= r <= b: return k
    return None


def pct(d, ids):
    vals = np.array([d[i] * DIR_cur for i in ids], float)
    return dict(zip(ids, stats.rankdata(vals) / len(vals) * 100))


out = {"left": {}, "right": {}}
left, right = {}, {}
for name in ORDER:
    DIR_cur = DIR[name]
    ids_l = [i for i in S[name] if i in rating and group.get(i) in ("reject", "accept", "oral")]
    pl = pct(S[name], ids_l)
    rho = stats.spearmanr([rating[i] for i in ids_l], [pl[i] for i in ids_l])[0]
    left[name] = [float(np.mean([pl[i] for i in ids_l if band_of(rating[i]) == k])) for k in range(len(BANDS))]
    out["left"][name] = {"n": len(ids_l), "rho_vs_rating": round(float(rho), 3), "bands": [round(x, 1) for x in left[name]]}
    ids_r = [i for i in S[name] if i in group]
    pr = pct(S[name], ids_r)
    right[name] = [float(np.mean([pr[i] for i in ids_r if group[i] == g])) for g in GROUPS]
    out["right"][name] = {"n": {g: sum(1 for i in ids_r if group[i] == g) for g in GROUPS}, "groups": [round(x, 1) for x in right[name]]}
json.dump(out, open(f"{HERE}/results/fig_review_2panel.json", "w"), indent=1)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
for ax, data, xs, xt in [(ax1, left, list(range(4)), [f"{a}–{b}" for a, b in BANDS]), (ax2, right, list(range(4)), ["FARS\n(AI)", "ICLR\nreject", "ICLR\naccept", "ICLR\noral"])]:
    for name in ORDER:
        c, mk = STYLE[name]; ours = name == "scislop"
        ax.plot(xs, data[name], color=c, lw=2.2 if ours else 1.0, marker=mk, ms=5 if ours else 3, mec="white" if ours else c, mew=0.8 if ours else 0.5, zorder=5 if ours else 3, label=LABEL[name])
    ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=7.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555"); ax.tick_params(length=2.5, labelsize=7.5, color="#555555")
    ax.set_xlim(-0.3, 3.3)
ax1.set_ylim(30, 70); ax1.set_yticks([30, 40, 50, 60, 70]); ax1.set_xlabel("mean review score of ICLR papers", fontsize=7.5)
ax1.set_ylabel("AI-likeness (score percentile)", fontsize=7.5)
ax2.set_ylim(30, 95); ax2.set_yticks([30, 50, 70, 90]); ax2.set_xlabel("paper group", fontsize=7.5)
ax2.axvline(0.5, color="#cccccc", lw=0.6, ls=":", zorder=1)
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.30)
fig.savefig(f"{HERE}/results/fig_review_scores.pdf"); fig.savefig(f"{HERE}/results/fig_review_scores.png", dpi=300)
for name in ORDER:
    print(f"{name:12s} left n={out['left'][name]['n']:5d} rho={out['left'][name]['rho_vs_rating']:+.3f} bands={out['left'][name]['bands']} | right={out['right'][name]['groups']}")
