"""x = public review rating (within-year quintile), y = each system's score as a within-corpus percentile
oriented so that higher = more AI-like (detectors), lower overall = more AI-like (reviewers), higher = more slop (ours).
One line per baseline, one for the SciSlop aggregate. Reads whatever is available:
  results/detectors/{binoculars.s*.jsonl, detectgpt.s*.jsonl, fast_detectgpt.jsonl, nts.jsonl}   (fresh ICLR runs)
  results/reviews/{b3a,b2h}/*.json                                                                 (reviewer subset)
  --archive : fall back to claude/detectors/iclr9y_detectors.pkl (0908 Binoculars/DetectGPT, mixed tex/OCR text) for the overlap
Writes results/rating_systems.json and results/fig_rating_systems.{pdf,png}.
"""
import json, os, glob, sys, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SCISLOP_ROOT", ".")
ARCHIVE = "--archive" in sys.argv
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
iclr = {k: v for k, v in P.items() if v["src"] == "iclr" and v["group"] in ("reject", "accept", "oral") and v.get("rating_pct_in_year") is not None}
recs = {r["id"]: r for r in json.load(open(f"{HERE}/records/iclr_records.json"))["records"]}

# system scores keyed by iclr id; DIR = +1 if higher raw = more AI-like
S = collections.defaultdict(dict); DIR = {}; LABEL = {}; SRC = {}


def add(name, label, d, direction, src):
    for k, v in d.items():
        if k in iclr and v is not None:
            S[name][k] = v
    DIR[name], LABEL[name], SRC[name] = direction, label, src


def load_jsonl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)):
                out[r["id"]] = r[key]
    return out


add("binoculars", "Binoculars", load_jsonl(f"{HERE}/results/detectors/binoculars.*.jsonl", "binoculars"), -1, "fresh")
add("detectgpt", "DetectGPT", load_jsonl(f"{HERE}/results/detectors/detectgpt.*.jsonl", "detectgpt"), +1, "fresh")
add("fast_detectgpt", "Fast-DetectGPT", load_jsonl(f"{HERE}/results/detectors/fast_detectgpt.jsonl", "fast_detectgpt"), +1, "fresh")
add("nts", "NTS", load_jsonl(f"{HERE}/results/detectors/nts*.jsonl", "nts"), +1, "fresh")
for sysname, label in [("b3a", "AI Scientist reviewer"), ("b2h", "CycleReviewer")]:
    d = {}
    for f in glob.glob(f"{HERE}/results/reviews/{sysname}/*.json"):
        r = json.load(open(f)); fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)): d[r["item_id"]] = v
    add(f"rev_{sysname}", label, d, -1, "fresh")
if ARCHIVE:
    import pandas as pd
    df = pd.read_pickle(f"{ROOT}/paper/draft_v6/claude/detectors/iclr9y_detectors.pkl")
    by_note = {f"iclr{int(r.year)}_{r.pid}": r for r in df.itertuples()}
    for col, name, label, direction in [("binoculars", "binoculars_arch", "Binoculars (0908 archive)", -1), ("detectgpt", "detectgpt_arch", "DetectGPT (0908 archive)", +1)]:
        add(name, label, {k: (float(getattr(r, col)) if getattr(r, col) == getattr(r, col) else None) for k, r in by_note.items()}, direction, "archive")
OURS = "agg3" if "--ours-agg3" in sys.argv else "det4"    # --ours-agg3: aggregate without evidence_gap (macro, xsec, citation)
def agg3(v):
    if any(v.get(k) is None for k in ("macro_redund", "xsec_ref", "citation")): return None
    return ((v["macro_redund"] + v["xsec_ref"]) / 2 + v["citation"]) / 2
add("scislop_det4", "SciSlop (ours, 4 items)" if OURS == "det4" else "SciSlop (ours, 3 items, no evidence gap)",
    {k: (v.get("scislop_det4") if OURS == "det4" else agg3(v)) for k, v in iclr.items()}, +1, "ours")
S = {k: v for k, v in S.items() if len(v) >= 30}

# percentile (0-100) within each system's own sample, AI-likeness oriented
PCT = {}
for name, d in S.items():
    ids = list(d); vals = np.array([d[i] * DIR[name] for i in ids], float)
    pct = stats.rankdata(vals) / len(vals) * 100
    PCT[name] = dict(zip(ids, pct))
MIN_BIN = 20
BANDS = "--bands" in sys.argv          # --bands: two-point score bands 2-3 / 4-5 / 6-7 / 8-9 instead of single integers
def band(r):
    r = int(round(r))
    return {2: 2, 3: 2, 4: 4, 5: 4, 6: 6, 7: 6, 8: 8, 9: 8}.get(r, r) if BANDS else r
for k, v in iclr.items():
    v["q"] = band(v["rating"])   # x = mean public review score, rounded to the integer scale (1-10)

out = {"note": "y = within-sample percentile of the system score, oriented AI-like/slop-high; x = mean public review score rounded to the 1-10 scale, bins with >= 20 papers; rho is against the raw mean score", "systems": {}}
for name in S:
    ids = list(S[name]); x = [iclr[i]["rating"] for i in ids]; y = [PCT[name][i] for i in ids]
    rho, p = stats.spearmanr(x, y)
    qm = {}
    for q in range(1, 11):
        yy = [PCT[name][i] for i in ids if iclr[i]["q"] == q]
        if len(yy) >= MIN_BIN:
            b = np.random.default_rng(0).choice(yy, (1000, len(yy))).mean(1)
            qm[q] = {"n": len(yy), "mean": float(np.mean(yy)), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    out["systems"][name] = {"label": LABEL[name], "source": SRC[name], "n": len(ids), "spearman_vs_rating": round(float(rho), 3), "p": float(f"{p:.3g}"), "quintile": qm}
json.dump(out, open(f"{HERE}/results/rating_systems{'_archive' if ARCHIVE else ''}{'_agg3' if OURS == 'agg3' else ''}{'_bands' if BANDS else ''}.json", "w"), indent=1)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
# detectors = cool hues, reviewers = warm hues, ours = black (validated categorical palette of the dataviz skill)
STYLE = {"binoculars": ("#2a78d6", "o"), "detectgpt": ("#1baf7a", "s"), "fast_detectgpt": ("#7a6fd0", "^"), "nts": ("#eda100", "D"),
         "rev_b2h": ("#eb6834", "v"), "rev_b3a": ("#e87ba4", "P"), "binoculars_arch": ("#2a78d6", "o"), "detectgpt_arch": ("#1baf7a", "s"),
         "scislop_det4": ("#111111", "o")}
# plotted set = the baselines of Table 2 (Binoculars, DetectGPT, NTS, CycleReviewer, AI Scientist) + ours.
# Fast-DetectGPT is measured (rating_systems.json) but left out of the figure, as it is left out of Table 2 (0912).
TABLE2 = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "scislop_det4"]
grey_out = {n for n in S if n.endswith("_arch") and n[:-5] in S}      # archive only where no fresh run exists
order = [n for n in STYLE if n in S and n not in grey_out and (n in TABLE2 or (n.endswith("_arch") and n[:-5] in TABLE2))]
fig, ax = plt.subplots(figsize=(3.6, 2.5))
for name in order:
    qm = out["systems"][name]["quintile"]; qs = sorted(qm); y = [qm[q]["mean"] for q in qs]
    c, mk = STYLE[name]; ours = name == "scislop_det4"
    ax.plot(qs, y, color=c, lw=2.2 if ours else 1.1, marker=mk, ms=5 if ours else 3.2, mec="white" if ours else c, mew=0.8 if ours else 0.5,
            zorder=5 if ours else 3, solid_capstyle="round",
            label=(("SciSlop (ours)" if OURS == "det4" else "SciSlop (ours, 3 items)") if ours else LABEL[name].replace(" (0908 archive)", "*").replace(" reviewer", "")) + f"   ρ = {out['systems'][name]['spearman_vs_rating']:+.2f}")
allq = sorted({q for n_ in order for q in out["systems"][n_]["quintile"]})
ax.set_xticks(allq); ax.set_xticklabels([(f"{q}–{q+1}" if BANDS else str(q)) for q in allq], fontsize=7.5)
ax.set_xlabel("mean review score (ICLR, 1–10 scale)", fontsize=7.5)
ax.set_ylabel("AI-likeness (percentile of each system's score)", fontsize=7.5)
ax.annotate("more AI-like / more slop", xy=(0.012, 0.985), xycoords="axes fraction", fontsize=6.2, color="#555555", va="top", ha="left", style="italic")
ax.set_ylim(28, 72); ax.set_yticks([30, 40, 50, 60, 70])
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
ax.tick_params(length=2.5, labelsize=7.5, color="#555555"); ax.set_xlim(min(allq) - 0.3, max(allq) + 0.3)
ax.legend(frameon=False, fontsize=6, loc="lower left", handlelength=2.2, labelspacing=0.35, borderaxespad=0.3)
fig.tight_layout(pad=0.4)
suf = ("_archive" if ARCHIVE else "") + ("_agg3" if OURS == "agg3" else "") + ("_bands" if BANDS else "")
fig.savefig(f"{HERE}/results/fig_rating_systems{suf}.pdf"); fig.savefig(f"{HERE}/results/fig_rating_systems{suf}.png", dpi=300)
for name in order:
    s = out["systems"][name]; print(f"{name:18s} {s['source']:8s} n={s['n']:5d} rho={s['spearman_vs_rating']:+.3f} p={s['p']:.2g}  score->pct = " + " ".join(f"{q}:{s['quintile'][q]['mean']:.1f}" for q in sorted(s['quintile'])))
