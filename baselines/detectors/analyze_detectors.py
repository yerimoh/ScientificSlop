"""Head-to-head: real detectors (Binoculars, DetectGPT) vs our scientific-slop index,
on the SAME 1,294 ICLR 2017-2025 papers, against review score."""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_pickle(os.path.join(HERE, "..", "iclr9y.pkl"))
MOLDS = ["comp_pos","eval_surface","baseline","method","problem_borrowing",
         "cross_ref","macro_redundancy","external_number"]
STYLE = ["excess_vocab_rate","tricolon_rate","emdash_rate","burstiness_inv"]

def zin(df, col):
    return df.groupby("year")[col].transform(lambda v: (v - v.mean()) / v.std(ddof=0))

def load(pattern, col):
    rows = []
    for f in glob.glob(os.path.join(HERE, pattern)):
        for l in open(f):
            try: rows.append(json.loads(l))
            except Exception: pass
    if not rows: return None
    r = pd.DataFrame(rows).drop_duplicates("id")
    return r[["id", col, "source"]] if "source" in r else r[["id", col]]

b = load("binoculars*.jsonl", "binoculars")
g = load("detectgpt*.jsonl", "detectgpt")
if b is not None: d = d.merge(b.rename(columns={"id": "pid"}), on="pid", how="left")
if g is not None:
    g = g.drop(columns=[c for c in ("source",) if c in g.columns])
    d = d.merge(g.rename(columns={"id": "pid"}), on="pid", how="left")

z = d[MOLDS].copy()
for c in MOLDS: z[c] = zin(d, c)
d["M"] = z.mean(axis=1)
zs = d[STYLE].copy()
for c in STYLE: zs[c] = zin(d, c)
d["S"] = zs.mean(axis=1)
# sign so that "higher = more machine-like" for every detector
if "binoculars" in d: d["bino_ai"] = -zin(d.dropna(subset=["binoculars"]).reindex(d.index), "binoculars")
if "detectgpt" in d: d["dgpt_ai"] = zin(d.dropna(subset=["detectgpt"]).reindex(d.index), "detectgpt")

print("=" * 86)
print("SAME PAPERS, SAME OUTCOME: ICLR 2017-2025, association with mean review score")
print("=" * 86)
print(f"{'signal':44s} {'n':>6s} {'rho':>8s} {'p':>10s}")
print("-" * 86)
rows = [("REAL BASELINE  Binoculars (Hans+ 2024, zero-shot)", "bino_ai"),
        ("REAL BASELINE  DetectGPT (Mitchell+ 2023, zero-shot)", "dgpt_ai"),
        ("proxy          writing-style index (surface cues)", "S"),
        ("OURS           scientific slop index (8 measures)", "M")]
for lab, col in rows:
    if col not in d: print(f"{lab:44s} {'--':>6s}  not measured yet"); continue
    s = d.dropna(subset=[col, "score"])
    if len(s) < 30: print(f"{lab:44s} {len(s):6d}  too few"); continue
    r, p = stats.spearmanr(s[col], s.score)
    print(f"{lab:44s} {len(s):6d} {r:+8.3f} {p:10.2e}")

print()
print("independence: does our index carry something the detectors do not?")
print("-" * 86)
for col, lab in [("bino_ai", "Binoculars"), ("dgpt_ai", "DetectGPT"), ("S", "style index")]:
    if col not in d: continue
    s = d.dropna(subset=[col, "M"])
    if len(s) < 30: continue
    r, p = stats.spearmanr(s[col], s.M)
    print(f"  rho(ours, {lab:12s}) = {r:+.3f}  p={p:.2e}   n={len(s)}")

print()
print("partial: our index vs review score, controlling for each detector (Spearman on ranks)")
print("-" * 86)
def partial(y, x, z_):
    A = np.c_[np.ones(len(z_)), stats.rankdata(z_)]
    ry = stats.rankdata(y) - A @ np.linalg.lstsq(A, stats.rankdata(y), rcond=None)[0]
    rx = stats.rankdata(x) - A @ np.linalg.lstsq(A, stats.rankdata(x), rcond=None)[0]
    return stats.pearsonr(ry, rx)
for col, lab in [("bino_ai", "Binoculars"), ("dgpt_ai", "DetectGPT")]:
    if col not in d: continue
    s = d.dropna(subset=[col, "M", "score"])
    if len(s) < 30: continue
    r, p = partial(s.score.values, s.M.values, s[col].values)
    print(f"  ours vs score | {lab:12s} : rho = {r:+.3f}  p={p:.2e}  n={len(s)}")

print()
print("robustness: tex-sourced papers only (OCR text can shift detector scores)")
print("-" * 86)
if "source" in d:
    for col, lab in [("bino_ai","Binoculars"),("dgpt_ai","DetectGPT"),("S","style"),("M","ours")]:
        if col not in d: continue
        s = d[(d.source == "tex")].dropna(subset=[col, "score"])
        if len(s) < 30: continue
        r, p = stats.spearmanr(s[col], s.score)
        print(f"  {lab:12s} n={len(s):5d}  rho={r:+.3f}  p={p:.2e}")
d.to_pickle(os.path.join(HERE, "iclr9y_detectors.pkl"))
print("\nsaved iclr9y_detectors.pkl")
