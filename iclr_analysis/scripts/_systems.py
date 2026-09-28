"""Shared loader for the review-score figures (0916). Returns the paper table P (results/papers_stairs.jsonl, quality
filtered), every system's raw score keyed by paper id, its AI direction, its label, and the year x domain stratum of
each ICLR paper. Domain = arXiv primary category (records/iclr_primary_category.json, fill_categories.py) collapsed
to four groups so that year x domain cells stay populated:
  LG  = cs.LG, stat.*, cs.NE, cs.AI, math.*, cs.IT, cs.DS, cs.GT
  CV  = cs.CV, eess.IV, cs.GR, cs.MM
  CL  = cs.CL, cs.IR
  OTH = everything else (cs.RO, cs.CR, cs.SD, q-bio.*, physics.*, eess.AS, ...)"""
import json, os, glob
ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"; ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ["FARS", "reject", "accept", "oral"]
BANDS = [(2, 3), (4, 5), (6, 7), (8, 9)]
STYLE = {"binoculars": ("#2a78d6", "o"), "detectgpt": ("#1baf7a", "s"), "nts": ("#eda100", "D"), "rev_b2h": ("#eb6834", "v"), "rev_b3a": ("#e87ba4", "P"), "ours": ("#111111", "o")}
ORDER = ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "ours"]
ITEMS = [("macro_redund", "Macro redundancy", "#95627A", "-", "s"), ("xsec_ref", "Cross-section refs.", "#95627A", "--", "^"),
         ("argument_graph", "Argument graph", "#E4959E", "--", "P"), ("citation", "Citation isolation", "#E4959E", "-", "D"),
         ("fig_exposition", "Figure exposition", "#6D8A96", "--", "X"), ("evidence_gap", "Evidence gap", "#6D8A96", "-", "v")]


def domain_group(primary):
    if not primary: return "OTH"
    p = primary.lower()
    if p.startswith(("cs.lg", "stat.", "cs.ne", "cs.ai", "math.", "cs.it", "cs.ds", "cs.gt")): return "LG"
    if p.startswith(("cs.cv", "eess.iv", "cs.gr", "cs.mm")): return "CV"
    if p.startswith(("cs.cl", "cs.ir")): return "CL"
    return "OTH"


def band_of(r):
    r = int(round(r))
    for k, (a, b) in enumerate(BANDS):
        if a <= r <= b: return k
    return None


def mean_skip(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def jl(pattern, key):
    out = {}
    for f in glob.glob(pattern):
        for l in open(f):
            r = json.loads(l)
            if isinstance(r.get(key), (int, float)): out[r["id"]] = r[key]
    return out


def load(ours="agg5"):
    P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/results/papers_stairs.jsonl")}
    P = {k: v for k, v in P.items() if v["group"] in GROUPS}
    cat = json.load(open(f"{HERE}/records/iclr_primary_category.json"))
    for k, v in P.items():
        if v["src"] == "iclr":
            v["domain"] = domain_group((cat.get(k) or {}).get("primary"))
            v["stratum"] = f"{v['year']}_{v['domain']}"
            v["band"] = band_of(v["rating"]) if v.get("rating") is not None else None
    items = json.load(open(f"{B}/items165.json"))["items"]; ai_codes = {i["pair"] for i in items if i["label"] == 1}
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
    AGG = {"agg3": lambda v: mean_skip([mean_skip([v.get("macro_redund"), v.get("xsec_ref")]), v.get("citation")]) if None not in (v.get("macro_redund"), v.get("xsec_ref"), v.get("citation")) else None,
           "agg4": lambda v: v.get("scislop_det4"), "agg5": lambda v: v.get("scislop5")}
    for a, fn in AGG.items():
        S[a] = {k: fn(v) for k, v in P.items() if fn(v) is not None}; DIR[a] = +1; LABEL[a] = f"SciSlop ({a})"
    S["ours"] = S[ours]; DIR["ours"] = +1; LABEL["ours"] = "SciSlop (ours)"
    for k in [i[0] for i in ITEMS]:
        S[k] = {pid: v[k] for pid, v in P.items() if v.get(k) is not None}; DIR[k] = +1; LABEL[k] = dict((i[0], i[1]) for i in ITEMS)[k]
    S = {k: {i: x for i, x in d.items() if i in P} for k, d in S.items()}
    return P, S, DIR, LABEL
