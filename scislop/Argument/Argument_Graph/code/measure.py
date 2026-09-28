#!/usr/bin/env python3
"""Argument_Graph  -  Argument build-up in the introduction (Argument plane), v3.

Unit = KEY CLAIM: one sentence of the Introduction labelled superiority, prior_limitation or
design_choice. A key claim is ARGUED when the sentence that most raises its likelihood precedes it
in the text, DECLARED when that sentence follows it. slop_score = declared / key claims.

Why v3 (0911-0912). v2 asked a model which sentence grounds a claim. Three model families under
three prompts named a position: moving the grounding sentence to the end lost it in 87 to 100
percent of cases, and the support rate ran from 0.26 to 0.87 on identical text. The original mold's
longest path was the sentence count, because every edge kind but one runs forward by its own
extraction rule. Neither is asked now.

Pipeline
  1. Introduction sentences (shared splitter, >= 4 words).
  2. Every sentence labelled by a forced four-way choice (Qwen-32B, PROMPT_label.txt, 3 runs,
     majority). Nothing is selected; the model cannot skip a sentence.
  3. pmi(j -> i) = mean logP(s_i | s_j) - mean logP(s_i) for every ordered pair, the two sentences
     concatenated by THIS code in a fixed order (local LM, _common/lm_score.py). The scorer never
     sees positions. Cached per introduction.
  4. pred(i) = argmax_j pmi(j -> i).  Direction is read from the text: argued iff pred(i) < i.
  5. build_up(i) = longest chain of forward predictors ending at i (premise layers under the claim).
  6. Shuffle null: 300 permutations of sentence positions over the same matrix; ABU = observed mean
     build_up over key claims minus the permuted mean. Positional expectation of the argued share
     is reported beside it.
  7. slop_score = declared key claims / key claims; coverage = key claims / labelled sentences.

Stages. --stage labels needs the vLLM server; --stage pmi needs a GPU; --stage all runs both, each
reading its cache first. A paper whose labels are not cached and whose server is down is recorded
as labels_pending, never as zero; the same for pmi_pending.

Usage: python3 measure.py [--stage labels|pmi|all] [--ai-limit K --hu-limit K] [--only ids] [--runs 3]
       -> ../results/{papers.jsonl, claims.jsonl, summary.json}
"""
from __future__ import annotations
import argparse, json, math, os, random, re, statistics, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import load_doc, prose_view, sentences, MappedText
from records import instance, write_jsonl, compare_groups, rate, describe, confound_audit, slop_score, VIOLATION, CONSISTENT
import llm as LLM
import lm_score as LMS

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "argument_graph_v3"
KEY_KINDS = ("superiority", "prior_limitation", "design_choice")
LABELS = KEY_KINDS + ("none",)
MIN_WORDS, MIN_KEY, NPERM, K_DEFAULT = 4, 3, 300, 1
LABEL_TAG = "argument_graph_label_v3"
_rng = random.Random(0)


# --------------------------------------------------------------------------- input
# The unit of analysis. "intro" is the item as defined and is what the published numbers use.
# "intro_concl" adds the conclusion, where a paper restates the same key claims, because a rate out
# of four or five claims is too coarse to carry a difference of 0.07; the construct is unchanged,
# only the number of claims it is estimated from.
UNITS = {"intro": ("Intro",), "intro_concl": ("Intro", "Conclusion")}


def intro_text(doc, unit="intro"):
    roles = UNITS[unit]
    parts = [prose_view(doc.tex[s.body_start:s.end], base=s.body_start).text.strip()
             for s in doc.body_sections() if s.role in roles]
    parts = [p for p in parts if p]
    return "\n".join(parts) if parts else None


def intro_sentences(intro: str):
    """Sentences of the introduction with at least MIN_WORDS alphabetic words; view spans kept."""
    out = []
    for s in sentences(MappedText(intro)):
        if len(re.findall(r"[A-Za-z]+", s["text"])) >= MIN_WORDS:
            out.append({"text": s["text"], "span": s["view_span"]})
    return out


def numbered(sents):
    return "\n".join(f"[S{i+1}] {s['text']}" for i, s in enumerate(sents))


# --------------------------------------------------------------------------- stage 1: labels
_TPL = None


def label_one(sents, i, runs):
    """Forced four-way label for sentence i; returns (label, runs_list) or (None, []) if no server."""
    global _TPL
    if _TPL is None:
        _TPL = open(os.path.join(HERE, "PROMPT_label.txt")).read()
    prompt = _TPL.replace("{numbered}", numbered(sents)).replace("{i}", str(i + 1)).replace("{target}", sents[i]["text"])
    got = []
    for r in range(runs):
        try:
            o = LLM.llm_json(prompt, tag=LABEL_TAG, run=r, max_tokens=40)
        except RuntimeError:            # server down and not cached
            return None, []
        lab = (o or {}).get("label") if isinstance(o, dict) else None
        lab = lab.strip().lower() if isinstance(lab, str) else None
        got.append(lab if lab in LABELS else "none")
    return Counter(got).most_common(1)[0][0], got


def label_sentences(sents, runs):
    # one paper's sentences are independent, so the width is a throughput knob and nothing else
    with ThreadPoolExecutor(max_workers=int(os.environ.get("AG_LABEL_WORKERS", 8))) as ex:
        res = list(ex.map(lambda i: label_one(sents, i, runs), range(len(sents))))
    if any(lab is None for lab, _ in res):
        return None
    for s, (lab, got) in zip(sents, res):
        s["label"] = lab; s["label_runs"] = got; s["unanimous"] = len(set(got)) == 1
    return sents


# --------------------------------------------------------------------------- stage 2: pmi + graph
def predictors(M, k):
    n = len(M)
    return {i: sorted([j for j in range(n) if j != i and not math.isnan(M[j][i])], key=lambda j: -M[j][i])[:k] for i in range(n)}


def build_up_under(pos, pred, n):
    """DP in the given order: depth[s] = 1 + depth of the strongest forward predictor, else 0."""
    d = {}
    for s in sorted(range(n), key=lambda s: pos[s]):
        d[s] = max([1 + d[j] for j in pred[s] if pos[j] < pos[s]], default=0)
    return d


def graph_metrics(M, key_idx, k=K_DEFAULT, nperm=NPERM):
    n = len(M); pred = predictors(M, k)
    ident = {s: s for s in range(n)}
    d_obs = build_up_under(ident, pred, n)
    obs = statistics.mean([d_obs[i] for i in key_idx])
    nulls = []
    for _ in range(nperm):
        p = list(range(n)); _rng.shuffle(p)
        dd = build_up_under({s: p[s] for s in range(n)}, pred, n)
        nulls.append(statistics.mean([dd[i] for i in key_idx]))
    mu = statistics.mean(nulls); sd = statistics.pstdev(nulls) or 1e-9
    argued = [i for i in key_idx if pred[i] and pred[i][0] < i]           # strongest predictor precedes
    return {"pred": pred, "build_up": d_obs, "argued_idx": argued,
            "build_up_mean": obs, "build_up_null": mu, "ABU": obs - mu, "ABU_z": (obs - mu) / sd,
            "ABU_null_pct": sum(1 for x in nulls if x < obs) / nperm,
            "argued_rate": rate(len(argued), len(key_idx)),
            "argued_expected": statistics.mean([i / (n - 1) for i in key_idx]) if n > 1 else None,
            "deep_rate": rate(sum(1 for i in key_idx if d_obs[i] >= 2), len(key_idx))}


# --------------------------------------------------------------------------- per paper
def measure(paper, runs=3, stage="all", unit="intro"):
    pid = paper["id"]; base = {"corpus": paper["corpus"], "id": pid, "checker": CHECKER}
    doc = load_doc(paper); intro = intro_text(doc, unit)
    if not intro or len(intro.split()) < 80:
        return {**base, "status": "no_intro"}, []
    sents = intro_sentences(intro)
    if len(sents) < 4:
        return {**base, "status": "too_short", "n_sentences": len(sents)}, []
    # stage 1
    if stage in ("labels", "all"):
        if label_sentences(sents, runs) is None:
            return {**base, "status": "labels_pending", "n_sentences": len(sents)}, []
    else:
        if label_sentences(sents, runs) is None:       # pmi stage still needs the cached labels
            return {**base, "status": "labels_pending", "n_sentences": len(sents)}, []
    key_idx = [i for i, s in enumerate(sents) if s["label"] in KEY_KINDS]
    row = {**base, "status": "ok", "n_words_intro": len(intro.split()), "n_sentences": len(sents), "n_labelled": len(sents),
           "n_key_claims": len(key_idx), "key_by_kind": dict(Counter(sents[i]["label"] for i in key_idx)),
           "label_unanimity": rate(sum(1 for s in sents if s["unanimous"]), len(sents)), "label_runs": runs}
    if stage == "labels":
        return {**row, "status": "labels_only"}, []
    if not key_idx:
        return {**row, **slop_score(0, 0), "coverage": 0.0, "ABU": None}, []
    # stage 2
    try:
        M = LMS.pmi_matrix([s["text"] for s in sents])
    except Exception as e:
        return {**row, "status": "pmi_pending", "error": repr(e)[:200]}, []
    g = graph_metrics(M, key_idx, K_DEFAULT)
    declared = [i for i in key_idx if i not in set(g["argued_idx"])]
    row.update({**slop_score(len(declared), len(key_idx), min_units=MIN_KEY),
               "coverage": rate(len(key_idx), len(sents)),
               "argued_rate": g["argued_rate"], "argued_expected": g["argued_expected"],
               "build_up_mean": g["build_up_mean"], "build_up_null": g["build_up_null"], "ABU": g["ABU"], "ABU_z": g["ABU_z"],
               "ABU_null_pct": g["ABU_null_pct"], "deep_rate": g["deep_rate"],
               "predictor_adjacent_rate": rate(sum(1 for i in key_idx if abs(g["pred"][i][0] - i) == 1), len(key_idx)),
               "predictor_distance_median": statistics.median([abs(g["pred"][i][0] - i) for i in key_idx])})
    by_kind = defaultdict(list)
    for i in key_idx:
        by_kind[sents[i]["label"]].append(i)
    row["slop_by_kind"] = {k: rate(sum(1 for i in v if i in declared), len(v)) for k, v in by_kind.items()}
    row["build_up_by_kind"] = {k: statistics.mean([g["build_up"][i] for i in v]) for k, v in by_kind.items()}
    for kk in (2, 3):                                     # sensitivity, fewer permutations
        gk = graph_metrics(M, key_idx, kk, nperm=100)
        row[f"ABU_k{kk}"] = gk["ABU"]; row[f"argued_rate_k{kk}"] = gk["argued_rate"]
    # evidence records, one per key claim
    recs = []
    for n_, i in enumerate(key_idx):
        p = g["pred"][i][0]
        recs.append(instance(pid, "argument_graph", n_, CONSISTENT if i in set(g["argued_idx"]) else VIOLATION,
                             {"kind": "text", "span": sents[i]["span"], "quote": sents[i]["text"], "sentence": i + 1, "label": sents[i]["label"]},
                             {"kind": "text", "span": sents[p]["span"], "quote": sents[p]["text"], "sentence": p + 1,
                              "pmi": round(M[p][i], 4), "position": "before" if p < i else "after"},
                             expected="predictor before the claim", reported="before" if p < i else "after", tolerance=None,
                             checker=CHECKER, build_up=g["build_up"][i], corpus=paper["corpus"]))
    return row, recs


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ai-limit", type=int, default=0); ap.add_argument("--hu-limit", type=int, default=0)
    ap.add_argument("--only", default=""); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--unit", default="intro", choices=sorted(UNITS))
    ap.add_argument("--stage", choices=["labels", "pmi", "all"], default="all",
                    help="labels: server only, cache labels. pmi: GPU only, read cached labels. all: both")
    a = ap.parse_args()
    ai, hu = ai_papers(), hu_papers()
    if a.ai_limit: ai = ai[:a.ai_limit]
    if a.hu_limit: hu = hu[:a.hu_limit]
    papers = ai + hu
    if a.only:
        ids = set(a.only.split(",")); papers = [p for p in papers if p["id"] in ids]
    if a.limit: papers = papers[:a.limit]
    rows, recs, fails = [], [], []
    for p in papers:                       # sequential: the PMI stage owns the GPU; labels fan out inside
        try:
            r, rr = measure(p, a.runs, a.stage, a.unit); rows.append(r); recs.extend(rr)
            print(f"  {r['corpus']} {r['id']} {r['status']}" + (f"  key {r.get('n_key_claims')}  slop {r.get('slop_score')}  ABU {r.get('ABU') if r.get('ABU') is None else round(r['ABU'], 2)}" if r["status"] == "ok" else ""), flush=True)
        except Exception as ex:
            fails.append({"id": p["id"], "error": repr(ex)[:300]}); print(f"  FAIL {p['id']} {ex!r}"[:200], flush=True)
    rows.sort(key=lambda r: (r["corpus"], r["id"]))
    os.makedirs(RESULTS, exist_ok=True)
    write_jsonl(os.path.join(RESULTS, "papers.jsonl"), rows); write_jsonl(os.path.join(RESULTS, "claims.jsonl"), recs)
    ok = [r for r in rows if r["status"] == "ok" and r.get("slop_score") is not None]
    A = [r for r in ok if r["corpus"] == "AI"]; H = [r for r in ok if r["corpus"] == "HU"]
    summary = {"checker": CHECKER, "stage": a.stage, "n_ai": len(A), "n_hu": len(H), "failures": fails,
               "status_counts": dict(Counter(r["status"] for r in rows)), "metrics": {}}
    for m in ["slop_score", "coverage", "ABU", "ABU_z", "build_up_mean", "build_up_null", "argued_rate", "argued_expected", "deep_rate",
              "n_key_claims", "n_sentences", "label_unanimity", "predictor_adjacent_rate", "ABU_k2", "ABU_k3", "argued_rate_k2", "argued_rate_k3"]:
        summary["metrics"][m] = compare_groups(m, [r[m] for r in A if r.get(m) is not None], [r[m] for r in H if r.get(m) is not None])
    bands = {}
    for lo, hi in [(3, 5), (6, 9), (10, 99)]:
        bands[f"key_{lo}_{hi}"] = {c: describe([r["ABU"] for r in g if lo <= r["n_key_claims"] <= hi and r["ABU"] is not None]) for c, g in (("AI", A), ("HU", H))}
    summary["ABU_by_key_count_band"] = bands
    summary["slop_by_kind"] = {c: {k: describe([r["slop_by_kind"].get(k) for r in g if r.get("slop_by_kind", {}).get(k) is not None]) for k in KEY_KINDS} for c, g in (("AI", A), ("HU", H))}
    summary["weak_papers"] = {c: sum(1 for r in g if r.get("weak")) for c, g in (("AI", A), ("HU", H))}
    summary["low_unanimity_papers"] = [r["id"] for r in ok if (r.get("label_unanimity") or 1) < 0.9]
    for c, g in (("AI", A), ("HU", H)):                # within-corpus confounds
        summary[f"confound_{c}"] = {}
        for f in ("n_sentences", "n_words_intro", "n_key_claims"):
            xs = [r[f] for r in g]; ys = [r["ABU"] for r in g]
            summary[f"confound_{c}"][f] = _spearman(xs, ys)
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=1, ensure_ascii=False)[:2000])
    for m, s in summary["metrics"].items():
        print(f"  {m:26} AI {s['AI'].get('mean')} (n={s['AI'].get('n')}) | HU {s['HU'].get('mean')} (n={s['HU'].get('n')}) | cliff {s['cliff_delta_AI_minus_HU']} p {s['mwu_p']}")


def _spearman(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 4: return None
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(s): r[i] = p + 1
        return r
    rx, ry = rk([p[0] for p in pts]), rk([p[1] for p in pts]); mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry)); den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** .5
    return round(num / den, 3) if den else None


if __name__ == "__main__":
    main()
