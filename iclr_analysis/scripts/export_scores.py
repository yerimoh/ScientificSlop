r"""Write one row per paper with every system's score, for both review corpora, and a coverage snapshot.

Run it at any time, including while jobs are still running. It only reads, it never blocks a runner, and a missing
or half written source file is skipped rather than raised. Outputs, all overwritten atomically:

  results/scores_2026.csv        ICLR 2026, one row per paper in records/iclr2026_records.json
  results/scores_years.csv       ICLR 2017-2026 corpus, one row per paper in results/papers_stairs.jsonl
  results/coverage.csv           one row per run, how many papers each system has scored in each corpus

Columns hold the raw score of each system, not a percentile, plus the metadata an analysis needs (review score and
its sub-scores, decision, Pangram fraction, body length, area) and the quality flags. Direction is not applied here,
so a later analysis can orient the scores itself.
"""
import csv, json, glob, os, sys, time, datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SCISLOP_ROOT", ".")
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
ARCH = f"{ROOT}/artifact-ai2science/Evaluation/02_baselines_B"
ITEMS = ["macro_redund", "xsec_ref", "citation", "evidence_gap", "argument_graph", "fig_exposition"]


def jload(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def jl_rows(pattern, key):
    """id -> value, tolerant of a file being appended to right now."""
    out = {}
    for f in glob.glob(pattern):
        try:
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue          # last line of a file being written
                if isinstance(r.get(key), (int, float)):
                    out[r["id"]] = r[key]
        except Exception:
            pass
    return out


def reviewer_rows(pattern):
    out = {}
    for f in glob.glob(pattern):
        r = jload(f)
        if not r:
            continue
        fin = r.get("final") or {}
        v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
        if isinstance(v, (int, float)):
            out[os.path.splitext(os.path.basename(f))[0]] = v
    return out


def item_rows(dirname):
    out = {}
    p = f"{HERE}/results/slop/{dirname}/papers.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["id"]] = r
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def write_csv(path, rows, fields):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    os.replace(tmp, path)


# ------------------------------------------------------------------ ICLR 2026
def export_2026():
    recs = jload(f"{HERE}/records/iclr2026_records.json")
    if not recs:
        return 0, {}
    recs = {r["id"]: r for r in recs["records"]}
    words = {}
    for line in open(f"{HERE}/records/texts_iclr2026.jsonl"):
        try:
            r = json.loads(line); words[r["id"]] = r["body_words"]
        except Exception:
            pass
    it = {k: item_rows(f"{k}_2026") for k in ITEMS}
    det = {"binoculars": jl_rows(f"{HERE}/results/detectors/binoculars.p26*.jsonl", "binoculars"),
           "detectgpt": jl_rows(f"{HERE}/results/detectors/detectgpt.p26*.jsonl", "detectgpt"),
           "nts": jl_rows(f"{HERE}/results/detectors/nts.p26*.jsonl", "nts")}
    rev = {"rev_b2h": reviewer_rows(f"{HERE}/results/reviews/b2h_2026/*.json"),
           "rev_b3a": reviewer_rows(f"{HERE}/results/reviews/b3a_2026/*.json")}
    rows = []
    for pid, r in sorted(recs.items()):
        row = {"id": pid, "note_id": r["note_id"], "year": 2026, "corpus": "iclr2026",
               "rating": r["overall_mean"], "rating_min": r.get("rating_min"), "rating_max": r.get("rating_max"),
               "n_reviews": r.get("n_reviews"), "soundness": r.get("soundness"), "presentation": r.get("presentation"),
               "contribution": r.get("contribution"), "confidence": r.get("confidence"),
               "accept": int(bool(r.get("accept"))), "tier": r.get("tier"), "group": r.get("group4"),
               "pangram_fraction_ai": r.get("fraction_ai"), "area": r.get("primary_area"),
               "body_words": words.get(pid), "title": (r.get("title") or "")[:120]}
        for k in ITEMS:
            d = it.get(k, {}).get(pid) or {}
            row[k] = d.get("slop_score_agg") if k == "macro_redund" else d.get("slop_score")
            row[f"{k}_weak"] = int(bool(d.get("weak"))) if d else ""
            row[f"{k}_denominator"] = d.get("slop_denominator") if d else ""
        st = mean([row["macro_redund"], row["xsec_ref"]])
        row["agg3"] = mean([st, row["citation"]]) if None not in (st, row["citation"]) else None
        row["agg4"] = mean([st, row["citation"], row["evidence_gap"]]) if None not in (st, row["citation"]) else None
        for k, d in list(det.items()) + list(rev.items()):
            row[k] = d.get(pid)
        rows.append(row)
    fields = (["id", "note_id", "year", "corpus", "rating", "rating_min", "rating_max", "n_reviews", "soundness",
               "presentation", "contribution", "confidence", "accept", "tier", "group", "pangram_fraction_ai",
               "area", "body_words", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3", "agg4"]
              + ITEMS + [f"{k}_weak" for k in ITEMS] + [f"{k}_denominator" for k in ITEMS] + ["title"])
    write_csv(f"{HERE}/results/scores_2026.csv", rows, fields)
    cov = {k: sum(1 for r in rows if r.get(k) not in (None, "")) for k in
           ["pangram_fraction_ai", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"] + ITEMS}
    return len(rows), cov


# ------------------------------------------------------------------ ICLR 2017-2026
def export_years():
    src = f"{HERE}/results/papers_stairs.jsonl"
    if not os.path.exists(src):
        return 0, {}
    P = {}
    for line in open(src):
        try:
            r = json.loads(line); P[r["id"]] = r
        except Exception:
            pass
    recs = jload(f"{HERE}/records/iclr_records.json") or {"records": []}
    meta = {r["id"]: r for r in recs["records"]}
    det = {"binoculars": {}, "detectgpt": {}, "nts": {}}
    for k in det:
        det[k].update(jl_rows(f"{HERE}/results/detectors/{k}*.jsonl", k))
        bench = jl_rows(f"{B}/results/{k}.jsonl", k)
        for bid, v in bench.items():                       # FARS side of the staircase
            if bid.startswith("AI_"):
                det[k][bid[3:]] = v
    rev = {"rev_b2h": reviewer_rows(f"{HERE}/results/reviews/b2h/*.json"),
           "rev_b3a": reviewer_rows(f"{HERE}/results/reviews/b3a/*.json")}
    for sysname, dname in (("b2h", "B2h_cyclereviewer"), ("b3a", "B3a_ai_scientist")):
        for fp in glob.glob(f"{ARCH}/{dname}/runs/FA*/logs/*_{sysname}_R1_review.json"):
            r = jload(fp)
            v = ((r or {}).get("final") or {}).get("Overall")
            if isinstance(v, (int, float)):
                rev[f"rev_{sysname}"][os.path.basename(fp).split("_")[0]] = v
    excl = set((jload(f"{HERE}/records/exclude_quality.json") or {}).get("ids", []))
    rows = []
    for pid, v in sorted(P.items()):
        m = meta.get(pid, {})
        row = {"id": pid, "year": v.get("year"), "corpus": "fars" if v.get("group") == "FARS" else "iclr_years",
               "group": v.get("group"), "rating": v.get("rating"), "accept": m.get("accept"), "tier": m.get("tier"),
               "n_reviews": m.get("n_reviews"), "body_words": v.get("n_words"),
               "quality_excluded": int(pid in excl), "title": (m.get("title") or "")[:120]}
        for k in ITEMS + ["agg3", "agg4", "scislop_det4", "scislop5"]:
            if k in v:
                row[k] = v[k]
        st = mean([v.get("macro_redund"), v.get("xsec_ref")])
        row["agg3"] = mean([st, v.get("citation")]) if None not in (st, v.get("citation")) else None
        for k, d in list(det.items()) + list(rev.items()):
            row[k] = d.get(pid)
        rows.append(row)
    fields = (["id", "year", "corpus", "group", "rating", "accept", "tier", "n_reviews", "body_words",
               "quality_excluded", "binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3", "scislop_det4",
               "scislop5"] + ITEMS + ["title"])
    write_csv(f"{HERE}/results/scores_years.csv", rows, fields)
    cov = {k: sum(1 for r in rows if r.get(k) not in (None, "")) for k in
           ["binoculars", "detectgpt", "nts", "rev_b2h", "rev_b3a", "agg3"] + ITEMS}
    return len(rows), cov


def main():
    n26, c26 = export_2026()
    ny, cy = export_years()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = f"{HERE}/results/coverage.csv"
    new = not os.path.exists(path)
    keys = sorted(set(c26) | set(cy))
    with open(path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["time", "corpus", "n_papers"] + keys)
        wr.writerow([stamp, "iclr2026", n26] + [c26.get(k, "") for k in keys])
        wr.writerow([stamp, "iclr_years", ny] + [cy.get(k, "") for k in keys])
    print(f"{stamp}  scores_2026.csv {n26} rows, scores_years.csv {ny} rows")
    print("  2026 coverage:", {k: v for k, v in c26.items() if v})
    print("  years coverage:", {k: v for k, v in cy.items() if v})


if __name__ == "__main__":
    main()
