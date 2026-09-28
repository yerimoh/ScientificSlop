"""fig_exposition on the ICLR corpus, scored by the item's own code (slop/Artifacts/fig_exposition/code/measure.py:
read_kinds and measure), from the ICLR transcripts. Also rescoring of the benchmark pairs on the five
pattern kinds, so FARS / bench-human / ICLR groups are compared on one instrument.

The sixth kind, experimental_content, is hand-verified per figure in the item and cannot be given to
1.4k ICLR figures, so every row carries two scores:
  slop_score            kinds / 6 with the hand kind absent for ICLR (a lower bound, as the item's own rows)
  slop_score_pattern5   pattern kinds / 5, for all corpora alike (the comparable number)
-> results/slop/fig_exposition/papers.jsonl (ICLR rows), results/slop/fig_exposition/bench_pattern5.jsonl,
   results/slop/fig_exposition/summary.json
"""
import json, os, sys, glob, importlib.util
from collections import Counter
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/fig_exposition/code"
sys.path.insert(0, FE)
spec = importlib.util.spec_from_file_location("fe_measure", f"{FE}/measure.py"); FEM = importlib.util.module_from_spec(spec); spec.loader.exec_module(FEM)
PATTERN_KINDS = [k for k in FEM.KINDS if k != "experimental_content"]
OUT = f"{R}/results/slop/fig_exposition"; os.makedirs(OUT, exist_ok=True)


def score_lines(pid, lines, corpus, extra):
    fig = {"corpus": corpus, "id": pid, "pair": None, "path": extra.get("path", "")}
    row, insts = FEM.measure(fig, lines)
    pk = [k for k in row["kinds"] if k in PATTERN_KINDS]
    row["kinds_pattern5"] = pk
    row["slop_score_pattern5"] = round(len(pk) / len(PATTERN_KINDS), 4)
    row["hand_kind_available"] = pid in FEM.LEAK.CLEAR or corpus != "ICLR"
    row.update(extra)
    return row


def main():
    manifest = {}
    for f in glob.glob(f"{R}/results/figexp/manifest.s*.json"):
        for m in json.load(open(f)):
            manifest[m["id"]] = m
    tr = {}
    for f in sorted(glob.glob(f"{R}/results/figexp/transcripts*.jsonl")):
        for l in open(f):
            j = json.loads(l)
            if (j.get("result") or {}).get("lines") or j["id"] not in tr:
                tr[j["id"]] = j
    recs = {r["id"]: r for r in json.load(open(f"{R}/records/iclr_records.json"))["records"]}
    rows = []
    for pid, rec in recs.items():
        m = manifest.get(pid)
        base = {"corpus": "ICLR", "id": pid, "group4": rec["group4"], "year": rec["year"], "rating": rec["overall_mean"]}
        if m is None:
            rows.append({**base, "applicable": False, "status": "no_pdf_or_captions", "slop_score": None, "slop_score_pattern5": None}); continue
        if m["status"] != "ok":
            rows.append({**base, "applicable": False, "status": m["status"], "pick": m.get("pick"), "slop_score": None, "slop_score_pattern5": None}); continue
        t = tr.get(pid)
        if not t or not (t.get("result") or {}).get("lines"):
            rows.append({**base, "applicable": False, "status": "not_transcribed" if not t else "transcript_empty", "slop_score": None, "slop_score_pattern5": None}); continue
        lines = [str(x) for x in t["result"]["lines"]]
        row = score_lines(pid, lines, "ICLR", {"path": m["path"], "fig_num": m["fig_num"], "caption": m["caption"], "parse": t.get("parse"), "hit_cap": t.get("hit_cap")})
        rows.append({**base, **row, "status": "ok"})
    FEM.write_jsonl(f"{OUT}/papers.jsonl", rows)
    # benchmark pairs on the same five pattern kinds (transcripts of the item's own run)
    cache = FEM.load_cache()
    brows = []
    for f in FEM.figures():
        if f["key"] in cache:
            brows.append(score_lines(f["id"], cache[f["key"]], f["corpus"], {"path": f["path"], "pair": f["pair"]}))
    FEM.write_jsonl(f"{OUT}/bench_pattern5.jsonl", brows)
    ok = [r for r in rows if r["status"] == "ok"]
    def kinds_table(rs):
        return {k: sum(1 for r in rs if k in r["kinds"]) for k in FEM.KINDS}
    summ = {"checker": FEM.CHECKER + "+pattern5", "n_iclr_records": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
            "iclr": {"n": len(ok), "mean_pattern5": round(sum(r["slop_score_pattern5"] for r in ok) / len(ok), 4) if ok else None,
                     "any_kind": sum(1 for r in ok if r["n_kinds"] > 0), "kinds": kinds_table(ok),
                     "by_group4": {g: {"n": len(v), "mean_pattern5": round(sum(r["slop_score_pattern5"] for r in v) / len(v), 4) if v else None,
                                       "any_kind": sum(1 for r in v if r["n_kinds"] > 0), "kinds": kinds_table(v)}
                                   for g in ("reject", "accept", "oral") for v in [[r for r in ok if r["group4"] == g]]}},
            "bench": {c: {"n": len(v), "mean_pattern5": round(sum(r["slop_score_pattern5"] for r in v) / len(v), 4) if v else None,
                          "mean_6kind": round(sum(r["slop_score"] for r in v) / len(v), 4) if v else None,
                          "any_kind": sum(1 for r in v if len(r["kinds_pattern5"]) > 0), "kinds": kinds_table(v)}
                      for c in ("AI", "HU") for v in [[r for r in brows if r["corpus"] == c]]}}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    main()
