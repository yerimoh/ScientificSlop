"""Build the Hugging Face release of SciSlopBench (user/Scientific_Slop).

Three tables, written as Parquet under repo/data/:
  pairs.parquet   390 rows, one per AI/human pair, with the pairing provenance (pool, tier, H1-H4, ranks)
  papers.parquet  773 rows, one per unique paper (7 human anchors serve both halves), with the two views the measures read (body.tex, body.txt)
  scores.parquet  780 rows (one per pair side), SciSlop measure scores, detector baselines and reviewer scores

Sources are the frozen benchmark builds bench165/ (FARS half, 143 pairs) and benchA4S/ (Agents4Science
half, 247 pairs) plus their pair manifests. Score files are the ones table_pooled.py reads, so Tables 2/3
of the paper are recomputable from scores.parquet alone (see validate() at the bottom).

Usage: python3 build_hf_release.py [--no-human-text]
  --no-human-text  keep human-side rows in papers.parquet but blank body_tex/body_txt (arXiv IDs remain)
"""
import argparse, glob, json, os, statistics as st, sys
import pyarrow as pa, pyarrow.parquet as pq

SB = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench"
ARCH = os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/Evaluation/02_baselines_B"
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
A4S_META = os.environ.get("SCISLOP_ROOT", ".") + "/Agents4Science/2025_full/metadata.jsonl"
OUT = f"{SB}/hf_release/repo/data"
MEASURES = ["macro_redund", "xsec_ref", "argument_graph", "citation", "fig_exposition", "evidence_gap"]
PLANES = {"macro_redund": "structure", "xsec_ref": "structure", "argument_graph": "argument",
          "citation": "argument", "fig_exposition": "artifacts", "evidence_gap": "artifacts"}

HALVES = [  # tag, root, items file, manifest, baselines subdir, source name
    ("165", f"{SB}/bench165", "items165.json", f"{SB}/data/pairs165_0911/pairs165_draft.json", "", "fars"),
    ("a4s", f"{SB}/benchA4S", "itemsA4S.json", f"{SB}/data/Agents4Science/pairs_a4s.json", "baselines/", "agents4science"),
]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def fnum(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def load_halves():
    """items + manifest per half -> list of pair dicts and per-paper dicts (without text)."""
    a4s_meta = {r["submission_id"]: r for r in jl(A4S_META)}
    topics165 = json.load(open(f"{SB}/bench165/results/bench_stats_topic_labels.json"))
    pairs, papers, seen = [], [], {}
    for tag, root, itf, manf, _sub, source in HALVES:
        items = json.load(open(f"{root}/{itf}"))["items"]
        man = {p["code"]: p for p in json.load(open(manf))["pairs"]}
        byp = {}
        for i in items:
            byp.setdefault(i["pair"], {})[i["label"]] = i
        for code, d in sorted(byp.items()):
            ai, hu = d[1], d[0]
            m = man[code]
            hp = m["human_primary"]
            row = {
                "pair_id": code, "source": source,
                "ai_paper_id": ai["item_id"], "human_paper_id": hu["item_id"],
                "ai_title": m["ai"].get("title"), "human_title": hp.get("title"),
                "human_arxiv": hu["arxiv"], "human_venue": hu.get("venue"),
                "human_v1_year": hp.get("v1_year"), "human_iclr_rating": fnum(hu.get("iclr_rating")),
                "contribution_type": m.get("fars_type") or m.get("a4s_type"),
                "human_contribution_type": hp.get("paper_type"),
                "same_contribution_type": bool(hp.get("paper_type") and (m.get("fars_type") or m.get("a4s_type")) == hp.get("paper_type")),
                "pool": ai["pool"], "sim_tier": int(ai["sim_tier"]),
                "relation": hp.get("relation"), "tier_reason": hp.get("tier_reason"),
                "H1_human": hp.get("H1_human"), "H1_by": hp.get("H1_by"), "H2_top_tier": hp.get("H2_top_tier"),
                "H3_type_match": hp.get("H3_type_match"), "H4_source": hp.get("H4_source"),
                "specter2_cosine": fnum(hp.get("specter2")), "tfidf_similarity": fnum(hp.get("tfidf_sim")),
                "match_rank": hp.get("v3_rank", hp.get("match_rank")),
                "human_pangram_ai_fraction": fnum(hp.get("pangram_ai_frac")),
                "ai_body_words": ai["body_words"], "human_body_words": hu["body_words"],
                "body_length_ratio": fnum(hp.get("body_ratio")),
                "topic": None, "secondary_topic": None,
                "a4s_submission_id": None, "a4s_openreview_url": None, "a4s_decision": None,
                "a4s_reviewer_scores": None, "a4s_ai_involvement": None, "ai_tex_source": None,
            }
            if source == "fars":
                row["topic"] = topics165.get(code)
                row["ai_tex_source"] = "shipped_tex"
                if row["H3_type_match"] is None:  # the FARS manifest applied H3 as a hard filter but only recorded it for some rows
                    row["H3_type_match"] = "pass"
            else:
                sid = m["submission_id"]; mm = a4s_meta.get(sid, {})
                row.update({"topic": mm.get("primary_topic"), "secondary_topic": mm.get("secondary_topic"),
                            "a4s_submission_id": sid, "a4s_openreview_url": mm.get("openreview_url") or f"https://openreview.net/forum?id={sid}",
                            "a4s_decision": ai.get("a4s_status"),
                            "a4s_reviewer_scores": json.dumps(ai.get("scores")) if ai.get("scores") else None,
                            "a4s_ai_involvement": json.dumps(ai.get("autonomy")) if ai.get("autonomy") else None,
                            "ai_tex_source": ai.get("ai_source")})
            for k in ("topic", "secondary_topic"):
                if row[k] == "":
                    row[k] = None
            pairs.append(row)
            for it, role, title in ((ai, "ai", m["ai"].get("title")), (hu, "human", hp.get("title"))):
                pid = it["item_id"]
                if pid in seen:  # a human anchor shared by both halves: identical views and scores, keep one row
                    seen[pid]["pair_ids"].append(code); seen[pid]["source"] += "+" + source
                    continue
                seen[pid] = {"paper_id": pid, "pair_ids": [code], "source": source,
                             "label": int(it["label"]), "role": role, "title": title,
                             "arxiv": it.get("arxiv"), "venue": it.get("venue"),
                             "body_words": it["body_words"], "_views": f"{root}/views/{pid}"}
                papers.append(seen[pid])
    return pairs, papers


def load_scores(pairs):
    """One row per pair side (780 rows). SciSlop measures are per paper and identical across halves; the detector
    and LLM-reviewer baselines were run once per half, so a human anchor shared by both halves keeps one row per
    half, which is what reproduces Tables 2/3 (table_pooled.py) exactly."""
    rows = {}
    for p in pairs:
        for pid, lb, role in ((p["ai_paper_id"], 1, "ai"), (p["human_paper_id"], 0, "human")):
            rows[(p["source"], pid)] = {"paper_id": pid, "pair_id": p["pair_id"], "source": p["source"], "label": lb, "role": role}
    for tag, root, itf, _m, sub, source in HALVES:
        items = json.load(open(f"{root}/{itf}"))["items"]
        key = {}
        for i in items:  # (corpus, id) as the measure outputs name them -> paper_id
            k = i["pair"] if i["label"] == 1 else i["arxiv"]
            key[("AI" if i["label"] == 1 else "HU", k)] = i["item_id"]
        def put(pid, col, val):
            r = rows.get((source, pid))
            if r is not None and val is not None:
                r[col] = float(val)
        for name in MEASURES:
            p = f"{root}/results/slop/{name}/papers.jsonl"
            if not os.path.isfile(p) and name == "fig_exposition" and tag == "165":
                p = f"{SLOP}/Artifacts/fig_exposition/results/papers.jsonl"
            if not os.path.isfile(p):
                continue
            for r in jl(p):
                put(key.get((r["corpus"], r["id"])), f"scislop_{name}", fnum(r.get("slop_score")))
        for fname, field, col in [("binoculars.jsonl", "binoculars", "binoculars"), ("detectgpt.jsonl", "detectgpt", "detectgpt"),
                                  ("nts.jsonl", "nts", "nts"), ("fast_detectgpt.jsonl", "fast_detectgpt", "fast_detectgpt"),
                                  ("pangram.jsonl", "pangram", "pangram")]:
            p = f"{root}/results/{sub}{fname}"
            if not os.path.isfile(p):
                print("  missing", p); continue
            for r in jl(p):
                put(r["id"], col, r.get(field))
        for sysname, arch_dir, col in [("b2h", "B2h_cyclereviewer", "cyclereviewer_overall"), ("b3a", "B3a_ai_scientist", "ai_scientist_overall")]:
            for f in glob.glob(f"{root}/results/reviews/{sysname}/*.json"):
                r = json.load(open(f)); fin = r.get("final")
                v = fin.get("Overall", fin.get("Rating")) if isinstance(fin, dict) else None
                if isinstance(v, (int, float)):
                    put(r["item_id"], col, v)
            if tag == "165":  # FARS AI-side reviews live in the survival-run archive
                for i in items:
                    if i["label"] != 1:
                        continue
                    fp = f"{ARCH}/{arch_dir}/runs/{i['pair']}/logs/{i['pair']}_{sysname}_R1_review.json"
                    if os.path.isfile(fp):
                        fin = json.load(open(fp)).get("final")
                        v = fin.get("Overall") if isinstance(fin, dict) else None
                        if isinstance(v, (int, float)):
                            put(i["item_id"], col, v)
    out = []
    for p in pairs:
        for pid in (p["ai_paper_id"], p["human_paper_id"]):
            r = rows[(p["source"], pid)]
            planes = {}
            for name in MEASURES:
                v = r.get(f"scislop_{name}")
                if v is not None:
                    planes.setdefault(PLANES[name], []).append(v)
            r["scislop_aggregate"] = round(st.mean(st.mean(v) for v in planes.values()), 6) if planes else None
            for name in MEASURES:
                r.setdefault(f"scislop_{name}", None)
            for col in ["binoculars", "detectgpt", "nts", "fast_detectgpt", "pangram", "cyclereviewer_overall", "ai_scientist_overall"]:
                r.setdefault(col, None)
            out.append(r)
    return out


def pair_metrics(scores, pairs, col, direction):
    by = {}
    for r in scores:
        if r.get(col) is not None:
            by.setdefault(r["pair_id"], {})[r["label"]] = r[col]
    both = [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]
    w = sum(1 for a, h in both if (a - h) * direction > 0); t = sum(1 for a, h in both if a == h)
    ai = [v[1] for v in by.values() if 1 in v]; hu = [v[0] for v in by.values() if 0 in v]
    au = sum(1.0 if (a - b) * direction > 0 else 0.5 if a == b else 0.0 for a in ai for b in hu) / (len(ai) * len(hu))
    return round((w + 0.5 * t) / len(both), 3), round(au, 3), len(both)


def write(rows, name, schema=None):
    cols = list(rows[0].keys())
    tbl = pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in rows], schema=schema)
    pq.write_table(tbl, f"{OUT}/{name}.parquet", compression="zstd")
    print(f"  wrote {name}.parquet  rows={tbl.num_rows} cols={tbl.num_columns}  {os.path.getsize(f'{OUT}/{name}.parquet')/1e6:.1f} MB")
    return tbl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-human-text", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pairs, papers = load_halves()
    print(f"pairs={len(pairs)} papers={len(papers)}")
    for p in papers:
        v = p.pop("_views")
        blank = a.no_human_text and p["role"] == "human"
        p["body_tex"] = None if blank else open(f"{v}/body.tex", encoding="utf-8", errors="replace").read()
        p["body_txt"] = None if blank else open(f"{v}/body.txt", encoding="utf-8", errors="replace").read()
    scores = load_scores(pairs)
    write(pairs, "pairs"); write(papers, "papers"); write(scores, "scores")
    print("validation (PairAcc, AUROC, n pairs) -- paper Table 2: SciSlop 0.859, Binoculars 0.687")
    for col, d in [("scislop_aggregate", +1), ("binoculars", -1), ("detectgpt", +1), ("nts", +1), ("fast_detectgpt", +1),
                   ("pangram", +1), ("cyclereviewer_overall", -1), ("ai_scientist_overall", -1)] + [(f"scislop_{m}", +1) for m in MEASURES]:
        print(f"  {col:<28}", pair_metrics(scores, pairs, col, d))
    json.dump({"n_pairs": len(pairs), "n_papers": len(papers), "human_text_included": not a.no_human_text},
              open(f"{OUT}/../../build_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
