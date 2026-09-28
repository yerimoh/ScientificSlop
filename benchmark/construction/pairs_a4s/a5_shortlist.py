"""Step 5 (Agents4Science): hard filters H1/H2/H4 + a heuristic type screen, then per-paper
judging shortlists.

H1 human-verified : ICLR 2026 submission (arXiv id or title) with fraction_ai <= 0.05,
                    else arXiv v1 year <= 2024, else fail.
H2 top-tier main  : ICLR 2026 accept record, or the arXiv comment / journal_ref, or the
                    OpenAlex publication venue of the resolved reference.
H3 (screen only)  : the A4S type comes from step 0 (cache/a4s_types.json); the candidate type is
                    a keyword heuristic here and is replaced by the judged type in a6.
H4 usable source  : e-print with body >= 2500 prose words and >= 4 sections.

Outputs: cache/filters.json, cache/shortlists.json, cache/judge_batches/batch_XX.json
"""
import os, re, sys, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"

PANGRAM_MAX = 0.05
MIN_BODY, MIN_SEC = 2500, 4
SHORTLIST_N = 6

idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
pool = json.load(open(f"{OUTD}/cache/pool.json"))
meta = json.load(open(f"{OUTD}/cache/arxiv_meta_a4s.json"))
oa = json.load(open(f"{OUTD}/cache/openalex.json")) if os.path.exists(f"{OUTD}/cache/openalex.json") else {}
cm = json.load(open(f"{OUTD}/cache/cand_metrics_a4s.json")) if os.path.exists(f"{OUTD}/cache/cand_metrics_a4s.json") else {}
# step 0 of the rule: the A4S paper's own type, read from its title and abstract (not heuristic)
types = json.load(open(f"{OUTD}/cache/a4s_types.json"))


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# OpenAlex records are keyed by the parsed reference title; re-key them by arXiv id
oa_by_id = {}
for v in oa.values():
    if v and v.get("arxiv"):
        oa_by_id.setdefault(re.sub(r"v\d+$", "", v["arxiv"]), v)

iclr, iclr_bytitle = {}, {}
with open(f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("arxiv_id"):
            iclr[re.sub(r"v\d+$", "", r["arxiv_id"])] = r
        iclr_bytitle[norm_title(r["title"])] = r

BAD = re.compile(r"workshop|findings|tiny paper|non.?archival|rejected|withdraw|arXiv|preprint", re.I)
VENWORDS = (r"ICLR|NeurIPS|Neural Information Processing|ICML|International Conference on Machine Learning|"
            r"ACL\b|Association for Computational Linguistics|EMNLP|Empirical Methods in Natural Language|"
            r"NAACL|COLM|CVPR|Computer Vision and Pattern Recognition|ICCV|ECCV|AAAI|IJCAI|AISTATS|KDD|VLDB|"
            r"SIGIR|SIGMOD|The Web Conference|WWW '?2\d|USENIX Security|CCS\b|IEEE S&P|Oakland|TPAMI|"
            r"Trans[a-z.]* on Pattern Analysis|TMLR|Transactions on Machine Learning Research|JMLR|PNAS|"
            r"Nature\b|Science\b|ICSE|FSE\b|MLSys|COLING|Interspeech|ICASSP|ICRA|IROS|CoRL|RSS\b|"
            r"MICCAI|Bioinformatics|Nucleic Acids Research|Cell\b|Lancet|NEJM|JAMA")
ACC = re.compile(r"(accept|to appear|appear[s]?|published|camera.?ready|presented|oral|spotlight|poster)"
                 r"[^.;]{0,80}?\b(" + VENWORDS + r")", re.I)
VEN = re.compile(r"\b(" + VENWORDS + r")\s*[,;]?\s*(20\d\d|'\d\d)\b")
OAVEN = re.compile(r"\b(" + VENWORDS + r")", re.I)


def h2_venue(m, rec, o):
    com = " ".join(x for x in [m.get("comment"), m.get("journal_ref")] if x)
    if rec and rec.get("accept"):
        return True, f"ICLR 2026 {rec.get('tier', '')}".strip(), "iclr26_pool"
    if com and not BAD.search(com) and (ACC.search(com) or VEN.search(com)):
        mm = VEN.search(com) or ACC.search(com)
        return True, re.sub(r"\s+", " ", mm.group(0))[:40], "arxiv_comment"
    if o and o.get("venue") and not BAD.search(o["venue"]) and OAVEN.search(o["venue"]) \
            and o.get("type") in ("article", "book-chapter", "proceedings-article", None):
        return True, f"{o['venue'][:34]} {o.get('year') or ''}".strip(), "openalex_venue"
    return False, (com[:40] if com else ((o or {}).get("venue") or "preprint")[:40]), "none"


def heur_type(title, abstract):
    t = (title or "").lower()
    a = (abstract or "").lower()
    ta = t + " " + a[:900]
    if re.search(r"\bsurvey\b|systematic review|literature review", t):
        return "survey"
    if re.search(r"\bbench(mark)?\b|leaderboard|testbed|evaluation suite|a suite of", t):
        return "benchmark"
    if re.search(r"\bdataset\b|\bcorpus\b", t) and not re.search(r"method|approach|learning", t):
        return "dataset"
    if re.search(r"^(do|does|are|is|can|why|how|what|when|should)\b|\?$", t) or re.search(
            r"(is|are) (insufficient|not (necessary|enough|sufficient))|does not (improve|help|transfer)|"
            r"\ban (empirical |systematic )?(study|analysis)\b|understanding\b|investigat|rethinking|revisiting|"
            r"^on the\b|measuring\b|characteriz|dissect|reveals\b|a case study", t):
        return "analysis"
    if re.search(r"framework for (evaluat|analy|assess)|evaluation framework|a metric\b|new metric|protocol for", ta):
        return "framework"
    if re.search(r"we (propose|introduce|present|develop|design)", a):
        return "method"
    if re.search(r"benchmark", a[:400]) and re.search(r"we (introduce|present|release|construct|curate)", a[:400]):
        return "benchmark"
    return "method"


TYPE_INCOMPAT = {("method", "survey"), ("method", "dataset"), ("method", "benchmark"),
                 ("analysis", "survey"), ("analysis", "dataset"),
                 ("framework", "survey"), ("framework", "dataset")}


def main():
    filters, shortlists = {}, {}
    for code, P in pool.items():
        f = idx[code]
        ftype = types[code]
        rows = {}
        for aid, src in P.items():
            m = meta.get(aid) or {}
            if m.get("missing") or not m.get("title"):
                rows[aid] = dict(arxiv=aid, error="no arxiv meta")
                continue
            yr = int(m["published"][:4]) if m.get("published") else None
            rec = iclr.get(aid) or iclr_bytitle.get(norm_title(m["title"]))
            o = oa_by_id.get(aid)
            pg = rec["fraction_ai"] if rec else None
            if pg is not None and float(pg) <= PANGRAM_MAX:
                h1, h1by = "pass", "pangram"
            elif yr and yr <= 2024:
                h1, h1by = "pass", "year<=2024"
            else:
                h1, h1by = "fail", ("pangram>0.05" if pg is not None else "2025+ unverified")
            ok2, venue, how2 = h2_venue(m, rec, o)
            c = cm.get(aid)
            if c and c.get("metrics"):
                mt = c["metrics"]
                h4 = "pass" if mt["body_words"] >= MIN_BODY and mt["n_sections"] >= MIN_SEC else "fail"
            else:
                h4 = "fail" if (c and c.get("error")) else "unknown"
            roles = src.get("cite_roles") or {}
            rows[aid] = dict(arxiv=aid, title=m["title"], v1_year=yr, venue=venue, venue_by=how2,
                             heur_type=heur_type(m["title"], m.get("abstract", "")), cite_roles=roles,
                             relation="parent" if "experiments" in roles else "cited",
                             ref_idx=src.get("ref_idx", []), id_source=";".join(sorted(set(src["sources"]))),
                             H1_human=h1, H1_by=h1by, H2_top_tier="pass" if ok2 else "fail", H4_source=h4,
                             pangram_ai_frac=pg,
                             iclr_rating=rec.get("rating_mean") if rec else None,
                             iclr_tier=rec.get("tier") if rec else None,
                             oa_citations=(o or {}).get("citations"),
                             body_words=c["metrics"]["body_words"] if c and c.get("metrics") else None,
                             n_sections=c["metrics"]["n_sections"] if c and c.get("metrics") else None)
        filters[code] = rows
        surv = [r for r in rows.values() if not r.get("error")
                and r["H1_human"] == "pass" and r["H2_top_tier"] == "pass" and r["H4_source"] != "fail"
                and (ftype, r["heur_type"]) not in TYPE_INCOMPAT and (r["heur_type"], ftype) not in TYPE_INCOMPAT]
        surv.sort(key=lambda r: (-(r["relation"] == "parent"), -(r["heur_type"] == ftype),
                                 -(r["H4_source"] == "pass"), -(r["iclr_rating"] is not None),
                                 -(r["oa_citations"] or 0), -(r["v1_year"] or 0)))
        shortlists[code] = dict(a4s_type=ftype, shortlist=[r["arxiv"] for r in surv[:SHORTLIST_N]],
                                n_hard_survivors=len(surv))
    json.dump(filters, open(f"{OUTD}/cache/filters.json", "w"), ensure_ascii=False)
    json.dump(shortlists, open(f"{OUTD}/cache/shortlists.json", "w"), ensure_ascii=False, indent=1)

    import collections
    ns = sorted(s["n_hard_survivors"] for s in shortlists.values())
    print("survivors min/med/max:", ns[0], ns[len(ns) // 2], ns[-1],
          "| zero-survivor papers:", sum(1 for s in shortlists.values() if s["n_hard_survivors"] == 0))
    print("a4s types (step 0):", collections.Counter(s["a4s_type"] for s in shortlists.values()))

    bd = f"{OUTD}/cache/judge_batches"
    os.makedirs(bd, exist_ok=True)
    todo = [(c, shortlists[c]["shortlist"]) for c in sorted(shortlists) if shortlists[c]["shortlist"]]
    B = 10
    for bi in range(0, len(todo), B):
        batch = []
        for code, aids in todo[bi:bi + B]:
            f = idx[code]
            items = [dict(arxiv=a, title=meta[a]["title"], year=filters[code][a]["v1_year"],
                          venue=filters[code][a]["venue"], relation=filters[code][a]["relation"],
                          abstract=(meta[a].get("abstract") or "")[:900]) for a in aids]
            batch.append(dict(code=code, a4s_title=f["title"], a4s_abstract=f["abstract"][:1200],
                              a4s_type=shortlists[code]["a4s_type"], candidates=items))
        json.dump(batch, open(f"{bd}/batch_{bi // B:02d}.json", "w"), ensure_ascii=False, indent=1)
    print(f"judge batches: {(len(todo) + B - 1) // B} | papers to judge {len(todo)} | "
          f"(paper,candidate) pairs {sum(len(a) for _, a in todo)}")


if __name__ == "__main__":
    main()
