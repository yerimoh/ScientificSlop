"""Step 5: apply hard filters H1/H2/H4 + heuristic type screen, build per-FARS judging shortlists.

H1 human-verified : ICLR2026 submission (match by arxiv_id or normalized title) with fraction_ai<=0.05,
                    else arXiv v1 year <= 2024, else fail.
H2 top-tier main  : candidates.json venue, or arXiv comment/journal_ref parsing, or ICLR2026 accept tier.
H3 (screen only)  : keyword heuristic type on title+abstract; obvious mismatches vs the FARS heuristic
                    type are dropped from the judging shortlist (final H3 uses model-judged types).
H4 usable source  : e-print with body>=2500 words, >=4 sections (unknown if no e-print yet).

Outputs: cache/filters.json     {code: {aid: row}}   every pool candidate with filter columns
         cache/shortlists.json  {code: {fars_heur_type, shortlist: [aid,...]}}
         cache/judge_batches/batch_XX.json  inputs for tier-judging agents
"""
import os, re, sys, json

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/pairs165_0911"

PANGRAM_MAX = 0.05
MIN_BODY, MIN_SEC = 2500, 4
SHORTLIST_N = 7

idx = json.load(open(f"{OUTD}/cache/fars_index.json"))
pool = json.load(open(f"{OUTD}/cache/pool.json"))
meta = json.load(open(f"{OUTD}/cache/arxiv_meta_165.json"))
t2i = json.load(open(f"{OUTD}/cache/title2id.json")) if os.path.exists(f"{OUTD}/cache/title2id.json") else {}
cm = json.load(open(f"{OUTD}/cache/cand_metrics_165.json"))


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# ---- ICLR 2026 pool index ----
iclr = {}
iclr_bytitle = {}
for line in open(f"{ROOT}/artifact-ai2science/Evaluation/ICLR2026_Pangram/data/papers_2026.jsonl"):
    r = json.loads(line)
    if r.get("arxiv_id"):
        iclr[re.sub(r"v\d+$", "", r["arxiv_id"])] = r
    iclr_bytitle[norm_title(r["title"])] = r

# ---- venue / H2 ----
# candidates.json is the pre-filtered top-tier citation pool: every non-null venue there counts
# (NeurIPS ICLR ICML EMNLP ACL NAACL CVPR ICCV ECCV AAAI IJCAI AISTATS KDD VLDB SIGIR WWW
#  USENIX-Sec CCS S&P TPAMI TMLR JMLR PNAS Nature). For bib-derived candidates we parse the
# arXiv comment / journal_ref with the same venue vocabulary.
BAD = re.compile(r"workshop|findings|tiny paper|non.?archival|rejected|withdraw", re.I)
VENWORDS = (r"ICLR|NeurIPS|Neural Information Processing|ICML|International Conference on Machine Learning|"
            r"ACL\b|Association for Computational Linguistics|EMNLP|Empirical Methods in Natural Language|"
            r"NAACL|COLM|CVPR|Computer Vision and Pattern Recognition|ICCV|ECCV|AAAI|IJCAI|AISTATS|KDD|VLDB|"
            r"SIGIR|SIGMOD|The Web Conference|WWW '?2\d|USENIX Security|CCS\b|IEEE S&P|Oakland|TPAMI|"
            r"Trans[a-z.]* on Pattern Analysis|TMLR|Transactions on Machine Learning Research|JMLR|PNAS|"
            r"Nature\b|Science\b|ICSE|FSE\b|MLSys|COLING|Interspeech|ICASSP")
ACC = re.compile(r"(accept|to appear|appear[s]?|published|camera.?ready|presented|oral|spotlight|poster)"
                 r"[^.;]{0,80}?\b(" + VENWORDS + r")", re.I)
VEN = re.compile(r"\b(" + VENWORDS + r")\s*[,;]?\s*(20\d\d|'\d\d)\b")


def h2_venue(aid, cand_meta, m, rec):
    """returns (pass?, venue string, how)"""
    com = " ".join(x for x in [m.get("comment"), m.get("journal_ref")] if x)
    if rec and rec.get("accept"):
        return True, f"ICLR 2026 {rec.get('tier', '')}".strip(), "iclr26_pool"
    if cand_meta and cand_meta.get("venue"):
        return True, f"{cand_meta['venue']} {cand_meta.get('year', '')}".strip(), "candidates.json"
    if com and not BAD.search(com):
        if ACC.search(com) or VEN.search(com):
            mm = VEN.search(com) or ACC.search(com)
            return True, re.sub(r"\s+", " ", mm.group(0))[:40], "arxiv_comment"
    return False, (com[:40] if com else "preprint"), "none"


# ---- heuristic paper type ----
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
    filters = {}
    shortlists = {}
    for code, P in pool.items():
        f = idx[code]
        ftype = heur_type(f["title"], f["abstract"])
        rows = {}
        for aid, src in P.items():
            m = meta.get(aid) or {}
            if m.get("missing") or not m.get("title"):
                rows[aid] = dict(arxiv=aid, error="no arxiv meta")
                continue
            yr = int(m["published"][:4]) if m.get("published") else None
            rec = iclr.get(aid) or iclr_bytitle.get(norm_title(m["title"]))
            pg = rec["fraction_ai"] if rec else None
            if pg is not None and pg <= PANGRAM_MAX:
                h1, h1by = "pass", "pangram"
            elif yr and yr <= 2024:
                h1, h1by = "pass", "year<=2024"
            else:
                h1, h1by = "fail", ("pangram>0.05" if pg is not None else "2025+ unverified")
            ok2, venue, how2 = h2_venue(aid, src.get("cand_meta"), m, rec)
            c = cm.get(aid)
            if c and c.get("metrics"):
                mt = c["metrics"]
                h4 = "pass" if mt["body_words"] >= MIN_BODY and mt["n_sections"] >= MIN_SEC else "fail"
            else:
                h4 = "unknown" if not (c and c.get("error")) else "fail"
            roles = {}
            for k in src.get("bib_keys", []):
                for sec, n in f["cite_roles"].get(k, {}).items():
                    roles[sec] = roles.get(sec, 0) + n
            if not roles:  # candidates.json-only entry: find bib key by title
                for k, b in f["bib"].items():
                    if k in f["cite_roles"] and norm_title(b["title"]) == norm_title(m["title"]):
                        roles = f["cite_roles"][k]
                        break
            htype = heur_type(m["title"], m.get("abstract", ""))
            rows[aid] = dict(arxiv=aid, title=m["title"], v1_year=yr, venue=venue, venue_by=how2,
                             heur_type=htype, cite_roles=roles,
                             relation="parent" if "experiments" in roles else "cited",
                             H1_human="pass" if h1 == "pass" else "fail", H1_by=h1by,
                             H2_top_tier="pass" if ok2 else "fail", H4_source=h4,
                             pangram_ai_frac=pg,
                             iclr_rating=rec.get("rating_mean") if rec else None,
                             iclr_tier=rec.get("tier") if rec else None,
                             body_words=c["metrics"]["body_words"] if c and c.get("metrics") else None,
                             n_sections=c["metrics"]["n_sections"] if c and c.get("metrics") else None)
        filters[code] = rows

        # shortlist for tier judging: H1+H2 pass, H4 pass or unknown, heuristic types not obviously incompatible
        surv = [r for r in rows.values() if not r.get("error")
                and r["H1_human"] == "pass" and r["H2_top_tier"] == "pass" and r["H4_source"] != "fail"
                and (ftype, r["heur_type"]) not in TYPE_INCOMPAT and (r["heur_type"], ftype) not in TYPE_INCOMPAT]
        surv.sort(key=lambda r: (-(r["relation"] == "parent"), -(r["heur_type"] == ftype),
                                 -(r["H4_source"] == "pass"), -(r["iclr_rating"] is not None),
                                 -(r["v1_year"] or 0)))
        shortlists[code] = dict(fars_heur_type=ftype, shortlist=[r["arxiv"] for r in surv[:SHORTLIST_N]],
                                n_hard_survivors=len(surv))
    json.dump(filters, open(f"{OUTD}/cache/filters.json", "w"), ensure_ascii=False)
    json.dump(shortlists, open(f"{OUTD}/cache/shortlists.json", "w"), ensure_ascii=False, indent=1)

    ns = sorted(s["n_hard_survivors"] for s in shortlists.values())
    print("survivor counts min/med/max:", ns[0], ns[len(ns)//2], ns[-1])
    print("zero-survivor codes:", [c for c, s in shortlists.items() if s["n_hard_survivors"] == 0])
    import collections
    print("fars heur types:", collections.Counter(s["fars_heur_type"] for s in shortlists.values()))

    # ---- judge batch files ----
    round2 = "--round2" in sys.argv
    done = set()
    if round2:
        import glob as _g
        for bf in sorted(_g.glob(f"{OUTD}/cache/judge_batches/batch_*.json")):
            for rec in json.load(open(bf)):
                for c in rec["candidates"]:
                    done.add((rec["code"], c["arxiv"]))
    bd = f"{OUTD}/cache/judge_batches2" if round2 else f"{OUTD}/cache/judge_batches"
    os.makedirs(bd, exist_ok=True)
    todo = []
    for code in sorted(shortlists):
        aids = [a for a in shortlists[code]["shortlist"] if (code, a) not in done]
        if aids:
            todo.append((code, aids))
    B = 12
    nb = 0
    for bi in range(0, len(todo), B):
        batch = []
        for code, aids in todo[bi:bi + B]:
            f = idx[code]
            items = []
            for aid in aids:
                m = meta[aid]
                items.append(dict(arxiv=aid, title=m["title"], year=filters[code][aid]["v1_year"],
                                  venue=filters[code][aid]["venue"], relation=filters[code][aid]["relation"],
                                  abstract=(m.get("abstract") or "")[:1100]))
            batch.append(dict(code=code, fars_title=f["title"], fars_abstract=f["abstract"][:1400],
                              fars_heur_type=shortlists[code]["fars_heur_type"], candidates=items))
        json.dump(batch, open(f"{bd}/batch_{bi//B:02d}.json", "w"), ensure_ascii=False, indent=1)
        nb += 1
    print("judge batches:", nb, "in", bd, "| new (code,cand) pairs:", sum(len(a) for _, a in todo))


if __name__ == "__main__":
    main()
