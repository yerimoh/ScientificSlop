#!/usr/bin/env python3
"""citation  -  Comparative positioning of prior work (Argument plane), v3.

Unit = RELATION CLAIM: a statement in the Introduction or Related Work in which the paper asserts a
relation to prior work. Four kinds, named after Jurgens et al. (2018) citation functions:
    limitation  (MOTIVATION)              prior work falls short or leaves a gap
    comparison  (COMPARES_OR_CONTRASTS)   a similarity or difference between works, or with this paper
    use         (USES)                    this paper uses a method / dataset / model / metric of a work
    extension   (EXTENDS)                 this paper extends or adapts a work
Sentences that only introduce a work (BACKGROUND) or point to future work (FUTURE) are NOT units.
A relation claim need not carry a \\cite ("Existing methods cannot handle long documents.").

Why v3 (0910 review). v2 scored every citation context and counted background citations as
failures, which penalised the corpus with more background prose and contradicted the item's own
reading rule; a quote-matching failure counted as the paper's failure; and the paragraph-level
rw_chain rate was written into the score field when the LLM layer was off. All three are gone.

Pipeline
  1. Paragraphs of Intro + Related (light tex cleaning, \\cite keys kept).
  2. Qwen2.5-32B greedy, PROMPT_relation.txt, one call per paragraph with the preceding paragraph
     as context; the model lists claims {kind, quote, target_phrase, targets, criterion, content,
     self_involved}.
  3. Deterministic verification. Cite macros are stripped from quote and paragraph before matching;
     the quote must be in the target paragraph. Each target must be a cite key of the target or
     context paragraph, or a name occurring in their text.
  4. Verdict from a fixed table (REQUIRED below), never from the model:
       inconclusive  quote not found | targets given but none verifies | bad kind   (extraction loss)
       vague         targets empty | content empty | comparison without criterion    (paper's omission)
       concrete      otherwise
  5. Dedup: same kind + same verified target set + same normalised criterion (or content).
  6. slop_score = vague / (concrete + vague); coverage = (concrete + vague) / all claims.
Layer M (0909 probe regex, unchanged) is reported under its own names and never as the score.

Usage: python3 measure.py [--ai-limit K --hu-limit K] [--only ids] [--claims] [--runs 3]
       -> ../results/{papers.jsonl, claims.jsonl, paragraphs.jsonl, summary.json}
"""
from __future__ import annotations
import argparse, itertools, json, os, re, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
from corpus import ai_papers, hu_papers
from views import load_doc, strip_comments, CITE_RE
from records import instance, write_jsonl, compare_groups, rate, describe, confound_audit, slop_score
import llm as LLM

RESULTS = os.path.join(HERE, "..", "results")
CHECKER = "citation_v3"
KINDS = {"limitation": "MOTIVATION", "comparison": "COMPARES_OR_CONTRASTS", "use": "USES", "extension": "EXTENDS"}
# required fields for a concrete verdict, per kind (the table in citation.md step 4)
REQUIRED = {"limitation": ("target", "content"), "comparison": ("target", "criterion", "content"),
            "use": ("target", "content"), "extension": ("target", "content")}
CONCRETE, VAGUE, INCONCLUSIVE, NOT_UNIT = "concrete", "vague", "inconclusive", "not_a_unit"
OWN_PHRASE_RE = re.compile(r"\b(our|ours|we|this (work|paper|study|approach|method|framework)|the proposed|proposed (method|approach|framework))\b", re.I)
# a phrase that names a class of prior work: the model's "general" label is overridden by this (deterministic)
PRIOR_CLASS_RE = re.compile(r"\b(prior|previous|existing|past|recent|earlier|current|conventional|traditional|standard|most|many|these|those|other)\s+(\w+[- ]){0,3}(works?|methods?|approach(es)?|studies|techniques?|models?|systems?|notions?|efforts?|lines?|attempts?|solutions?|baselines?|literature|frameworks?|architectures?|schemes?|linearizing|linearization)\b", re.I)
# a comparison must compare: without a cue like these and without the paper as one side, the sentence describes a work
LIMIT_CUE_RE = re.compile(r"\b(fail\w*|cannot|can not|do(es)? not|lack\w*|limited|limitation|insufficient|suboptimal|waste\w*|degrade\w*|ignore\w*|overlook\w*|neglect\w*|only|however|but|assum\w+|restrict\w*|require\w*|struggle\w*|unable|expensive|costly|prohibitive|bottleneck|drawback|shortcoming|gap|miss\w*|still)\b", re.I)
COMPARE_CUE_RE = re.compile(r"\b(unlike|whereas|while|although|though|compar\w*|differ\w*|similar\w*|contrast|than|both|however|but|instead|rather|outperform\w*|improv\w*|lag\w*|better|worse|superior|inferior|faster|slower|cheaper|larger|smaller|higher|lower|more|less|same|akin|analogous|versus|vs)\b", re.I)

# ---------------------------------------------------------------- layer M (probe_cp_linear_0909, unchanged)
SELF_RE = re.compile(r"\b(our|we|ours|this work|this paper|the present (work|paper|study))\b", re.I)
CUE_RE = re.compile(r"\b(however|unlike|in contrast|by contrast|whereas|while|although|though|differ\w*|rather than|instead of|"
                    r"similar(ly)?( to)?|compar(ed|es|ing|able|ison)( to| with)?|both|complement\w*|orthogonal|akin to|analogous|"
                    r"build(s|ing)? (on|upon)|based on|extend(s|ed|ing)?|follow(s|ing)?|improv\w+ (on|upon|over)|"
                    r"generali[sz]\w+|in common|share[sd]?|combin\w+|unif\w+|closest to|most (closely )?related|contrary|conversely|"
                    r"in line with|consistent with|departs? from|advantage over|limitation|connected to|relat(es|ed) to|relevant to|address\w*|gap|fill|bridge|motivat\w+|first to|none|neither)\b", re.I)
REL1_RE = re.compile(r"\b(unlike|in contrast to|compared (to|with)|in comparison (to|with)|similar(ly)? to|as in|akin to|analogous to|"
                     r"build(s|ing)? (up)?on|extend(s|ed|ing)? (the|this|these|their|prior|previous|earlier|work|idea|approach|method)|"
                     r"improv\w+ (up)?on|improv\w+ over|differ(s|ent|ing)? from|complement(s|ary to|ing)?|orthogonal to|"
                     r"follow(s|ing)? (the|this|their|prior|previous)|inspired by|closely related to|generali[sz]\w+ (the|this|their|prior|previous)|"
                     r"(same|similar|different) (idea|approach|spirit|line|principle) (as|to|of|from)|in the (same |similar )?(spirit|vein) (of|as)|alternative to|"
                     r"variant of|special case of|instance of|whereas|by contrast|conversely|in contrast,|however,? (their|this|these|such))\b", re.I)
GROUP_RE = re.compile(r"\b((these|those|both|all|such|existing|prior|previous|current|the above|the aforementioned|most|many|none|neither|each) (of )?(the |these )?"
                      r"(\w+[- ])*(methods?|approach(es)?|works?|studies|techniques?|systems?|models?|lines?|efforts?|directions?|frameworks?|strategies|schemes?|benchmarks?)|"
                      r"they (all|both|each)|the (former|latter)|(across|between|among) (these|those|them|the two|the three))\b", re.I)
ANAPH_RE = re.compile(r"\b(this|these|those|that|such) (\w+ )?(task|method|approach|technique|finding|result|line|idea|setting|phenomen\w*|challenge|limitation|gap|problem|"
                      r"direction|observation|insight|framework|formulation|assumption|strategy|scheme|model|benchmark|dataset|issue|concern|question)s?\b", re.I)
REL1_INIT_RE = re.compile(r"^\W*(similarly|likewise|analogously|in a similar vein|along (this|the same) line|following this|building on this|relatedly|complementary to this|in the same spirit)\b", re.I)
SYNTH_INIT_RE = re.compile(r"^\W*(in contrast|by contrast|conversely|overall|taken together|collectively|in summary|together,|unlike|whereas|however,? (these|those|existing|such|none|all|most|neither|current|prior|previous)|"
                           r"(while|although|though|despite) (these|those|such|existing|all|prior|previous))\b", re.I)
USE_RE = re.compile(r"\b(we|our \w+) (use|used|adopt|adopted|employ|employed|leverage|leveraged|apply|applied|rely|relied|follow|followed|build|built|base|based|take|took|borrow|borrowed|instantiate|evaluate|evaluated|reuse|reused|draw|drew)\b", re.I)
ABBR = re.compile(r"\b(et al|e\.g|i\.e|cf|vs|Fig|Eq|Sec|Tab|resp|approx|no|etc|al)\.$", re.I)


def light_tex(t: str) -> str:
    """Keep \\cite macros (the keys are the target vocabulary); drop floats, math, refs, formatting."""
    t = strip_comments(t)
    t = re.sub(r"\\begin\{(figure\*?|table\*?|algorithm\*?|tabular\*?|wrapfigure)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}", " xxmathxx ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " xxmathxx ", t)
    t = re.sub(r"\\(sub)*section\*?\{[^}]*\}", "\n\n", t)
    t = re.sub(r"\\paragraph\*?\{([^}]*)\}", r"\1. ", t)
    t = re.sub(r"\\(label|vspace|hspace|noindent|footnote|includegraphics)\s*(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"\\[A-Za-z]*[Rr]ef[A-Za-z]*\s*\{[^}]*\}", " xxrefxx ", t)
    t = re.sub(r"\\(textbf|textit|emph|texttt|textsc)\{([^}]*)\}", r"\2", t)
    return t


def sentences_raw(par: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\\\(\[~0-9])", par.strip())
    out = []
    for s in raw:
        if out and ABBR.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return [s for s in out if s.strip()]


def keys_of(text: str) -> list[list[str]]:
    groups = []
    for m in CITE_RE.finditer(text):
        ks = [k.strip() for k in re.split(r"[,\s]+", m.group(1)) if k.strip()]
        if ks:
            groups.append(ks)
    return groups


def tag_paragraph(par: str) -> dict:
    """0909 probe logic, unchanged: per-sentence tags and the rw_chain paragraph verdict."""
    sents = sentences_raw(par)
    pre = [(s, keys_of(s)) for s in sents]
    pre = [(s, g, set(itertools.chain.from_iterable(g))) for s, g in pre]
    rows, seen = [], set()
    for i, (s, groups, works) in enumerate(pre):
        body = CITE_RE.sub(" ", s)
        cue, rel1 = bool(CUE_RE.search(body)), bool(REL1_RE.search(body))
        selfr, group, anaph = bool(SELF_RE.search(body)), bool(GROUP_RE.search(body)), bool(ANAPH_RE.search(body))
        next_cited = i + 1 < len(pre) and bool(pre[i + 1][1])
        tags = set()
        if any(len(set(g)) >= 2 for g in groups): tags.add("BUNDLE")
        if len(groups) >= 2 and len(works) >= 2: tags.add("MULTI")
        if len(works) >= 2 and cue: tags.add("REL")
        if len(works) == 1 and (rel1 or REL1_INIT_RE.search(body) or (group and len(seen) >= 2)): tags.add("REL1")
        if selfr and (cue or anaph or (groups and USE_RE.search(body))) and (groups or seen or next_cited): tags.add("SELF")
        if not groups and not selfr and len(seen) >= 2 and (SYNTH_INIT_RE.search(body) or (group and GROUP_RE.search(body[:70]))): tags.add("SYNTH")
        rows.append({"works": sorted(works), "tags": sorted(tags)})
        seen |= works
    all_works = set(itertools.chain.from_iterable(r["works"] for r in rows))
    tagset = set(itertools.chain.from_iterable(r["tags"] for r in rows))
    return {"n_sentences": len(rows), "n_citing_sentences": sum(1 for r in rows if r["works"]), "n_works": len(all_works),
            "tag_counts": dict(Counter(t for r in rows for t in r["tags"])),
            "n_weaving_sentences": sum(1 for r in rows if set(r["tags"]) & {"BUNDLE", "MULTI", "REL"}),
            "n_self_sentences": sum(1 for r in rows if "SELF" in r["tags"]),
            "eligible": len(all_works) >= 3, "rw_chain": len(all_works) >= 3 and not (tagset & {"BUNDLE", "MULTI", "REL"})}


# ---------------------------------------------------------------- paragraphs
def merge_paragraphs(chunks: list[str]) -> list[str]:
    """A blank line inside a sentence (a stripped comment, a list item continuation) must not split a
    paragraph: a chunk is appended to its predecessor when that one does not end a sentence or the chunk
    starts in lower case or with a digit. Chunks with fewer than 8 alphabetic words (math-only, headings)
    are dropped."""
    out = []
    for c in chunks:
        if out and (not re.search(r"[.!?:]\s*[\)\]\"']*$", out[-1]) or re.match(r"^[a-z0-9]", c)):
            out[-1] = out[-1] + " " + c
        else:
            out.append(c)
    return [c for c in out if len(re.findall(r"[A-Za-z]{2,}", CITE_RE.sub(" ", c))) >= 8]


def paragraphs_of(doc) -> list[dict]:
    """Intro + Related paragraphs in order, each with its predecessor (same section) as context."""
    out = []
    for sec in doc.body_sections():
        if sec.role not in ("Intro", "Related"):
            continue
        t = light_tex(doc.tex[sec.body_start:sec.end])
        pars = merge_paragraphs([p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()])
        for pi, par in enumerate(pars):
            keys = sorted(set(itertools.chain.from_iterable(keys_of(par))))
            ctx = pars[pi - 1] if pi > 0 else ""
            ctx_keys = sorted(set(itertools.chain.from_iterable(keys_of(ctx)))) if ctx else []
            out.append({"section": sec.title, "role": sec.role, "para": pi, "text": par, "keys": keys, "context": ctx, "ctx_keys": ctx_keys,
                        **{f"m_{k}": v for k, v in tag_paragraph(par).items()}})
    return out


# ---------------------------------------------------------------- layer L
def strip_cites(s: str) -> str:
    return CITE_RE.sub(" ", s)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%]+", " ", s.lower())).strip()


def verify_claim(c: dict, par: dict) -> dict | None:
    """Deterministic verification + verdict from the REQUIRED table. Returns the claim record or None
    when the object is not a dict."""
    if not isinstance(c, dict):
        return None
    kind = c.get("kind") if c.get("kind") in KINDS else None
    quote = (c.get("quote") or "").strip() if isinstance(c.get("quote"), str) else ""
    targets = [t.strip() for t in (c.get("targets") or []) if isinstance(t, str) and t.strip()]
    rec = {"kind": kind or "invalid", "function": KINDS.get(kind, "UNPARSED"), "quote": quote,
           "target_phrase": (c.get("target_phrase") or "").strip() if isinstance(c.get("target_phrase"), str) else "",
           "targets": targets, "criterion": (c.get("criterion") or "").strip() if isinstance(c.get("criterion"), str) else "",
           "content": (c.get("content") or "").strip() if isinstance(c.get("content"), str) else "",
           "self_involved": bool(c.get("self_involved")),
           "other_side": c.get("other_side") if c.get("other_side") in ("prior_work", "own_work", "general") else "unspecified"}
    # targets: cite key of target or context paragraph, or a name occurring in their text
    window_keys = {k.lower() for k in par["keys"] + par["ctx_keys"]}
    window_text = _norm(strip_cites(par["text"] + " " + par["context"]))
    rec["targets_verified"] = [t for t in targets if t.lower() in window_keys or (len(_norm(t)) >= 3 and _norm(t) in window_text)]
    # unit guard 1: the other side must be prior work by others. A target phrase naming the paper itself
    # is decisive; the model's own_work label is accepted; its general label is accepted only when the
    # phrase names no class of prior work and no target verifies (deterministic override).
    phrase = rec["target_phrase"]
    if phrase and OWN_PHRASE_RE.search(phrase):
        why = "own_phrase"
    elif rec["other_side"] == "own_work":
        why = "own_work"
    elif rec["other_side"] == "general" and not rec["targets_verified"] and not CITE_RE.search(quote) and not (phrase and PRIOR_CLASS_RE.search(phrase)):
        why = "general"
    # unit guard 2: a "comparison" that involves neither the paper nor a comparison cue is either a limitation
    # the model mislabelled (a limitation cue is present; the kind is corrected) or a description of a work
    # (BACKGROUND, not a unit)
    elif kind == "comparison" and not rec["self_involved"] and not COMPARE_CUE_RE.search(strip_cites(quote)):
        if LIMIT_CUE_RE.search(strip_cites(quote)):
            kind = "limitation"; rec["kind"], rec["function"], rec["kind_reclassified"] = kind, KINDS[kind], "comparison->limitation"; why = None
        else:
            why = "description"
    else:
        why = None
    if why:
        rec["verdict"], rec["reason"] = NOT_UNIT, why
        rec["quote_verified"] = None
        return rec
    # quote: strip cite macros on both sides, then locate in the TARGET paragraph. A quote with an ellipsis
    # is verified fragment by fragment (each fragment of three or more words must be found).
    frags = [f.strip() for f in re.split(r"\.\.\.|…", strip_cites(quote)) if len(f.split()) >= 3] if quote else []
    rec["quote_verified"] = bool(frags) and all(LLM.find_quote(f, strip_cites(par["text"])) is not None for f in frags)
    # verdict
    if kind is None:
        rec["verdict"], rec["reason"] = INCONCLUSIVE, "bad_kind"
    elif not rec["quote_verified"]:
        rec["verdict"], rec["reason"] = INCONCLUSIVE, "quote_unverified"
    elif targets and not rec["targets_verified"]:
        rec["verdict"], rec["reason"] = INCONCLUSIVE, "target_unverified"
    else:
        have = {"target": bool(rec["targets_verified"]), "criterion": bool(rec["criterion"]), "content": bool(rec["content"])}
        missing = [f for f in REQUIRED[kind] if not have[f]]
        if missing:
            rec["verdict"], rec["reason"] = VAGUE, "no_" + "+no_".join(missing)
        else:
            rec["verdict"], rec["reason"] = CONCRETE, ""
    return rec


def dedup_key(rec: dict):
    return (rec["kind"], tuple(sorted(t.lower() for t in rec["targets_verified"])), _norm(rec["criterion"] or rec["content"])[:60])


def extract_paper(paper, pars: list[dict], tpl: str, run: int = 0) -> tuple[list[dict], dict]:
    claims, audit = [], Counter()
    for par in pars:
        prompt = (tpl.replace("{keys}", ", ".join(par["keys"]) or "(none)").replace("{ctx_keys}", ", ".join(par["ctx_keys"]) or "(none)")
                  .replace("{context}", par["context"][:3000] or "(no preceding paragraph)").replace("{target}", par["text"][:4000]))
        out = LLM.llm_json(prompt, tag="citation_relation_v3", run=run, max_tokens=2500)
        if not isinstance(out, dict) or not isinstance(out.get("claims"), list):
            audit["paragraphs_unparsed"] += 1; par["n_claims_raw"] = None; continue
        par["n_claims_raw"] = 0
        for c in out["claims"]:
            rec = verify_claim(c, par)
            if rec is None:
                audit["claims_dropped_not_dict"] += 1; continue
            if rec["verdict"] == NOT_UNIT:
                audit["claims_not_a_unit_" + rec["reason"]] += 1; continue
            if rec.get("kind_reclassified"):
                audit["claims_kind_reclassified"] += 1
            par["n_claims_raw"] += 1
            rec.update({"section": par["section"], "role": par["role"], "para": par["para"]})
            claims.append(rec)
    audit["n_before_dedup"] = len(claims)
    # dedup across the paper: same kind + verified targets + criterion/content
    kept, by_key = [], {}
    for k, rec in enumerate(claims):
        if rec["verdict"] == INCONCLUSIVE:
            rec["occurrences"] = [(rec["section"], rec["para"])]; kept.append(rec); continue
        dk = dedup_key(rec)
        if dk in by_key:
            by_key[dk]["occurrences"].append((rec["section"], rec["para"])); audit["claims_merged_duplicates"] += 1; continue
        rec["occurrences"] = [(rec["section"], rec["para"])]; by_key[dk] = rec; kept.append(rec)
    return kept, dict(audit)


def paper_metrics(claims: list[dict], pars: list[dict]) -> dict:
    vc = Counter(c["verdict"] for c in claims)
    dec = vc[CONCRETE] + vc[VAGUE]
    decided = [c for c in claims if c["verdict"] != INCONCLUSIVE]
    by_kind = {k: rate(sum(1 for c in decided if c["kind"] == k and c["verdict"] == CONCRETE), sum(1 for c in decided if c["kind"] == k)) for k in KINDS}
    selfc = [c for c in decided if c["self_involved"]]
    by_sec = {}
    for role in ("Intro", "Related"):
        d = [c for c in decided if c["role"] == role]
        by_sec[role] = {"n_decided": len(d), **{k: v for k, v in slop_score(sum(1 for c in d if c["verdict"] == VAGUE), len(d)).items() if k in ("slop_score", "weak")}}
    elig_rw = [p for p in pars if p["role"] == "Related" and p["m_eligible"]]
    return {"n_relation_claims": len(claims), "n_concrete": vc[CONCRETE], "n_vague": vc[VAGUE], "n_inconclusive": vc[INCONCLUSIVE],
            "concrete_rate": rate(vc[CONCRETE], dec), "vague_rate": rate(vc[VAGUE], dec), "inconclusive_rate": rate(vc[INCONCLUSIVE], len(claims)),
            "vague_reasons": dict(Counter(c["reason"] for c in claims if c["verdict"] == VAGUE)),
            "inconclusive_reasons": dict(Counter(c["reason"] for c in claims if c["verdict"] == INCONCLUSIVE)),
            "kind_counts": dict(Counter(c["kind"] for c in decided)), "concrete_rate_by_kind": by_kind,
            "n_self_claims": len(selfc), "self_claim_concrete_rate": rate(sum(1 for c in selfc if c["verdict"] == CONCRETE), len(selfc)),
            "by_section": by_sec,
            "quote_unverified_rate": rate(sum(1 for c in claims if c["quote"] and not c["quote_verified"]), sum(1 for c in claims if c["quote"])),
            "target_unverified_rate": rate(sum(1 for c in claims if c["targets"] and not c["targets_verified"]), sum(1 for c in claims if c["targets"])),
            "no_claim_eligible_paragraph_rate": rate(sum(1 for p in elig_rw if p.get("n_claims_raw") == 0), len(elig_rw)),
            # v3 score, demoted in v4 to an observation of the claims layer (see citation.md, version note)
            "claims_vague_score": rate(vc[VAGUE], dec), "claims_coverage": rate(dec, len(claims))}


MIN_CIT = 8   # registered: below this many citing sentences the score is weak (reported, not interpreted)


def layer_m_metrics(pars: list[dict]) -> dict:
    """Deterministic layer. Since v4 this carries the item score:
    slop_score = isolated citing sentences / citing sentences (Intro + Related Work),
    a citing sentence being isolated when it carries none of the weaving tags
    {BUNDLE, MULTI, REL} that relate the cited work to another prior work."""
    n_cit = sum(p["m_n_citing_sentences"] for p in pars)
    n_weave = sum(p["m_n_weaving_sentences"] for p in pars)
    rw = [p for p in pars if p["role"] == "Related"]
    elig = [p for p in rw if p["m_eligible"]]
    return {"n_paragraphs": len(pars), "n_related_paragraphs": len(rw), "n_citing_sentences": n_cit,
            "n_weaving_sentences": n_weave,
            "co_citation_rate": rate(n_weave, n_cit),
            "self_positioning_rate": rate(sum(p["m_n_self_sentences"] for p in pars), n_cit),
            "n_eligible_paragraphs": len(elig), "n_rw_chain_paragraphs": sum(1 for p in elig if p["m_rw_chain"]),
            "rw_chain_paragraph_rate": rate(sum(1 for p in elig if p["m_rw_chain"]), len(elig)),
            "tag_counts": dict(sum((Counter(p["m_tag_counts"]) for p in pars), Counter())),
            # slop level (SLOP_SCORE.md): isolated / citing sentences; every citing sentence is decidable
            **slop_score(n_cit - n_weave, n_cit, min_units=MIN_CIT),
            "coverage": 1.0 if n_cit else None}


def measure(paper, tpl: str | None, runs: int = 1):
    doc = load_doc(paper)
    pars = paragraphs_of(doc)
    base = {"corpus": paper["corpus"], "id": paper["id"], "n_words_body": doc.n_words_body, "source_incomplete": bool(doc.report["missing"])}
    if not pars:
        return {**base, "status": "no_sections"}, [], pars
    row = {**base, "status": "ok", **layer_m_metrics(pars)}
    if tpl is None:
        return {**row, "status": "ok"}, [], pars
    outs = []
    for r in range(runs):
        claims, audit = extract_paper(paper, pars, tpl, run=r)
        m = paper_metrics(claims, pars)
        outs.append((m["claims_vague_score"], claims, m, audit))
    vals = [o[0] for o in outs if o[0] is not None]
    if vals:
        med = statistics.median(vals); best = min([o for o in outs if o[0] is not None], key=lambda o: abs(o[0] - med))
    else:
        best = outs[0]
    _, claims, m, audit = best
    if not any(p.get("n_claims_raw") is not None for p in pars):
        return {**row, "status": "extraction_failed"}, [], pars
    row.update({**m, "audit": audit, "runs": len(outs), "score_per_run": [o[0] for o in outs],
                "rate_run_spread": (max(vals) - min(vals)) if len(vals) > 1 else 0.0})
    return row, claims, pars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ai-limit", type=int, default=0); ap.add_argument("--hu-limit", type=int, default=0)
    ap.add_argument("--only", default=""); ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--claims", action="store_true", help="also run the LLM relation-claim layer (evidence, claims_vague_score)")
    ap.add_argument("--layer-m-only", action="store_true", help="deprecated alias of the default deterministic run")
    a = ap.parse_args()
    ai, hu = ai_papers(), hu_papers()
    if a.ai_limit: ai = ai[:a.ai_limit]
    if a.hu_limit: hu = hu[:a.hu_limit]
    papers = ai + hu
    if a.only:
        ids = set(a.only.split(",")); papers = [p for p in papers if p["id"] in ids]
    if a.limit:
        papers = papers[:a.limit]
    tpl = open(os.path.join(HERE, "PROMPT_relation.txt")).read() if a.claims else None
    rows, claim_rows, par_rows, fails = [], [], [], []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as ex_:
        futs = {ex_.submit(measure, p, tpl, a.runs): p for p in papers}
        for f in as_completed(futs):
            p = futs[f]
            try:
                r, cs, ps = f.result(); rows.append(r)
                for k, c in enumerate(cs):
                    claim_rows.append(instance(p["id"], "citation_relation", k, c["verdict"], {"section": c["section"], "para": c["para"], "quote": c["quote"]},
                                               {"targets_verified": c["targets_verified"], "criterion": c["criterion"], "content": c["content"]},
                                               checker=CHECKER, corpus=p["corpus"], **{kk: v for kk, v in c.items() if kk not in ("section", "para", "quote", "targets_verified", "criterion", "content")}))
                for ps_ in ps:
                    par_rows.append({"id": p["id"], "corpus": p["corpus"], **{k: v for k, v in ps_.items() if k not in ("text", "context")}})
            except Exception as ex:
                fails.append({"id": p["id"], "error": repr(ex)[:300]})
    rows.sort(key=lambda r: (r["corpus"], r["id"]))
    write_jsonl(os.path.join(RESULTS, "papers.jsonl"), rows); write_jsonl(os.path.join(RESULTS, "claims.jsonl"), claim_rows); write_jsonl(os.path.join(RESULTS, "paragraphs.jsonl"), par_rows)
    ok = [r for r in rows if r["status"] in ("ok", "layer_m_only")]
    A = [r for r in ok if r["corpus"] == "AI"]; H = [r for r in ok if r["corpus"] == "HU" and not r["source_incomplete"]]
    summary = {"checker": CHECKER, "n_ai": len(A), "n_hu": len(H), "failures": fails, "status_counts": dict(Counter(r["status"] for r in rows)), "metrics": {}}
    mets = ["co_citation_rate", "self_positioning_rate", "rw_chain_paragraph_rate", "n_eligible_paragraphs", "n_citing_sentences"]
    if tpl is not None:
        mets = ["slop_score", "coverage", "n_relation_claims", "concrete_rate", "vague_rate", "inconclusive_rate", "self_claim_concrete_rate",
                "no_claim_eligible_paragraph_rate", "quote_unverified_rate", "target_unverified_rate", "rate_run_spread"] + mets
    for m in mets:
        summary["metrics"][m] = compare_groups(m, [r[m] for r in A if r.get(m) is not None], [r[m] for r in H if r.get(m) is not None])
    if tpl is not None:
        summary["verdict_totals"] = {c: {v: sum(r.get(f"n_{v}", 0) for r in g) for v in (CONCRETE, VAGUE, INCONCLUSIVE)} for c, g in (("AI", A), ("HU", H))}
        summary["vague_reasons"] = {c: dict(sum((Counter(r.get("vague_reasons", {})) for r in g), Counter())) for c, g in (("AI", A), ("HU", H))}
        summary["inconclusive_reasons"] = {c: dict(sum((Counter(r.get("inconclusive_reasons", {})) for r in g), Counter())) for c, g in (("AI", A), ("HU", H))}
        summary["kind_counts"] = {c: dict(sum((Counter(r.get("kind_counts", {})) for r in g), Counter())) for c, g in (("AI", A), ("HU", H))}
        summary["concrete_rate_by_kind"] = {c: {k: describe([r["concrete_rate_by_kind"].get(k) for r in g if r.get("concrete_rate_by_kind", {}).get(k) is not None]) for k in KINDS} for c, g in (("AI", A), ("HU", H))}
        summary["by_section"] = {c: {role: describe([r["by_section"][role]["slop_score"] for r in g if r.get("by_section") and r["by_section"][role]["slop_score"] is not None]) for role in ("Intro", "Related")} for c, g in (("AI", A), ("HU", H))}
        bands = {}
        for lo, hi in [(1, 3), (4, 8), (9, 999)]:
            bands[f"claims_{lo}_{hi}"] = {c: describe([r["slop_score"] for r in g if r.get("slop_score") is not None and lo <= (r["n_concrete"] + r["n_vague"]) <= hi]) for c, g in (("AI", A), ("HU", H))}
        summary["claim_count_bands"] = bands
        summary["audit_totals"] = {c: dict(sum((Counter(r.get("audit", {})) for r in g), Counter())) for c, g in (("AI", A), ("HU", H))}
    pos = [r for r in A if r.get("n_rw_chain_paragraphs")]; neg = [r for r in A if r.get("n_eligible_paragraphs") and not r.get("n_rw_chain_paragraphs")]
    summary["confound_audit_AI_rw_chain"] = {f: confound_audit([r[f] for r in pos], [r[f] for r in neg], f) for f in ["n_words_body", "n_citing_sentences", "n_eligible_paragraphs"]}
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=1, ensure_ascii=False)[:3000])
    for m in mets:
        s = summary["metrics"][m]
        print(f"  {m:34} AI mean {s['AI'].get('mean')} (n={s['AI'].get('n')}) | HU mean {s['HU'].get('mean')} (n={s['HU'].get('n')}) | cliff {s['cliff_delta_AI_minus_HU']} p {s['mwu_p']}")


if __name__ == "__main__":
    main()
