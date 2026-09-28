"""
Fetch full text (arXiv LaTeX + PDF) for a Pangram-stratified sample of ICLR 2026 submissions,
so the substance molds (which need \\section/\\cite/\\ref structure) can be measured on the
same papers Pangram scored.

OpenReview PDFs are Cloudflare-blocked from this host (403 on both api and api2), so arXiv is
the only text route: known arxiv_id for accepted papers, title+author search otherwise.
Sampling is stratified over Pangram fraction_ai bands x accept/reject so the high-AI tail --
which is overwhelmingly rejected, and therefore rarely has a pre-registered arxiv_id -- is
represented. Resumable: re-running skips ids already on disk.

Usage: python3 collect_texts.py --per-band 100
"""
import argparse
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TEXTS = os.path.join(DATA, "texts")
ICLR = os.path.abspath(os.path.join(HERE, "..", "ICLR"))
sys.path.insert(0, ICLR)
import collect_iclr_dataset as C  # noqa: E402

BANDS = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.25), (0.25, 0.50), (0.50, 0.639), (0.639, 1.01)]


def band_of(fa):
    for i, (lo, hi) in enumerate(BANDS):
        if lo == hi:
            if fa == 0:
                return i
        elif lo < fa <= hi:
            return i
    return None


def have(sid):
    d = os.path.join(TEXTS, sid)
    return os.path.exists(os.path.join(d, "meta.json"))


S2_MATCH = "https://api.semanticscholar.org/graph/v1/paper/search/match"
S2_SLEEP = 1.5   # unauthenticated pool is ~1 request/second, shared across workers


def s2_arxiv_id(title):
    """Title -> arXiv id via Semantic Scholar.

    arXiv's own search API (export.arxiv.org) rate-limits hard enough to 429 a couple of
    workers, and it indexes rejected preprints unevenly; S2 answers the same question, so the
    search step goes here and only the e-print download still touches arxiv.org.
    """
    q = urllib.parse.urlencode({"query": title, "fields": "title,externalIds"})
    req = urllib.request.Request(f"{S2_MATCH}?{q}", headers={"User-Agent": "science-bench/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except Exception:
        return None, 0.0
    hits = body.get("data") or []
    if not hits:
        return None, 0.0
    aid = (hits[0].get("externalIds") or {}).get("ArXiv")
    if not aid:
        return None, 0.0
    return aid, C.sim(C.norm(title), C.norm(hits[0].get("title") or ""))


def collect_one(p):
    sid = p["submission_id"]
    d = os.path.join(TEXTS, sid)
    os.makedirs(d, exist_ok=True)
    aid, simv = p.get("arxiv_id"), 1.0
    if not aid:
        aid, simv = s2_arxiv_id(p["title"])
        time.sleep(S2_SLEEP)
        if not aid or simv < 0.9:
            return None
    tex_ok = C.download_tex(aid, Path(d) / "tex")
    pdf_ok = C.download_pdf(aid, Path(d) / "paper.pdf")
    if not (tex_ok or pdf_ok):
        return None
    meta = {k: p.get(k) for k in ("submission_id", "submission_number", "title", "fraction_ai",
                                  "avg_rating", "rating_mean", "soundness_mean", "presentation_mean",
                                  "contribution_mean", "accept", "tier", "review_ai_frac")}
    meta.update(arxiv_id=aid, arxiv_sim=round(float(simv), 3), tex_ok=bool(tex_ok), pdf_ok=bool(pdf_ok),
                band=band_of(p["fraction_ai"]))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=100)
    ap.add_argument("--max-tries-per-band", type=int, default=400)
    ap.add_argument("--bands", default="", help="comma-separated band indices; default all")
    ap.add_argument("--class", dest="klass", choices=["accept", "reject"], default=None,
                    help="restrict to one decision class; default fills both at half the quota each")
    ap.add_argument("--shard", default="", help="i/n — take every n-th candidate starting at i, so "
                                                "several workers can hunt the same cell without overlap")
    a = ap.parse_args()

    os.makedirs(TEXTS, exist_ok=True)
    papers = [json.loads(l) for l in open(f"{DATA}/papers_2026.jsonl") if json.loads(l)["fraction_ai"] is not None]
    # Stratify by band AND decision. Only accepted papers carry a pre-registered arxiv_id, so
    # sorting by "arxiv_id first" inside a band silently yields an all-accepted sample; rejected
    # papers have to be found by title search and need their own quota.
    buckets = {(i, c): [] for i in range(len(BANDS)) for c in (True, False)}
    for p in papers:
        b = band_of(p["fraction_ai"])
        if b is not None:
            buckets[(b, bool(p["accept"]))].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda p: (p.get("arxiv_id") is None, p["submission_id"]))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        buckets = {k: v[i::n] for k, v in buckets.items()}

    want = [int(x) for x in a.bands.split(",") if x.strip()] or list(range(len(BANDS)))
    classes = {"accept": [True], "reject": [False]}.get(a.klass, [True, False])
    per_cell = a.per_band if a.klass else max(1, a.per_band // 2)
    done = {k: sum(1 for p in buckets[k] if have(p["submission_id"])) for k in buckets}
    for b in want:
        for c in classes:
            k = (b, c)
            tries = 0
            for p in buckets[k]:
                if done[k] >= per_cell or tries >= a.max_tries_per_band:
                    break
                if have(p["submission_id"]):
                    continue
                tries += 1
                try:
                    m = collect_one(p)
                except Exception:
                    traceback.print_exc()
                    m = None
                if m:
                    done[k] += 1
                    print(f"band{b}/{'acc' if c else 'rej'} {done[k]}/{per_cell} {p['submission_id']} "
                          f"fa={p['fraction_ai']:.3f} tex={m['tex_ok']} {m['arxiv_id']}", flush=True)
            print(f"== band {b} {BANDS[b]} {'accept' if c else 'reject'}: "
                  f"{done[k]}/{per_cell} ({tries} tried) ==", flush=True)


if __name__ == "__main__":
    main()
