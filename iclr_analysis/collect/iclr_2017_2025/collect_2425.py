"""
Collect ICLR 2024/2025 papers into the existing data/{year}/{id}/ layout, using HF labels
(_labels_20XX.jsonl) + arXiv for tex+pdf (OpenReview PDF is Cloudflare-blocked for these years).
Stratified sample by score (1-10, target 20 each) + awards; arxiv_id-known papers first (fast).

Per paper: meta.json (year, decision{accept,tier,venue}, reviews[overall/soundness/confidence/text],
overall_mean, arxiv_id, tex_ok, pdf_ok) + tex/ + paper.pdf. Resumable.
"""
import argparse
import glob
from pathlib import Path
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_iclr_dataset as C   # arxiv_match, download_tex, download_pdf, ARXIV_SLEEP
DATA = os.path.join(HERE, "data")


def score_of(om):
    return max(1, min(10, round(om))) if om is not None else None


def load_labels(year):
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"_labels_{year}.jsonl"))]
    for r in rows:
        r["score"] = score_of(r.get("overall_mean"))
    return rows


def collect_year(year, per_bucket, max_tries):
    ydir = os.path.join(DATA, str(year))
    os.makedirs(ydir, exist_ok=True)
    have = {}
    for d in glob.glob(f"{ydir}/*/meta.json"):
        m = json.load(open(d))
        have[m["note_id"]] = m
    # bucket counts from disk
    from collections import Counter
    cnt = Counter()
    for m in have.values():
        s = score_of(m.get("overall_mean"))
        if s:
            cnt[("s", s)] += 1
        aw = (m.get("decision") or {}).get("tier")
        if aw in ("oral", "spotlight"):
            cnt[("a", aw)] += 1
    rows = load_labels(year)
    # order: arxiv_id-known first, then by score present; shuffle-ish by id for spread
    rows = [r for r in rows if r.get("score") is not None and r["title"]]
    rows.sort(key=lambda r: (r.get("arxiv_id") is None, r["id"]))

    def need(r):
        s = r["score"]; aw = r["tier"]
        if cnt[("s", s)] < per_bucket:
            return True
        if aw in ("oral", "spotlight") and cnt[("a", aw)] < per_bucket:
            return True
        return False

    tries = 0
    for r in rows:
        if tries >= max_tries:
            print(f"[{year}] max_tries {max_tries} reached"); break
        pid = r["id"]
        if pid in have:
            continue
        if not need(r):
            continue
        tries += 1
        aid = r.get("arxiv_id")
        try:
            if not aid:
                aid, simv = C.arxiv_match(r["title"], "")
                if not aid or simv < 0.9:
                    time.sleep(C.ARXIV_SLEEP); continue
            pdir = os.path.join(ydir, pid)
            os.makedirs(pdir, exist_ok=True)
            tex_ok = C.download_tex(aid, Path(pdir) / "tex")
            pdf_ok = C.download_pdf(aid, Path(pdir) / "paper.pdf")
            if not (tex_ok or pdf_ok):
                for f in glob.glob(f"{pdir}/**/*", recursive=True):
                    if os.path.isfile(f):
                        os.remove(f)
                time.sleep(C.ARXIV_SLEEP); continue
            meta = {"note_id": pid, "year": year, "title": r["title"],
                    "decision": {"accept": r["accept"], "tier": r["tier"], "venue": r.get("decision_raw")},
                    "reviews": r["reviews"], "overall_mean": r["overall_mean"],
                    "soundness_mean": r.get("soundness_mean"), "arxiv_id": aid,
                    "tex_ok": tex_ok, "pdf_ok": pdf_ok, "pdf_source": "arxiv"}
            json.dump(meta, open(os.path.join(pdir, "meta.json"), "w"), ensure_ascii=False, indent=1)
            have[pid] = meta
            s = r["score"]; cnt[("s", s)] += 1
            if r["tier"] in ("oral", "spotlight"):
                cnt[("a", r["tier"])] += 1
            print(f"[{year}] OK s{s} {r['tier']} tex={tex_ok} pdf={pdf_ok} arxiv={aid} :: {r['title'][:45]}", flush=True)
            time.sleep(C.ARXIV_SLEEP)
        except Exception as e:
            print(f"[{year}] ERR {pid}: {str(e)[:70]}", flush=True)
            time.sleep(C.ARXIV_SLEEP)
    tot = len(glob.glob(f"{ydir}/*/meta.json"))
    print(f"==== [{year}] done: {tot} papers on disk ====", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2025,2024")
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--max-tries", type=int, default=600)
    a = ap.parse_args()
    for y in [int(x) for x in a.years.split(",")]:
        collect_year(y, a.per_bucket, a.max_tries)


if __name__ == "__main__":
    main()
