"""
collect_strata.py — ICLR papers stratified by (year x integer score) and (year x award).

Per the 0704 request:
  * real integer score 1..10  -> 20 papers each, per year   (score = round(mean review rating))
  * awards oral / spotlight   -> 20 papers each, per year
  * PDF for every paper (official OpenReview PDF via api.openreview.net — works from this
    server, unlike openreview.net which is Cloudflare-blocked); tex added when the paper is
    on arXiv ("if that fails, at least the PDF" — pdf always, tex best-effort).

Storage (deduplicated pool + symlink views, so an oral paper that is also score-7 is stored
once and linked into both views):
  data/{year}/{note_id}/         meta.json + paper.pdf + tex/(optional)
  data/by_score/{year}/{s}/{note_id}   -> symlink into pool
  data/by_award/{year}/{award}/{note_id} -> symlink into pool
  data/strata_index.csv          note_id, year, score, award, tex_ok, pdf_source

Reuses helpers from collect_iclr_dataset.py. Resumable (counts existing pool per stratum).
Years 2017-2023 (2015/16 absent; 2024-26 on Cloudflare-blocked api2 — run from another IP).
"""
import argparse
import collections
import csv
import glob
import json
import os
import time
from pathlib import Path

import collect_iclr_dataset as C  # fetch_submissions, arxiv_match, download_tex, http, DATA, ...

DATA = C.DATA
OR_PDF = "https://api.openreview.net/pdf?id={}"
PER = 20
SCORES = list(range(1, 11))
AWARDS = ("oral", "spotlight")


def score_of(om):
    return max(1, min(10, round(om)))


def award_of(decision):
    """Map the raw decision string to an honor tier. Handles ICLR 2023 'notable-top-5/25%'."""
    t = str((decision or {}).get("venue") or "").lower()
    if "oral" in t or "talk" in t or "top-5" in t or "top 5" in t:  # 2017-22 oral/talk, 2023 top-5%
        return "oral"
    if "spotlight" in t or "notable" in t or "top-25" in t or "top 25" in t:  # spotlight / 2023 top-25%
        return "spotlight"
    return None  # poster / reject


def download_or_pdf(note_id, dest_file):
    try:
        b = C.http(OR_PDF.format(note_id), binary=True)
    except Exception:
        return False
    if not b or b[:5] != b"%PDF-":
        return False
    dest_file.write_bytes(b)
    return True


def seed_counts(year):
    """Count already-saved pool papers into (score, award) strata."""
    hs, ha = collections.Counter(), collections.Counter()
    for mp in glob.glob(f"{DATA}/{year}/*/meta.json"):
        try:
            m = json.loads(open(mp).read())
        except Exception:
            continue
        if m.get("overall_mean") is None:
            continue
        hs[score_of(m["overall_mean"])] += 1
        aw = award_of(m.get("decision"))
        if aw:
            ha[aw] += 1
    return hs, ha


def strata_full(hs, ha):
    return all(hs[s] >= PER for s in SCORES) and all(ha[a] >= PER for a in AWARDS)


def collect_year(year, pool, max_tries):
    ydir = DATA / str(year)
    ydir.mkdir(parents=True, exist_ok=True)
    hs, ha = seed_counts(year)
    print(f"[{year}] resume scores={dict(sorted(hs.items()))} awards={dict(ha)}")

    subs = C.fetch_submissions(year, pool)
    subs = [s for s in subs if s["overall_mean"] is not None and s["decision"]["accept"] is not None]
    print(f"[{year}] {len(subs)} decided submissions with scores (pool={pool})")

    tries = 0
    for s in subs:
        if strata_full(hs, ha):
            print(f"[{year}] all strata full")
            break
        sc = score_of(s["overall_mean"])
        aw = award_of(s["decision"])
        need = (hs[sc] < PER) or (aw and ha[aw] < PER)
        if not need:
            continue
        pdir = ydir / s["note_id"]
        if (pdir / "meta.json").exists():
            continue  # already in pool (counted in seed)
        if tries >= max_tries:
            print(f"[{year}] hit max_tries={max_tries}")
            break
        tries += 1
        try:
            # PDF is mandatory (official OpenReview); tex is best-effort (arXiv)
            pdir.mkdir(parents=True, exist_ok=True)
            pdf_ok = download_or_pdf(s["note_id"], pdir / "paper.pdf")
            pdf_source = "openreview"
            title = s.get("title") or ""
            authors = s.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            fa = authors[0] if authors else ""
            aid, sim = C.arxiv_match(title, fa)
            tex_ok = False
            if aid and sim >= 0.9:
                tex_ok = C.download_tex(aid, pdir / "tex")
                if not tex_ok:
                    # drop empty tex dir
                    td = pdir / "tex"
                    if td.exists():
                        for f in td.rglob("*"):
                            if f.is_file():
                                f.unlink()
            if not pdf_ok and not tex_ok:
                # nothing retrievable; drop
                for f in pdir.rglob("*"):
                    if f.is_file():
                        f.unlink()
                if pdir.exists():
                    os.rmdir(pdir) if not any(pdir.iterdir()) else None
                print(f"[{year}] drop (no pdf/tex) :: {title[:45]}")
                time.sleep(C.ARXIV_SLEEP)
                continue
            s["score"] = sc
            s["award"] = aw
            s["arxiv_id"] = aid if tex_ok else ""
            s["tex_ok"] = tex_ok
            s["pdf_ok"] = pdf_ok
            s["pdf_source"] = pdf_source
            (pdir / "meta.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
            hs[sc] += 1
            if aw:
                ha[aw] += 1
            print(f"[{year}] OK s{sc}/{hs[sc]} aw={aw or '-'} tex={tex_ok} pdf={pdf_ok} :: {title[:42]}")
            time.sleep(C.ARXIV_SLEEP)
        except Exception as e:
            import traceback
            print(f"[{year}] SKIP {s.get('note_id')} :: {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(C.ARXIV_SLEEP)
    return hs, ha


def build_views_and_index():
    """Rebuild symlink views + strata_index.csv from the pool."""
    for v in ("by_score", "by_award"):
        vp = DATA / v
        if vp.exists():
            for link in vp.rglob("*"):
                if link.is_symlink():
                    link.unlink()
    rows = []
    for mp in sorted(glob.glob(f"{DATA}/20*/*/meta.json")):
        p = mp.split("/")
        year, nid = p[-3], p[-2]
        if not year.isdigit():
            continue
        m = json.loads(open(mp).read())
        if m.get("overall_mean") is None:
            continue
        sc = score_of(m["overall_mean"])
        aw = award_of(m.get("decision"))
        tex_ok = bool(glob.glob(f"{DATA}/{year}/{nid}/tex/*.tex"))
        pool_rel = os.path.relpath(f"{DATA}/{year}/{nid}", f"{DATA}/by_score/{year}/{sc}")
        sdir = DATA / "by_score" / year / str(sc)
        sdir.mkdir(parents=True, exist_ok=True)
        link = sdir / nid
        if not link.exists():
            os.symlink(pool_rel, link)
        if aw:
            adir = DATA / "by_award" / year / aw
            adir.mkdir(parents=True, exist_ok=True)
            arel = os.path.relpath(f"{DATA}/{year}/{nid}", adir)
            al = adir / nid
            if not al.exists():
                os.symlink(arel, al)
        rows.append(dict(year=year, note_id=nid, score=sc, award=aw or "",
                         overall_mean=m["overall_mean"], tier=(m.get("decision") or {}).get("tier"),
                         tex_ok=tex_ok, pdf_source=m.get("pdf_source", "arxiv")))
    with open(DATA / "strata_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # summary
    byscore = collections.defaultdict(collections.Counter)
    byaward = collections.defaultdict(collections.Counter)
    for r in rows:
        byscore[r["year"]][r["score"]] += 1
        if r["award"]:
            byaward[r["year"]][r["award"]] += 1
    lines = ["# ICLR strata summary", "", "## by score (papers per year x integer score)",
             "| year | " + " | ".join(f"s{s}" for s in SCORES) + " |",
             "|" + "---|" * (len(SCORES) + 1)]
    for y in sorted(byscore):
        lines.append(f"| {y} | " + " | ".join(str(byscore[y].get(s, 0)) for s in SCORES) + " |")
    lines += ["", "## by award", "| year | oral | spotlight |", "|---|---|---|"]
    for y in sorted(byaward):
        lines.append(f"| {y} | {byaward[y].get('oral',0)} | {byaward[y].get('spotlight',0)} |")
    (DATA / "STRATA_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"index: {len(rows)} papers -> strata_index.csv ; views: by_score/ by_award/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017,2018,2019,2020,2021,2022,2023")
    ap.add_argument("--pool", type=int, default=4000)
    ap.add_argument("--max-tries", type=int, default=500)
    ap.add_argument("--views-only", action="store_true", help="just rebuild symlink views + index")
    args = ap.parse_args()
    if args.views_only:
        build_views_and_index()
        return
    for y in [int(x) for x in args.years.split(",") if x.strip()]:
        try:
            hs, ha = collect_year(y, args.pool, args.max_tries)
            print(f"==== [{y}] done scores={dict(sorted(hs.items()))} awards={dict(ha)} ====")
        except Exception as e:
            print(f"[{y}] ERROR {e}")
        build_views_and_index()


if __name__ == "__main__":
    main()
