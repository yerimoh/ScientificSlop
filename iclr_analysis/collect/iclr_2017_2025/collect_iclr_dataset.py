"""
collect_iclr_dataset.py — per-year ICLR dataset: tex + pdf + review scores, 50 per year.

Layout (target):
  Evaluation/ICLR/data/
    labels.csv                       every candidate considered (incl. tex_ok, decision, scores)
    {year}/{note_id}/
        meta.json                    openreview content + parsed reviews(scores) + decision + arxiv_id
        tex/                         arXiv e-print source (main.tex + sections + .bbl)
        paper.pdf                    arXiv pdf
    SUMMARY.md                       per-year counts, accept/reject balance, arXiv match-rate

Reachability from this server (measured 2026-07-04):
  * v1 api.openreview.net : ICLR 2017-2023 OK.  2015 = 0 structured, 2016 conference = 0.
  * api2 + openreview.net PDF host : Cloudflare 403 (2024/2025/2026) -> NOT collectable here.
  * arXiv : OK -> tex (e-print) AND pdf both come from arXiv (one match serves both).

So a qualifying paper = arXiv-matched (gives tex+pdf) AND has OpenReview review scores.
Per year we aim for a balanced 25 accept / 25 reject (=50), falling back to whatever is
available. Resumable (skips note_ids already saved). arXiv rate-limit 3s.
"""
import argparse
import csv
import io
import json
import os
import re
import tarfile
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
V1 = "https://api.openreview.net"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_SRC = "https://arxiv.org/e-print/{}"
ARXIV_PDF = "https://arxiv.org/pdf/{}"
UA = {"User-Agent": "Mozilla/5.0 (research; ICLR mold dataset)"}
NS = {"a": "http://www.w3.org/2005/Atom"}
OR_SLEEP, ARXIV_SLEEP, TIMEOUT, RETRIES = 0.7, 3.0, 40, 4


def invitation(year):
    if year == 2017:
        return f"ICLR.cc/{year}/conference/-/submission"
    return f"ICLR.cc/{year}/Conference/-/Blind_Submission"  # 2018-2023


# ---------------- http ----------------
def http(url, binary=False):
    last = None
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=TIMEOUT) as r:
                return r.read() if binary else r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (i + 1))  # linear backoff (helps on transient 503)
    raise last


def http_json(url):
    return json.loads(http(url))


# ---------------- openreview parse ----------------
def cval(c, k):
    v = c.get(k)
    return v["value"] if isinstance(v, dict) and "value" in v else v


def first_int(s):
    if s is None:
        return None
    m = re.search(r"-?\d+", str(s))
    return int(m.group(0)) if m else None


SCORE_KEYS = ("rating", "recommendation", "soundness", "correctness",
              "technical_novelty_and_significance", "empirical_novelty_and_significance",
              "presentation", "contribution", "confidence")
# review free-text fields (names vary by year); keep every string field as-is
TEXT_KEYS = ("summary", "review", "main_review", "summary_of_the_paper",
             "summary_of_the_review", "strengths_and_weaknesses", "strengths",
             "weaknesses", "questions", "limitations", "comment", "detailed_comments",
             "clarity_quality_novelty_and_reproducibility", "title")


def parse_review(content):
    g = lambda k: cval(content, k)
    overall = first_int(g("rating")) if g("rating") is not None else first_int(g("recommendation"))
    soundness = None
    for k in ("soundness", "correctness", "technical_novelty_and_significance"):
        if g(k) is not None:
            soundness = first_int(g(k))
            break
    scores = {k: g(k) for k in SCORE_KEYS if g(k) is not None}
    # full review text: named text fields + any other long string field, verbatim
    text = {k: g(k) for k in TEXT_KEYS if isinstance(g(k), str) and g(k).strip()}
    for k in content:
        if k in text or k in SCORE_KEYS:
            continue
        v = g(k)
        if isinstance(v, str) and len(v.strip()) > 40:
            text[k] = v
    return {"overall": overall, "soundness": soundness, "confidence": first_int(g("confidence")),
            "scores_raw": scores, "text": text}


def reply_kind(reply):
    """Classify an OpenReview reply by its invitation into 'review' | 'decision' | 'other'."""
    inv = reply.get("invitation", "").split("/")[-1].lower()
    if "meta_review" in inv or "decision" in inv or "acceptance" in inv:
        return "decision"
    if "review" in inv:  # official_review / review
        return "review"
    c = reply.get("content", {})
    if "rating" in c and "comment" not in c:  # fallback for odd invitation names
        return "review"
    return "other"


def _tier_from(s):
    sl = (s or "").lower()
    for t in ("oral", "spotlight", "notable", "poster"):
        if t in sl:
            return t
    return None


def resolve_decision(content, replies):
    """Primary: a Decision/Meta_Review/acceptance reply's decision string
    (`decision` for 2017/2020-2023, `recommendation` for 2019 Meta_Review) —
    'Accept (Poster)'/'Reject'. Fallback: content.venue."""
    dstr = None
    for r in replies:
        if reply_kind(r) != "decision":
            continue
        rc = r.get("content", {})
        dstr = str(cval(rc, "decision") or cval(rc, "recommendation") or "")
        if dstr:
            break
    if dstr:
        dl = dstr.lower()
        accept = dl.startswith("accept") or ("accept" in dl and "reject" not in dl)
        return {"accept": bool(accept), "tier": _tier_from(dstr), "venue": dstr}
    # venue fallback
    v = str(cval(content, "venue") or "")
    vl = v.lower()
    if not vl or vl.startswith("submitted") or "reject" in vl:
        accept = None if not vl else False
    elif vl.startswith("iclr") or _tier_from(vl):
        accept = True
    else:
        accept = None
    return {"accept": accept, "tier": _tier_from(vl), "venue": v}


def agg(reviews, f):
    xs = [r[f] for r in reviews if r.get(f) is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def fetch_submissions(year, pool):
    """Yield up to `pool` submissions (with parsed reviews+decision), newest pages first."""
    inv = invitation(year)
    out, offset = [], 0
    while len(out) < pool:
        d = http_json(f"{V1}/notes?invitation={inv}&details=directReplies&limit=100&offset={offset}")
        notes = d.get("notes", [])
        if not notes:
            break
        for n in notes:
            c = n.get("content", {})
            replies = (n.get("details", {}) or {}).get("directReplies", [])
            reviews = [parse_review(r.get("content", {})) for r in replies
                       if reply_kind(r) == "review"]
            dec = resolve_decision(c, replies)
            out.append({
                "note_id": n.get("id"), "year": year,
                "title": cval(c, "title"), "authors": cval(c, "authors") or [],
                "abstract": cval(c, "abstract"), "keywords": cval(c, "keywords"),
                "reviews": reviews, "decision": dec,
                "overall_mean": agg(reviews, "overall"),
                "soundness_mean": agg(reviews, "soundness"),
                "confidence_mean": agg(reviews, "confidence"),
            })
        offset += 100
        time.sleep(OR_SLEEP)
    return out


# ---------------- arxiv match + download ----------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def sim(a, b):
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def surname(a):
    toks = norm(a).split()  # norm() can empty non-ASCII names -> guard the [-1]
    return toks[-1] if toks else ""


def _query(q):
    try:
        return ET.fromstring(http(f"{ARXIV_API}?search_query={urllib.parse.quote(q)}&max_results=8"))
    except Exception:
        return None


def _best(root, title, fa):
    best, bs = None, 0.0
    if root is None:
        return best, bs
    for e in root.findall("a:entry", NS):
        ct = (e.findtext("a:title", "", NS) or "").strip()
        s = sim(title, ct)
        aus = {surname(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)}
        if fa and fa in aus:
            s += 0.05
        if s > bs:
            m = re.search(r"abs/([^v]+)", e.findtext("a:id", "", NS) or "")
            best, bs = (m.group(1) if m else None), s
    return best, bs


def arxiv_match(title, first_author):
    fa = surname(first_author)
    best, bs = _best(_query(f'ti:"{title[:200]}"'), title, fa)
    if bs < 0.9:
        toks = " ".join(w for w in norm(title).split() if len(w) > 2)[:220]
        if toks:
            time.sleep(ARXIV_SLEEP)
            b2, s2 = _best(_query(f"all:{toks}"), title, fa)
            if s2 > bs:
                best, bs = b2, s2
    return best, round(bs, 3)


def is_main(txt):
    return "\\documentclass" in txt and "\\begin{document}" in txt


def download_tex(arxiv_id, dest):
    dest.mkdir(parents=True, exist_ok=True)
    try:
        blob = http(ARXIV_SRC.format(arxiv_id), binary=True)
    except Exception:
        return False
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    except tarfile.ReadError:
        import gzip
        try:
            txt = gzip.decompress(blob).decode("utf-8", "ignore")
            (dest / "main.tex").write_text(txt)
            return is_main(txt)
        except Exception:
            return False
    found, mains = False, []
    for m in tf.getmembers():
        if not m.isfile() or not re.search(r"\.(tex|bbl|bib|sty|cls)$", os.path.basename(m.name), re.I):
            continue
        try:
            data = tf.extractfile(m).read()
        except Exception:
            continue
        outp = dest / m.name
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_bytes(data)
        if m.name.lower().endswith(".tex"):
            found = True
            if is_main(data.decode("utf-8", "ignore")):
                mains.append((len(data), m.name))
    if mains:
        mains.sort(reverse=True)
        best = dest / mains[0][1]
        if best.name != "main.tex":
            (dest / "main.tex").write_bytes(best.read_bytes())
    return found


def download_pdf(arxiv_id, dest_file):
    try:
        b = http(ARXIV_PDF.format(arxiv_id), binary=True)
    except Exception:
        return False
    if b[:5] != b"%PDF-":
        return False
    dest_file.write_bytes(b)
    return True


# ---------------- driver ----------------
def collect_year(year, target, per_class, pool, max_tries, labels_rows):
    ydir = DATA / str(year)
    ydir.mkdir(parents=True, exist_ok=True)
    have = {"accept": 0, "reject": 0}
    for pdir in ydir.iterdir():
        if pdir.is_dir() and (pdir / "meta.json").exists():
            m = json.loads((pdir / "meta.json").read_text())
            k = "accept" if (m.get("decision") or {}).get("accept") else "reject"
            have[k] += 1
    print(f"[{year}] resume: have accept={have['accept']} reject={have['reject']}")

    subs = fetch_submissions(year, pool)
    # only papers that actually have review scores + a decided outcome
    subs = [s for s in subs if s["overall_mean"] is not None and s["decision"]["accept"] is not None]
    print(f"[{year}] {len(subs)} submissions with scores+decision (pool={pool})")

    tries = 0
    for s in subs:
        if have["accept"] >= per_class and have["reject"] >= per_class:
            break
        if have["accept"] + have["reject"] >= target:
            break
        k = "accept" if s["decision"]["accept"] else "reject"
        if have[k] >= per_class:
            continue
        pdir = ydir / s["note_id"]
        if (pdir / "meta.json").exists():
            continue
        if tries >= max_tries:
            print(f"[{year}] hit max_tries={max_tries}")
            break
        tries += 1
        try:
            title = s.get("title") or ""
            authors = s.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            fa = authors[0] if authors else ""
            aid, score = arxiv_match(title, fa)
            row = {"year": year, "note_id": s["note_id"], "title": title.replace("\n", " ")[:200],
                   "class": k, "overall_mean": s["overall_mean"], "soundness_mean": s["soundness_mean"],
                   "confidence_mean": s["confidence_mean"], "tier": s["decision"]["tier"],
                   "arxiv_id": aid or "", "sim": score, "tex_ok": False}
            labels_rows.append(row)
            if not aid or score < 0.9:
                print(f"[{year}] {k} miss sim={score} :: {title[:55]}")
                time.sleep(ARXIV_SLEEP)
                continue
            pdir.mkdir(parents=True, exist_ok=True)
            tex_ok = download_tex(aid, pdir / "tex")
            pdf_ok = download_pdf(aid, pdir / "paper.pdf")
            if tex_ok:
                s["arxiv_id"] = aid
                s["pdf_ok"] = pdf_ok
                (pdir / "meta.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
                have[k] += 1
                row["tex_ok"] = True
                print(f"[{year}] {k} OK {have[k]}/{per_class} arxiv={aid} pdf={pdf_ok} :: {title[:45]}")
            else:
                for f in pdir.rglob("*"):  # no usable tex: drop the dir
                    if f.is_file():
                        f.unlink()
                print(f"[{year}] {k} no-tex arxiv={aid} :: {title[:45]}")
            time.sleep(ARXIV_SLEEP)
        except Exception as e:  # one bad record must not abort the year
            import traceback
            print(f"[{year}] SKIP {s.get('note_id')} :: {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(ARXIV_SLEEP)
    return have


def count_disk(year):
    """Authoritative accept/reject counts from saved meta.json (survives mid-run errors)."""
    ydir = DATA / str(year)
    have = {"accept": 0, "reject": 0}
    if not ydir.exists():
        return have
    for pdir in ydir.iterdir():
        mp = pdir / "meta.json"
        if pdir.is_dir() and mp.exists():
            try:
                m = json.loads(mp.read_text())
                have["accept" if (m.get("decision") or {}).get("accept") else "reject"] += 1
            except Exception:
                pass
    return have


def write_summary(years):
    lines = ["# ICLR dataset — collection summary", "",
             "| year | accept | reject | total |", "|---|---|---|---|"]
    for y in sorted(years):
        h = count_disk(y)
        lines.append(f"| {y} | {h['accept']} | {h['reject']} | {h['accept']+h['reject']} |")
    (DATA / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017,2018,2019,2020,2021,2022,2023")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--per-class", type=int, default=25, help="cap per accept/reject bucket")
    ap.add_argument("--pool", type=int, default=600, help="submissions to pull per year before matching")
    ap.add_argument("--max-tries", type=int, default=180, help="arXiv match attempts per year (runtime bound)")
    args = ap.parse_args()

    labels_path = DATA / "labels.csv"
    labels_rows = []
    years = [int(x) for x in args.years.split(",") if x.strip()]
    for y in years:
        try:
            collect_year(y, args.target, args.per_class, args.pool, args.max_tries, labels_rows)
        except Exception as e:  # noqa
            print(f"[{y}] ERROR {e}")
        # flush labels + summary each year (resumable/progress-visible)
        if labels_rows:
            cols = list(labels_rows[0].keys())
            new = not labels_path.exists()
            with open(labels_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                if new:
                    w.writeheader()
                w.writerows(labels_rows)
            labels_rows.clear()
        write_summary(years)
        h = count_disk(y)  # disk = source of truth
        print(f"==== [{y}] done: accept={h['accept']} reject={h['reject']} ====")


if __name__ == "__main__":
    main()
