"""Top up the ICLR tex corpus so that every integer review score (2..8) has >= TARGET papers with
usable tex; score 9 (and 1, 10) takes everything that exists. Candidates come from the full
submission lists (data/_paperlists 2017-2023, data/_labels_2024/2025.jsonl), are sampled at
random (seed 0, round-robin over years), matched to arXiv through Semantic Scholar title match
(fallback: arXiv API), and the e-print tex is stored in the usual data/{year}/{note_id}/ layout
with a minimal meta.json (pdf_source = "extend_0915"). Withdrawn / desk-rejected papers are skipped.
Selection never looks at any measured score. Resumable; log to data/extend_0915.log.
"""
import json, glob, os, re, random, sys, time, collections, urllib.request, urllib.parse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_iclr_dataset as C

DATA = C.DATA
TARGET = 150
MARGIN = 1.15           # quality filter removes ~15 % afterwards
NEED_OVERRIDE = {}      # score -> extra papers wanted (computed below from the review corpus)
S2 = "https://api.semanticscholar.org/graph/v1/paper/search/match?query={}&fields=externalIds,title"
random.seed(0)


def mean_rating(p):
    r = p.get("rating_avg")
    if isinstance(r, list) and r:
        try: return float(r[0])
        except Exception: pass
    if isinstance(r, (int, float)): return float(r)
    s = p.get("rating")
    if isinstance(s, str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        return sum(nums) / len(nums) if nums else None
    return None


def decision_of(status):
    t = (status or "").lower()
    acc = any(k in t for k in ("poster", "spotlight", "oral", "notable", "accept", "talk"))
    tier = "oral" if ("oral" in t or "talk" in t or "top-5" in t) else ("spotlight" if ("spotlight" in t or "top-25" in t or "notable" in t) else ("poster" if acc else None))
    return {"accept": acc, "tier": tier, "venue": status}


def candidates():
    have = {os.path.basename(d) for y in range(2017, 2026) for d in glob.glob(f"{DATA}/{y}/*")}
    out = []
    for y in range(2017, 2024):
        for p in json.load(open(f"{DATA}/_paperlists/iclr{y}.json")):
            st = p.get("status") or ""
            if st in ("Withdraw", "Withdrawn", "Desk Reject", "") or p["id"] in have: continue
            r = mean_rating(p)
            if r is None: continue
            out.append(dict(year=y, note_id=p["id"], title=p["title"], first_author=(p.get("author") or "").split(";")[0],
                            overall_mean=r, decision=decision_of(st), arxiv_id=None, n_reviews=len((p.get("rating") or "").split(";"))))
    for y in (2024, 2025):
        for l in open(f"{DATA}/_labels_{y}.jsonl"):
            p = json.loads(l)
            if p["id"] in have or p.get("overall_mean") is None: continue
            dr = p.get("decision_raw") or ""
            if not dr or "withdraw" in dr.lower(): continue
            out.append(dict(year=y, note_id=p["id"], title=p["title"], first_author="", overall_mean=p["overall_mean"],
                            decision={"accept": bool(p.get("accept")), "tier": p.get("tier") or ("poster" if p.get("accept") else None), "venue": dr},
                            arxiv_id=p.get("arxiv_id"), n_reviews=len(p.get("reviews") or [])))
    for c in out: c["score"] = max(1, min(10, round(c["overall_mean"])))
    return out


C.UA.clear(); C.UA.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})   # arXiv answers 406 to the old research UA from urllib
OA = "https://api.openalex.org/works?filter=title.search:{}&per-page=3&select=id,title,locations&mailto=user@example.org"
_ARX = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})")



def _curl_http(url, binary=False):
    """arXiv answers 406 to urllib for many e-prints but 200 to curl; route downloads through curl."""
    import subprocess
    r = subprocess.run(["curl", "-s", "-L", "-m", "120", "-A", "Mozilla/5.0", url], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"curl rc={r.returncode}")
    return r.stdout if binary else r.stdout.decode("utf-8", "ignore")
C.http = _curl_http

DC = "https://api.datacite.org/dois?query={}&client-id=arxiv.content&page[size]=5&fields[dois]=doi,titles"
_last_dc = [0.0]


def oa_match(title):
    """DataCite title query restricted to the arXiv client (arXiv DOIs 10.48550/arXiv.<id>); <= 2 req/s; 429 -> back off.
    Returns the arXiv id of the first result whose title matches (Jaccard >= 0.9), else None."""
    import urllib.error
    q = urllib.parse.quote('titles.title:"' + re.sub(r"[^\w\s]", " ", title)[:200] + '"')
    for attempt in range(4):
        wait = 0.5 - (time.time() - _last_dc[0])
        if wait > 0: time.sleep(wait)
        _last_dc[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(DC.format(q), headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.api+json"}), timeout=30) as r:
                j = json.loads(r.read().decode("utf-8", "ignore"))
            for d in j.get("data", []):
                at = d.get("attributes", {})
                ts = [x.get("title", "") for x in at.get("titles", [])]
                if any(C.sim(title, t) >= 0.9 for t in ts):
                    m = re.search(r"10\.48550/arxiv\.(.+)$", at.get("doi", ""), re.I)
                    if m: return m.group(1)
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"DC 429, sleeping {60 * (attempt + 1)}s", file=sys.stderr, flush=True); time.sleep(60 * (attempt + 1)); continue
            return None
        except Exception:
            return None
    return "RATE_LIMITED"


def main():
    need = {int(k): v for k, v in json.loads(sys.argv[1]).items()}   # {"2":127,"3":81,...}
    cands = candidates()
    by = collections.defaultdict(list)
    for c in cands: by[c["score"]].append(c)
    log = open(f"{DATA}/extend_0915{sys.argv[2] if len(sys.argv) > 2 else ''}.log", "a")
    got = collections.Counter(); tried = collections.Counter()
    for s, n_need in sorted(need.items()):
        pool = by[s]; random.shuffle(pool)
        # round-robin over years so no year dominates
        byy = collections.defaultdict(list)
        for c in pool: byy[c["year"]].append(c)
        order = []
        while any(byy.values()):
            for y in sorted(byy):
                if byy[y]: order.append(byy[y].pop())
        for c in order:
            if got[s] >= n_need: break
            tried[s] += 1
            ax = c["arxiv_id"] or oa_match(c["title"])
            if ax == "RATE_LIMITED":
                print(f"[s{s}] ratelimited {c['year']} {c['note_id']}", file=log, flush=True); time.sleep(120); continue      # S2 only; the arXiv search API fallback was too slow (retries + backoff)
            if not ax:
                print(f"[s{s}] nomatch {c['year']} {c['note_id']} :: {c['title'][:60]}", file=log, flush=True); continue
            d = Path(f"{DATA}/{c['year']}/{c['note_id']}")
            time.sleep(4)
            ok = C.download_tex(ax, d / "tex")
            main_ok = ok and any(C.is_main(open(f, errors="ignore").read()) for f in glob.glob(str(d / "tex" / "*.tex")))
            if not main_ok:
                import shutil; shutil.rmtree(d, ignore_errors=True)
                print(f"[s{s}] notex {c['year']} {c['note_id']} arxiv={ax}", file=log, flush=True); continue
            meta = dict(note_id=c["note_id"], year=c["year"], title=c["title"], decision=c["decision"], reviews=[{"n": c["n_reviews"]}] * 0,
                        overall_mean=c["overall_mean"], n_reviews=c["n_reviews"], soundness_mean=None, arxiv_id=ax, tex_ok=True, pdf_ok=False,
                        pdf_source="extend_0915", arxiv_primary_category=None, arxiv_categories=[])
            json.dump(meta, open(d / "meta.json", "w"), indent=1)
            got[s] += 1
            print(f"[s{s}] OK {got[s]}/{n_need} {c['year']} {c['note_id']} arxiv={ax} :: {c['title'][:50]}", file=log, flush=True)
        print(f"==== score {s}: got {got[s]} of {n_need} wanted, tried {tried[s]} ====", file=log, flush=True)
    print("EXTEND_DONE", dict(got), file=log, flush=True)


if __name__ == "__main__":
    main()
