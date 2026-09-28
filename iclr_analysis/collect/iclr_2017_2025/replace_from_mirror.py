"""Replacement collection from papercopilot/paperlists (GitHub mirror; no OpenReview needed).
Buckets (year x score x accept) computed from data_retired/. Candidates get strict arXiv
match (>=0.88) + tex+pdf; meta saves ratings-derived reviews + keywords/abstract/primary_area."""
import glob, json, os, re, time, threading, urllib.request
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import socket
socket.setdefaulttimeout(45)
import collect_iclr_dataset as C
C.ARXIV_SLEEP = 3.5   # arXiv ToS: 1 req/3s — overspeed caused silent 503s
_lock = threading.Lock()
FAILS=[0]
FAILED_CACHE="data/_replace_failed.txt"
FAILED=set(open(FAILED_CACHE).read().split()) if os.path.exists(FAILED_CACHE) else set()
_ff=open(FAILED_CACHE,"a")
UA = {"User-Agent": "Mozilla/5.0"}
PL = "data/_paperlists"; os.makedirs(PL, exist_ok=True)
ACCEPT = {"poster", "oral", "spotlight", "talk", "top-5%", "top-25%", "top 5%", "top 25%"}

def tier_of(status):
    s = (status or "").lower()
    if s in ("oral", "talk", "top-5%", "top 5%"): return "oral"
    if s in ("spotlight", "top-25%", "top 25%"): return "spotlight"
    return "none"

def score_of(om): return max(1, min(10, round(om))) if om is not None else None

def load_paperlist(y):
    p = f"{PL}/iclr{y}.json"
    if not os.path.exists(p):
        u = f"https://raw.githubusercontent.com/papercopilot/paperlists/main/iclr/iclr{y}.json"
        open(p, "w").write(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode())
    return json.load(open(p))

# need from retired papers
need = {}
existing = set()
for mp in glob.glob("data/20*/*/meta.json") + glob.glob("data_retired/*/*/meta.json"):
    existing.add(os.path.basename(os.path.dirname(mp)))
for mp in glob.glob("data_retired/*/*/meta.json"):
    m = json.load(open(mp)); y = int(mp.split("/")[1])
    b = (y, score_of(m.get("overall_mean")), bool((m.get("decision") or {}).get("accept")))
    need[b] = need.get(b, 0) + 1
for mp in glob.glob("data/20*/*/meta.json"):   # already-collected replacements reduce need
    m = json.load(open(mp))
    if m.get("replacement_for_texless"):
        y = int(mp.split("/")[1])
        b = (y, score_of(m.get("overall_mean")), bool((m.get("decision") or {}).get("accept")))
        if need.get(b, 0) > 0: need[b] -= 1
need = {b: n for b, n in need.items() if n > 0 and b[1] is not None}
print("need total:", sum(need.values()), flush=True)

def attempt(y, cand):
    ratings = [int(x) for x in re.findall(r"\d+", str(cand.get("rating") or ""))]
    if not ratings: return
    om = round(sum(ratings)/len(ratings), 3)
    st = (cand.get("status") or "").strip()
    if st.lower() in ("withdraw", "desk reject", "workshop", ""): return
    acc = st.lower() in ACCEPT
    b = (y, score_of(om), acc)
    pid = cand["id"]
    with _lock:
        if need.get(b, 0) <= 0 or pid in existing or pid in FAILED: return
        need[b] -= 1; existing.add(pid)
    try:
        title = cand.get("title") or ""
        au = (cand.get("author") or "").split(";")[0] if cand.get("author") else ""
        aid, sim = C.arxiv_match(title, au)
        if not aid or sim < 0.88: raise ValueError("nomatch")
        pdir = Path(f"data/{y}/{pid}"); pdir.mkdir(parents=True, exist_ok=True)
        tex = C.download_tex(aid, pdir/"tex"); pdf = C.download_pdf(aid, pdir/"paper.pdf")
        if not tex:
            for f in pdir.rglob("*"):
                if f.is_file(): f.unlink()
            raise ValueError("notex")
        meta = {"note_id": pid, "year": y, "title": title,
                "decision": {"accept": acc, "tier": tier_of(st), "venue": st},
                "reviews": [{"overall": r, "confidence": None, "soundness": None, "text": ""} for r in ratings],
                "overall_mean": om, "soundness_mean": None, "arxiv_id": aid, "tex_ok": True,
                "pdf_ok": bool(pdf), "pdf_source": "arxiv", "replacement_for_texless": True,
                "source": "papercopilot", "keywords": cand.get("keywords"),
                "abstract": cand.get("abstract"), "primary_area": cand.get("primary_area")}
        json.dump(meta, open(pdir/"meta.json", "w"), ensure_ascii=False, indent=1)
        print(f"[{y}] REPLACED s{b[1]}/{'acc' if acc else 'rej'} arxiv={aid} :: {title[:45]}", flush=True)
    except Exception as e:
        with _lock:
            need[b] = need.get(b, 0) + 1; existing.discard(pid)
            FAILED.add(pid); _ff.write(pid+"\n"); _ff.flush()
            FAILS[0] += 1
            if FAILS[0] % 25 == 0:
                print(f"  ..fails={FAILS[0]} last={type(e).__name__}:{str(e)[:40]}", flush=True)

from concurrent.futures import ThreadPoolExecutor
for y in sorted({b[0] for b in need}):
    ny = sum(n for b, n in need.items() if b[0] == y)
    if ny == 0: continue
    try:
        cands = load_paperlist(y)
    except Exception as e:
        print(f"[{y}] paperlist FAIL {str(e)[:60]}", flush=True); continue
    print(f"[{y}] need {ny}, candidates {len(cands)}", flush=True)
    for _c in cands:
        th = threading.Thread(target=attempt, args=(y, _c), daemon=True)
        th.start(); th.join(300)
        if th.is_alive():
            pid_=_c.get("id")
            print(f"  ..TIMEOUT(300s) abandon {pid_}", flush=True)
            with _lock:
                FAILED.add(pid_); _ff.write(str(pid_)+"\n"); _ff.flush()
    print(f"[{y}] remaining need: {sum(n for b,n in need.items() if b[0]==y)}", flush=True)
print("MIRROR REPLACE DONE; remaining:", sum(need.values()), flush=True)
