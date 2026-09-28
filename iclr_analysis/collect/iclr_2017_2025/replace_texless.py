"""Replace still-tex-less papers with same-bucket (year x score x accept) papers that DO have
arXiv tex+pdf. 2017-23 candidates from OpenReview v1; strict match (>=0.88). Additive only
(old papers retired later by the orchestrator). Resumable via replacement markers."""
import glob, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_iclr_dataset as C
C.ARXIV_SLEEP=1.5
import threading
_lock=threading.Lock()
def score_of(om): return max(1,min(10,round(om))) if om is not None else None
# 1) need per bucket
need={}
existing=set()
for mp in glob.glob("data/20*/*/meta.json"):
    d=os.path.dirname(mp); pid=d.split("/")[-1]; y=int(d.split("/")[1]); existing.add(pid)
    m=json.load(open(mp))
    if m.get("replacement_for_texless"): continue
    if glob.glob(f"{d}/tex/*.tex"): continue
    b=(y,score_of(m.get("overall_mean")),bool((m.get("decision") or {}).get("accept")))
    need[b]=need.get(b,0)+1
# subtract replacements already collected
for mp in glob.glob("data/20*/*/meta.json"):
    m=json.load(open(mp))
    if m.get("replacement_for_texless"):
        d=os.path.dirname(mp); y=int(d.split("/")[1])
        b=(y,score_of(m.get("overall_mean")),bool((m.get("decision") or {}).get("accept")))
        if need.get(b,0)>0: need[b]-=1
need={b:n for b,n in need.items() if n>0}
print("replacement need (buckets):",sum(need.values()),flush=True)
for y in sorted({b[0] for b in need}):
    if y>=2024: continue   # 24/25 tex-less is negligible
    ny=sum(n for b,n in need.items() if b[0]==y)
    if ny==0: continue
    print(f"[{y}] need {ny}; fetching candidates...",flush=True)
    subs=None
    for att in range(12):
        try:
            subs=C.fetch_submissions(y,2500); break
        except Exception as e:
            print(f"[{y}] fetch retry{att+1}: {str(e)[:60]}",flush=True); time.sleep(1800)  # 403 rate-ban: wait 30min
    if subs is None:
        print(f"[{y}] fetch FAILED, skip year",flush=True); continue
    subs=[s for s in subs if s["overall_mean"] is not None and s["decision"]["accept"] is not None
          and s["note_id"] not in existing]
    def attempt(s):
        b=(y,score_of(s["overall_mean"]),bool(s["decision"]["accept"]))
        with _lock:
            if need.get(b,0)<=0 or s["note_id"] in existing: return
            need[b]-=1; existing.add(s["note_id"])   # reserve
        try:
            title=s.get("title") or ""; au=(s.get("authors") or [""])[0] if isinstance(s.get("authors"),list) else ""
            aid,sim=C.arxiv_match(title,au)
            if not aid or sim<0.88: raise ValueError("nomatch")
            pdir=Path(f"data/{y}/{s['note_id']}"); pdir.mkdir(parents=True,exist_ok=True)
            tex=C.download_tex(aid,pdir/"tex"); pdf=C.download_pdf(aid,pdir/"paper.pdf")
            if not tex:
                for f in pdir.rglob("*"):
                    if f.is_file(): f.unlink()
                raise ValueError("notex")
            meta={"note_id":s["note_id"],"year":y,"title":title,"decision":s["decision"],
                  "reviews":s["reviews"],"overall_mean":s["overall_mean"],
                  "soundness_mean":s.get("soundness_mean"),"arxiv_id":aid,"tex_ok":True,
                  "pdf_ok":bool(pdf),"pdf_source":"arxiv","replacement_for_texless":True,
                  "keywords":s.get("keywords"),"abstract":s.get("abstract")}
            json.dump(meta,open(pdir/"meta.json","w"),ensure_ascii=False,indent=1)
            print(f"[{y}] REPLACED s{b[1]}/{'acc' if b[2] else 'rej'} arxiv={aid} :: {title[:45]}",flush=True)
        except Exception:
            with _lock: need[b]=need.get(b,0)+1; existing.discard(s["note_id"])  # release
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(attempt, subs[:1500]))
print("REPLACE DONE; remaining need:",sum(need.values()),flush=True)
