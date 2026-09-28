"""Record keywords + domain for EVERY paper.
- arxiv_primary_category / arxiv_categories: arXiv API id_list batches (uniform domain label).
- keywords: keep if present; backfill <=2023 via OpenReview v1 /notes?id=;
  2025 via HF ai-conferences/ICLR2025 (keywords + primary_area); 2024 via HF ICLR2024/ICLR2024-papers if usable.
Resumable: skips fields already filled."""
import glob, json, os, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
NS={"a":"http://www.w3.org/2005/Atom","ar":"http://arxiv.org/schemas/atom"}
UA={"User-Agent":"Mozilla/5.0"}
def http(u):
    return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read().decode("utf-8","ignore")
metas={}
for mp in sorted(glob.glob("data/20*/*/meta.json")):
    metas[mp]=json.load(open(mp))
# ---- pass 1: arXiv categories in batches of 80 ----
need=[(mp,m["arxiv_id"]) for mp,m in metas.items() if m.get("arxiv_id") and not m.get("arxiv_primary_category")]
print("arxiv-domain to fill:",len(need),flush=True)
for i in range(0,len(need),80):
    batch=need[i:i+80]
    ids=",".join(a for _,a in batch)
    try:
        root=ET.fromstring(http(f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(ids)}&max_results=100"))
    except Exception as e:
        print("batch err",str(e)[:60],flush=True); time.sleep(5); continue
    bycid={}
    for e in root.findall("a:entry",NS):
        idr=e.findtext("a:id","",NS); m=re.search(r"abs/([^v]+)",idr)
        if not m: continue
        pc=e.find("ar:primary_category",NS)
        cats=[c.get("term") for c in e.findall("a:category",NS)]
        bycid[m.group(1)]={"primary":(pc.get("term") if pc is not None else (cats[0] if cats else None)),"cats":cats}
    for mp,aid in batch:
        info=bycid.get(aid) or bycid.get(aid.split("v")[0])
        if info:
            metas[mp]["arxiv_primary_category"]=info["primary"]
            metas[mp]["arxiv_categories"]=info["cats"]
            json.dump(metas[mp],open(mp,"w"),ensure_ascii=False,indent=1)
    print(f"  arxiv batch {i//80+1}/{(len(need)+79)//80}",flush=True)
    time.sleep(3.2)
# ---- pass 2a: keywords backfill <=2023 via OpenReview v1 ----
need_kw=[(mp,m) for mp,m in metas.items() if not m.get("keywords") and int(mp.split("/")[1])<=2023]
print("keywords backfill <=2023:",len(need_kw),flush=True)
for j,(mp,m) in enumerate(need_kw,1):
    nid=m.get("note_id") or os.path.basename(os.path.dirname(mp))
    try:
        d=json.loads(http(f"https://api.openreview.net/notes?id={nid}"))
        kw=(d.get("notes") or [{}])[0].get("content",{}).get("keywords")
        if kw:
            m["keywords"]=kw; json.dump(m,open(mp,"w"),ensure_ascii=False,indent=1)
    except Exception: pass
    if j%50==0: print(f"  or-kw {j}/{len(need_kw)}",flush=True)
    time.sleep(0.7)
# ---- pass 2b: 2024/25 keywords + primary_area via HF ----
try:
    from datasets import load_dataset
    kmap={}
    for r in load_dataset("ai-conferences/ICLR2025",split="train"):
        kmap[("2025",r["paper_id"])]={"keywords":r.get("keywords"),"primary_area":r.get("primary_area")}
    try:
        for r in load_dataset("ICLR2024/ICLR2024-papers",split="train"):
            pid=r.get("paper_id") or r.get("id")
            if pid: kmap[("2024",pid)]={"keywords":r.get("keywords"),"primary_area":r.get("primary_area")}
    except Exception as e: print("2024 HF kw unavailable:",str(e)[:60],flush=True)
    n=0
    for mp,m in metas.items():
        y=mp.split("/")[1]
        if y not in ("2024","2025"): continue
        info=kmap.get((y,m.get("note_id") or os.path.basename(os.path.dirname(mp))))
        if not info: continue
        ch=False
        if not m.get("keywords") and info.get("keywords"): m["keywords"]=info["keywords"]; ch=True
        if not m.get("primary_area") and info.get("primary_area"): m["primary_area"]=info["primary_area"]; ch=True
        if ch: json.dump(m,open(mp,"w"),ensure_ascii=False,indent=1); n+=1
    print("HF kw/area filled:",n,flush=True)
except Exception as e:
    print("HF pass skipped:",str(e)[:70],flush=True)
# summary
import collections
c=collections.Counter()
for mp in glob.glob("data/20*/*/meta.json"):
    m=json.load(open(mp))
    c["kw"]+=bool(m.get("keywords")); c["dom"]+=bool(m.get("arxiv_primary_category")); c["n"]+=1
print(f"DONE: keywords {c['kw']}/{c['n']}, arxiv-domain {c['dom']}/{c['n']}",flush=True)
