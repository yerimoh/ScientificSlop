"""Retry arXiv tex for tex-less papers with smarter matching:
multiple query forms + relaxed sim (>=0.85, or >=0.72 with publication-year sanity check).
Logs every decision -> data/retry_tex.log; downloads tex on success. Resumable."""
import glob, json, os, re, time, sys, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_iclr_dataset as C
NS={"a":"http://www.w3.org/2005/Atom"}
def norm(s): return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()
def sim(a,b):
    ta,tb=set(norm(a).split()),set(norm(b).split())
    return len(ta&tb)/len(ta|tb) if ta and tb else 0.0
def query(q,n=10):
    try:
        u=f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&max_results={n}"
        return ET.fromstring(C.http(u))
    except Exception: return None
def candidates(root):
    out=[]
    if root is None: return out
    for e in root.findall("a:entry",NS):
        t=(e.findtext("a:title","",NS) or "").strip()
        idr=e.findtext("a:id","",NS) or ""
        m=re.search(r"abs/([^v]+)",idr)
        pub=e.findtext("a:published","",NS)[:4]
        out.append((m.group(1) if m else None, t, int(pub) if pub.isdigit() else None))
    return out
def best_match(title, year):
    cands=[]
    toks=norm(title).split()
    for q in (f'ti:"{title[:200]}"', "all:"+" ".join(w for w in toks if len(w)>2)[:220]):
        cands+=candidates(query(q)); time.sleep(2)
    best=(None,0,None)
    for aid,t,py in cands:
        s=sim(title,t)
        if s>best[1]: best=(aid,s,py)
    aid,s,py=best
    if aid and (s>=0.85 or (s>=0.72 and py and year-2<=py<=year+1)):
        return aid,s
    return None,s
def one(args):
    d,m=args
    title=m.get("title") or ""; year=int(d.split("/")[1])
    aid,sc=best_match(title,year)
    ok=False
    if aid:
        ok=C.download_tex(aid,Path(d)/"tex")
        if ok:
            m["arxiv_id"]=aid; m["tex_ok"]=True
            json.dump(m,open(f"{d}/meta.json","w"),ensure_ascii=False,indent=1)
    return d,sc,aid,ok,title

def main():
    todo=[]
    for mp in sorted(glob.glob("data/20*/*/meta.json")):
        d=os.path.dirname(mp)
        if glob.glob(f"{d}/tex/*.tex"): continue
        m=json.load(open(mp)); todo.append((d,m))
    print(f"tex-less: {len(todo)}",flush=True)
    got=0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i,(d,sc,aid,ok,title) in enumerate(ex.map(one,todo),1):
            got+=ok
            print(f"[{i}/{len(todo)}] sim={sc:.2f} arxiv={aid} tex={ok} :: {title[:50]}",flush=True)
    print(f"RECOVERED {got}/{len(todo)}",flush=True)
main()
