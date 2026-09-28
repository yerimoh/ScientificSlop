"""Fetch the arXiv PDF of every ICLR record that has no paper.pdf next to its tex (the 0915 top-up
records were collected tex-only). Same source and user agent as collect_iclr_dataset.py, so every
ICLR PDF in the figure pipeline is the arXiv version that matches the tex. -> results/figexp/pdfs/<id>.pdf"""
import json, os, sys, time, urllib.request
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (research; ICLR mold dataset)"}
recs = json.load(open(f"{R}/records/iclr_records.json"))["records"]
out = f"{R}/results/figexp/pdfs"; ok = miss = have = 0
for r in recs:
    p0 = os.path.join(os.path.dirname(r["root"]), "paper.pdf")
    if os.path.exists(p0):
        have += 1; continue
    dst = f"{out}/{r['id']}.pdf"
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        ok += 1; continue
    aid = r.get("arxiv_id")
    if not aid:
        miss += 1; continue
    for t in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers=UA), timeout=60).read()
            if b[:5] == b"%PDF-":
                open(dst, "wb").write(b); ok += 1; break
        except Exception as e:
            time.sleep(5 * (t + 1))
    else:
        miss += 1; print("FAIL", r["id"], aid, flush=True)
    time.sleep(1.0)
print("have_local", have, "fetched", ok, "missing", miss)
