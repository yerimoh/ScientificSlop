"""Caption list of every ICLR PDF, with the same reader as the paired census (pdftotext -bbox,
pairs165_locate_hu_figs.find_captions). -> results/figexp/captions/<id>.json
  python3 figexp_captions.py [shard nshards]"""
import json, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/candidte/_census_0914")
import pairs165_locate_hu_figs as L


def pdf_of(r):
    p0 = os.path.join(os.path.dirname(r["root"]), "paper.pdf")
    if os.path.exists(p0):
        return p0
    p1 = f"{R}/results/figexp/pdfs/{r['id']}.pdf"
    return p1 if os.path.exists(p1) else None


def main():
    shard, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    recs = json.load(open(f"{R}/records/iclr_records.json"))["records"]
    out = f"{R}/results/figexp/captions"; os.makedirs(out, exist_ok=True)
    done = 0
    for k, r in enumerate(recs):
        if k % n != shard:
            continue
        dst = f"{out}/{r['id']}.json"
        if os.path.exists(dst):
            continue
        pdf = pdf_of(r)
        if not pdf:
            continue
        try:
            pages = L.words_by_page(pdf)
            caps = L.find_captions(pages)
        except Exception as e:
            json.dump({"id": r["id"], "pdf": pdf, "error": repr(e)[:200]}, open(dst, "w")); continue
        json.dump({"id": r["id"], "pdf": pdf, "n_pages": len(pages),
                   "captions": [{"num": c["num"], "page": c["page"], "text": c["text"]} for c in caps]}, open(dst, "w"))
        done += 1
    print("shard", shard, "done", done, flush=True)


if __name__ == "__main__":
    main()
