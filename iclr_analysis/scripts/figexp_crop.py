"""Crop the picked method figure of every ICLR paper from its PDF, with the census crop rule
(pairs165_locate_hu_figs.crop_region / render: caption column, upward to the first body paragraph,
full-width fallback, 150 dpi). -> results/figexp/crops/<id>.png, results/figexp/manifest.json
  python3 figexp_crop.py [shard nshards]"""
import json, os, sys, glob
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/candidte/_census_0914")
import pairs165_locate_hu_figs as L
OUT = f"{R}/results/figexp/crops"; os.makedirs(OUT, exist_ok=True)


def main():
    shard, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    picks = json.load(open(f"{R}/results/figexp/picks.json"))
    rows = []
    for k, (pid, p) in enumerate(sorted(picks.items())):
        if k % n != shard:
            continue
        rec = {"id": pid, "key": f"ICLR_{pid}", "pick": p["pick"], "n_captions": p["n_captions"]}
        if p["pick"] in (None, "unanswered", "error"):
            rec["status"] = "no_method_figure" if p["pick"] is None else p["pick"]; rows.append(rec); continue
        cap = json.load(open(f"{R}/results/figexp/captions/{pid}.json"))
        pdf = cap["pdf"]; out = f"{OUT}/{pid}.png"
        if os.path.exists(out) and os.path.exists(out + ".json"):
            rows.append(json.load(open(out + ".json"))); continue
        try:
            pages = L.words_by_page(pdf)
            caps = L.find_captions(pages)
            c = next((c for c in caps if c["num"] == p["pick"]), None)
            if c is None:
                raise RuntimeError("caption_not_found")
            region, vtext = L.crop_region(pdf, pages, c)
            L.render(pdf, c["page"], region, out)
            from PIL import Image
            im = Image.open(out)
            rec.update({"status": "ok", "path": out, "page": c["page"] + 1, "fig_num": c["num"], "caption": c["text"][:200],
                        "region_pt": region, "crop_size": list(im.size), "n_pdf_words": len(vtext.split())})
        except Exception as e:
            rec["status"] = f"error: {e}"[:120]
        json.dump(rec, open(out + ".json", "w"))
        rows.append(rec)
    json.dump(rows, open(f"{R}/results/figexp/manifest.s{shard}.json", "w"), indent=0)
    import collections
    print("shard", shard, collections.Counter(r["status"].split(":")[0] for r in rows))


if __name__ == "__main__":
    main()
