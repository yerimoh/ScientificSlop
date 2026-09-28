"""Dump per-sentence Argument_Graph labels for one judge tag (env MODEL_TAG), pure cache reads.
Usage: MODEL_TAG=<tag> python3 dump_labels165.py <out.json>"""
import os, sys, json
ROOT = os.environ.get("SCISLOP_ROOT", ".")
SLOP = f"{ROOT}/paper/draft_v6/slop"
B = f"{ROOT}/paper/draft_v6/scislopbench/bench165"
sys.path.insert(0, f"{SLOP}/_common"); sys.path.insert(0, f"{ROOT}/artifact-ai2science/_llm")
import corpus  # noqa: E402
items = json.load(open(f"{B}/items165.json"))["items"]
ai_codes = {i["pair"] for i in items if i["label"] == 1}
hu_records = []
for i in items:
    if i["label"] != 0: continue
    main = os.path.join(i["tex_dir"], i["tex_root"])
    hu_records.append({"corpus": "HU", "id": i["arxiv"], "root": i["tex_dir"], "main_tex": main,
                       "paper_dir": i["tex_dir"], "exp_dir": None, "diagram": None,
                       "anchor_of": i["pair"], "anchor_year": None, "anchor_venue": i.get("venue")})
_orig = corpus.ai_papers
corpus.ai_papers = lambda: [r for r in _orig() if r["id"] in ai_codes]
corpus.hu_papers = lambda: hu_records
cdir = f"{SLOP}/Argument/Argument_Graph/code"
sys.path.insert(0, cdir); os.chdir(cdir)
import measure as M  # noqa: E402
out = {}
for p in corpus.ai_papers() + corpus.hu_papers():
    iid = ("AI_" if p["corpus"] == "AI" else "HU_") + p["id"]
    try:
        doc = M.load_doc(p); intro = M.intro_text(doc)
        if not intro or len(intro.split()) < 80: continue
        sents = M.intro_sentences(intro)
        if len(sents) < 4: continue
        if M.label_sentences(sents, 3) is None:
            out[iid] = None; continue
        out[iid] = [s["label"] for s in sents]
    except Exception as e:
        out[iid] = f"ERR:{str(e)[:60]}"
json.dump(out, open(sys.argv[1], "w"))
done = sum(1 for v in out.values() if isinstance(v, list))
print(os.environ.get("MODEL_TAG"), "docs labelled:", done, "/", len(out))
