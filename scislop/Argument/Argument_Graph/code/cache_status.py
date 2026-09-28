#!/usr/bin/env python3
"""How much of the label and PMI cache exists for a list of paper records (exit 0 when complete)."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "_common"))
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/artifact-ai2science/_llm")
import importlib.util
_spec = importlib.util.spec_from_file_location("ag_measure", os.path.join(HERE, "measure.py")); AG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AG)
from views import load_doc
import llm as LLM, lm_score as LMS
TPL = open(os.path.join(HERE, "PROMPT_label.txt")).read()
tot = lab = pmi = na = 0
for f in sys.argv[1:]:
    for rec in json.load(open(f)):
        try:
            doc = load_doc(rec); intro = AG.intro_text(doc)
        except Exception:
            na += 1; continue
        if not intro or len(intro.split()) < 80:
            na += 1; continue
        sents = AG.intro_sentences(intro)
        if len(sents) < 4:
            na += 1; continue
        tot += 1
        ok = True
        for i in range(len(sents)):
            p = TPL.replace("{numbered}", AG.numbered(sents)).replace("{i}", str(i + 1)).replace("{target}", sents[i]["text"])
            if not all(os.path.exists(os.path.join(LLM.CACHE_DIR, LLM._key(p, None, f"{AG.LABEL_TAG}#{r}") + ".json")) for r in range(3)):
                ok = False; break
        lab += ok
        texts = [s["text"] for s in sents]
        key = hashlib.sha256((LMS.MODEL + "\n" + "\n\x1e".join(texts)).encode()).hexdigest()
        pmi += os.path.exists(os.path.join(LMS.CACHE_DIR, key + ".json"))
print(json.dumps({"papers": tot, "labels_complete": lab, "pmi_complete": pmi, "not_applicable": na}))
sys.exit(0 if (lab == tot and pmi == tot) else 1)
