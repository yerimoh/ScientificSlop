"""Whole-figure transcription of the ICLR method-figure crops with the item's instrument
(Qwen2.5-VL-32B-Instruct, PROMPT_transcribe.txt, full answer kept, census parser). Resumable.
-> results/figexp/transcripts.jsonl ; marker results/figexp/transcribe.done"""
import json, os, sys, glob, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOP = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop"
sys.path.insert(0, f"{SLOP}/Artifacts/candidte/_census_0914")
import pairs165_retranscribe as RT
from pairs165_transcribe import load
PROMPT = open(f"{SLOP}/Artifacts/fig_exposition/code/PROMPT_transcribe.txt").read().strip()
OUTP = f"{R}/results/figexp/transcripts.jsonl"


def manifest():
    rows = []
    for f in sorted(glob.glob(f"{R}/results/figexp/manifest.s*.json")):
        rows += json.load(open(f))
    return [r for r in rows if r.get("status") == "ok"]


def main():
    while not os.path.exists(f"{R}/results/figexp/crops.done"):
        print("waiting for crops.done", flush=True); time.sleep(120)
    figs = manifest()
    done = set()
    if os.path.exists(OUTP):
        done = {json.loads(l)["key"] for l in open(OUTP)}
    todo = [f for f in figs if f["key"] not in done]
    print(f"figures {len(figs)}, done {len(done)}, todo {len(todo)}", flush=True)
    t0 = time.time()
    with open(OUTP, "a", encoding="utf-8") as fo:
        for k, f in enumerate(todo):
            try:
                answer, ntok = RT.ask_full(load(f["path"]), PROMPT)
                lines, how = RT.parse(answer)
                rec = {"key": f["key"], "id": f["id"], "path": f["path"], "result": {"lines": lines}, "parse": how,
                       "n_tokens": ntok, "hit_cap": ntok >= RT.MAX_NEW, "full_raw": answer}
            except Exception as e:
                rec = {"key": f["key"], "id": f["id"], "path": f["path"], "result": {"error": repr(e)[:200]}}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            if k % 20 == 0:
                print(f"{k}/{len(todo)} {f['key']} {rec.get('parse')} {len(rec['result'].get('lines', []))} lines  {(time.time() - t0) / 60:.0f} min", flush=True)
    open(f"{R}/results/figexp/transcribe.done", "w").write("ok\n")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
