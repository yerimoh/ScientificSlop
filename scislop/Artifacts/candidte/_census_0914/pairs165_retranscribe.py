"""Re-transcribe every figure whose first transcription did not parse into a list of lines.

vision_calls.ask() keeps only the first 200 characters when the model's answer is not a JSON object,
and Qwen2.5-VL-32B often answers with a fenced bare list, so those figures were read only in part.
This script sends the same prompt with a larger token budget, keeps the full answer, and parses fenced
objects, fenced lists, truncated lists (quoted strings) and plain lines. Output:
pairs165_full/transcripts_fix.jsonl (resumable), which pairs165_G.py overlays on transcripts.jsonl.
A marker file pairs165_full/retranscribe.done is written when every target is covered.
"""
import json, os, re, sys
import torch
sys.path.insert(0, os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/slop/Artifacts/fig_graph/roi_extract")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_calls as V
from pairs165_transcribe import PROMPT, load, OUT

MAX_NEW = 1500


def ask_full(img, prompt):
    m, proc = V.model()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[:, inp.input_ids.shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0], int(gen.shape[1])


def _unquote(x):
    try:
        return json.loads('"' + x + '"')
    except Exception:
        return x


def parse(s):
    t = re.sub(r"^\s*```[a-zA-Z]*\s*", "", s)
    t = re.sub(r"\s*```\s*$", "", t)
    pairs = sorted([("{", "}"), ("[", "]")], key=lambda ab: (t.find(ab[0]) if t.find(ab[0]) >= 0 else 10 ** 9))
    for a, b in pairs:
        i, j = t.find(a), t.rfind(b)
        if i >= 0 and j > i:
            try:
                obj = json.loads(t[i:j + 1])
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
                return [str(x) for x in obj["lines"]], "json_object"
            if isinstance(obj, list):
                return [str(x) for x in obj], "json_list"
    quoted = [_unquote(x) for x in re.findall(r'"((?:[^"\\]|\\.)*)"', t) if x != "lines"]
    if quoted:
        return quoted, "quoted_strings"
    return [x.strip() for x in t.splitlines() if x.strip()], "plain_lines"


def targets_and_paths():
    last, paths = {}, {}
    for l in open(f"{OUT}/transcripts.jsonl"):
        j = json.loads(l)
        last[j["key"]] = j["result"]
        paths[j["key"]] = j["path"]
    tg = [k for k, r in last.items() if not (isinstance(r, dict) and isinstance(r.get("lines"), list) and r["lines"])]
    return sorted(tg), paths


def main():
    targets, paths = targets_and_paths()
    outp = f"{OUT}/transcripts_fix.jsonl"
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["key"] for l in open(outp)}
    print(f"targets {len(targets)}, already done {len(done & set(targets))}", flush=True)
    with open(outp, "a") as fo:
        for k in targets:
            if k in done:
                continue
            s, ntok = ask_full(load(paths[k]), PROMPT)
            lines, how = parse(s)
            fo.write(json.dumps({"key": k, "path": paths[k], "result": {"lines": lines}, "parse": how,
                                 "n_tokens": ntok, "hit_cap": ntok >= MAX_NEW, "full_raw": s}, ensure_ascii=False) + "\n")
            fo.flush()
            print(k, how, len(lines), ntok, flush=True)
    done = {json.loads(l)["key"] for l in open(outp)}
    if set(targets) <= done:
        open(f"{OUT}/retranscribe.done", "w").write("ok\n")
        print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
