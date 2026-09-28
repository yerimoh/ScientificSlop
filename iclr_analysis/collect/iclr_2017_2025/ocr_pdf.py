"""
ocr_pdf.py — convert collected paper.pdf -> text with **Mistral OCR**
(the tool named in the 0704 meeting: "use Mistral OCR, that one is the best", 50:29-50:45).

Mistral OCR REST API (no SDK needed):
  POST https://api.mistral.ai/v1/ocr
  model = mistral-ocr-latest ; document = data:application/pdf;base64,...
  -> {"pages":[{"index":0,"markdown":"...",...}, ...]}
Auth: env MISTRAL_API_KEY.

Writes per paper:  data/{year}/{note_id}/ocr.md   (concatenated page markdown)
                   data/{year}/{note_id}/ocr.meta.json  (pages, model, chars)
Resumable (skips papers that already have ocr.md). Errors -> data/ocr_errors.log.

Default target = papers WITHOUT arXiv tex (`--which pdf_only`, the 498 that need OCR to be
analyzable); `--which all` OCRs every paper for a uniform text layer.
Run `--dry-run` (no key needed) to see counts + a rough page/cost estimate first.
"""
import argparse
import base64
import glob
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
API = "https://api.mistral.ai/v1/ocr"
MODEL = "mistral-ocr-latest"
ERRLOG = DATA / "ocr_errors.log"


def targets(which):
    out = []
    for mp in sorted(glob.glob(f"{DATA}/20*/*/meta.json")):
        d = Path(mp).parent
        pdf = d / "paper.pdf"
        if not (pdf.exists() and pdf.stat().st_size > 1000):
            continue
        has_tex = bool(glob.glob(f"{d}/tex/*.tex"))
        if which == "pdf_only" and has_tex:
            continue
        out.append(d)
    return out


def ocr_call(pdf_bytes, key, timeout=300):
    b64 = base64.b64encode(pdf_bytes).decode()
    body = json.dumps({
        "model": MODEL,
        "document": {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"},
        "include_image_base64": False,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(dirs, key, sleep):
    ok = fail = 0
    for i, d in enumerate(dirs, 1):
        out = d / "ocr.md"
        if out.exists() and out.stat().st_size > 0:
            continue
        pdf = (d / "paper.pdf").read_bytes()
        try:
            res = ocr_call(pdf, key)
            pages = res.get("pages", [])
            md = "\n\n".join(p.get("markdown", "") for p in pages)
            out.write_text(md)
            (d / "ocr.meta.json").write_text(json.dumps(
                {"model": MODEL, "n_pages": len(pages), "chars": len(md)}, indent=1))
            ok += 1
            print(f"[{i}/{len(dirs)}] OK {d.name} pages={len(pages)} chars={len(md)}")
        except Exception as e:
            fail += 1
            msg = getattr(e, "read", lambda: b"")() if isinstance(e, urllib.error.HTTPError) else b""
            with open(ERRLOG, "a") as f:
                f.write(f"{d}\t{type(e).__name__}: {e}\t{msg[:200]}\n")
            print(f"[{i}/{len(dirs)}] FAIL {d.name} :: {e}")
        time.sleep(sleep)
    print(f"\n=== OCR done: ok={ok} fail={fail} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pdf_only", "all"], default="pdf_only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="count targets + cost estimate; no API calls")
    args = ap.parse_args()

    dirs = targets(args.which)
    todo = [d for d in dirs if not (d / "ocr.md").exists()]
    if args.limit:
        todo = todo[: args.limit]

    if args.dry_run:
        # rough page estimate: ~110KB pdf/page heuristic
        est_pages = sum(max(1, round((d / "paper.pdf").stat().st_size / 110_000)) for d in todo)
        print(f"which={args.which}  eligible={len(dirs)}  to-OCR(no ocr.md yet)={len(todo)}")
        print(f"rough pages ~{est_pages}  (Mistral OCR ~US$1/1000 pages -> ~${est_pages/1000:.2f})")
        print("set MISTRAL_API_KEY and re-run without --dry-run.")
        return

    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("MISTRAL_API_KEY not set. Export it, or run --dry-run for an estimate.")
        return
    print(f"OCR {len(todo)} papers (which={args.which}) via {MODEL}")
    run(todo, key, args.sleep)


if __name__ == "__main__":
    main()
