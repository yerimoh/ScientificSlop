"""
Pangram v3 async client for the ICLR 2017-2025 corpus — the years ICLR 2026's free dashboard
does not cover.

The API key authenticates but the account has no credits (`402 Insufficient credits`,
re-confirmed 2026-08-05), so this ships with `--dry-run` as the default: it extracts exactly
the text that would be sent, reports word counts and the cost at Pangram's published rates,
and touches the network only when you pass `--go`.

Text is taken from the rendered PDF via the same extractor the fable census uses, so detector
scores and mold scores read identical prose (LaTeX/OCR residue distorts detectors).

  python3 pangram_api.py --years 2023,2024,2025 --per-year 200            # cost preview
  PANGRAM_API_KEY=... python3 pangram_api.py --years 2023,2024,2025 --per-year 200 --go

Results are cached one JSON per paper under data/pangram_api/, keyed by paper id, so a re-run
resumes and never pays twice.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pangram_api")
ICLR_DATA = os.path.abspath(os.path.join(HERE, "..", "ICLR", "data"))
HARNESS = os.path.abspath(os.path.join(HERE, "..", "..", "TEMP", "_harness"))
sys.path.insert(0, HARNESS)
import fable_common as FC  # noqa: E402

ENDPOINT = "https://text.external-api.pangram.com/task"
# published rates, USD per word; bulk discount applied separately
RATES = {"pangram3": 0.05 / 1000, "pangram4": 0.05 / 100}
BULK_DISCOUNT = 0.20


def paper_text(d, max_chars):
    """Prose for one paper dir, preferring the rendered PDF (uniform with the fable census)."""
    pdf = os.path.join(d, "paper.pdf")
    if os.path.exists(pdf):
        try:
            t = FC.extract_prose(pdf, max_pages=99, max_chars=max_chars)
            if t and len(FC.words(t)) >= 150:
                return t
        except Exception:
            pass
    ocr = os.path.join(d, "ocr.md")
    if os.path.exists(ocr):
        return FC.normalize(open(ocr, errors="ignore").read())[:max_chars]
    return None


def submit(text, key):
    req = urllib.request.Request(
        ENDPOINT, method="POST",
        data=json.dumps({"text": text, "public_dashboard_link": True}).encode(),
        headers={"x-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def poll(task_id, key, timeout=600):
    req = urllib.request.Request(f"{ENDPOINT}/{task_id}", headers={"x-api-key": key})
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.load(r)
        stage = body.get("stage") or ""
        if stage == "STAGE_SUCCESS":
            return body
        if "FAIL" in stage or "ERROR" in stage:
            raise RuntimeError(f"{task_id}: {stage}")
        time.sleep(5)
    raise TimeoutError(task_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2023,2024,2025")
    ap.add_argument("--per-year", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=60000)
    ap.add_argument("--go", action="store_true", help="actually call the API (costs money)")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    key = os.environ.get("PANGRAM_API_KEY", "")
    jobs, words = [], 0
    for y in [s.strip() for s in a.years.split(",") if s.strip()]:
        dirs = sorted(os.path.dirname(m) for m in glob.glob(f"{ICLR_DATA}/{y}/*/meta.json"))
        taken = 0
        for d in dirs:
            if taken >= a.per_year:
                break
            pid = f"{y}/{os.path.basename(d)}"
            cache = os.path.join(OUT, pid.replace("/", "__") + ".json")
            if os.path.exists(cache):
                taken += 1
                continue
            t = paper_text(d, a.max_chars)
            if not t:
                continue
            jobs.append((pid, cache, t))
            words += len(FC.words(t))
            taken += 1

    n = len(jobs)
    print(f"{n} papers to score, {words:,} words "
          f"({words/max(n,1):,.0f} avg)")
    for name, rate in RATES.items():
        print(f"  {name}: ${words*rate:,.2f}  (bulk -{BULK_DISCOUNT:.0%}: ${words*rate*(1-BULK_DISCOUNT):,.2f})")
    if not a.go:
        print("\ndry run — pass --go with PANGRAM_API_KEY set to submit")
        return
    if not key:
        sys.exit("PANGRAM_API_KEY is not set")

    for i, (pid, cache, text) in enumerate(jobs, 1):
        try:
            task = submit(text, key)
            res = poll(task["task_id"], key)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"[{i}/{n}] {pid} HTTP {e.code} {body}", flush=True)
            if e.code == 402:
                sys.exit("out of credits — stopping so the remaining papers stay unbilled")
            continue
        except Exception as e:
            print(f"[{i}/{n}] {pid} {type(e).__name__}: {e}", flush=True)
            continue
        res["_paper_id"] = pid
        res["_text_sha1"] = hashlib.sha1(text.encode()).hexdigest()
        res["_n_words"] = len(FC.words(text))
        json.dump(res, open(cache, "w"))
        print(f"[{i}/{n}] {pid} ok", flush=True)


if __name__ == "__main__":
    main()
