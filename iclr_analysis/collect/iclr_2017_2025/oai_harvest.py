"""Harvest arXiv metadata (id, title, dates, categories) through OAI-PMH for offline title matching.
Sets: cs and stat, from 2016-01-01. Resumable (token + partial output kept). Output: data/_arxiv_oai/<set>.jsonl
Polite: one page (<=1000 records) per request, 503 Retry-After honoured, 2 s between pages.
"""
import os, re, sys, time, json, urllib.request, urllib.error, html
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "_arxiv_oai")
os.makedirs(OUT, exist_ok=True)
BASE = "http://export.arxiv.org/oai2"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
REC = re.compile(r"<record>(.*?)</record>", re.S)


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After", "10")) if e.code in (503, 429) else 30 * (attempt + 1)
            print(f"HTTP {e.code}, waiting {wait}s", flush=True); time.sleep(wait)
        except Exception as ex:
            print("ERR", repr(ex)[:80], flush=True); time.sleep(15)
    raise SystemExit("giving up")


def field(rec, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", rec, re.S)
    return html.unescape(re.sub(r"\s+", " ", m.group(1))).strip() if m else None


def harvest(setname, frm):
    out = f"{OUT}/{setname}.jsonl"; tokf = f"{OUT}/{setname}.token"
    token = open(tokf).read().strip() if os.path.exists(tokf) else None
    n = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    with open(out, "a") as w:
        while True:
            url = f"{BASE}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}" if token else f"{BASE}?verb=ListRecords&metadataPrefix=arXiv&set={setname}&from={frm}"
            xml = get(url)
            for rec in REC.findall(xml):
                if "<header status=\"deleted\"" in rec: continue
                w.write(json.dumps({"id": field(rec, "id"), "title": field(rec, "title"), "created": field(rec, "created"),
                                    "categories": field(rec, "categories")}) + "\n"); n += 1
            w.flush()
            m = re.search(r"<resumptionToken[^>]*>([^<]*)</resumptionToken>", xml)
            token = m.group(1).strip() if m and m.group(1).strip() else None
            print(f"[{setname}] {n} records, token={'yes' if token else 'none'}", flush=True)
            if not token:
                if os.path.exists(tokf): os.remove(tokf)
                break
            open(tokf, "w").write(token); time.sleep(2)


if __name__ == "__main__":
    import urllib.parse
    for s in sys.argv[1:] or ["cs", "stat"]:
        harvest(s, "2016-01-01")
    print("OAI_DONE", flush=True)
