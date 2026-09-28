"""Step 3: resolve arXiv metadata.

Phase A: id_list batches of 20 for every pool id (seeded from ../cache/arxiv_meta.json).
Phase B: title search for cited bib entries with no arXiv id (cache/title_queue.json).
3 s between requests, exponential backoff on failure, incremental saves every 10 requests.

Outputs: cache/arxiv_meta_165.json  {id: {title, abstract, published, updated, comment,
                                         journal_ref, primary, version}}
         cache/title2id.json        {normalized title: id or null}
"""
import os, re, sys, json, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/pairs165_0911"
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
API = "https://export.arxiv.org/api/query"
SLEEP = 3.0


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def fetch(url, tries=5):
    delay = SLEEP
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return r.read()
        except Exception as e:
            print("  retry", i + 1, type(e).__name__, file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def parse_entries(xml):
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for e in root.findall("a:entry", NS):
        idu = e.findtext("a:id", "", NS)
        m = re.search(r"abs/([\w.\-/]+?)(v\d+)?$", idu)
        if not m:
            continue
        aid = m.group(1)
        pc = e.find("ar:primary_category", NS)
        out.append((aid, dict(
            arxiv=aid, version=(m.group(2) or ""),
            title=re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip(),
            abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", NS)).strip(),
            published=e.findtext("a:published", "", NS), updated=e.findtext("a:updated", "", NS),
            comment=e.findtext("ar:comment", None, NS), journal_ref=e.findtext("ar:journal_ref", None, NS),
            primary=pc.get("term") if pc is not None else None)))
    return out


def main():
    meta_p = f"{OUTD}/cache/arxiv_meta_165.json"
    t2i_p = f"{OUTD}/cache/title2id.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    seed_p = f"{DATA}/cache/arxiv_meta.json"
    if os.path.exists(seed_p):
        for k, v in json.load(open(seed_p)).items():
            meta.setdefault(k, v)
    t2i = json.load(open(t2i_p)) if os.path.exists(t2i_p) else {}

    ids = json.load(open(f"{OUTD}/cache/all_ids.json"))
    todo = [i for i in ids if i not in meta]
    print(f"phase A: {len(todo)} ids to fetch")
    nreq = 0
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        xml = fetch(f"{API}?id_list={','.join(batch)}&max_results=20")
        nreq += 1
        if xml:
            got = dict(parse_entries(xml))
            for b in batch:
                meta[b] = got.get(b) or {"arxiv": b, "missing": True}
        if nreq % 10 == 0 or i + 20 >= len(todo):
            json.dump(meta, open(meta_p, "w"), ensure_ascii=False)
            print(f"  A {i + len(batch)}/{len(todo)} saved")
        time.sleep(SLEEP)

    queue = json.load(open(f"{OUTD}/cache/title_queue.json"))
    todo = [q for q in queue if norm_title(q["title"]) not in t2i]
    print(f"phase B: {len(todo)} title lookups")
    for n, q in enumerate(todo):
        t = q["title"]
        nt = norm_title(t)
        # quoted-phrase title search; fall back to all-words query
        quoted = urllib.parse.quote(f'ti:"{t}"')
        xml = fetch(f"{API}?search_query={quoted}&max_results=5")
        best = None
        if xml:
            for aid, m in parse_entries(xml):
                if norm_title(m["title"]) == nt:
                    best = aid
                    meta.setdefault(aid, m)
                    break
        if best is None and xml is not None:
            words = [w for w in re.findall(r"[a-z0-9]+", nt) if len(w) > 2][:12]
            time.sleep(SLEEP)
            xml = fetch(f"{API}?search_query={urllib.parse.quote(' AND '.join('ti:' + w for w in words))}&max_results=5")
            if xml:
                for aid, m in parse_entries(xml):
                    a, b = set(norm_title(m["title"]).split()), set(nt.split())
                    if a and b and len(a & b) / len(a | b) >= 0.85:
                        best = aid
                        meta.setdefault(aid, m)
                        break
        t2i[nt] = best
        if (n + 1) % 10 == 0 or n + 1 == len(todo):
            json.dump(t2i, open(t2i_p, "w"), ensure_ascii=False)
            json.dump(meta, open(meta_p, "w"), ensure_ascii=False)
            print(f"  B {n + 1}/{len(todo)} resolved={sum(1 for v in t2i.values() if v)}")
        time.sleep(SLEEP)

    json.dump(meta, open(meta_p, "w"), ensure_ascii=False)
    json.dump(t2i, open(t2i_p, "w"), ensure_ascii=False)
    print("done. meta:", len(meta), "resolved titles:", sum(1 for v in t2i.values() if v), "/", len(t2i))


if __name__ == "__main__":
    main()
