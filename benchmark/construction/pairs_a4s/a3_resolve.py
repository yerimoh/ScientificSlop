"""Step 3 (Agents4Science): resolve reference titles to arXiv ids, then fetch arXiv metadata.

Phase 1  arXiv title search per unique reference title: a quoted-title query, then an all-words
         fallback, accepted only on a title match. OpenAlex was the first choice and 429s this
         host persistently, so the FARS route is used instead; the consequence is that H2 has
         no publication-venue source beyond the arXiv comment / journal_ref.
Phase 2  arXiv id_list batches of 20 for every pool id (H1 needs the v1 year, H2 the comment /
         journal_ref). Seeded from the FARS build's arxiv_meta caches.

Outputs: cache/title2id.json, cache/arxiv_meta_a4s.json
Resumable: every phase skips what is already cached.
"""
import os, re, sys, json, time, urllib.parse, urllib.request, urllib.error
import xml.etree.ElementTree as ET

ROOT = os.environ.get("SCISLOP_ROOT", ".")
DATA = f"{ROOT}/paper/draft_v6/scislopbench/data"
OUTD = f"{DATA}/Agents4Science"
MAIL = "user@example.org"
UA = {"User-Agent": f"scislopbench-a4s/0.1 (mailto:{MAIL})"}
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
OA_SEL = "id,doi,title,publication_year,type,primary_location,locations,cited_by_count"
AXID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})|10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A and B else 0.0


def title_hit(parsed, cand):
    """Accept an exact-ish match, or a parsed title that is a subset of the real one: the
    reference parser truncates at "?" and at venue words, so subset matching is needed."""
    P, C = set(parsed.split()), set(cand.split())
    if not P or not C:
        return False
    if len(P & C) / len(P | C) >= 0.85:
        return True
    return len(P) >= 4 and len(P & C) / len(P) >= 0.92


def get(url, tries=3, base=0.6):
    delay = base
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(delay); delay = min(delay * 2, 90); continue
            return None
        except Exception:
            time.sleep(delay); delay = min(delay * 2, 60)
    return None


def oa_arxiv(w):
    for loc in (w.get("locations") or []) + [w.get("primary_location") or {}]:
        for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
            m = AXID.search(u or "")
            if m:
                return m.group(1) or m.group(2)
    m = AXID.search(w.get("doi") or "")
    return (m.group(1) or m.group(2)) if m else None


def parse_entries(xml):
    out = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out
    for e in root.findall("a:entry", NS):
        m = re.search(r"abs/([\w.\-/]+?)(v\d+)?$", e.findtext("a:id", "", NS))
        if not m:
            continue
        pc = e.find("ar:primary_category", NS)
        out.append((m.group(1), dict(
            arxiv=m.group(1), version=(m.group(2) or ""),
            title=re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip(),
            abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", NS)).strip(),
            published=e.findtext("a:published", "", NS), updated=e.findtext("a:updated", "", NS),
            comment=e.findtext("ar:comment", None, NS), journal_ref=e.findtext("ar:journal_ref", None, NS),
            primary=pc.get("term") if pc is not None else None)))
    return out


def save(obj, path):
    json.dump(obj, open(path, "w"), ensure_ascii=False)


def main():
    t2i_p, oa_p, meta_p = (f"{OUTD}/cache/title2id.json", f"{OUTD}/cache/openalex.json",
                           f"{OUTD}/cache/arxiv_meta_a4s.json")
    t2i = json.load(open(t2i_p)) if os.path.exists(t2i_p) else {}
    oa = json.load(open(oa_p)) if os.path.exists(oa_p) else {}
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    for seed in (f"{DATA}/pairs165_0911/cache/arxiv_meta_165.json", f"{DATA}/cache/arxiv_meta.json"):
        if os.path.exists(seed):
            for k, v in json.load(open(seed)).items():
                meta.setdefault(k, v)

    queue = json.load(open(f"{OUTD}/cache/title_queue.json"))

    # ---- phase 1: arXiv title search, batched ----
    # OpenAlex was the first choice and 429s this host persistently. The arXiv API allows a
    # disjunction of quoted-title phrases in one query, so titles go 8 at a time (pass A) and
    # the leftovers get an all-words query, 4 groups at a time (pass B). 3 s between requests.
    def run_pass(items, build_query, batch, max_results, tag):
        got = 0
        for i in range(0, len(items), batch):
            grp = items[i:i + batch]
            xml = get(f"https://export.arxiv.org/api/query?search_query="
                      f"{urllib.parse.quote(build_query(grp))}&max_results={max_results}", base=3.0)
            ents = parse_entries(xml) if xml else []
            for nt, title in grp:
                hit = None
                for aid, m in ents:
                    if title_hit(nt, norm_title(m["title"])):
                        hit = aid
                        meta.setdefault(aid, m)
                        break
                if hit or nt not in t2i:
                    t2i[nt] = hit
                got += 1 if hit else 0
            if (i // batch) % 10 == 0 or i + batch >= len(items):
                save(t2i, t2i_p); save(meta, meta_p)
                print(f"  {tag}: {min(i + batch, len(items))}/{len(items)} newly resolved={got}", flush=True)
            time.sleep(3.0)
        return got

    todo = [(norm_title(q["title"]), q["title"]) for q in queue if norm_title(q["title"]) not in t2i]
    todo = list({nt: (nt, t) for nt, t in todo}.values())
    print(f"phase 1A (quoted titles, 8 per query): {len(todo)} titles", flush=True)
    run_pass(todo, lambda g: " OR ".join(f'ti:"{t[:180]}"' for _, t in g), 8, 80, "1A")

    left = [(nt, t) for nt, t in todo if not t2i.get(nt)]
    print(f"phase 1B (all-words fallback, 4 per query): {len(left)} titles", flush=True)

    def words_q(g):
        parts = []
        for nt, _ in g:
            ws = [w for w in nt.split() if len(w) > 2][:10]
            if ws:
                parts.append("(" + " AND ".join("ti:" + w for w in ws) + ")")
        return " OR ".join(parts) or 'ti:"zzzzzzzz"'

    run_pass(left, words_q, 4, 60, "1B")

    # ---- phase 2: arXiv metadata for the pool ----
    os.system(f"cd {OUTD} && python3 scripts/a2_pool.py")
    ids = json.load(open(f"{OUTD}/cache/all_ids.json"))
    todo = [i for i in ids if i not in meta]
    print(f"phase 2 (arXiv metadata): {len(todo)} ids", flush=True)
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        xml = get(f"https://export.arxiv.org/api/query?id_list={','.join(batch)}&max_results=20", base=3.0)
        got = dict(parse_entries(xml)) if xml else {}
        for b in batch:
            meta[b] = got.get(b) or {"arxiv": b, "missing": True}
        save(meta, meta_p)
        print(f"  2: {i + len(batch)}/{len(todo)}", flush=True)
        time.sleep(3.0)
    print("done | titles resolved", sum(1 for v in t2i.values() if v), "/", len(t2i),
          "| arxiv meta", len(meta), flush=True)


if __name__ == "__main__":
    main()
