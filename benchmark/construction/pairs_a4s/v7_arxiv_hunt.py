"""Do the A4S submissions also exist on arXiv? If so, arXiv gives the real LaTeX e-print and the
both-sides-LaTeX ceiling stops being 15.

Quoted-title search, eight titles per query, 3 s apart, accepted only on a title match.
Output: cache/a4s_arxiv_hits.json
"""
import json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

OUTD = os.environ.get("SCISLOP_ROOT", ".") + "/paper/draft_v6/scislopbench/data/Agents4Science"
NS = {"a": "http://www.w3.org/2005/Atom"}


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def hit(parsed, cand):
    P, C = set(parsed.split()), set(cand.split())
    if not P or not C:
        return False
    return len(P & C) / len(P | C) >= 0.8 or (len(P) >= 5 and len(P & C) / len(P) >= 0.9)


def fetch(url, tries=3):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "scislopbench-a4s/0.2 (user@example.org)"}), timeout=40).read()
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def entries(xml):
    out = []
    try:
        root = ET.fromstring(xml)
    except Exception:
        return out
    for e in root.findall("a:entry", NS):
        m = re.search(r"abs/([\w.\-/]+?)(v\d+)?$", e.findtext("a:id", "", NS))
        if m:
            out.append((m.group(1), re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip(),
                        e.findtext("a:published", "", NS)))
    return out


def main():
    idx = json.load(open(f"{OUTD}/cache/a4s_index.json"))
    out_p = f"{OUTD}/cache/a4s_arxiv_hits.json"
    hits = json.load(open(out_p)) if os.path.exists(out_p) else {}
    todo = [(c, idx[c]["title"]) for c in sorted(idx) if c not in hits]
    print(f"{len(todo)} papers to look up", flush=True)
    B = 8
    for i in range(0, len(todo), B):
        grp = todo[i:i + B]
        q = " OR ".join(f'ti:"{t[:170]}"' for _, t in grp)
        xml = fetch(f"https://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&max_results=80")
        ents = entries(xml) if xml else []
        for code, t in grp:
            nt = norm(t)
            found = None
            for aid, at, pub in ents:
                if hit(nt, norm(at)):
                    found = dict(arxiv=aid, arxiv_title=at, published=pub)
                    break
            hits[code] = found
        json.dump(hits, open(out_p, "w"), ensure_ascii=False, indent=1)
        print(f"  {min(i + B, len(todo))}/{len(todo)} cumulative hits {sum(1 for v in hits.values() if v)}", flush=True)
        time.sleep(3)
    print("Done. Found on arXiv:", sum(1 for v in hits.values() if v), "/", len(hits))


if __name__ == "__main__":
    main()
