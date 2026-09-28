"""Failing-unit lists for the slop-aware arms, built by running the four checkers on the
current round tree and reading their unit files. Output is text for the prompt plus a JSON
record for the log."""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ITEMS, SLOP, hu_record  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = sys.executable


def measure_dir(code: str, tree: Path, out: Path, items=ITEMS):
    """Run the checkers for one paper tree into out/<item>/."""
    for it in items:
        o = out / it
        if o.exists():
            shutil.rmtree(o)
        r = subprocess.run([PY, str(HERE / 'measure_tree.py'), '--item', it, '--out', str(o),
                            '--ai', f'{code}={tree}'], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            (out / f'{it}.err').write_text(r.stdout[-3000:] + '\n' + r.stderr[-3000:])


def _rows(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _sec_title(tree: Path, code: str, idx):
    """Title of top-level body section idx from the checker's own loader."""
    try:
        sys.path.insert(0, str(SLOP / '_common'))
        from views import load_doc
        from common import ai_record
        doc = load_doc(ai_record(code, tree))
        tops = [s for s in doc.sections if s.level == 1 and not s.in_appendix]
        return tops[idx].title if 0 <= idx < len(tops) else f'section {idx}'
    except Exception:
        return f'section {idx}'


def failing_units(code: str, tree: Path, out: Path, items=ITEMS) -> dict:
    """Returns {item: [unit strings]} and per-item scores; empty list when the item passes."""
    tree = Path(tree).resolve(); out = Path(out).resolve()
    measure_dir(code, tree, out, items)
    units, scores = {}, {}
    # macro_redund: recycled sentences
    if 'macro_redund' in items:
        rows = _rows(out / 'macro_redund' / 'papers.jsonl')
        scores['macro_redund'] = rows[0].get('slop_score') if rows else None
        ss = [s for s in _rows(out / 'macro_redund' / 'sentences.jsonl') if s.get('recycled')]
        units['macro_redund'] = [f'[{s["section"]}] "{s["text"][:260]}" (repeats the {s["source_section"]})' for s in ss]
    if 'xsec_ref' in items:
        rows = _rows(out / 'xsec_ref' / 'papers.jsonl')
        scores['xsec_ref'] = rows[0].get('slop_score') if rows else None
        objs = [o for o in _rows(out / 'xsec_ref' / 'objects.jsonl') if not o.get('reused')]
        titles = {}
        us = []
        for o in objs:
            hs = o.get('home_section')
            if hs not in titles:
                titles[hs] = _sec_title(tree, code, hs)
            if o['kind'] == 'section':
                us.append(f'section "{o["title"]}" is never referred to from any other section')
            else:
                us.append(f'{o["kind"]} \\label{{{o["title"]}}} (in "{titles[hs]}") is never referred to from any other section')
        units['xsec_ref'] = us
    if 'citation' in items:
        rows = _rows(out / 'citation' / 'papers.jsonl')
        scores['citation'] = rows[0].get('slop_score') if rows else None
        units['citation'] = isolated_sentences(code, tree)
    if 'evidence_gap' in items:
        rows = _rows(out / 'evidence_gap' / 'papers.jsonl')
        r = rows[0] if rows else {}
        scores['evidence_gap'] = r.get('slop_score')
        if r.get('applicable') and r.get('slop_score') == 1.0:
            units['evidence_gap'] = ['The paper has result tables but displays no concrete instance anywhere '
                                     '(no example input/output, case, failure, worked example, or quoted specimen).']
        else:
            units['evidence_gap'] = []
    return {'units': units, 'scores': scores}


def isolated_sentences(code: str, tree: Path) -> list[str]:
    """Citing sentences of Intro/Related with none of the weaving tags, from the citation
    checker's own paragraph and sentence functions (v4 logic, unchanged)."""
    import importlib.util
    cdir = SLOP / 'Argument' / 'citation' / 'code'
    sys.path.insert(0, str(SLOP / '_common')); sys.path.insert(0, str(cdir))
    import corpus
    from common import ai_record
    corpus.ai_papers = lambda: []; corpus.hu_papers = lambda: []
    spec = importlib.util.spec_from_file_location('measure_citation_units', str(cdir / 'measure.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    from views import load_doc
    doc = load_doc(ai_record(code, tree))
    out = []
    for par in m.paragraphs_of(doc):
        sents = m.sentences_raw(par['text'])
        tags = m.tag_paragraph(par['text'])
        # recompute per-sentence rows exactly as tag_paragraph does (it returns only counts)
        rows = _tag_rows(m, par['text'])
        for s, r in zip(sents, rows):
            if r['works'] and not (set(r['tags']) & {'BUNDLE', 'MULTI', 'REL'}):
                clean = re.sub(r'\s+', ' ', s).strip()
                out.append(f'[{par["section"]}] "{clean[:300]}"')
    return out


def _tag_rows(m, par: str):
    import itertools
    sents = m.sentences_raw(par)
    pre = [(s, m.keys_of(s)) for s in sents]
    pre = [(s, g, set(itertools.chain.from_iterable(g))) for s, g in pre]
    rows, seen = [], set()
    for i, (s, groups, works) in enumerate(pre):
        body = m.CITE_RE.sub(' ', s)
        cue, rel1 = bool(m.CUE_RE.search(body)), bool(m.REL1_RE.search(body))
        selfr, group, anaph = bool(m.SELF_RE.search(body)), bool(m.GROUP_RE.search(body)), bool(m.ANAPH_RE.search(body))
        next_cited = i + 1 < len(pre) and bool(pre[i + 1][1])
        tags = set()
        if any(len(set(g)) >= 2 for g in groups): tags.add('BUNDLE')
        if len(groups) >= 2 and len(works) >= 2: tags.add('MULTI')
        if len(works) >= 2 and cue: tags.add('REL')
        if len(works) == 1 and (rel1 or m.REL1_INIT_RE.search(body) or (group and len(seen) >= 2)): tags.add('REL1')
        if selfr and (cue or anaph or (groups and m.USE_RE.search(body))) and (groups or seen or next_cited): tags.add('SELF')
        rows.append({'works': sorted(works), 'tags': sorted(tags)})
        seen |= works
    return rows


if __name__ == '__main__':
    code, tree = sys.argv[1], Path(sys.argv[2])
    r = failing_units(code, tree, Path('/tmp/eor_units_test'))
    print(json.dumps(r['scores']))
    for k, v in r['units'].items():
        print(k, len(v)); [print('  ', u[:160]) for u in v[:4]]
