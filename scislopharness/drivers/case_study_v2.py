"""Case-study candidates, per instance. For each instance in the location list (SLOP_FINDINGS.md), check with per-item rules whether a repair
our harness passed through the gate actually fixed that instance, and judge deterministically how the R3 manuscripts of the four baseline arms (a1_base generic revision, a2_code Claude
Code, a3_review review-based, a4_slop definitions+locations) handled the same instance.

  xsec_ref        is label L referenced from another section; is the referring sentence pointer-only (roadmap) or does it use the object
  macro_redund    did the recycled sentence disappear; if so, did citations/numbers disappear with it at that spot
  citation        did the isolated citing sentence change; did the new sentence add a citation, erase one, or only merge with while/and
  argument_graph  does the cue sentence now precede the claim sentence (position comparison within the introduction)
  evidence_gap    is there an explicit-gap sentence (ACK_RE) / did a new display environment appear (egv2 also records the fabrication audit result)
  fig_exposition  diagram edit record (baselines cannot touch images; only checks whether the diagram was erased)

  python3 case_study_v2.py --out ../final_v14/CASE_STUDY/candidates_v2.json
"""
from __future__ import annotations
import argparse, difflib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
DRAFT = API.parent.parent.parent
EOR = DRAFT / 'Effects_of_revision'
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import harness as H   # noqa: E402

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
ITEMS = ['xsec_ref', 'macro_redund', 'citation', 'argument_graph', 'evidence_gap', 'fig_exposition']
CITE_RE = re.compile(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])?\{([^}]*)\}')
NUM_RE = re.compile(r'(?<![\w.\-])\d+(?:\.\d+)?%?(?![\w.])')
HEAD = {'Cross-section references': 'xsec_ref', 'Macro redundancy': 'macro_redund', 'Argument graph': 'argument_graph',
        'Citation': 'citation', 'Figure exposition': 'fig_exposition', 'Evidence gap': 'evidence_gap'}


def norm(t): return re.sub(r'\s+', ' ', re.sub(r'\\(cite[a-zA-Z]*|ref|cref|Cref|label)\*?\{[^}]*\}', ' ', t or '')).strip().lower()
def cites(t): return set(k.strip() for m in CITE_RE.finditer(t or '') for k in m.group(1).split(','))
def nums(t): return set(NUM_RE.findall(re.sub(r'\\(?:cite[a-zA-Z]*|ref|cref|label)\{[^}]*\}', ' ', t or '')))
def sentences(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', re.sub(r'\s+', ' ', t)) if s.strip()]
def tree_text(tree: Path) -> str: return '\n\n'.join(f.read_text(errors='ignore') for f in H.tex_files(tree))


def parse_findings(p: Path) -> dict:
    out = {}
    if not p.exists(): return out
    cur = None
    for line in p.read_text(errors='ignore').splitlines():
        m = re.match(r'## (.+)', line)
        if m: cur = HEAD.get(m.group(1).strip()); continue
        if cur and line.startswith('- ') and 'instances.' not in line:
            out.setdefault(cur, []).append(line[2:].strip())
    return out


def fuzzy_in(sent: str, text: str, thr: float = 0.85) -> bool:
    s = norm(sent)[:300]
    if not s: return False
    if s in norm(text): return True
    T = norm(text)
    # sliding window over sentences of text
    for cand in sentences(text):
        c = norm(cand)[:300]
        if c and difflib.SequenceMatcher(None, s, c, autojunk=False).ratio() >= thr: return True
    return False


def sentence_with(text: str, needle_re) -> list[str]:
    return [s for s in sentences(text) if needle_re.search(s)]


def pointer_only(s: str) -> bool:
    body = re.sub(r'\\(c|C)?ref\{[^}]*\}|Section|Sec\.|Table|Figure|Fig\.|Eq\.|Equation|Algorithm|Appendix|~|[()]', ' ', s)
    body = re.sub(r'\b(see|shown|presented|described|discussed|provided|detailed|summarized|reported|in|as|is|are|we|the|of|and|for|to|this|these|further|below|above|more|details|results|section|our)\b', ' ', body, flags=re.I)
    return len(re.findall(r'[a-zA-Z]{4,}', body)) <= 3 or bool(re.match(r'^\s*(See|As (shown|described|discussed) in|Details? (are|is) in|We (describe|present|discuss|provide))', s))


def home_of(tree: Path, label: str) -> Path | None:
    for f in H.tex_files(tree):
        if f'\\label{{{label}}}' in f.read_text(errors='ignore'): return f
    return None


def refs_outside_home(tree: Path, label: str) -> list[str]:
    home = home_of(tree, label)
    out = []
    for f in H.tex_files(tree):
        if home is not None and f == home: continue
        t = f.read_text(errors='ignore')
        for s in sentences(t):
            if re.search(r'\\(c|C)?ref\{[^}]*' + re.escape(label) + r'[^}]*\}', s): out.append(s)
    return out


def baseline_scores(arm: str, code: str, item: str):
    p = EOR / 'results' / 'slop' / arm / 'R3' / item / 'papers.jsonl'
    if not p.exists(): return None
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get('id') == code and r.get('corpus', 'AI') == 'AI': return r.get('slop_score')
    return None


def load_changes(gate_dir: Path) -> dict:
    out = {}
    t = (gate_dir / 'CHANGES.md').read_text(errors='ignore') if (gate_dir / 'CHANGES.md').exists() else ''
    for m in re.finditer(r'## Change (\d+)\. (\S+) \((\w+)\)\n\n### original\n\n(.*?)\n\n### revised\n\n(.*?)(?=\n## Change |\n# Context|\Z)', t, re.S):
        out[int(m.group(1))] = {'file': m.group(2), 'op': m.group(3), 'original': m.group(4).strip(), 'revised': m.group(5).strip()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(API / 'runs_haiku_sonnetgate_v14' / 'A-loc'))
    ap.add_argument('--out', default=str(API / 'final_v14' / 'CASE_STUDY' / 'candidates_v2.json'))
    a = ap.parse_args()
    runs = Path(a.runs)
    egv2 = {x['code']: x for x in json.load(open(EOR / 'results' / 'summary' / 'egv2_audit.egv2.json'))}
    cands = {it: [] for it in ITEMS}
    for tp in sorted(runs.glob('*/trajectory.json')):
        code = tp.parent.name; t = json.load(open(tp))
        m0 = H.fars_paper_dir(code); m0_text = tree_text(m0)
        final = tp.parent / f"R{t['rounds'][-1]['round']}"; fin_text = tree_text(final)
        btrees = {arm: (EOR / 'runs' / arm / code / 'R3') for arm in ARMS}
        btext = {arm: (tree_text(p) if p.exists() else '') for arm, p in btrees.items()}
        scores = lambda it: {'ours_R0': t['r0']['scores'].get(it), 'ours_R3': t['rounds'][-1]['scores_after'].get(it),
                             **{f'{arm}_R3': baseline_scores(arm, code, it) for arm in ARMS}}
        for r in t['rounds']:
            g = r.get('gate') or {}
            gate_dir = tp.parent / 'gate' / f"R{r['round']}"
            ch = load_changes(gate_dir)
            findings = parse_findings(tp.parent / f"R{r['round']}" / 'SLOP_FINDINGS.md')
            kept = [(v, ch.get(v['id'], {})) for v in g.get('verdicts', []) if v['verdict'] == 'keep']
            # ---- xsec_ref: label instances
            for inst in findings.get('xsec_ref', []):
                m = re.search(r'\\label\{([^}]+)\}', inst)
                if not m: continue
                label = m.group(1)
                for v, c in kept:
                    o, rv = c.get('original', ''), c.get('revised', '')
                    if re.search(r'\\(c|C)?ref\{[^}]*' + re.escape(label), rv) and not re.search(r'\\(c|C)?ref\{[^}]*' + re.escape(label), o):
                        ours_sents = [s for s in sentences(rv) if label in s]
                        base = {}
                        for arm in ARMS:
                            ss = refs_outside_home(btrees[arm], label) if btrees[arm].exists() else []
                            base[arm] = {'referenced': bool(ss), 'sentences': ss[:3], 'pointer_only': [pointer_only(s) for s in ss[:3]]}
                        cands['xsec_ref'].append({'code': code, 'round': r['round'], 'id': v['id'], 'instance': inst, 'label': label, 'file': c.get('file'),
                                                  'original': o, 'ours': rv, 'ours_sentences': ours_sents, 'ours_pointer_only': [pointer_only(s) for s in ours_sents],
                                                  'reason': v.get('reason'), 'baselines': base, 'scores': scores('xsec_ref')})
            # ---- macro_redund: recycled sentence instances
            for inst in findings.get('macro_redund', []):
                m = re.search(r'"(.+?)"\s*\(repeats', inst)
                if not m: continue
                sent = m.group(1)
                for v, c in kept:
                    o, rv = c.get('original', ''), c.get('revised', '')
                    if fuzzy_in(sent, o) and not fuzzy_in(sent, rv):
                        base = {}
                        for arm in ARMS:
                            bt = btext[arm]
                            still = fuzzy_in(sent, bt) if bt else None
                            # what happened to the paragraph
                            bp, sc = None, 0
                            for p in re.split(r'\n[ \t]*\n', bt):
                                s_ = difflib.SequenceMatcher(None, norm(o)[:500], norm(p)[:500], autojunk=False).ratio()
                                if s_ > sc: bp, sc = p.strip(), s_
                            base[arm] = {'sentence_still_present': still, 'para': bp if sc >= 0.5 else None, 'match': round(sc, 2),
                                         'cite_removed': sorted(cites(o) - cites(bp or '')) if sc >= 0.5 else None,
                                         'numbers_removed': sorted(nums(o) - nums(bp or '')) if sc >= 0.5 else None}
                        cands['macro_redund'].append({'code': code, 'round': r['round'], 'id': v['id'], 'instance': inst, 'sentence': sent, 'file': c.get('file'),
                                                      'original': o, 'ours': rv, 'reason': v.get('reason'),
                                                      'ours_cite_removed': sorted(cites(o) - cites(rv)), 'ours_numbers_removed': sorted(nums(o) - nums(rv)),
                                                      'baselines': base, 'scores': scores('macro_redund')})
            # ---- citation: isolated citing sentence instances
            for inst in findings.get('citation', []):
                m = re.search(r'\]\s*"?(.+?)"?\s*$', inst)
                if not m: continue
                sent = m.group(1).strip().strip('"')
                for v, c in kept:
                    o, rv = c.get('original', ''), c.get('revised', '')
                    if fuzzy_in(sent, o, 0.8) and not fuzzy_in(sent, rv, 0.97):
                        new_s = [s for s in sentences(rv) if difflib.SequenceMatcher(None, norm(sent)[:200], norm(s)[:200], autojunk=False).ratio() > 0.5]
                        base = {}
                        for arm in ARMS:
                            bt = btext[arm]
                            bs = [s for s in sentences(bt) if difflib.SequenceMatcher(None, norm(sent)[:200], norm(s)[:200], autojunk=False).ratio() > 0.5] if bt else []
                            base[arm] = {'sentence_unchanged': fuzzy_in(sent, bt, 0.97) if bt else None, 'sentences': bs[:2],
                                         'cites_added': sorted(set().union(*[cites(s) for s in bs[:2]]) - cites(sent) - cites(o)) if bs else None,
                                         'cites_removed': sorted(cites(sent) - set().union(*[cites(s) for s in bs[:2]])) if bs else None,
                                         'merge_connector': bool(bs and re.search(r'\b(while|whereas|and)\b', bs[0]) and len(cites(bs[0])) >= 2)}
                        cands['citation'].append({'code': code, 'round': r['round'], 'id': v['id'], 'instance': inst, 'sentence': sent, 'file': c.get('file'),
                                                  'original': o, 'ours': rv, 'ours_sentences': new_s[:2], 'reason': v.get('reason'), 'baselines': base, 'scores': scores('citation')})
            # ---- argument_graph: claim/cue order
            for inst in findings.get('argument_graph', []):
                m = re.search(r'"(.+?)" is stated before its strongest cue "(.+?)"', inst)
                if not m: continue
                claim, cue = m.group(1), m.group(2)
                def order(text):
                    T = norm(text); ci, cu = T.find(norm(claim)[:80]), T.find(norm(cue)[:80])
                    if ci < 0 or cu < 0: return None
                    return 'cue_first' if cu < ci else 'claim_first'
                if order(m0_text) == 'claim_first' and order(fin_text) == 'cue_first':
                    hit = [(v, c) for v, c in kept if fuzzy_in(claim, c.get('original', ''), 0.7) or fuzzy_in(cue, c.get('original', ''), 0.7) or fuzzy_in(claim, c.get('revised', ''), 0.7)]
                    v, c = hit[0] if hit else ({}, {})
                    cands['argument_graph'].append({'code': code, 'round': r['round'], 'id': v.get('id'), 'instance': inst, 'claim': claim, 'cue': cue, 'file': c.get('file'),
                                                    'original': c.get('original', ''), 'ours': c.get('revised', ''), 'reason': v.get('reason'),
                                                    'baselines': {arm: {'order': order(btext[arm]) if btext[arm] else None} for arm in ARMS}, 'scores': scores('argument_graph')})
            # ---- evidence_gap
            for v, c in kept:
                rv = c.get('revised', '')
                if H.ACK_RE.search(re.sub(r'\s+', ' ', rv)) and not H.ACK_RE.search(re.sub(r'\s+', ' ', c.get('original', ''))):
                    ack = [s for s in sentences(rv) if H.ACK_RE.search(s)]
                    base = {}
                    for arm in ARMS:
                        bt = btext[arm]
                        base[arm] = {'ack_sentence': bool(bt and H.ACK_RE.search(re.sub(r'\s+', ' ', bt))),
                                     'display_env_added': bool(bt) and bool(re.search(r'\\begin\{(verbatim|lstlisting|minted|tcolorbox|mdframed|quoting)\}', bt)) and not re.search(r'\\begin\{(verbatim|lstlisting|minted|tcolorbox|mdframed|quoting)\}', m0_text)}
                    base['a4s_evidence_gap.egv2'] = egv2.get(code)
                    cands['evidence_gap'].append({'code': code, 'round': r['round'], 'id': v['id'], 'file': c.get('file'), 'original': c.get('original', ''), 'ours': rv,
                                                  'ack_sentences': ack, 'reason': v.get('reason'), 'baselines': base, 'scores': scores('evidence_gap')})
            # ---- fig_exposition
            for v, c in kept:
                if v.get('category') != 'figure': continue
                rep = tp.parent / 'rounds' / f"R{r['round']}" / 'EDIT_REPORT.json'
                rj = json.loads(rep.read_text()) if rep.exists() else {}
                base = {}
                for arm in ARMS:
                    bt = btext[arm]
                    base[arm] = {'figure_kept': ('includegraphics' in bt) if bt else None, 'tikz_added': bool(bt) and ('tikzpicture' in bt) and ('tikzpicture' not in m0_text)}
                cands['fig_exposition'].append({'code': code, 'round': r['round'], 'id': v['id'], 'reason': v.get('reason'), 'erased': [e.get('phrase') for e in (rj.get('erased') or [])],
                                                'preservation': rj.get('preservation'), 'baselines': base, 'scores': scores('fig_exposition'),
                                                'images': {'R0': str(tp.parent / 'rounds' / 'R0' / 'figure.jpeg'), f"R{r['round']}": str(tp.parent / 'rounds' / f"R{r['round']}" / 'figure.jpeg')}})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands, indent=1, ensure_ascii=False))
    for it, l in cands.items(): print(it, len(l))


if __name__ == '__main__':
    main()
