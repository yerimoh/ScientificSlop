"""Case-study candidate extraction. Per item, collect the repairs our harness kept (keep) and find the same paragraph in the baseline
(a1_base generic revision, a2_code Claude Code, a3_review review-based, a4_slop definitions+locations only) R3 trees, placing them side by side.
How the baseline paragraph differs from the original is marked with deterministic flags (citation removed, numbers removed/added, pointer-only sentence, unchanged).
Attempts our editor produced in the same round but the gate reverted are included too (editor without a gate = one more baseline).

  python3 case_study_extract.py --out ../final_v14/CASE_STUDY/candidates.json
"""
from __future__ import annotations
import argparse, difflib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
API = HERE.parent
ROOT = API.parent.parent.parent          # paper/draft_v6
EOR = ROOT / 'Effects_of_revision' / 'runs'
sys.path.insert(0, str(API.parent / 'temp' / 'code')); sys.path.insert(0, str(HERE))
import harness as H   # noqa: E402

ARMS = ['a1_base', 'a2_code', 'a3_review', 'a4_slop']
KW = {'xsec_ref': re.compile(r'\\(c|C)?ref|refer|pointer|Table~|Figure~|table it points|figure', re.I),
      'macro_redund': re.compile(r'recycl|repeat|redundan|duplicate|verbatim|abstract|remov\w+ the (sentence|repetition)', re.I),
      'citation': re.compile(r'relation|cited works|two cited|both works|already stated|positioning', re.I),
      'argument_graph': re.compile(r'reorder|moves? |moved|precede|ahead of|order of|before the claim|limitation .* before', re.I),
      'evidence_gap': re.compile(r'aggregate|individual|acknowledg|evidence.gap', re.I),
      'fig_exposition': re.compile(r'.', re.I)}
CITE_RE = re.compile(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])?\{([^}]*)\}')
NUM_RE = re.compile(r'(?<![\w.\-])\d+(?:\.\d+)?%?(?![\w.])')
POINTER_ONLY = re.compile(r'^[^.]{0,60}(Section|Sec\.|\\S)~?\\(c|C)?ref\{[^}]*\}[^.]{0,80}\.$')


def cites(t): return set(k.strip() for m in CITE_RE.finditer(t or '') for k in m.group(1).split(','))
def nums(t): return set(NUM_RE.findall(re.sub(r'\\(?:cite[a-zA-Z]*|ref|cref|label)\{[^}]*\}', ' ', t or '')))


def paras(txt: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n[ \t]*\n', txt) if p.strip()]


def find_para(tree: Path, rel: str, original: str) -> tuple[str | None, float]:
    f = tree / rel
    if not f.exists():
        return None, 0.0
    best, score = None, 0.0
    o = re.sub(r'\s+', ' ', original)
    for p in paras(f.read_text(errors='ignore')):
        s = difflib.SequenceMatcher(None, o[:600], re.sub(r'\s+', ' ', p)[:600], autojunk=False).ratio()
        if s > score:
            best, score = p, s
    return best, score


def flags(original: str, other: str) -> list[str]:
    if other is None:
        return ['paragraph_not_found']
    o, x = re.sub(r'\s+', ' ', original).strip(), re.sub(r'\s+', ' ', other).strip()
    if o == x:
        return ['unchanged']
    out = []
    if cites(o) - cites(x): out.append('cite_removed:' + ','.join(sorted(cites(o) - cites(x)))[:60])
    if cites(x) - cites(o): out.append('cite_added:' + ','.join(sorted(cites(x) - cites(o)))[:60])
    if nums(o) - nums(x): out.append('numbers_removed:' + ','.join(sorted(nums(o) - nums(x)))[:40])
    if nums(x) - nums(o): out.append('numbers_added:' + ','.join(sorted(nums(x) - nums(o)))[:40])
    for s in re.split(r'(?<=[.!?])\s+', x):
        if POINTER_ONLY.match(s.strip()) and s.strip() not in o:
            out.append('pointer_only_sentence_added'); break
    if len(x) < 0.5 * len(o): out.append('mostly_deleted')
    return out or ['changed']


def load_changes(gate_dir: Path) -> dict:
    """From CHANGES.md, id -> (file, original, revised) full text."""
    out = {}
    t = (gate_dir / 'CHANGES.md').read_text(errors='ignore') if (gate_dir / 'CHANGES.md').exists() else ''
    for m in re.finditer(r'## Change (\d+)\. (\S+) \((\w+)\)\n\n### original\n\n(.*?)\n\n### revised\n\n(.*?)(?=\n## Change |\n# Context|\Z)', t, re.S):
        out[int(m.group(1))] = {'file': m.group(2), 'op': m.group(3), 'original': m.group(4).strip(), 'revised': m.group(5).strip()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(API / 'runs_haiku_sonnetgate_v14' / 'A-loc'))
    ap.add_argument('--out', default=str(API / 'final_v14' / 'CASE_STUDY' / 'candidates.json'))
    a = ap.parse_args()
    runs = Path(a.runs)
    cands = {it: [] for it in KW}
    for tp in sorted(runs.glob('*/trajectory.json')):
        code = tp.parent.name
        t = json.load(open(tp))
        m0 = H.fars_paper_dir(code)
        for r in t['rounds']:
            g = r.get('gate') or {}
            gate_dir = tp.parent / 'gate' / f"R{r['round']}"
            ch = load_changes(gate_dir)
            reverted = [v for v in g.get('verdicts', []) if v['verdict'] != 'keep']
            for v in g.get('verdicts', []):
                if v['verdict'] != 'keep':
                    continue
                reason = v.get('reason') or ''
                c = ch.get(v['id'], {})
                orig, rev = c.get('original', v.get('original', '')), c.get('revised', v.get('revised', ''))
                for it, rx in KW.items():
                    if it == 'fig_exposition':
                        if v.get('category') != 'figure': continue
                    elif v.get('category') != 'faithful' or not rx.search(reason):
                        continue
                    if not orig and it != 'fig_exposition':
                        continue      # an added paragraph has nothing to compare against
                    base = {}
                    for arm in ARMS:
                        tree = EOR / arm / code / 'R3'
                        if not tree.exists(): tree = EOR / arm / code / 'R1'
                        p, s = find_para(tree, v['file'], orig) if orig else (None, 0.0)
                        base[arm] = {'para': p, 'match': round(s, 3), 'flags': flags(orig, p) if s >= 0.5 else ['paragraph_not_found']}
                    scores = {'R0': t['r0']['scores'].get(it), 'R3': (t['rounds'][-1]['scores_after'] or {}).get(it)}
                    cands[it].append({'code': code, 'round': r['round'], 'id': v['id'], 'file': v['file'], 'reason': reason,
                                      'original': orig, 'ours': rev, 'baselines': base, 'scores_ours': scores,
                                      'reverted_same_round': [{'file': x['file'], 'category': x.get('category'), 'reason': (x.get('reason') or '')[:300]} for x in reverted][:6]})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands, indent=1, ensure_ascii=False))
    for it, l in cands.items():
        bad = sum(1 for c in l if any(f for arm in ARMS for f in c['baselines'][arm]['flags'] if f.startswith(('cite_removed', 'numbers_removed', 'pointer_only', 'mostly_deleted'))))
        print(f'{it}: {len(l)} candidates, {bad} where some baseline damaged the same paragraph')


if __name__ == '__main__':
    main()
