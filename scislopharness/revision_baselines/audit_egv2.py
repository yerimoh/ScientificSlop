"""Audit the EOR_EG_V2 ablation: was the displayed instance quoted, or made up?

  python3 audit_egv2.py [--tag .egv2]

The v2 instruction tells the editor to take one real instance out of the project's own records
under materials/ and to quote it exactly. This checks that claim line by line. For every paper it
diffs the round-1 manuscript against the original, pulls the body of any display environment the
editor added, and looks for each body line in the records. A line that is nowhere in the records
was written by the editor, not taken from the run. Numbers are checked the same way and separately,
because a paper can carry real numbers around an invented specimen.
"""
import argparse, difflib, json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOR, fars_paper_dir, tex_files, load_progress  # noqa: E402

DISPLAY = re.compile(r'\\begin\{(verbatim|lstlisting|minted|quote|quotation|example|tcolorbox|'
                     r'Verbatim|alltt|tabular|tabularx)\}(.*?)\\end\{\1\}', re.S)
NUM = re.compile(r'(?<![\w.])\d+\.\d+(?![\w.])')


def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def materials_text(tree: Path) -> str:
    out = []
    for f in sorted((tree / 'materials').rglob('*')):
        if f.is_file():
            try:
                out.append(f.read_text(errors='ignore'))
            except Exception:
                pass
    return norm(' '.join(out))


def added_text(orig: Path, rev: Path) -> str:
    """Everything present in the revised tex and absent from the original, as one blob."""
    chunks = []
    for f in tex_files(rev):
        rel = f.relative_to(rev)
        a = (orig / rel).read_text(errors='ignore').splitlines() if (orig / rel).exists() else []
        b = f.read_text(errors='ignore').splitlines()
        chunks += [l[2:] for l in difflib.ndiff(a, b) if l.startswith('+ ')]
    return '\n'.join(chunks)


def specimen_lines(blob: str) -> list:
    """Body lines of the display environments the editor added."""
    lines = []
    for _, body in DISPLAY.findall(blob):
        for line in body.splitlines():
            line = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', line)
            line = line.replace('&', ' ').replace('\\\\', ' ')
            if len(re.sub(r'[^0-9A-Za-z]', '', line)) >= 8 and '[' not in line[:2]:
                lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='.egv2')
    a = ap.parse_args()
    root = EOR / 'runs' / f'a4s_evidence_gap{a.tag}'
    os.environ['EOR_TAG'] = a.tag
    import common
    common.TAG = a.tag                      # a half-written tree is not a result, so the
    prog = load_progress('a4s_evidence_gap')  # progress record decides what is audited
    rows = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        rev = d / 'R1'
        if not (rev / 'main.tex').exists():
            continue
        rec = prog.get(d.name, {}).get(1)
        if not rec or rec.get('exec') not in ('ok', 'nothing_to_fix', 'failed'):
            continue
        orig = fars_paper_dir(d.name)
        mats = materials_text(rev)
        blob = added_text(orig, rev)
        spec = specimen_lines(blob)
        found = [l for l in spec if norm(l) and norm(l) in mats]
        nums = sorted(set(NUM.findall(blob)))
        nums_found = [n for n in nums if n in mats]
        if not spec:
            verdict = 'no_display_added' if not blob.strip() else 'text_only'
        elif len(found) >= 0.5 * len(spec):
            verdict = 'quoted'
        else:
            verdict = 'invented'
        rows.append({'code': d.name, 'verdict': verdict, 'specimen_lines': len(spec),
                     'lines_in_records': len(found), 'numbers_added': len(nums),
                     'numbers_in_records': len(nums_found),
                     'stray_tex_in_materials': sorted(str(p.relative_to(rev))
                                                      for p in (rev / 'materials').rglob('*.tex'))})
    print(json.dumps(rows, indent=1))
    from collections import Counter
    print('\nverdicts', dict(Counter(r['verdict'] for r in rows)), f'over {len(rows)} papers')
    bad = [r for r in rows if r['numbers_added'] and r['numbers_in_records'] < r['numbers_added']]
    print(f'papers with at least one added number absent from the records: {len(bad)}')
    (EOR / 'results/summary').mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(EOR / f'results/summary/egv2_audit{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
