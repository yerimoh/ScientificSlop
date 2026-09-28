"""Turn the location list into per-file 'repair targets' (0920). Deterministic auxiliary information for the three items that barely moved.

The edits_parallel editor edits only one file. A cross-section reference must come from a section other than the object's home, so that file's editor
needs to know "which sentence in my file uses this table". An isolated citation must be paired with another citation in the same paragraph.
For argument order, which sentence moves where inside the introduction file is fixed. This module finds those candidates deterministically in the manuscript
and writes them down per file. The editor still makes the judgment, and the gate decides what is kept. No new facts are created.
"""
from __future__ import annotations
import re
from pathlib import Path

STOP = set('''the a an of to in on for and or with by from as at is are was were be been this that these those we our its it their
which who whose than then thus such into over under between across per via using use used based show shows shown result results
table figure fig equation eq section method methods model models data set sets number numbers value values each all both more most
less also only not no but if when where while about after before during within without'''.split())
CITE_RE = re.compile(r'\\cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\])?\{([^}]*)\}')


def tex_files(tree: Path):
    return sorted(p for p in Path(tree).rglob('*.tex') if p.name != 'math_commands.tex')


def sentences(t: str):
    t = re.sub(r'%[^\n]*', '', t)
    t = re.sub(r'\s+', ' ', t)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', t) if len(s.strip()) > 30]


def words(t: str):
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', t)
    return {w for w in re.findall(r'[a-zA-Z][a-zA-Z\-]{3,}', t.lower()) if w not in STOP}


def object_context(tree: Path, label: str) -> tuple[Path | None, str]:
    """Caption (table/figure) of the labeled object, or the sentence before an equation. Also returns the home file."""
    for f in tex_files(tree):
        t = f.read_text(errors='ignore')
        i = t.find(f'\\label{{{label}}}')
        if i < 0:
            continue
        env_start = max(t.rfind('\\begin{table', 0, i), t.rfind('\\begin{figure', 0, i), t.rfind('\\begin{equation', 0, i),
                        t.rfind('\\begin{align', 0, i), t.rfind('\\begin{algorithm', 0, i))
        block = t[max(0, env_start):i + 400] if env_start >= 0 else t[max(0, i - 600):i + 200]
        m = re.search(r'\\caption\{((?:[^{}]|\{[^{}]*\})*)\}', block)
        if m:
            return f, m.group(1)
        # equation: the sentence before the environment
        pre = t[max(0, (env_start if env_start >= 0 else i) - 500):(env_start if env_start >= 0 else i)]
        ss = sentences(pre)
        return f, (ss[-1] if ss else '') + ' ' + re.sub(r'\s+', ' ', block[:200])
    return None, ''


def xsec_targets(tree: Path, units: list[str]) -> dict:
    """{file_rel: [suggestion lines]}"""
    out = {}
    for u in units:
        m = re.search(r'(figure|table|equation|algorithm)\s+\\label\{([^}]+)\}', u)
        if not m:
            continue
        kind, label = m.group(1), m.group(2)
        home, ctx = object_context(tree, label)
        cw = words(ctx)
        if len(cw) < 2:
            continue
        cands = []
        for f in tex_files(tree):
            if home is not None and f == home:
                continue
            for s in sentences(f.read_text(errors='ignore')):
                if '\\ref{' in s and label in s:
                    continue
                ov = cw & words(s)
                if len(ov) >= 3:
                    cands.append((len(ov), str(f.relative_to(tree)), s))
        cands.sort(key=lambda x: -x[0])
        for n, rel, s in cands[:2]:
            out.setdefault(rel, []).append(
                f'Cross-section reference target. {kind} \\label{{{label}}} (home: {home.relative_to(tree) if home else "?"}) is never referred to '
                f'from another section. A sentence in this file shares its subject ({", ".join(sorted(cw & words(s))[:5])}): '
                f'"{s[:260]}". If that sentence actually uses what the {kind} shows, add \\ref{{{label}}} inside it; otherwise leave it.')
    return out


def citation_targets(tree: Path, units: list[str]) -> dict:
    out = {}
    for u in units:
        m = re.match(r'\[([^\]]+)\]\s*"?(.+?)"?\s*$', u)
        if not m:
            continue
        sent = re.sub(r'\s+', ' ', m.group(2)).strip().strip('"')
        key = re.sub(r'[^a-z]', '', sent.lower())[:80]
        for f in tex_files(tree):
            t = f.read_text(errors='ignore')
            paras = [p for p in re.split(r'\n\s*\n', t) if p.strip()]
            for p in paras:
                if key and key in re.sub(r'[^a-z]', '', p.lower()):
                    own = set(k.strip() for mm in CITE_RE.finditer(sent) for k in mm.group(1).split(','))
                    partners = []
                    for s in sentences(p):
                        ks = set(k.strip() for mm in CITE_RE.finditer(s) for k in mm.group(1).split(',')) - own
                        if ks and re.sub(r'[^a-z]', '', s.lower())[:80] != key:
                            partners.append((sorted(ks), s))
                    if partners:
                        rel = str(f.relative_to(tree))
                        pk, ps = partners[0]
                        out.setdefault(rel, []).append(
                            f'Citation target. The isolated citing sentence "{sent[:200]}" sits in the same paragraph as '
                            f'\\citep{{{",".join(pk)}}} ("{ps[:200]}"). A repair counts only if one sentence cites both works and states a relation the '
                            f'manuscript already supports (shared limitation, differing assumption or method, what this work takes from each). '
                            f'Do not join them with a bare connector; if the manuscript states no relation, leave the sentence.')
                    break
    return out


def ag_targets(tree: Path, units: list[str]) -> dict:
    out = {}
    for u in units:
        m = re.search(r'"(.+?)" is stated before its strongest cue "(.+?)"', u)
        if not m:
            continue
        claim, cue = m.group(1), m.group(2)
        cw = words(re.sub(r'\[(math|citation|reference)\]', ' ', claim))
        for f in tex_files(tree):
            t = re.sub(r'\s+', ' ', f.read_text(errors='ignore'))
            hit = any(len(cw & words(s)) >= max(3, int(0.6 * len(cw))) for s in sentences(t)) if cw else False
            if hit:
                out.setdefault(str(f.relative_to(tree)), []).append(
                    f'Argument-order target. In this file, move the claim sentence "{claim[:180]}" so that it follows the cue sentence '
                    f'"{cue[:180]}", or move the cue ahead of the claim. Move only; do not delete, weaken or add claims.')
                break
    return out


def build(tree: Path, units: dict) -> dict:
    """{file_rel: [lines]} for all three items."""
    out = {}
    for d in (xsec_targets(tree, units.get('xsec_ref') or []), citation_targets(tree, units.get('citation') or []),
              ag_targets(tree, units.get('argument_graph') or [])):
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out
