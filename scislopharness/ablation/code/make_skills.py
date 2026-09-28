"""Component ablation of the editor's information (§5: "which component carries the gain").

Derives three reduced skill files from the shipped SciSlop_v0.5.md so that every retained field is
verbatim and only the ablated fields are removed.

  (a) name     entry headings only
  (b) def      + Definition
  (c) defloc   + Units-as-reported format, and the harness attaches SLOP_FINDINGS.md (locations)
  (d) full     = SciSlop_v0.5.md + SLOP_FINDINGS.md (the shipped method, already run: final_v14)

Global rules 1-4 (no fabrication, no citation changes, keep results, smallest edit) are constant
across arms: they are constraints, not information about the patterns, and the general-revision
baselines receive the same constraints. Rule 5 is the only rule that talks about locations, so it
is rewritten per arm. Section headers (Structure / Argument / Artifacts) are kept.

  python3 make_skills.py            # writes ../skills/SciSlop_v0.5_{name,def,defloc}.md
"""
from __future__ import annotations
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL = HERE.parent
SRC = ABL.parent / 'ver1' / 'temp' / 'skill' / 'SciSlop_v0.5.md'
OUT = ABL / 'skills'

# Pattern names as the paper names them (paper tables / Fig. SciSlopHarness). The shipped heading of entry 4 is
# "Citation (isolated citations in the Introduction and Related Work)"; the parenthetical is a gloss, i.e. definition,
# so the name-only arm must not carry it. "Citation isolation" is the paper's name for the item.
NAMES = {'Cross-section references': 'Cross-section references', 'Macro redundancy': 'Macro redundancy',
         'Argument graph': 'Argument graph', 'Citation': 'Citation isolation',
         'Figure exposition': 'Figure exposition', 'Evidence gap': 'Evidence gap'}

DESCRIPTION = {
    'name': 'Names of six scientific slop patterns. Read this file before revising a research manuscript. Remove these patterns from the manuscript. Never invent evidence.',
    'def': 'Definitions of six scientific slop patterns. Read this file before revising a research manuscript. Fix what the definitions describe. Never invent evidence.',
    'defloc': 'Definitions of six scientific slop patterns, each with the format in which located instances are reported. Read this file before revising a research manuscript. Fix what the definitions describe. Never invent evidence.',
}
INTRO = {
    'name': 'Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives the name of the pattern.',
    'def': 'Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives what the pattern is and when a unit of the manuscript counts as an instance.',
    'defloc': 'Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives what the pattern is, when a unit of the manuscript counts as an instance, and how instances are reported to you when an external check has located them.',
}
RULE5 = {
    'name': '5. No locations are listed. Read the manuscript and find the instances yourself using the names below. Repair an instance where the repair reads naturally. When the only available repair would add text that the argument does not need, leave that instance unchanged.',
    'def': '5. No locations are listed. Read the manuscript and find the instances yourself using the definitions below. Repair an instance where the repair reads naturally. When the only available repair would add text that the argument does not need, leave that instance unchanged.',
    'defloc': None,   # keep the shipped rule 5 verbatim
}
USED = {
    'name': 'An editor or agent reads this file once, then revises the manuscript. After a revision, the manuscript is checked again. The procedure stops when nothing remains, when a revision changes nothing, or when a round limit is reached.',
    'def': 'An editor or agent reads this file once, then revises the manuscript. After a revision, the manuscript is checked again. The procedure stops when nothing remains, when a revision changes nothing, or when a round limit is reached.',
    'defloc': None,   # keep verbatim
}
KEEP_FIELDS = {'name': (), 'def': ('Definition',), 'defloc': ('Definition', 'Units as reported')}


def build(arm: str, txt: str) -> str:
    fm_m = re.match(r'---\n(.*?)\n---\n', txt, re.S)
    fm = fm_m.group(1)
    fm = re.sub(r'^description:.*$', 'description: ' + DESCRIPTION[arm], fm, flags=re.M)
    fm = re.sub(r'^version:.*$', f'version: 0.5-ablation-{arm} (2026-09-22; component ablation of the editor information. Derived from 0.5 by removing fields; every retained field is verbatim from 0.5)', fm, flags=re.M)
    body = txt[fm_m.end():]
    # intro paragraph (first paragraph after the H1)
    body = re.sub(r'(# SciSlop\. Scientific slop and how to remove it\n\n)(.*?)(\n\n## Global rules)', lambda m: m.group(1) + INTRO[arm] + m.group(3), body, count=1, flags=re.S)
    if RULE5[arm]:
        body = re.sub(r'^5\. .*?$', RULE5[arm], body, count=1, flags=re.M)
    if USED[arm]:
        body = re.sub(r'(## How this file is used\n\n)(.*?)(\n*\Z)', lambda m: m.group(1) + USED[arm] + '\n', body, count=1, flags=re.S)

    def entry(m):
        num, head, paren, ebody = m.group('num'), m.group('head').strip(), m.group('paren'), m.group('body')
        name = NAMES[head]
        fields = []
        for field in ('Definition', 'Fix', 'Units as reported'):
            fm2 = re.search(r'\*\*' + field + r'\.\*\*\s*(.*?)(?=\n\s*\n\*\*|\n\s*\n---|\n\s*\n###|\Z)', ebody, re.S)
            if fm2 and field in KEEP_FIELDS[arm]:
                fields.append(f'**{field}.** {fm2.group(1).strip()}')
        return f'### {num}. {name}\n\n' + ('\n\n'.join(fields) + '\n\n' if fields else '')

    body = re.sub(r'^### (?P<num>\d+)\. (?P<head>[^(\n]+?)\s*(?:\((?P<paren>[^)]*)\))?\s*\n(?P<body>.*?)(?=^### |^---|\Z)', entry, body, flags=re.S | re.M)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return f'---\n{fm}\n---\n' + body


def main():
    txt = SRC.read_text()
    OUT.mkdir(exist_ok=True)
    for arm in ('name', 'def', 'defloc'):
        out = OUT / f'SciSlop_v0.5_{arm}.md'
        out.write_text(build(arm, txt))
        print(out, len(out.read_text()), 'chars')


if __name__ == '__main__':
    main()
