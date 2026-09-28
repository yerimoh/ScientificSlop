"""Prompt texts for every arm. Kept verbatim here so the appendix can quote them."""

# ---- neutral editor system prompt (a3_review, a4_slop, a4s_*) ------------------------
# Same as the B-series wrapper minus the sentence that restricted edits to the
# experiments and method sections (see DESIGN.md section 1).
AGENT_SYSTEM = ('You are revising a LaTeX research paper. Source files (main.tex, sections/*.tex, '
                'math_commands.tex, a .bib file) are in your working directory. Use Read/Edit/Write to '
                'modify the .tex files in place. Do not create files outside the paper directory. '
                'When done, finish your turn; do not run latex and do not summarize.')

import os

NO_FAB = ('Do not fabricate experiments, numbers, or citations; if evidence is missing, narrow the '
          'claim instead. Do not add citations that are not already in the paper, and do not change '
          'reported numbers.')

# ---- generic improvement instruction (a1_base, a2_code) -------------------------------
GENERIC = ('Revise this paper so that it is a better research paper: improve its clarity, '
           'organization, argumentation, and presentation, and make sure every part of the paper '
           'does its job for the reader. Keep the method, the experiments, and the reported results '
           'as they are. ' + NO_FAB)

A2_PROMPT = (GENERIC + ' Make the changes directly by editing the .tex files in the current '
             'directory; do not just give advice.')

A1_SYSTEM = ('You are an expert editor of machine-learning research papers written in LaTeX. You '
             'return revised LaTeX source only.')


def a1_prompt(files: list[tuple[str, str]], n: int) -> str:
    """One call per round: the whole paper in, the whole paper back, file by file.
    files = [(relative path, content)] in document order."""
    body = '\n'.join(f'===== FILE {rel} =====\n{txt}' for rel, txt in files)
    names = ', '.join(rel for rel, _ in files)
    return (f'{GENERIC}\n\nThis is revision round {n}. The complete LaTeX source of the paper follows, '
            f'file by file. Return the revised source of every file, in the same order, using exactly '
            f'this format and nothing else:\n\n'
            f'<<<FILE {{path}}>>>\n(the complete revised content of that file)\n<<<END {{path}}>>>\n\n'
            f'Return all {len(files)} files ({names}). Keep every \\input, \\label, \\cite, \\ref, '
            f'environment and preamble line that a file already has, unless the revision moves or removes '
            f'prose; never remove \\begin{{document}}, \\end{{document}}, \\documentclass, or \\input lines. '
            f'Do not abbreviate or elide any part of a file.\n\n' + body + '\n\n')


def a1_prompt_file(files: list[tuple[str, str]], rel: str, n: int) -> str:
    """Per-file variant (the 0914 original design, one call per file, recovered verbatim from
    runs/_archive/a1_base_perfile_0914 logs). Used by the open-model backend (0916), whose editor
    does not keep the multi-file format of a1_prompt: the whole paper is context, one file comes back."""
    body = '\n'.join(f'%%%%% FILE {r}\n{txt}' for r, txt in files)
    return (f'{GENERIC}\n\nThis is revision round {n}. The complete LaTeX source of the paper is given below for '
            f'context. Then revise ONE file, `{rel}`, and return its complete revised content between the markers '
            f'<<<BEGIN FILE>>> and <<<END FILE>>>, with nothing else. Keep every \\input, \\label, \\cite, \\ref, '
            f'environment, and preamble line that the file already has, unless the revision needs to move or remove '
            f'prose; never remove \\begin{{document}}, \\end{{document}}, or \\input lines.\n\n'
            f'===== COMPLETE PAPER (context) =====\n' + body + '\n')


A1_FORMAT = ('The complete LaTeX source of the paper follows, file by file. Return the revised source of every '
             'file, in the same order, using exactly this format and nothing else:\n\n'
             '<<<FILE {path}>>>\n(the complete revised content of that file)\n<<<END {path}>>>\n\n'
             'Return all {n_files} files ({names}). Keep every \\input, \\label, \\cite, \\ref, environment and '
             'preamble line that a file already has, unless the revision moves or removes prose; never remove '
             '\\begin{{document}}, \\end{{document}}, \\documentclass, or \\input lines. Do not abbreviate or elide '
             'any part of a file.\n\n')


def a4_prompt_text(units: dict, items: list[str], n: int, files: list[tuple[str, str]], max_units: int = 12):
    """Text-in / text-out form of the slop-aware instruction (open-model backend, 0917): the same issue list as
    a4_prompt, with the agent sentence 'Edit the .tex files in place.' replaced by the a1 file-return format."""
    fb = a4_prompt(units, items, n, max_units)
    if fb is None:
        return None
    fb = fb.replace(' Edit the .tex files in place.\n', '\n')
    body = '\n'.join(f'===== FILE {rel} =====\n{txt}' for rel, txt in files)
    names = ', '.join(rel for rel, _ in files)
    return fb + '\n' + A1_FORMAT.format(path='{path}', n_files=len(files), names=names) + body + '\n\n'


# ---- a3_review: B3 prompts verbatim ---------------------------------------------------
REVIEW_PROMPT = (
    "You are a knowledgeable, critical, but fair reviewer for a top machine-learning "
    "conference. Read the paper draft below and write a review of its overall quality: "
    "the significance of the problem, the soundness of the approach, the strength of "
    "the evidence, and the clarity of the writing.\n\n"
    "Respond with ONLY compact JSON in exactly this format:\n"
    '{"summary": "<2-4 sentence summary of the paper and its contributions>",\n'
    ' "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],\n'
    ' "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],\n'
    ' "questions": ["<question 1>", "<question 2>", "<question 3>"],\n'
    ' "rating": <integer 1-10, where 1 = trivial or wrong and 10 = seminal>}\n\n'
    "PAPER (LaTeX source, possibly truncated):\n")


def a3_prompt(review: dict) -> str:
    import json
    return ("A reviewer left the following review of this paper. Revise the paper to "
            "address the review.\n\n" + json.dumps(review, indent=2) +
            "\n\nDo not fabricate experiments, numbers, or citations; if evidence is "
            "missing, narrow the claim instead.")


# ---- a4_slop / a4s_<item>: item definitions and fix directions -------------------------
# Definitions follow slop/SLOP_SCORE.md; no score formula or threshold is given.
ITEM_TEXT = {
    'xsec_ref': {
        'name': 'Objects never used outside their own section',
        'definition': ('A paper declares objects: its body sections and the labelled figures, tables, '
                       'equations, and algorithms inside them. An object is unused when no section other '
                       'than the one that contains it ever refers to it, so the rest of the paper never '
                       'builds on it.'),
        'fix': ('Where the paper\'s argument actually relies on an object, refer to it from the section '
                'that relies on it (for example, discuss the result of a table in the section whose claim '
                'it supports, and point back to the method\'s equation where the experiments use it). Do '
                'not add pointers that the surrounding text does not use.'),
        'unit_label': 'unused objects (label, kind, home section)'},
    'macro_redund': {
        'name': 'Sentences recycled from an earlier section',
        'definition': ('A sentence is recycled when at least half of it repeats, almost word for word, '
                       'text that an earlier, different section already contained.'),
        'fix': ('Rewrite each recycled sentence so that it adds what its own section needs (a new '
                'detail, a consequence, a qualification), or remove it if the section does not need it. '
                'Do not paraphrase merely to disguise the repetition.'),
        'unit_label': 'recycled sentences (section; sentence; source section)'},
    'citation': {
        'name': 'Isolated citations in the Introduction and Related Work',
        'definition': ('A citing sentence is isolated when it cites a single prior work and neither '
                       'groups it with another work, nor mentions another work elsewhere in the sentence, '
                       'nor states a relation between two works. A related-work section made of isolated '
                       'sentences lists prior work instead of positioning it.'),
        'fix': ('Where the paper\'s positioning depends on a cited work, state how it relates to other '
                'works already cited in the paper: a shared limitation, a difference in assumption or '
                'method, or what this paper takes from each. Only use works already cited in the paper; '
                'do not add new citations.'),
        'unit_label': 'isolated citing sentences (section; sentence)'},
    'evidence_gap': {
        'name': 'No concrete instance is displayed',
        'definition': ('The paper reports aggregate results but never displays a single concrete instance '
                       'of what it counts: an example input and output, a case, a failure, a worked '
                       'example, or a quoted specimen.'),
        'fix': ('If the paper\'s own materials in this directory contain such an instance (for example a '
                'prompt, a generated output, a case in the appendix), display one where the reader needs '
                'it and refer to it from the text. If no instance is available in the materials, do not '
                'invent one; instead say explicitly, in the relevant section, that no instance is shown.'),
        'unit_label': 'paper-level observation'},
}

# 0917 ablation. The shipped evidence_gap instruction offers two routes and the second one, saying
# that no instance is shown, cannot change the score. It also points at "materials in this
# directory", but stage() copies only .tex, .bib, .bst and .sty, so the round tree never held the
# project's experiment records. EOR_EG_V2=1 stages those records under materials/ and replaces the
# instruction with a single imperative route. The ban on fabrication is untouched, and the editor
# is told to quote the record exactly and to name the file it came from, so every added instance
# is auditable.
EG_V2_FIX = ('Display one concrete instance in the paper. The directory materials/ holds this '
             'project\'s own experiment records. Read them, take one real instance from them, '
             'for example an input and the output it produced, a single case, a failure, or a '
             'logged example, and typeset it where the reader needs it as a figure, a table, a '
             'listing or an example block, then refer to it from the text. Quote it exactly as it '
             'appears in the record and do not invent, alter, round or embellish any part of it. '
             'Name the file you took it from in the caption. If the records hold nothing that can '
             'be displayed, leave the paper unchanged and do not write about the absence.')

if os.environ.get('EOR_EG_V2') == '1':
    ITEM_TEXT['evidence_gap'] = {**ITEM_TEXT['evidence_gap'], 'fix': EG_V2_FIX}

A4_HEAD = ('An automated check of this manuscript found the following issues. Revise the paper to '
           'address them. ' + NO_FAB + ' Edit the .tex files in place.\n')


def a4_prompt(units: dict, items: list[str], n: int, max_units: int = 12) -> str:
    parts = [A4_HEAD, f'This is revision round {n}.\n']
    k = 0
    for it in items:
        us = units.get(it) or []
        if not us:
            continue
        k += 1
        t = ITEM_TEXT[it]
        parts.append(f'\n## Issue {k}: {t["name"]}\n{t["definition"]}\nWhat to do: {t["fix"]}\n')
        if it == 'evidence_gap':
            parts.append(f'Finding: {us[0]}\n')
        else:
            parts.append(f'Findings ({len(us)} total; {t["unit_label"]}):\n')
            for u in us[:max_units]:
                parts.append(f'- {u}\n')
            if len(us) > max_units:
                parts.append(f'- (and {len(us) - max_units} more of the same kind; treat them the same way)\n')
    if k == 0:
        return None
    return ''.join(parts)
