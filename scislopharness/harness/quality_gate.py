"""Quality gate for Recursive Slop Mitigation (0918 evening).

After an editor round, every changed paragraph is reviewed against the original manuscript and the project's own
experiment records by a second model acting as a co-author. A change is kept only if it repairs a slop pattern
without saying anything the record does not support, without moving a number, and without adding a pointer or a
citation relation that the sentence does not use. Reverted paragraphs are restored before re-measurement, and the
reasons are handed to the editor in the next round.

Reviewer = Claude Code agent (default claude-sonnet-5), read-only over a staging directory:
  ORIGINAL/   M_0 sources          REVISED/   M_r sources        materials/   project experiment records
  SciSlop.md  the skill file       CHANGES.md numbered changes   -> VERDICTS.json written by the reviewer
"""
from __future__ import annotations
import difflib, json, os, re, shutil, sys, time
from pathlib import Path
import harness as H
sys.path.insert(0, str(H.EOR_CODE))
from run_arm import stage_materials  # noqa: E402

GATE_MODEL = H.os.environ.get('SH_GATE_MODEL', 'claude-sonnet-5')
GATE_SETTINGS = H.HERE / 'gate_settings.json'
GATE_SETTINGS.write_text(json.dumps({'permissions': {'allow': ['Read', 'Glob', 'Grep', 'Write'], 'deny': ['WebSearch', 'WebFetch', 'Edit', 'Bash']}}))


def raw_paras(txt: str) -> list[str]:
    return re.split(r'(\n[ \t]*\n)', txt)     # keep separators so the file can be rebuilt byte-exact


def changes_between(m0: Path, mr: Path) -> list[dict]:
    out = []
    for f in H.a1_files(m0):
        rel = str(f.relative_to(m0)); g = mr / rel
        if not g.exists():
            continue
        a, b = raw_paras(f.read_text(errors='ignore')), raw_paras(g.read_text(errors='ignore'))
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'equal':
                continue
            oa, ob = ''.join(a[i1:i2]).strip(), ''.join(b[j1:j2]).strip()
            if oa == ob:
                continue
            out.append({'id': len(out) + 1, 'file': rel, 'op': op, 'i1': i1, 'i2': i2, 'j1': j1, 'j2': j2, 'original': oa, 'revised': ob})
    fc = figure_change(m0, mr)
    if fc:
        fc['id'] = len(out) + 1
        out.append(fc)
    return out


def figure_change(prev: Path, mr: Path) -> dict | None:
    """If the method-diagram image was edited in this round, that is also one numbered change (op = image).

    0919 evening. Figure edits leave no trace in the tex, so they never appeared in CHANGES.md and the reviewer's rule 6
    had nothing to judge. If EDIT_REPORT.json differs from the previous round, we list it as a change; on a revert, the
    image and the record are restored to the previous round's."""
    rp, rr = Path(prev) / 'figures' / 'EDIT_REPORT.json', Path(mr) / 'figures' / 'EDIT_REPORT.json'
    if not rr.exists():
        return None
    try:
        new = json.loads(rr.read_text()) or {}
    except Exception:
        return None
    old = {}
    if rp.exists():
        try:
            old = json.loads(rp.read_text()) or {}
        except Exception:
            old = {}
    if old.get('erased_cumulative', old.get('erased')) == new.get('erased_cumulative', new.get('erased')) and old.get('lines_after') == new.get('lines_after'):
        return None
    imgs = [f for f in (Path(mr) / 'figures').iterdir() if f.suffix.lower() in ('.jpeg', '.jpg', '.png', '.pdf')]
    name = imgs[0].name if imgs else 'method figure'
    this_round = (new.get('history') or [{}])[-1] if new.get('history') else new
    erased = this_round.get('erased') or new.get('erased') or []
    pres = this_round.get('preservation') or new.get('preservation') or {}
    prev_desc = ('the image before this round, PREVIOUS/figures/' + name + (
        '. Phrases erased in earlier rounds: ' + '; '.join(str(e.get('phrase')) for e in (old.get('erased_cumulative') or old.get('erased') or [])) if old else ' (the original, unedited image)'))
    rev_desc = ('REVISED/figures/' + name + '. This round erased ' + str(len(erased)) + ' phrase(s) by inpainting their rectangles: '
                + '; '.join(f'"{e.get("phrase")}" at {e.get("box_used")}' for e in erased)
                + f'. Pixels changed outside the target rectangles: {pres.get("changed_outside")} of the image '
                + f'(ratio {pres.get("outside_ratio")}). Text lines remaining after the edit: {len(new.get("lines_after") or [])}. '
                + 'Open both images with Read and check that every component, arrow and label of the mechanism is still there '
                + 'and that only the listed phrases are gone.')
    return {'file': f'figures/{name}', 'op': 'image', 'original': prev_desc, 'revised': rev_desc}


REVIEW_INSTRUCTION = """You are the second author of this manuscript, checking a colleague's revision before it is accepted. The directory holds ORIGINAL/ (the manuscript as it was before any revision round; the ground truth for what the manuscript says), PREVIOUS/ when present (the manuscript before this round; CHANGES.md is the difference between PREVIOUS/ and REVISED/), REVISED/ (after this round), materials/ (this project's own experiment records, the only ground truth for numbers and claims beyond the manuscript), SciSlop.md (the instructions the colleague followed), and CHANGES.md (every changed paragraph, numbered, original above and revised below).

Decide for every change in CHANGES.md whether to keep it or revert it to the original paragraph. Read the surrounding sections in ORIGINAL/ and REVISED/ and, when a number or a factual statement is involved, look for it in materials/ with Grep.

Revert a change when any of these holds.
1. Beyond the record. The revised text states, specifies, or implies something that neither the original manuscript nor materials/ supports. This includes narrowing a general statement to a specific one (a comparison target, a cause, a mechanism), adding a new contribution or a new interpretation of results, and describing a cited work in a way the manuscript never did.
2. Numbers. A reported number is changed, rounded, dropped, or newly introduced without an exact match in the original manuscript or in materials/.
3. Placement. A cross-reference, a pointer to a section, or a citation is added to a sentence that does not actually use it, a sentence whose only content is pointers is added, or a roadmap sentence that only announces what another section does is added.
4. Citation relation. Two cited works are joined by a stated relation (extends, builds on, shares a limitation, differs in assumption) that the original manuscript did not state and that cannot be verified from it. Grouping is not such a relation. A sentence that gathers works the manuscript already describes and names the property they share, where that property is read off the descriptions the manuscript itself gives them and each work keeps its own description, is the repair the citation item asks for and it stays. What this criterion forbids is a new fact about a cited work, or about how two of them stand to each other, that the manuscript never stated and that materials/ cannot confirm. Check the grouped works' own sentences in ORIGINAL/ before reverting: if each work is still described as it was and only the arrangement changed, keep it.
5. Damage. The change removes a figure, table, label, citation command, or reported result, or breaks LaTeX. Deleting a citing sentence that SLOP_FINDINGS.md did not flag is damage too, even when the citation survives elsewhere. It removes a compliant sentence and leaves the flagged ones standing, which improves a rate by shrinking what it is measured out of rather than by repairing anything.
6. Figures. The method figure is repaired by erasing the expository phrases from the original image, not by
redrawing it. The editor only lists the phrases; a deterministic step locates them and paints over those rectangles
with the surrounding background, so every other pixel is the original and nothing is generated. Check
REVISED/figures/EDIT_REPORT.json and open REVISED/figures/ and PREVIOUS/figures/ (ORIGINAL/figures/ in round 1) with
Read to compare the images yourself. The image edit appears in CHANGES.md as its own numbered change (op image); keep
or revert it like any other change. Revert the edit when a
listed phrase is not one the located instances named, when the erased rectangle covered a component, an arrow or a
label that is part of the mechanism, or when the figure was replaced by a drawing instead of being edited. A number
that labels a component is part of the mechanism and stays. Material taken out of the figure, a setting or a result
or a claim, belongs in the text near it, and reverting is right when it simply disappeared. Notation keys, check
marks, "key idea" boxes and step numbering are exposition, not information, and may simply go.
7. Exhibits. A newly displayed example, case, listing, schema, prompt, output, or quoted specimen counts as evidence only if it is an exact copy of something in materials/ (when SPECIMEN.md is present in this directory, that is the one specimen the record holds, with its source file) or of a produced record already shown in the manuscript, and the text names where it comes from. An illustrative or hypothetical instance ("might contain", "for example, a schema like", a made-up input), or a real fragment embedded in an invented frame, is not evidence and must be reverted even when every word in it appears somewhere in the manuscript.
9. Acknowledging the evidence gap. A sentence that states that no individual case was retained or inspected and that the results rest on aggregate scores is a legitimate repair when the record indeed holds no instance (SPECIMEN.md absent); keep it unless it contradicts the record or the manuscript. Narrowing a sentence that described how individual cases behave to what the aggregate supports is also legitimate, provided no reported number changes.
8. Merging. Two or more citing sentences are joined into one with a connector such as "while", "and", "whereas" or a semicolon without stating a relation between the cited works that the original manuscript supports (a shared limitation, a difference in assumption or method, what this manuscript takes from each). Joining sentences so that they sit together is not positioning and only makes the isolated-citation count go down; revert it. The same holds for any edit whose only effect is to make a check pass without changing what the manuscript says.

Keep a change when it repairs one of the patterns in SciSlop.md with material that is already in the manuscript or in materials/, in a sentence that genuinely uses what it refers to, and leaves every number and cited work as it was. Moving a sentence, restating a result that a table in the manuscript already reports, or referring to a figure from the sentence that interprets it are all fine.

A change may mix acceptable and unacceptable edits. Then decide by the worst part: if reverting would lose a good repair but keeping would admit an unsupported statement, revert and say in the reason which part was unsupported, so the colleague can redo only the good part.

Then decide which located instances should simply be left alone. SLOP_FINDINGS.md, when present, lists the instances the external check found. An instance you leave alone is removed from the list for the rest of the procedure and will never be repaired, so the bar is high and one failed attempt is not enough to clear it.

Before leaving an instance alone, look for a home for it. Read every other section of REVISED/ and ask where a sentence already does something with the object. The setup sentence of the experiments that applies a method's figure or equation, the sentence that interprets a result the table reports, the limitation the figure illustrates, the contribution the result supports. Leave the instance alone only when you have checked those places and none of them holds a sentence that would use the object. In your reason, name the sections you checked and say what is in them, so that the judgment can be audited. A bad attempt by the editor is evidence about that attempt, not about the instance, so never conclude from one clumsy pointer that no home exists.

For an unused figure, table, equation, or algorithm, the presumption is that a home exists, because the manuscript produced the object for a reason. Leave one alone only when the object is genuinely local, for example a notation table used only where it is defined. A section is different. A section is rarely used by another section, and pointing at one is usually the bare pointer this file forbids, so leaving a section alone is ordinary. An instance whose only available repair would state something the record does not support is also left alone, and you say which record you checked.

Write VERDICTS.json in this directory as a JSON list with one object per change: {"id": <number from CHANGES.md>, "verdict": "keep" or "revert", "category": one of "faithful", "beyond_record", "number", "placement", "citation_relation", "damage", "figure", "exhibit", "merging", "other", "reason": one or two sentences quoting the exact words that decided it}. Cover every change id. Also write RETIRE.json as a JSON list, one object per instance you are leaving alone: {"instance": "<the instance line from SLOP_FINDINGS.md, copied exactly>", "sections_checked": ["<section name>", ...], "reason": "<what you found in each section you checked and why none of them can host the object>"}; write an empty list when every remaining instance still deserves a repair. Do not modify any other file. Do not ask questions and do not summarize; finish your turn when VERDICTS.json exists."""


def stage_review(m0: Path, prev: Path, mr: Path, code: str, skill_path: Path, changes: list[dict], out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    H.stage(m0, out / 'ORIGINAL'); H.stage(mr, out / 'REVISED')
    if os.environ.get('SH_EXTRA_ITEMS') == '1' and 'fig_exposition' in H.EXTRA_ORDER:
        # To check that the redrawn diagram keeps the original mechanism, the reviewer must see the original image too. A
        # list of text labels alone cannot tell whether an edge exists, and then a revert reflects missing information, not an editor failure.
        sys.path.insert(0, str(H.TEMP.parent / 'api' / 'code'))
        import extra_items as _X
        _X.stage_method_figure(code, out / 'ORIGINAL')
    fnd = mr / 'SLOP_FINDINGS.md'
    if fnd.exists():
        shutil.copy2(fnd, out / 'SLOP_FINDINGS.md')
    if prev != m0:
        H.stage(prev, out / 'PREVIOUS')
    # To judge a figure edit the reviewer must see the before and after images and the edit record directly.
    for src_tree, name in ((mr, 'REVISED'), (prev, 'PREVIOUS')):
        fd = Path(src_tree) / 'figures'
        if fd.is_dir() and (out / name).is_dir():
            (out / name / 'figures').mkdir(exist_ok=True)
            for f in fd.iterdir():
                if f.is_file() and f.name != '_prior_lines.json':
                    shutil.copy2(f, out / name / 'figures' / f.name)
    if (mr / 'materials' / 'SPECIMEN.md').exists():
        shutil.copy2(mr / 'materials' / 'SPECIMEN.md', out / 'SPECIMEN.md')
    n_mat = stage_materials(H.fars_paper_dir(code), out)
    shutil.copy2(skill_path, out / 'SciSlop.md')
    md = [f'# Changes ({len(changes)}). materials/ holds {n_mat} record files.\n']
    for c in changes:
        md.append(f'## Change {c["id"]}. {c["file"]} ({c["op"]})\n\n### original\n\n{c["original"] or "(nothing; paragraph added)"}\n\n### revised\n\n{c["revised"] or "(nothing; paragraph deleted)"}\n')
    (out / 'CHANGES.md').write_text('\n'.join(md))
    return out


def _max_turns(env: str) -> list:
    v = os.environ.get(env, '')
    return ['--max-turns', v] if v.isdigit() and int(v) > 0 else []


def _effort(env: str) -> list:
    v = os.environ.get(env, '').strip().lower()
    return ['--effort', v] if v in ('low', 'medium', 'high') else []


def _context_block(d: Path, changes: list[dict]) -> str:
    """Inline the REVISED full text of each file touched by this chunk's changes into CHANGES.md. This saves the reviewer
    the turns of opening files one by one; the verdict criteria are unchanged. Originals and records are opened only when needed."""
    out = []
    seen = set()
    for c in changes:
        rel = c['file']
        if rel in seen or c.get('op') == 'image':
            continue
        seen.add(rel)
        for side in ('REVISED', 'ORIGINAL'):
            f = d / side / rel
            if f.exists():
                t = f.read_text(errors='ignore')
                if len(t) > 60000:
                    t = t[:60000] + '\n... (truncated)'
                label = 'full text after this round' if side == 'REVISED' else 'the original manuscript, ground truth for what it says'
                out.append(f'## {side}/{rel} ({label})\n\n```latex\n{t}\n```\n')
    return '\n'.join(out)


RETIRE_START = 'Then decide which located instances should simply be left alone.'
WRITE_START = 'Write VERDICTS.json in this directory'


def chunk_instruction(k: int, n: int, first: bool) -> str:
    """Adapt the review instruction for a chunk. The change list is chunk_k/CHANGES.md, the verdicts go to chunk_k/VERDICTS.json.
    Only the first chunk does the retire judgment. All other wording is unchanged."""
    t = REVIEW_INSTRUCTION
    head, rest = t.split(RETIRE_START, 1)
    retire_part, write_part = rest.split(WRITE_START, 1)
    head = head.replace('and CHANGES.md (every changed paragraph, numbered, original above and revised below).',
                        f'and CHANGES.md (every changed paragraph, numbered, original above and revised below). The changes are '
                        f'reviewed in {n} parts in parallel; you are part {k} and your part is chunk_{k}/CHANGES.md. Review only '
                        f'the changes listed there. Below the changes, that file also carries the full REVISED and ORIGINAL text of each '
                        f'changed file, so you should not need to open them yourself; open materials/ only when a number has to be '
                        f'checked against the record, and decide from what is in front of you otherwise.')
    head = head.replace('Decide for every change in CHANGES.md', f'Decide for every change in chunk_{k}/CHANGES.md')
    write_part = write_part.replace('as a JSON list with one object per change', f'as a JSON list with one object per change in chunk_{k}/CHANGES.md', 1)
    write_part = write_part.replace('Cover every change id.', f'Cover every change id in chunk_{k}/CHANGES.md.')
    if first:
        body = head + RETIRE_START + retire_part + f'Write chunk_{k}/VERDICTS.json' + write_part
    else:
        write_part = re.sub(r' Also write RETIRE\.json.*?write an empty list when every remaining instance still deserves a repair\.',
                            ' Do not write RETIRE.json; another part does that.', write_part, flags=re.S)
        write_part = write_part.replace('finish your turn when VERDICTS.json exists', f'finish your turn when chunk_{k}/VERDICTS.json exists')
        body = head + f'Write chunk_{k}/VERDICTS.json' + write_part
    body = body.replace(f'Write chunk_{k}/VERDICTS.json in this directory', f'Write chunk_{k}/VERDICTS.json')
    return body


def retire_instruction() -> str:
    """Instruction for the retire judgment only. Drops the change-verdict part and has the model write only RETIRE.json."""
    t = REVIEW_INSTRUCTION
    head, rest = t.split(RETIRE_START, 1)
    retire_part, write_part = rest.split(WRITE_START, 1)
    intro = head.split('Decide for every change in CHANGES.md', 1)[0]
    intro += ('Other reviewers are judging the individual changes in CHANGES.md in parallel. Your job is only the second '
              'decision below.\n\n')
    m = re.search(r'Also write RETIRE\.json.*?write an empty list when every remaining instance still deserves a repair\.', write_part, re.S)
    write = ('Write RETIRE.json in this directory' + m.group(0)[len('Also write RETIRE.json'):] if m else
             'Write RETIRE.json in this directory as a JSON list, one object per instance you are leaving alone: '
             '{"instance": "<the instance line from SLOP_FINDINGS.md, copied exactly>", "sections_checked": [...], "reason": "..."}; '
             'write an empty list when every remaining instance still deserves a repair.')
    return (intro + RETIRE_START + retire_part + write +
            ' Do not write VERDICTS.json and do not modify any other file. Do not ask questions and do not summarize; finish '
            'your turn when RETIRE.json exists.')


REVIEW_SYSTEM = ('You are the second author of a machine-learning manuscript reviewing a colleague\'s revision. You answer with '
                 'JSON only, exactly in the format requested, and nothing else.')
NUM_TOKEN = re.compile(r'(?<![\w.\-])\d+(?:\.\d+)?%?(?![\w.])')


def materials_evidence(d: Path, changes: list[dict], cap_lines: int = 40) -> str:
    """So the reviewer need not Grep itself, pre-print for every number in the revised text the materials/ lines that contain it."""
    mats = d / 'materials'
    if not mats.is_dir():
        return '(no materials directory)'
    nums = set()
    for c in changes:
        nums |= set(NUM_TOKEN.findall(c.get('revised') or ''))
    nums = {n for n in nums if len(n.replace('.', '').replace('%', '')) >= 2 and n not in ('10', '20', '100')}
    if not nums:
        return '(the revised text introduces no numeric tokens to check)'
    files = [f for f in mats.rglob('*') if f.is_file() and f.suffix.lower() in ('.json', '.md', '.txt', '.csv', '.log', '.yaml', '.yml')]
    hits = []
    for f in files:
        try:
            for i, line in enumerate(f.read_text(errors='ignore').splitlines()):
                found = [n for n in nums if n.rstrip('%') in line]
                if found:
                    hits.append(f'{f.relative_to(mats)}:{i + 1}: {line.strip()[:200]}')
                    if len(hits) >= cap_lines:
                        break
        except Exception:
            continue
        if len(hits) >= cap_lines:
            break
    listing = ', '.join(sorted(str(f.relative_to(mats)) for f in files)[:60])
    return (f'Record files: {listing}\n\nNumeric tokens in the revised text: {", ".join(sorted(nums))}\n\n'
            'Lines of the record that contain any of them (file:line: text):\n' + ('\n'.join(hits) if hits else '(none found)'))


def text_gate_instruction(k: int, n: int) -> str:
    """Instruction for the tool-less reviewer. Verdict rules are those of REVIEW_INSTRUCTION; everything to read is inlined below and the answer is JSON."""
    t = REVIEW_INSTRUCTION
    before_retire = t.split(RETIRE_START, 1)[0]
    rules = before_retire[before_retire.index('Decide for every change in CHANGES.md'):]
    rules = rules.replace(
        'Decide for every change in CHANGES.md whether to keep it or revert it to the original paragraph. Read the surrounding '
        'sections in ORIGINAL/ and REVISED/ and, when a number or a factual statement is involved, look for it in materials/ with Grep.',
        'Decide for every change below whether to keep it or revert it to the original paragraph. The full REVISED and ORIGINAL text '
        'of each changed file follows the changes, and after that the lines of the project record that contain any number the revised '
        'text uses. Judge from what is in this message.')
    rules = re.sub(r'Check\s+REVISED/figures/EDIT_REPORT\.json.*?keep\s+or revert it like any other change\.',
                   'For an image change, judge from its description (which phrases were erased, how many pixels changed outside the '
                   'target rectangles, how many text lines remain).', rules, flags=re.S)
    write = t.split(WRITE_START, 1)[1].split('Also write RETIRE.json', 1)[0].strip()
    write = 'Answer with a JSON list' + write[len('as a JSON list'):] if write.startswith('as a JSON list') else 'Answer with a JSON list ' + write
    write = write.replace('Cover every change id.', 'Cover every change id below.')
    return (f'You are the second author of this manuscript, checking a colleague\'s revision before it is accepted. You have no tools; '
            f'everything you may consult is in this message. This is part {k} of {n}. The original manuscript is the ground truth for '
            f'what the manuscript says; the project record is the only ground truth for numbers and claims beyond the manuscript.\n\n'
            + rules.rstrip() + '\n\n' + write + ' Output the JSON list and nothing else.')


def review_text_chunk(d: Path, part: list[dict], k: int, n: int, model: str) -> dict:
    ctx = _context_block(d, part)
    md = [f'# Changes, part {k} of {n} ({len(part)} changes)\n']
    for c in part:
        md.append(f'## Change {c["id"]}. {c["file"]} ({c["op"]})\n\n### original\n\n{c["original"] or "(nothing; paragraph added)"}\n\n### revised\n\n{c["revised"] or "(nothing; paragraph deleted)"}\n')
    prompt = (text_gate_instruction(k, n) + '\n\n' + '\n'.join(md) + '\n# Context\n\n' + ctx +
              '\n# Record evidence\n\n' + materials_evidence(d, part))
    cd = d / f'chunk_{k}'
    cd.mkdir(exist_ok=True)
    (cd / 'CHANGES.md').write_text('\n'.join(md))
    (cd / 'prompt.txt').write_text(prompt)
    res = H.run_cli_json(['-p', prompt, '--model', model, '--tools', '', '--system-prompt', REVIEW_SYSTEM] + _effort('SH_GATE_EFFORT'),
                         d, timeout=1200, env_extra=H.think_env('SH_GATE_THINK'))
    out = res.get('result') or ''
    (cd / 'reviewer_stdout.txt').write_text(out + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    m = re.search(r'\[.*\]', re.sub(r'^```(?:json)?|```$', '', out.strip(), flags=re.M), re.S)
    try:
        verdicts = json.loads(m.group(0)) if m else []
    except Exception:
        verdicts = []
    (cd / 'VERDICTS.json').write_text(json.dumps(verdicts, indent=1, ensure_ascii=False))
    return res


def retire_text(d: Path, model: str) -> dict:
    """Tool-less retire judgment. Inline the full manuscript and the location list, and receive the RETIRE list as JSON."""
    body = retire_instruction()
    body = body.replace('Write RETIRE.json in this directory', 'Answer with a JSON list').replace(
        'finish your turn when RETIRE.json exists', 'output the JSON list and nothing else').replace(
        'Do not write VERDICTS.json and do not modify any other file.', 'You have no tools; the whole REVISED manuscript and SLOP_FINDINGS.md follow.')
    parts = []
    for f in H.a1_files(d / 'REVISED'):
        parts.append(f'## REVISED/{f.relative_to(d / "REVISED")}\n\n```latex\n{f.read_text(errors="ignore")[:60000]}\n```\n')
    fnd = (d / 'SLOP_FINDINGS.md').read_text(errors='ignore') if (d / 'SLOP_FINDINGS.md').exists() else ''
    prompt = body + '\n\n' + '\n'.join(parts) + '\n## SLOP_FINDINGS.md\n\n' + fnd
    (d / 'retire_prompt.txt').write_text(prompt)
    # The retire judgment is simple: it applies only to section pointers and evidence-gap observations. The model can be chosen separately (default: same as the reviewer).
    model = os.environ.get('SH_RETIRE_MODEL', '') or model
    res = H.run_cli_json(['-p', prompt, '--model', model, '--tools', '', '--system-prompt', REVIEW_SYSTEM] + _effort('SH_GATE_EFFORT'),
                         d, timeout=1200, env_extra=H.think_env('SH_RETIRE_THINK') or H.think_env('SH_GATE_THINK'))
    out = res.get('result') or ''
    (d / 'retire_stdout.txt').write_text(out + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    m = re.search(r'\[.*\]', re.sub(r'^```(?:json)?|```$', '', out.strip(), flags=re.M), re.S)
    try:
        rr = json.loads(m.group(0)) if m else []
    except Exception:
        rr = []
    (d / 'RETIRE.json').write_text(json.dumps(rr, indent=1, ensure_ascii=False))
    return res


def review_parallel(d: Path, changes: list[dict], model: str, chunk: int) -> dict:
    """Split the changes into groups of `chunk` and call the reviewer in parallel. Wall clock becomes the single longest chunk (0919 evening, target of 5 minutes per round).

    Every chunk sees the same staging directory (ORIGINAL, PREVIOUS, REVISED, materials) and judges only its own CHANGES.md.
    The retire judgment reads the whole manuscript, so only the first chunk does it. Verdicts go to chunk_k/VERDICTS.json and are merged here."""
    import threading
    parts = [changes[i:i + chunk] for i in range(0, len(changes), chunk)]
    n = len(parts)
    results = [None] * n

    text_mode = os.environ.get('SH_GATE_TEXT', '0') == '1'

    def one(k, part):
        if text_mode:
            results[k - 1] = review_text_chunk(d, part, k, n, model); return
        cd = d / f'chunk_{k}'
        cd.mkdir(exist_ok=True)
        md = [f'# Changes, part {k} of {n} ({len(part)} of {len(changes)} changes). Ids are those of ../CHANGES.md.\n']
        for c in part:
            md.append(f'## Change {c["id"]}. {c["file"]} ({c["op"]})\n\n### original\n\n{c["original"] or "(nothing; paragraph added)"}\n\n### revised\n\n{c["revised"] or "(nothing; paragraph deleted)"}\n')
        if os.environ.get('SH_GATE_INLINE', '1') == '1':
            md.append('\n# Context\n\n' + _context_block(d, part))
        (cd / 'CHANGES.md').write_text('\n'.join(md))
        res = H.run_cli_json(['-p', chunk_instruction(k, n, k == 1 and not split_retire), '--model', model, '--permission-mode', 'acceptEdits',
                              '--add-dir', str(d), '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit',
                              '--settings', str(GATE_SETTINGS), '--restricted'] + _max_turns('SH_GATE_MAX_TURNS')
                             + _effort('SH_GATE_EFFORT'), d, timeout=2400, env_extra=H.think_env('SH_GATE_THINK'))
        (cd / 'reviewer_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
        results[k - 1] = res

    split_retire = os.environ.get('SH_GATE_RETIRE_SPLIT', '1') == '1' and (d / 'SLOP_FINDINGS.md').exists()
    retire_res = {}

    def retire_only():
        if text_mode:
            retire_res['res'] = retire_text(d, model); return
        # The retire judgment reads the whole manuscript and was the long tail of the first chunk. Split it off to run concurrently.
        # In v4 this call took 8 minutes (repeated Read per section). Inline the REVISED full text and the location list into the instruction and cap the turns.
        body = retire_instruction()
        if os.environ.get('SH_GATE_INLINE', '1') == '1':
            parts = []
            for f in H.a1_files(d / 'REVISED'):
                parts.append(f'## REVISED/{f.relative_to(d / "REVISED")}\n\n```latex\n{f.read_text(errors="ignore")[:60000]}\n```\n')
            fnd = (d / 'SLOP_FINDINGS.md').read_text(errors='ignore') if (d / 'SLOP_FINDINGS.md').exists() else ''
            body += ('\n\nEverything you need is below, so you should not need to open files. The full REVISED manuscript follows, '
                     'then SLOP_FINDINGS.md.\n\n' + '\n'.join(parts) + '\n## SLOP_FINDINGS.md\n\n' + fnd)
        res = H.run_cli_json(['-p', body, '--model', model, '--permission-mode', 'acceptEdits',
                              '--add-dir', str(d), '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit',
                              '--settings', str(GATE_SETTINGS), '--restricted'] + _max_turns('SH_RETIRE_MAX_TURNS')
                             + _effort('SH_GATE_EFFORT'), d, timeout=2400, env_extra=H.think_env('SH_GATE_THINK'))
        (d / 'retire_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
        retire_res['res'] = res

    ts = [threading.Thread(target=one, args=(k + 1, part)) for k, part in enumerate(parts)]
    if split_retire:
        ts.append(threading.Thread(target=retire_only))
    for t in ts:
        t.start(); time.sleep(1.5)
    for t in ts:
        t.join()
    merged = []
    for k in range(1, n + 1):
        vf = d / f'chunk_{k}' / 'VERDICTS.json'
        if vf.exists():
            try:
                merged += [v for v in json.loads(re.sub(r'^```(?:json)?|```$', '', vf.read_text().strip(), flags=re.M)) if isinstance(v, dict)]
            except Exception as e:
                (d / f'chunk_{k}' / 'parse_error.txt').write_text(repr(e))
    (d / 'VERDICTS.json').write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    rs = [r for r in results if r] + ([retire_res['res']] if retire_res.get('res') else [])
    usage = {}
    for r in rs:
        for kk, v in (r.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[kk] = usage.get(kk, 0) + v
    return {'rc': max((r.get('rc') or 0) for r in rs) if rs else -1,
            'wall_s': round(max((r.get('wall_s') or 0) for r in rs), 1) if rs else None,
            'cost_usd': sum((r.get('cost_usd') or 0) for r in rs), 'num_turns': sum((r.get('num_turns') or 0) for r in rs),
            'usage': usage, 'is_error': any(r.get('is_error') for r in rs), 'limited': any(r.get('limited') for r in rs),
            'chunks': [{'k': i + 1, 'n_changes': len(parts[i]), 'wall_s': (r or {}).get('wall_s'), 'cost_usd': (r or {}).get('cost_usd'),
                        'num_turns': (r or {}).get('num_turns')} for i, r in enumerate(results)]
                       + ([{'k': 'retire', 'n_changes': 0, 'wall_s': retire_res['res'].get('wall_s'), 'cost_usd': retire_res['res'].get('cost_usd'),
                            'num_turns': retire_res['res'].get('num_turns')}] if retire_res.get('res') else [])}


def review(m0: Path, prev: Path, mr: Path, code: str, skill_path: Path, workdir: Path, model: str = GATE_MODEL) -> dict:
    """Changes = diff(prev, mr) (this round only); ground truth for claims = m0 (ORIGINAL/) and materials/."""
    changes = changes_between(prev, mr)
    rec = {'model': model, 'n_changes': len(changes), 'verdicts': [], 'call': None}
    if not changes:
        return rec
    d = stage_review(m0, prev, mr, code, skill_path, changes, workdir)
    chunk = int(os.environ.get('SH_GATE_CHUNK', '0') or 0)
    if chunk and (len(changes) > chunk or os.environ.get('SH_GATE_TEXT', '0') == '1'):
        res = review_parallel(d, changes, model, chunk)
    else:
        res = H.run_cli_json(['-p', REVIEW_INSTRUCTION, '--model', model, '--permission-mode', 'acceptEdits', '--add-dir', str(d),
                              '--disallowed-tools', 'WebSearch,WebFetch,Bash,Edit', '--settings', str(GATE_SETTINGS), '--restricted']
                             + _max_turns('SH_GATE_MAX_TURNS') + _effort('SH_GATE_EFFORT'), d, timeout=2400, env_extra=H.think_env('SH_GATE_THINK'))
        (d / 'reviewer_stdout.txt').write_text((res.get('result') or '') + '\n\n=== STDERR ===\n' + (res.get('stderr') or ''))
    rec['call'] = {k: res.get(k) for k in ('rc', 'wall_s', 'cost_usd', 'num_turns', 'usage', 'is_error', 'limited', 'chunks')}
    rf = d / 'RETIRE.json'
    rec['retire'] = []
    if rf.exists():
        try:
            rr = json.loads(re.sub(r'^```(?:json)?|```$', '', rf.read_text().strip(), flags=re.M))
            rec['retire'] = [x for x in rr if isinstance(x, dict) and x.get('instance')]
        except Exception as e:
            rec['retire_parse_error'] = repr(e)
    vf = d / 'VERDICTS.json'
    verdicts = []
    if vf.exists():
        try:
            verdicts = json.loads(re.sub(r'^```(?:json)?|```$', '', vf.read_text().strip(), flags=re.M))
        except Exception as e:
            rec['parse_error'] = repr(e)
    byid = {int(v.get('id', -1)): v for v in verdicts if isinstance(v, dict)}
    for c in changes:
        v = byid.get(c['id'])
        rec['verdicts'].append({'id': c['id'], 'file': c['file'], 'verdict': (v or {}).get('verdict', 'missing'), 'category': (v or {}).get('category'),
                                'reason': (v or {}).get('reason'), 'original': c['original'][:400], 'revised': c['revised'][:400]})
    rec['_changes'] = changes
    return rec


def apply_verdicts(prev: Path, mr: Path, rec: dict) -> dict:
    """Restore the previous-round paragraphs of every reverted (or unreviewed) change. Returns counts."""
    m0 = prev
    changes = rec.get('_changes') or []
    verdict = {v['id']: v['verdict'] for v in rec['verdicts']}
    reverted = [c for c in changes if verdict.get(c['id']) != 'keep']
    for c in [c for c in reverted if c.get('op') == 'image']:
        # Restore the image and the edit record to the previous round's. If the previous round has no record, restore the original figure.
        name = Path(c['file']).name
        src_img = m0 / 'figures' / name
        if src_img.exists():
            shutil.copy2(src_img, mr / 'figures' / name)
        src_rep = m0 / 'figures' / 'EDIT_REPORT.json'
        dst_rep = mr / 'figures' / 'EDIT_REPORT.json'
        if src_rep.exists():
            shutil.copy2(src_rep, dst_rep)
        elif dst_rep.exists():
            dst_rep.unlink()
    reverted_text = [c for c in reverted if c.get('op') != 'image']
    by_file = {}
    for c in reverted_text:
        by_file.setdefault(c['file'], []).append(c)
    for rel, cs in by_file.items():
        a = raw_paras((m0 / rel).read_text(errors='ignore')); b = raw_paras((mr / rel).read_text(errors='ignore'))
        # rebuild b from the opcodes, substituting original spans for reverted changes (indices refer to the same a, b)
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        rev_keys = {(c['i1'], c['i2'], c['j1'], c['j2']) for c in cs}
        out = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != 'equal' and (i1, i2, j1, j2) in rev_keys:
                out.append(''.join(a[i1:i2]))
            else:
                out.append(''.join(b[j1:j2]))
        (mr / rel).write_text(''.join(out))
    n_keep = sum(1 for v in rec['verdicts'] if v['verdict'] == 'keep')
    return {'kept': n_keep, 'reverted': len(reverted), 'reverted_ids': [c['id'] for c in reverted]}


def _norm_instance(t: str) -> str:
    """Normalize an instance line for matching. The reviewer copies it from the findings file, which may add a
    heading, a list marker, or the 'Present.' prefix of the manuscript-level observation."""
    t = re.sub(r'\s+', ' ', t or '').strip()
    t = re.sub(r'^#+\s*[^-]*?(?=-|$)', '', t).strip()      # a copied '## Heading' before the line
    t = t.lstrip('-').strip()
    t = re.sub(r'^(evidence\s*gap)\s*[:.-]\s*', '', t, flags=re.I)   # a copied entry label
    t = re.sub(r'^Present\.\s*', '', t)                    # evidence_gap is printed as '- Present. <observation>'
    return t


SECTION_INSTANCE = re.compile(r'^section\s+"', re.I)


def retire_in_scope(x: dict) -> bool:
    """Which retirement proposals the harness applies (0918 evening, from two failed variants).

    The reviewer is reliable about two kinds and unreliable about the rest. It correctly leaves alone a section,
    which another section rarely uses, and the manuscript-level evidence observation, which it settles by reading
    the project records. Asked about a figure, a table or a citing sentence it over-generalizes from one clumsy
    repair attempt and retires an instance that a later attempt does place naturally, so those proposals are
    recorded in the trace and not applied. The loop still converges on them, because the gate reverts every forced
    repair and the round ends as a gated no-op."""
    t = _norm_instance(x.get('instance', ''))
    return bool(SECTION_INSTANCE.match(t)) or t.lower().startswith('the paper has result tables') or t.lower().startswith('the manuscript has result tables')


def drop_retired(units: dict, rec: dict, retired: list) -> dict:
    """Remove instances the reviewer retired (this round and earlier) from the located-unit lists. A unit whose text
    changed since it was retired is kept, so a rewritten sentence is measured again."""
    keys = [_norm_instance(x['instance']) for x in retired if retire_in_scope(x)]
    keys = [k for k in keys if len(k) >= 25]
    out = {}
    for it, us in (units or {}).items():
        keep = []
        for u in us:
            n = _norm_instance(u)
            if any(n[:80] in k or k[:80] in n for k in keys):
                continue
            keep.append(u)
        out[it] = keep
    return out


def notes_for_editor(rec: dict) -> str | None:
    rv = [v for v in rec['verdicts'] if v['verdict'] != 'keep']
    if not rv and not rec.get('retire'):
        return None
    lines = ['Reviewer notes from the previous round. These changes were reverted to the original text; do not repeat them. '
             'Repair the same instances only in a way that stays within what the manuscript and its records already say.\n']
    for v in rv:
        lines.append(f'- {v["file"]} [{v.get("category")}]: {v.get("reason")}')
    applied = [x for x in (rec.get('retire') or []) if retire_in_scope(x)]
    if applied:
        lines.append('\nInstances the reviewer judged better left unrepaired. They have been removed from the list of '
                     'located instances and must not be repaired.')
        for x in applied:
            lines.append(f'- {x["instance"][:200]} : {x.get("reason")}')
    return '\n'.join(lines) + '\n'
