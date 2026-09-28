---
name: SciSlop
description: Definitions of six scientific slop patterns, each with the format in which located instances are reported. Read this file before revising a research manuscript. Fix what the definitions describe. Never invent evidence.
version: 0.5-ablation-defloc (2026-09-22; component ablation of the editor information. Derived from 0.5 by removing fields; every retained field is verbatim from 0.5)
location_cap: 40
location_mode: candidates
---

# SciSlop. Scientific slop and how to remove it

Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives what the pattern is, when a unit of the manuscript counts as an instance, and how instances are reported to you when an external check has located them.

## Global rules

1. Do not fabricate experiments, numbers, citations, or examples. If evidence is missing, narrow the claim instead of inventing support.
2. Do not add citations that are not already in the manuscript, and do not remove cited works. Do not change reported numbers.
3. Keep the method, the experiments, and the reported results as they are. Repair how the manuscript organizes, substantiates, and presents them.
4. Prefer the smallest edit that removes the pattern. Do not paraphrase to disguise a pattern, do not delete a section or a citation command to make a count go away, and do not add text whose only purpose is to make a check pass. A repair must read as something the manuscript needed anyway.
5. When instances are listed with locations, they are the places an external check found the pattern, not orders to change every one of them. Consider each listed instance. Repair it where the repair reads naturally. When the only available repair would add text that the argument does not need, leave that instance unchanged. When no locations are listed, read the manuscript and find the instances yourself using the definitions below.

---

## Structure

### 1. Cross-section references

**Definition.** A manuscript declares objects. Its body sections and the labelled figures, tables, equations, and algorithms inside them. An object is unused when no section other than the one that contains it ever refers to it, so the rest of the manuscript never builds on it.

**Units as reported.** Unused objects, each given as label, kind, and home section.

### 2. Macro redundancy

**Definition.** A sentence is recycled when at least half of it repeats, almost word for word, text that an earlier, different section already contained.

**Units as reported.** Recycled sentences, each given as section, sentence, and source section.

---

## Argument

### 3. Argument graph

**Definition.** A key claim of the Introduction is a sentence asserting superiority, a prior limitation, or a design choice. The claim's strongest contextual cue is the Introduction sentence that most raises the claim's plausibility. The claim is declared rather than argued when that cue occurs after the claim, so the Introduction states the conclusion before building toward it.

**Units as reported.** Declared claims, each given as the claim sentence and the cue sentence that currently follows it.

### 4. Citation isolation

**Definition.** A citing sentence is isolated when it cites a single prior work and does not put that work in relation to another work that the same sentence also cites. Naming a method without citing it does not count, so the relation must hold between two citations. A related-work section made of isolated sentences lists prior work instead of positioning it.

**Units as reported.** Isolated citing sentences, each given as section and sentence.

---

## Artifacts

### 5. Figure exposition

**Definition.** A method diagram should depict the mechanism. Its components and how they connect. A box, panel, or caption element is expository when it carries material that belongs to the text rather than the mechanism. Experimental settings, dataset names, numeric results, interpretive claims, a legend that restates the prose, or a step-by-step narration of the pipeline.

**Units as reported.** Expository elements, each given as figure, element type, and the transcribed text of the element.

### 6. Evidence gap

**Definition.** The manuscript reports aggregate results but never displays a single concrete instance of what it counts. An example input and output, a case, a failure, a worked example, or a quoted specimen. The gap is silent when the manuscript also never says that no instance is shown; a silent gap lets the reader assume that individual cases were inspected.

**Units as reported.** A single manuscript-level observation. Silent, acknowledged, or exhibited.

---

## How this file is used

An editor or agent reads this file once, then revises the manuscript. Optionally, an external check reports located instances in the format given under each entry. After a revision, the manuscript is checked again and only remaining instances are reported. The procedure stops when nothing remains, when a revision changes nothing, or when a round limit is reached.
