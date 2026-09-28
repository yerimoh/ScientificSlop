---
name: SciSlop
description: Definitions of six scientific slop patterns, each with the direction of a repair. Read this file before revising a research manuscript. Fix what the definitions describe. Never invent evidence.
version: 0.5 (2026-09-20; evidence gap gains a second repair, acknowledging the gap when the record holds no instance, so that the item no longer needs a specimen to move; 0.4 was 2026-09-19; figure exposition may be repaired by redrawing a raster method figure as a source figure, so the item is no longer frozen; 0.3 was the same day; citation definition aligned with the measurer so that a relation counts only between two citations, location cap raised to 40; 0.2-reasonable was 2026-09-18; same definitions as 0.1; fix wording asks for placement where the text uses the object, and located instances are candidates rather than orders)
location_cap: 40
location_mode: candidates
---

# SciSlop. Scientific slop and how to remove it

Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives what the pattern is, when a unit of the manuscript counts as an instance, in which direction to repair it, and how instances are reported to you when an external check has located them.

## Global rules

1. Do not fabricate experiments, numbers, citations, or examples. If evidence is missing, narrow the claim instead of inventing support.
2. Do not add citations that are not already in the manuscript, and do not remove cited works. Do not change reported numbers.
3. Keep the method, the experiments, and the reported results as they are. Repair how the manuscript organizes, substantiates, and presents them.
4. Prefer the smallest edit that removes the pattern. Do not paraphrase to disguise a pattern, do not delete a section or a citation command to make a count go away, and do not add text whose only purpose is to make a check pass. A repair must read as something the manuscript needed anyway.
5. When instances are listed with locations, they are the places an external check found the pattern, not orders to change every one of them. Consider each listed instance. Repair it where the repair reads naturally. When the only available repair would add text that the argument does not need, leave that instance unchanged. When no locations are listed, read the manuscript and find the instances yourself using the definitions below.

---

## Structure

### 1. Cross-section references (objects never used outside their own section)

**Definition.** A manuscript declares objects. Its body sections and the labelled figures, tables, equations, and algorithms inside them. An object is unused when no section other than the one that contains it ever refers to it, so the rest of the manuscript never builds on it.

**Fix.** Refer to an object from a sentence that uses it. A sentence that states what the table shows, reads a number off it, applies the equation, or draws the conclusion the figure supports. Put that sentence where the reader needs the object, which is usually where the result is interpreted or the method is applied, and not in a summary that lists everything. One reference from one sentence that genuinely uses the object is enough. Do not collect several pointers into one sentence, do not write a sentence whose only content is pointers, and do not add a pointer to a section merely because the section exists. If no other part of the manuscript actually depends on the object, leave it unreferenced. An object used only where it is defined is not by itself a defect.

**Units as reported.** Unused objects, each given as label, kind, and home section.

### 2. Macro redundancy (sentences recycled from an earlier section)

**Definition.** A sentence is recycled when at least half of it repeats, almost word for word, text that an earlier, different section already contained.

**Fix.** Rewrite each recycled sentence so that it adds what its own section needs. A new detail, a consequence, or a qualification. Remove it if the section does not need it. Do not paraphrase merely to disguise the repetition.

**Units as reported.** Recycled sentences, each given as section, sentence, and source section.

---

## Argument

### 3. Argument graph (claims stated before the context that supports them)

**Definition.** A key claim of the Introduction is a sentence asserting superiority, a prior limitation, or a design choice. The claim's strongest contextual cue is the Introduction sentence that most raises the claim's plausibility. The claim is declared rather than argued when that cue occurs after the claim, so the Introduction states the conclusion before building toward it.

**Fix.** Reorder so that the context precedes the claim it supports. Move the limitation of prior work, the observation, or the design constraint ahead of the sentence that asserts the contribution, or move the assertion down to follow it. Do not delete claims and do not add new claims. Do not weaken a claim to avoid arguing for it.

**Units as reported.** Declared claims, each given as the claim sentence and the cue sentence that currently follows it.

### 4. Citation (isolated citations in the Introduction and Related Work)

**Definition.** A citing sentence is isolated when it cites a single prior work and does not put that work in relation to another work that the same sentence also cites. Naming a method without citing it does not count, so the relation must hold between two citations. A related-work section made of isolated sentences lists prior work instead of positioning it.

**Fix.** Where the manuscript's positioning depends on a cited work, state how it relates to another work the manuscript already cites, and cite both in the same sentence. A shared limitation, a difference in assumption or method, or what this manuscript takes from each. The relation must be one the manuscript already supports, drawn from what it already says about each work, and naming a method without its citation does not create the relation. Only use works already cited in the manuscript. Do not add new citations, do not drop existing ones, and do not merge sentences whose only purpose is to sit together.

**Units as reported.** Isolated citing sentences, each given as section and sentence.

---

## Artifacts

### 5. Figure exposition (method diagrams that explain instead of depict)

**Definition.** A method diagram should depict the mechanism. Its components and how they connect. A box, panel, or caption element is expository when it carries material that belongs to the text rather than the mechanism. Experimental settings, dataset names, numeric results, interpretive claims, a legend that restates the prose, or a step-by-step narration of the pipeline.

**Fix.** Remove the expository material from the method diagram and, where the reader needs it, state it once in the text near the figure. Keep only components of the method and their connections. If the figure has an editable source, edit that source. If the figure is a raster image, you may redraw the same mechanism as a TikZ figure in place of the image, keeping every component and connection the image shows and dropping only the expository material, and keeping the same label so that every reference to it still resolves. Redraw only after looking at the image itself. Remove exactly the elements listed as expository and keep every other element and every arrow the image has, including a number that labels a component. An element the check did not list is not yours to remove. Do not invent components the figure does not show, do not simplify the mechanism to make the drawing easier, and do not drop the figure. A figure that loses a component of the method is worse than one that carries a caption too many. Settings, results and claims taken out of the figure go into the text, not away.

**Units as reported.** Expository elements, each given as figure, element type, and the transcribed text of the element.

### 6. Evidence gap (no concrete instance is displayed)

**Definition.** The manuscript reports aggregate results but never displays a single concrete instance of what it counts. An example input and output, a case, a failure, a worked example, or a quoted specimen. The gap is silent when the manuscript also never says that no instance is shown; a silent gap lets the reader assume that individual cases were inspected.

**Fix.** Two repairs, in this order. (a) If the project's own materials contain such an instance (materials/SPECIMEN.md names it when one exists), display it where the reader needs it, quote it exactly as it appears in the record, name the file it came from, and refer to it from the text. (b) If no instance is available, make the gap explicit instead of leaving it silent. In the sentence or paragraph where the aggregate result is interpreted, state plainly that no individual case was retained or inspected and that the claims rest on aggregate scores only, and narrow any sentence that describes how individual cases behave (a mechanism, a failure pattern, "we observe that") to what the aggregate supports. One such statement is enough. Do not invent, alter, round, or embellish an instance, and do not write an example that was never produced.

**Units as reported.** A single manuscript-level observation. Silent, acknowledged, or exhibited.

---

## How this file is used

An editor or agent reads this file once, then revises the manuscript. Optionally, an external check reports located instances in the format given under each entry. After a revision, the manuscript is checked again and only remaining instances are reported. The procedure stops when nothing remains, when a revision changes nothing, or when a round limit is reached.
