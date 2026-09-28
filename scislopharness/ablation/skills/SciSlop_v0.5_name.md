---
name: SciSlop
description: Names of six scientific slop patterns. Read this file before revising a research manuscript. Remove these patterns from the manuscript. Never invent evidence.
version: 0.5-ablation-name (2026-09-22; component ablation of the editor information. Derived from 0.5 by removing fields; every retained field is verbatim from 0.5)
location_cap: 40
location_mode: candidates
---

# SciSlop. Scientific slop and how to remove it

Scientific slop is AI-generated scientific content that presents the form of a complete study but lacks the connections needed to organize, substantiate, and explain the research. This file names six such patterns. Each entry gives the name of the pattern.

## Global rules

1. Do not fabricate experiments, numbers, citations, or examples. If evidence is missing, narrow the claim instead of inventing support.
2. Do not add citations that are not already in the manuscript, and do not remove cited works. Do not change reported numbers.
3. Keep the method, the experiments, and the reported results as they are. Repair how the manuscript organizes, substantiates, and presents them.
4. Prefer the smallest edit that removes the pattern. Do not paraphrase to disguise a pattern, do not delete a section or a citation command to make a count go away, and do not add text whose only purpose is to make a check pass. A repair must read as something the manuscript needed anyway.
5. No locations are listed. Read the manuscript and find the instances yourself using the names below. Repair an instance where the repair reads naturally. When the only available repair would add text that the argument does not need, leave that instance unchanged.

---

## Structure

### 1. Cross-section references

### 2. Macro redundancy

---

## Argument

### 3. Argument graph

### 4. Citation isolation

---

## Artifacts

### 5. Figure exposition

### 6. Evidence gap

---

## How this file is used

An editor or agent reads this file once, then revises the manuscript. After a revision, the manuscript is checked again. The procedure stops when nothing remains, when a revision changes nothing, or when a round limit is reached.
