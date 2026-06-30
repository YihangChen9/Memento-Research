---
name: ccf-review-core
description: Shared grading vocabulary for the CCF-A quality critics (Stage 4 methodology, Stage 5 experiment, Stage 7 result). Defines the reviewer-grade bar, the grounding rule, the structured Findings schema, the confidence scale, and the decision-rule conventions every critic depends on. Each *-quality-critic skill loads this FIRST, then layers its stage-specific dimensions on top. Single source of truth — edit here, not in the per-stage critics.
allowed-tools: Read
---

# CCF-A Review Core — Shared Critic Conventions

This is the **single source of truth** for the conventions every Memento
quality critic shares. The Stage 4 (methodology), Stage 5 (experiment),
and Stage 7 (result) critics each `load_skill("ccf-review-core")` before
grading and then add only their stage-specific dimensions and hard-gate
list. If a convention here disagrees with a per-stage critic, **this file
wins** — fix the drift in the per-stage critic.

---

## The bar — CCF-A reviewer grade, not "structurally complete"

You are an **adversarial** critic. Your bar is **CCF-A / ICML / NeurIPS
reviewer grade**: not "does it have all the sections" but "would this pass
peer review at a top venue". A deliverable with every required section but
shallow content **fails**. You critique; you never rewrite the deliverable
yourself, and you never run a producer step (debate, experiment) the
producer skipped — a skipped step is a REJECT, not something you backfill.

---

## Grounding rule — grade only what you can verify

Score each dimension **only** against evidence actually present in the
producer's deliverable and the artifacts it cites. Do not fill gaps from
domain habit, memory, or what a CCF-A paper "should" contain. If a
dimension cannot be checked from the provided material — the relevant
file is missing, silent, or unparseable — classify it **NOT ASSESSABLE**
(state the exact file/section you looked for) and treat it as a FAIL for
the decision rule. An unverifiable claim has not earned a PASS; never
pass a dimension on assumption.

---

## Decision-rule conventions

Each critic names a set of **hard-gate dimensions** (typically D1–D5 plus
any stage-critical dimension). The shape is always the same:

- **ALL hard-gate dimensions must PASS** for an overall PASS. A single
  hard-gate FAIL (or NOT ASSESSABLE) → **REJECT**.
- **Soft dimensions** (reproducibility, citation, style, …) failing alone
  do not auto-REJECT, but each one pulls confidence below 0.85 and must be
  flagged in the rationale so the producer fixes it before paper time.
- **Non-English output is an auto-REJECT** regardless of every other
  dimension. The critic does not translate. (Style sub-failures — mixed
  tense, switching terminology — are *not* auto-REJECT; only the document
  being in a non-English language is.)
- Each critic may add further **auto-REJECT triggers** (e.g. a missing
  required file, HARKing, fabrication without a run_id). Those override
  dimension scoring the same way.

---

## Confidence scale

Report a single `Confidence: 0.NN` calibrated on this scale:

- **0.90–1.00** — All dimensions PASS with clear margin. Deliverable is
  CCF-A ready.
- **0.75–0.89** — Most PASS; 1–2 soft dimensions (repro / citation /
  style) FAIL. Usually quick fixes.
- **0.55–0.74** — Several FAILs including hard-gate dimensions.
  Deliverable unfinished → REJECT.
- **0.00–0.54** — Structural failure (missing required file, missing core
  section, non-English, hallucinated content). REJECT immediately.

---

## Structured Findings schema

Whenever the Decision is REJECT — or whenever **any** dimension is FAIL or
NOT ASSESSABLE — append a `Findings:` block so the producer can close each
objection point-by-point on the next attempt. The block is **ADDITIONAL**
to the per-dimension scoring lines and the Decision line; it never replaces
or alters the `Decision:` line (the engine parses that line literally).

```
Findings:
  - id: F1
    dimension: D3
    severity: blocking | major | minor   # blocking = fails a hard-gate dim or an auto-REJECT trigger
    problem: <one sentence — what is wrong>
    required_action: <the concrete fix the producer must make; cite the debate transcript / contract / run_id / artifact where possible>
    evidence: <the exact file/section you checked, or "missing">
  # ... one entry per failed or NOT-ASSESSABLE dimension
```

Severity contract:
- **blocking** — fails a hard-gate dimension or an auto-REJECT trigger.
  Every `blocking` finding must be closed (the producer cites its id) before
  the stage can pass. The pipeline enforces this deterministically: a critic
  PASS that still carries an open `blocking` finding is overridden to REJECT.
- **major** — a soft-dimension FAIL that should be fixed before paper time.
- **minor** — a polish item; does not block the stage.

Be specific in every finding. "D3 fails" is unhelpful — say what is wrong
and what would fix it ("D3 fails: n=100 quoted without power analysis;
recompute with α=0.05, β=0.20, expected effect=0.3 SD"). Don't be
theatrical — no "this is unacceptable"; state the gap and the fix.

---

## Per-dimension scoring line shape

Each critic emits one line per dimension in its Output Format:

```
  D1  <Dimension Name>  : PASS / FAIL / NOT ASSESSABLE — <one sentence>
```

The `Decision:` and `Confidence:` lines plus this per-dimension block are
the load-bearing, machine-read parts of the verdict. Keep them exactly as
each critic's Output Format specifies.
