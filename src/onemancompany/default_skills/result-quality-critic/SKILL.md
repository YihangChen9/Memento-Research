---
name: result-quality-critic
description: Grades the Stage 7 Result Analysis against the pre-registration locked in Stage 4/5 and the actual evidence captured in Stage 6. The contract is immutable — confirmatory analysis must use the locked tests and metrics, label exploratory claims separately, never HARK, and never conclude beyond Stage 6 coverage. Use this skill at every Stage 7 gate review.
allowed-tools: Read
---

# Stage 7 Quality Critic — Pre-Registration & Coverage Contract

You are reviewing `stage7_result_analyst.md` against the pre-registration
locked in Stage 4/5 and the actual coverage delivered in Stage 6. Pull
the four artifacts before scoring:

```
read("stage4_methodology_designer.md")
read("stage5_experiment_designer.md")
read("stage5_assignments.md")
read("stage6_experimentalist.md")
read("stage7_result_analyst.md")
```

## What You Are Grading

Stage 7 is where confirmatory analysis happens. The single most
common failure mode in published research is **HARKing**
(Hypothesising After Results are Known) — adjusting hypotheses or
tests post-hoc to fit the data. Your job is to make HARKing
impossible to slip past this gate.

Score 10 dimensions. **D1-D5 are hard gates** (any FAIL → REJECT
overall). D6-D10 reduce confidence but do not by themselves auto-
reject.

### D1 — Contract Fidelity
**This is the load-bearing check.** Compare Stage 7's Section 1
("Contract") with the pre-registered tests and thresholds in
`stage4_methodology_designer.md` and `stage5_experiment_designer.md`.

- ✅ Every pre-registered hypothesis appears in Stage 7's contract table.
- ✅ The test name for each hypothesis matches the Stage 4/5 wording
  exactly (e.g. "Page's L trend test", "GLMM with random intercepts
  for Problem and Model"). Verbatim, not paraphrased.
- ✅ The effect-size measure matches what was pre-registered.
- ✅ The α level and decision rule match Stage 4 D4 (and any multiple-
  comparison correction matches Stage 4's locked procedure).
- ❌ Any deviation from the pre-registered test = REJECT. Stage 7
  cannot upgrade, downgrade, or swap a test.
- ❌ A pre-registered hypothesis is silently dropped → REJECT.
- ❌ Stage 7 invents a new hypothesis or test absent from Stage 4/5 →
  REJECT (this is HARKing).

### D2 — Evidence Provenance
For every claim in Stage 7's confirmatory section:

- ✅ A specific Stage 6 `run_id` is cited.
- ✅ The cited run had terminal status `succeeded` (not `failed`,
  `rejected`, or only `submitted`).
- ✅ The metric value used is traceable back to Stage 6's captured
  metrics or log_tail excerpt (not invented).
- ❌ A confirmatory claim without a `run_id` → REJECT.
- ❌ A claim that "succeeded" runs reported `actual_cost: 0` AND
  `finished_at` < 30 seconds after `created_at` AND the log_tail
  contains an error message → that run was dead-on-arrival;
  treating it as evidence → REJECT.

### D3 — Effect Sizes & Confidence Intervals
- ✅ Every confirmatory decision is supported by an effect size + 95%
  CI, not a bare p-value.
- ✅ Confidence intervals use the metric specified in the pre-
  registration (raw difference / Cohen's h / Pearson r / slope etc.).
- ❌ Decisions resting on p-values alone, or on point estimates with
  no uncertainty quantification → REJECT.

### D4 — Manipulation Check
- ✅ Section 4 reports the observed manipulation-check value vs the
  pre-registered threshold.
- ✅ If the manipulation check failed, **every downstream hypothesis
  is correctly downgraded to INCONCLUSIVE** in Sections 3 and 9.
- ❌ A failed manipulation check that still leaves hypotheses
  reported as SUPPORTED → REJECT.

### D5 — Falsification Check
- ✅ Section 5 reports the falsification-check observed value vs the
  pre-registered bound, exactly as Stage 4/5 specified.
- ✅ If the falsification check fired, the primary hypothesis is
  reinterpreted per Stage 4's locked rule (e.g. "H1 reinterpreted as
  trade-off, per Stage 4 lock") — not silently dropped, not
  silently passed.
- ❌ Falsification check missing or its consequences ignored → REJECT.

### D6 — Coverage-Calibrated Verdict
- ✅ For every BLOCKED row in Stage 6's report, the corresponding
  hypothesis is marked NOT TESTED in Stage 7 — and the overall
  verdict in Section 9 is no stronger than `PARTIALLY CONFIRMED`
  unless every hypothesis got full coverage.
- ✅ "INCONCLUSIVE_DUE_TO_COVERAGE" is used when Stage 6 was BLOCKED
  for the load-bearing hypothesis.
- ❌ Stage 7 claiming CONFIRMED when Stage 6's verdict was PARTIAL on
  the matching hypothesis → confidence drop.

### D7 — Confirmatory / Exploratory Separation
- ✅ Section 3 (confirmatory) contains only pre-registered tests.
- ✅ Section 7 (exploratory) contains everything not pre-registered,
  with each item clearly labelled "exploratory".
- ❌ Mixing exploratory observations into Section 3, or omitting
  Section 7 while making post-hoc observations elsewhere → confidence
  drop.

### D8 — Sensitivity Robustness
- ✅ Pre-registered sensitivity analyses are reported with their
  deltas.
- ✅ Verdicts state robustness explicitly (e.g. "primary result
  unchanged ±0.5pp under alternative canonicalisation").
- ❌ Pre-registered sensitivity skipped → confidence drop.

### D9 — Provenance Citations
- ✅ Section 10 lists artifact citations linking every Section-3
  numeric claim back to a specific file + section in Stage 4-6
  artifacts.
- ❌ Section 10 missing or generic ("see Stage 6") → confidence drop.

### D10 — Language & Style
- ✅ English, academic register, terminology consistent with
  Stage 4-5. Notation matches Stage 4's lock.
- ❌ Non-English document → auto-REJECT.

### D11 — Result Figures
- ✅ At least one ``stage7_*.png`` figure exists in the project workspace
  AND is embedded in the report with a caption (``![Figure N: …](stage7_….png)``)
  AND the figure's values match the numbers in the Section-3 confirmatory tables.
- ❌ Numeric confirmatory results present but NO figures and no explicit
  "rendering impossible" note in §8 Limitations → confidence drop.
- ❌ A figure whose values contradict the tables → treat as a D2
  Evidence Provenance FAIL (figure is a view of the data, not a new
  analysis; value mismatch = fabrication risk).
- NOTE: the result-analysis-runbook Phase 5.5 instructs the analyst to
  render figures via the ``result-figures`` skill. If figures are absent,
  check whether Phase 5.5 was skipped before raising confidence.

## How to Run the Review

0. **Load the shared conventions first.** Call
   `load_skill("ccf-review-core")` and apply it — it defines the
   **grounding rule**, the **structured Findings schema**, the
   **confidence scale**, and the **decision-rule conventions** this review
   depends on. Skipping it makes the review invalid.
1. Read all five artifacts (Stage 4, 5, 5-assignments, 6, 7).
2. Reconstruct the pre-registration contract from Stage 4/5 (do NOT
   trust Stage 7's contract table; verify it against the source).
3. Walk D1-D11. For each, classify PASS / FAIL / NOT ASSESSABLE (per the
   core grounding rule) with a one-sentence justification.
4. Decide PASS / REJECT. State the failing dimension(s) on REJECT.

Grounding note: the core **grounding rule** applies to the Stage 4/5/6/7
files and RESULT_JSON — a dimension you cannot verify from them is NOT
ASSESSABLE (= FAIL), never a silent pass. This is the qualitative-judgment
companion to the deterministic data gate.

## Output Format

```
**Gate Review Complete — Stage 7 (Result Analysis)**

**Decision: PASS** (or **REJECT**)
**Confidence: 0.NN**

Per-dimension scoring:
  D1 Contract Fidelity            : PASS / FAIL — <one sentence>
  D2 Evidence Provenance          : PASS / FAIL — <one sentence>
  D3 Effect Sizes & CIs           : PASS / FAIL — <one sentence>
  D4 Manipulation Check           : PASS / FAIL — <one sentence>
  D5 Falsification Check          : PASS / FAIL — <one sentence>
  D6 Coverage-Calibrated Verdict  : PASS / FAIL — <one sentence>
  D7 Confirmatory/Exploratory Sep : PASS / FAIL — <one sentence>
  D8 Sensitivity Robustness       : PASS / FAIL — <one sentence>
  D9 Provenance Citations         : PASS / FAIL — <one sentence>
  D10 Language & Style            : PASS / FAIL — <one sentence>
  D11 Result Figures              : PASS / FAIL — <one sentence>

Rationale: <2-4 sentences summarising the verdict and pointing the
producer at any failing dimension>

[Findings block here — emit exactly as specified in `ccf-review-core`
 when REJECT, or whenever any dimension is FAIL / NOT ASSESSABLE. One
 entry per failed / NOT-ASSESSABLE dimension; `severity: blocking` for any
 D1-D5 (hard-gate) or auto-REJECT failure. ADDITIONAL to the lines above;
 never alters the Decision line. Every `blocking` finding must be closed
 (cite its id) before the stage can pass.]
```

## Decision Rule

ALL of D1, D2, D3, D4, D5 must PASS for an overall PASS. D6-D10
failures alone are not auto-REJECT but pull confidence below 0.85.

**Three auto-REJECT triggers regardless of dimensions**:
1. Any test in Stage 7 Section 3 that is not present verbatim in
   Stage 4/5 pre-registration (HARKing).
2. Any confirmatory claim without a real Stage 6 run_id (fabrication).
3. D10 caused by non-English output.

## What You Are NOT Doing

- **Not re-running the analysis.** You verify Stage 7 obeys the
  contract; you don't replicate the math.
- **Not lowering the bar for negative results.** A rejected
  hypothesis with proper test + effect size + CI is a clean result,
  not a failure to grade.
- **Not blocking on exploratory weakness.** Exploratory observations
  (Section 7) are not graded for rigour — they're graded only for
  being correctly labelled as exploratory.

## Key Principles

- **The pre-registration is the contract.** Stage 4 and Stage 5 set
  the terms. Stage 7's job is to honour them, not to negotiate.
- **Coverage is a ceiling, not a floor.** Stage 7 cannot claim
  certainty beyond what Stage 6 actually delivered.
- **HARKing is the worst failure mode.** Auto-REJECT, every time.
- **Effect sizes > p-values.** A p-value alone is not a conclusion.
