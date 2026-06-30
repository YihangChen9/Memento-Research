"""#138: the critic verdict parser must accept Markdown-bold PASS/REJECT,
and the on-disk fallback must match the real ``gate_review_stage{N}.md``
filename the critic actually writes.

Repro: Stage 1 produced a valid topic refinement, the critic wrote
``gate_review_stage1.md`` containing ``### 2. Decision: **PASS**``, but the
engine (a) globbed the inverted ``stage1_gate_review*.md`` so the file was
never read, and (b) couldn't parse the bolded label — verdict came back
ambiguous → defaulted to REJECT → 6 retries → ``stage_1_retries_exhausted``.
"""
from __future__ import annotations

import pytest

from onemancompany.core.pipeline_engine import PipelineEngine


@pytest.mark.parametrize(
    "text,expected",
    [
        # the literal content of the failing run's gate_review_stage1.md
        ("### 2. Decision: **PASS**", True),
        # bolded LABEL — the case that broke the separator match
        ("- **Decision**: PASS", True),
        ("**Decision**: PASS", True),
        # bolded VALUE
        ("Decision: **PASS**", True),
        # plain / table / mixed
        ("Decision: PASS", True),
        ("| Decision | PASS |", True),
        ("| **Decision** | **PASS** |", True),
        ("**Confidence**: 0.84 | **Decision**: PASS", True),
        ("Verdict: PASS", True),
        # REJECT variants
        ("**Decision**: REJECT", False),
        ("- **Decision**: REJECT", False),
        ("| Decision | REJECT |", False),
        # ambiguous — must NOT be coerced to a verdict (#19 guard)
        ("Some prose with no labeled verdict.", None),
        ("Auto-REJECT trigger check passed.", None),
        ("", None),
    ],
)
def test_verdict_from_text_handles_markdown_bold(text, expected):
    assert PipelineEngine._verdict_from_text(text) is expected


# ---------------------------------------------------------------------------
# Grounding rule (nature-skills #1): the critic may now classify a dimension
# "NOT ASSESSABLE" when it can't be verified from the provided material. This
# vocabulary must NOT be misread as a verdict, and the real Decision line must
# still parse cleanly when NOT ASSESSABLE lines are present in the body.
# ---------------------------------------------------------------------------

_PASS_BLOB = """\
Confidence: 0.88
Decision: PASS

Per-dimension scoring:
  D1 Research Question : PASS — clear and falsifiable
  D2 Hypotheses        : NOT ASSESSABLE — stage4 file silent on variables
  D3 Experimental Design : PASS — prose, internally consistent
"""

_REJECT_BLOB = """\
Confidence: 0.40
Decision: REJECT

Per-dimension scoring:
  D1 Research Question : NOT ASSESSABLE — could not locate the question
  D2 Hypotheses        : FAIL — no falsifiable H1
"""


@pytest.mark.parametrize(
    "text,expected",
    [
        (_PASS_BLOB, True),
        (_REJECT_BLOB, False),
        # NOT ASSESSABLE on its own is not a verdict signal (no PASS/REJECT)
        ("D2 Hypotheses : NOT ASSESSABLE — file missing", None),
    ],
)
def test_not_assessable_does_not_break_verdict_parse(text, expected):
    assert PipelineEngine._verdict_from_text(text) is expected


# ---------------------------------------------------------------------------
# Structured Findings block (nature-skills #2): the critic now appends a
# per-issue Findings list (id/severity/required_action). It must not corrupt
# verdict parsing — in particular, a finding whose required_action text
# happens to contain the word "reject" must NOT flip a PASS verdict.
# ---------------------------------------------------------------------------

_PASS_WITH_FINDINGS = """\
Confidence: 0.86
Decision: PASS

Per-dimension scoring:
  D1 Research Question : PASS — clear
  D6 Reproducibility   : FAIL — seed not pinned

Findings:
  - id: F1
    dimension: D6
    severity: minor
    problem: random seed is not pinned
    required_action: pin the seed; do not reject the null on an unseeded run
    evidence: stage5_experiment_designer.md §3
"""

_REJECT_WITH_FINDINGS = """\
Confidence: 0.42
Decision: REJECT

Findings:
  - id: F1
    dimension: D3
    severity: blocking
    problem: no control condition
    required_action: add a matched control arm
    evidence: missing
"""


@pytest.mark.parametrize(
    "text,expected",
    [
        # a "reject" buried in required_action must not override Decision: PASS
        (_PASS_WITH_FINDINGS, True),
        (_REJECT_WITH_FINDINGS, False),
        # a lone severity line is not a verdict
        ("  - severity: blocking", None),
    ],
)
def test_findings_block_does_not_break_verdict_parse(text, expected):
    assert PipelineEngine._verdict_from_text(text) is expected


# ---------------------------------------------------------------------------
# Findings gate (P2): _open_blocking_findings extracts ids the critic marked
# severity: blocking, so a PASS verdict carrying one can be overridden.
# ---------------------------------------------------------------------------

def test_open_blocking_findings_extracts_blocking_ids():
    text = (
        "Findings:\n"
        "  - id: F1\n    dimension: D3\n    severity: blocking\n"
        "  - id: F2\n    dimension: D6\n    severity: minor\n"
        "  - id: F3\n    dimension: D4\n    severity: blocking\n"
    )
    assert PipelineEngine._open_blocking_findings(text) == ["F1", "F3"]


def test_open_blocking_findings_ignores_template_placeholder():
    # The unfilled schema example uses the `a | b | c` form — must NOT trip.
    text = (
        "Findings:\n"
        "  - id: F1\n"
        "    severity: blocking | major | minor   # blocking = fails a hard-gate dim\n"
    )
    assert PipelineEngine._open_blocking_findings(text) == []


@pytest.mark.parametrize(
    "text",
    ["", "no findings here", "  - id: F1\n    severity: major\n"],
)
def test_open_blocking_findings_empty_when_no_blocker(text):
    assert PipelineEngine._open_blocking_findings(text) == []


def test_open_blocking_findings_does_not_trip_on_skill_template():
    """The literal `Findings:` schema — now centralised in ccf-review-core,
    and any residual reference in each *-quality-critic SKILL.md — must parse
    to zero blockers. The gate fires on real critic findings, never on the
    shipped documentation example (which uses the `a | b | c` placeholder)."""
    from pathlib import Path

    root = (
        Path(__file__).resolve().parents[3]
        / "src" / "onemancompany" / "default_skills"
    )
    for name in (
        "ccf-review-core",
        "methodology-quality-critic",
        "experiment-quality-critic",
        "result-quality-critic",
    ):
        text = (root / name / "SKILL.md").read_text(encoding="utf-8")
        assert PipelineEngine._open_blocking_findings(text) == [], name
