"""Structural tests for the shared ``ccf-review-core`` skill.

P1 (nature-skills ``_shared/core/`` pattern): the three CCF-A quality
critics (Stage 4 methodology, Stage 5 experiment, Stage 7 result) no longer
each carry their own copy of the grounding rule / Findings schema /
confidence scale / decision conventions. That vocabulary now lives **once**
here, and each critic loads it via ``load_skill("ccf-review-core")``. These
tests pin the canonical text so the single source of truth can't silently
rot, and so the engine findings-gate template guard has a stable target.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from onemancompany.core.pipeline_engine import PipelineEngine


SKILLS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src" / "onemancompany" / "default_skills"
)
CORE = SKILLS_ROOT / "ccf-review-core" / "SKILL.md"


class TestCoreExistsAndFrontmatter:
    def test_skill_md_exists(self):
        assert CORE.exists()

    def test_frontmatter_names_skill(self):
        text = CORE.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        head = text.split("---", 2)[1]
        assert "name: ccf-review-core" in head
        assert "description:" in head
        # Pure verification — Read only, no write/shell.
        assert "allowed-tools: Read" in head


class TestGroundingRuleCanonical:
    """The grounding rule (nature-skills #1) lives here verbatim now."""

    def test_has_grounding_rule(self):
        text = CORE.read_text(encoding="utf-8")
        assert "Grounding rule" in text
        assert "NOT ASSESSABLE" in text

    def test_treats_unverifiable_as_fail(self):
        flat = " ".join(CORE.read_text(encoding="utf-8").split())
        assert "treat it as a FAIL" in flat
        assert "never pass a dimension on assumption" in flat


class TestFindingsSchemaCanonical:
    """The structured Findings schema (nature-skills #2) lives here now."""

    def test_has_findings_block(self):
        text = CORE.read_text(encoding="utf-8")
        assert "Findings:" in text
        for field in ("id:", "severity:", "required_action:", "evidence:"):
            assert field in text, field

    def test_blocking_severity_contract(self):
        flat = " ".join(CORE.read_text(encoding="utf-8").split())
        assert "blocking" in flat
        assert "must be closed" in flat
        # The block is additive — it must not be described as replacing the
        # machine-read Decision line.
        assert "never replaces" in flat or "never alters" in flat

    def test_findings_template_does_not_trip_the_engine_gate(self):
        """The shipped schema example must parse to ZERO open blockers — the
        deterministic gate fires on real critic findings, never on this
        documentation template (it uses the ``a | b | c`` placeholder form)."""
        text = CORE.read_text(encoding="utf-8")
        assert PipelineEngine._open_blocking_findings(text) == []


class TestSharedConventions:
    def test_has_confidence_scale(self):
        text = CORE.read_text(encoding="utf-8")
        assert "Confidence scale" in text
        for band in ("0.90", "0.75", "0.55", "0.00"):
            assert band in text, band

    def test_non_english_auto_reject_convention(self):
        flat = " ".join(CORE.read_text(encoding="utf-8").split())
        assert "Non-English" in flat or "non-English" in flat
        assert "auto-REJECT" in flat

    def test_reviewer_grade_bar(self):
        text = CORE.read_text(encoding="utf-8")
        assert "CCF-A" in text
        # Bar is "would this pass peer review", not "structurally complete".
        assert "peer review" in text


class TestWiredIntoOnboarding:
    def test_core_injected_with_the_critics(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        runbooks = _SKILL_REQUIRED_RUNBOOKS.get("adversarial_review", [])
        assert "ccf-review-core" in runbooks
        # Must come before / alongside the critics that load it.
        for critic in (
            "methodology-quality-critic",
            "experiment-quality-critic",
            "result-quality-critic",
        ):
            assert critic in runbooks
