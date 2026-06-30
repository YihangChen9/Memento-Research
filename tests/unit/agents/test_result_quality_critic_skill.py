"""Structural tests for the Stage 7 result-quality-critic SKILL.

The critic grades stage7_result_analyst.md against the pre-registration
contract locked in Stage 4/5 and the actual evidence captured in Stage 6.
HARKing, fabrication, and non-English output are auto-REJECT regardless
of dimension scoring."""
from __future__ import annotations

from pathlib import Path


SKILLS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "onemancompany"
    / "default_skills"
)
RUNBOOK = SKILLS_ROOT / "result-quality-critic" / "SKILL.md"


class TestRunbookExists:
    def test_skill_folder_exists(self):
        assert RUNBOOK.parent.exists()

    def test_skill_md_exists(self):
        assert RUNBOOK.exists()

    def test_frontmatter_present(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
        head = text.split("---", 2)[1]
        assert "name: result-quality-critic" in head
        assert "description:" in head
        assert "allowed-tools:" in head

    def test_allowed_tools_include_read(self):
        """The critic is pure verification — only Read is needed; it must
        not be allowed to write or shell out."""
        text = RUNBOOK.read_text(encoding="utf-8")
        head = text.split("---", 2)[1]
        assert "Read" in head


class TestRunbookBehaviour:
    """The prose contract — these assertions catch silent drift if someone
    refactors the SKILL.md and loses the load-bearing checks."""

    def test_grades_against_pre_registration(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "pre-registration" in lowered or "pre-registered" in lowered, (
            "Stage 7 critic must explicitly grade Stage 7 against the "
            "Stage 4/5 pre-registration contract"
        )

    def test_lists_d1_through_d10(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        for i in range(1, 11):
            label = f"D{i} "
            assert label in text, (
                f"Stage 7 critic must define dimension {label.strip()} "
                f"explicitly"
            )

    def test_d1_is_contract_fidelity(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        # Find D1 and ensure "Contract Fidelity" appears in the same vicinity
        idx = text.find("D1")
        assert idx != -1, "D1 must be defined in the critic"
        # Look in the next ~200 chars for the Contract Fidelity label
        window = text[idx:idx + 400]
        assert "Contract Fidelity" in window, (
            "D1 must be the Contract Fidelity dimension — the load-bearing "
            "check against the pre-registration"
        )

    def test_three_auto_reject_triggers(self):
        """The decision rule must list HARKing, fabrication (no run_id),
        and non-English as auto-REJECT triggers regardless of dimension
        scoring."""
        text = RUNBOOK.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "hark" in lowered, "Must call out HARKing as auto-REJECT"
        # Fabrication is described as "without a real Stage 6 run_id"
        assert "fabrication" in lowered or "run_id" in text, (
            "Must call out fabrication (no run_id) as auto-REJECT"
        )
        assert "non-english" in lowered or "non english" in lowered, (
            "Must call out non-English output as auto-REJECT"
        )

    def test_d1_d5_are_hard_gates(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        # The phrasing in methodology/experiment critics is "D1, D2, D3, D4, D5
        # must PASS"; accept either that or "D1-D5 are hard gates" wording.
        assert (
            "D1, D2, D3, D4, D5 must PASS" in text
            or "D1-D5 are hard gates" in text
            or "D1–D5 are hard gates" in text
        ), (
            "Stage 7 critic must state that D1-D5 are hard gates / must "
            "PASS for an overall PASS"
        )

    def test_output_format_specified(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert "Output Format" in text, (
            "Stage 7 critic must specify an Output Format section so the "
            "gate review has a deterministic shape"
        )

    def test_decision_rule_specified(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert "Decision Rule" in text, (
            "Stage 7 critic must specify a Decision Rule section so the "
            "PASS/REJECT logic is unambiguous"
        )


class TestGroundingRule:
    """nature-skills #1 (P1 refactor): the verbatim grounding rule lives in
    ``ccf-review-core``. The Stage 7 critic loads it and keeps the
    result-specific note (RESULT_JSON, data-gate companion)."""

    def test_loads_core_and_references_grounding(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert 'load_skill("ccf-review-core")' in text
        assert "grounding" in text.lower()
        assert "NOT ASSESSABLE" in text

    def test_keeps_data_gate_companion_note(self):
        flat = " ".join(RUNBOOK.read_text(encoding="utf-8").split())
        assert "data gate" in flat.lower()


class TestStructuredFindings:
    """nature-skills #2 (P1 refactor): the Findings schema lives in
    ``ccf-review-core``; the Stage 7 critic points at it and keeps its own
    machine-read Decision line."""

    def test_references_core_findings_schema(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert "ccf-review-core" in text
        assert "Findings block" in text or "Findings:" in text

    def test_blocking_must_be_closed(self):
        flat = " ".join(RUNBOOK.read_text(encoding="utf-8").split())
        assert "blocking" in flat
        assert "must be closed" in flat

    def test_decision_line_preserved(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        assert "Decision: PASS" in text


class TestRunbookOnboardingWiring:
    """Cross-check that the critic runbook is wired into onboarding for
    the adversarial_review skill."""

    def test_listed_in_skill_required_runbooks_for_adversarial_review(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        runbooks = _SKILL_REQUIRED_RUNBOOKS.get("adversarial_review", [])
        assert "result-quality-critic" in runbooks
