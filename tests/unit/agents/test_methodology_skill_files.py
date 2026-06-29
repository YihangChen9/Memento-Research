"""Sanity tests for the two methodology-related SKILL.md files.

These don't try to validate prose quality — they check that required
sections / keywords exist so that accidental deletion or refactor of the
file is caught immediately."""
from __future__ import annotations

from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parents[3] / "src" / "onemancompany" / "default_skills"
QUALITY_CRITIC = SKILLS_ROOT / "methodology-quality-critic" / "SKILL.md"

# Note: methodology-debate-convener no longer lives in this repo. It is
# hosted as a separate Talent Market repo at
# https://github.com/YihangChen9/methodology-designer (cloned at hire time
# via hire_list.json source_repo). Tests that previously asserted its
# contents were removed alongside the deleted default_skills/ entry — they
# now belong in the talent repo itself.


# ---------------------------------------------------------------------------
# methodology-quality-critic — CCF-A grading
# ---------------------------------------------------------------------------

class TestQualityCriticSkillStructure:
    def test_file_exists(self):
        assert QUALITY_CRITIC.exists()

    def test_lists_eight_grading_dimensions(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        for dim_label in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"):
            assert dim_label in text

    def test_requires_transcript_check(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "transcript" in text.lower()
        assert "stage4_debate_transcript.md" in text
        # Critic must reject if transcript is missing
        assert "REJECT" in text and "debate not run" in text

    def test_specifies_output_format_with_decision(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "Confidence:" in text
        assert "Decision: PASS | REJECT" in text or "Decision: PASS" in text

    def test_decision_rule_requires_d1_through_d5(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # Decision rule: D1-D5 must PASS for overall PASS
        assert "D1, D2, D3, D4, D5" in text

    def test_mentions_ccf_a_or_icml_neurips_bar(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "CCF-A" in text or "NeurIPS" in text or "ICML" in text


# ---------------------------------------------------------------------------
# D9 — Language & Style enforcement (English default)
# ---------------------------------------------------------------------------

class TestEnglishDefaultAndLanguageDimension:
    def test_critic_has_d9_language_and_style(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "D9" in text, "Critic must include a D9 dimension for language/style"
        assert "Language" in text and "Style" in text

    def test_critic_d9_checks_english(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # D9 must demand the document be in English
        lower = text.lower()
        assert "english" in lower

    def test_critic_d9_is_not_auto_reject(self):
        """D9 failure should pull confidence but not auto-REJECT, matching
        the existing decision rule pattern for D6/D7/D8."""
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # Decision rule line should mention D9 alongside D6/D7/D8 (not in D1-D5)
        assert "D9" in text
        # The decision rule still names D1-D5 as the hard gate
        assert "D1, D2, D3, D4, D5" in text


# ---------------------------------------------------------------------------
# Stage 5 SKILL files — experiment-debate-convener + experiment-quality-critic
# ---------------------------------------------------------------------------

# experiment-debate-convener was moved to the talent repo at
# https://github.com/YihangChen9/experiment-designer; its content tests
# live in the talent repo now.
EXP_CRITIC = SKILLS_ROOT / "experiment-quality-critic" / "SKILL.md"



class TestExperimentCriticSkill:
    def test_file_exists(self):
        assert EXP_CRITIC.exists()

    def test_has_12_dimensions(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        for label in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12"):
            assert label in text

    def test_d10_is_coordination_plan(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        assert "D10" in text and "Coordination Plan" in text

    def test_d12_checks_english(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        assert "D12" in text
        assert "English" in text

    def test_requires_all_three_files(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        for path in (
            "stage5_experiment_designer.md",
            "stage5_assignments.md",
            "stage5_debate_transcript.md",
        ):
            assert path in text

    def test_decision_rule_includes_d10_non_negotiable(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        # D10 must be in the hard-gate list
        assert "D1, D2, D3, D4, D5, D8, D10" in text or "D10" in text
        assert "non-negotiable" in text.lower() or "auto-REJECT" in text

    def test_specifies_output_format_with_decision(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        assert "Confidence:" in text
        assert "Decision: PASS" in text



# ---------------------------------------------------------------------------
# D10 — Framework Figure (every CCF-A methodology must have a figure)
# ---------------------------------------------------------------------------

class TestCriticD10FrameworkFigure:
    """Every Stage 4 methodology must ship with a framework figure
    rendered via nano banana. The critic enforces this as a hard gate
    (auto-REJECT) on D10, equivalent to D1-D5."""

    def test_critic_lists_d10_dimension(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "D10" in text
        assert "Framework Figure" in text

    def test_critic_d10_is_hard_gate(self):
        """D10 must be in the decision rule alongside D1-D5 (not in the
        soft D6/D7/D8/D9 bucket). Without D10 hard gating, a producer
        can submit a methodology without a figure and PASS critic."""
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # The decision rule line names D10 as hard
        assert "D1, D2, D3, D4, D5, D10" in text

    def test_critic_warns_about_missing_figure_file(self):
        """If stage4_framework_figure.png is missing or not referenced,
        the critic must REJECT (not just dock confidence)."""
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # Normalise whitespace so the assertion survives soft line breaks
        # in the SKILL.md prose.
        flat = " ".join(text.split())
        assert "stage4_framework_figure.png" in flat
        assert "REJECT" in flat
        assert "framework figure missing" in flat

    def test_critic_requires_numbered_caption(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        # The caption must be numbered (Figure 1. / Figure 2. ...)
        assert "numbered caption" in text or "Figure 1." in text


class TestCriticGroundingRule:
    """nature-skills #1: both critics must instruct grading only against
    verifiable evidence, with a NOT ASSESSABLE escape hatch that counts as
    FAIL — never a silent pass on assumption."""

    def test_methodology_critic_has_grounding_rule(self):
        text = QUALITY_CRITIC.read_text(encoding="utf-8")
        assert "Grounding rule" in text
        assert "NOT ASSESSABLE" in text

    def test_experiment_critic_has_grounding_rule(self):
        text = EXP_CRITIC.read_text(encoding="utf-8")
        assert "Grounding rule" in text
        assert "NOT ASSESSABLE" in text

    def test_grounding_rule_treats_unverifiable_as_fail(self):
        for path in (QUALITY_CRITIC, EXP_CRITIC):
            flat = " ".join(path.read_text(encoding="utf-8").split())
            assert "treat it as a FAIL" in flat
            assert "never pass a dimension on assumption" in flat


class TestCriticStructuredFindings:
    """nature-skills #2: critics emit a per-issue Findings list (id /
    severity / required_action) so the producer can close objections
    point-by-point. The block must be additive — the Decision line stays."""

    def test_both_critics_have_findings_block(self):
        for path in (QUALITY_CRITIC, EXP_CRITIC):
            text = path.read_text(encoding="utf-8")
            assert "Findings:" in text
            for field in ("id:", "severity:", "required_action:"):
                assert field in text, f"{field} missing in {path.name}"

    def test_findings_blocking_must_be_closed(self):
        for path in (QUALITY_CRITIC, EXP_CRITIC):
            flat = " ".join(path.read_text(encoding="utf-8").split())
            assert "blocking" in flat
            assert "must be closed" in flat

    def test_findings_do_not_replace_decision_line(self):
        # The parseable Decision line must still be specified.
        for path in (QUALITY_CRITIC, EXP_CRITIC):
            text = path.read_text(encoding="utf-8")
            assert "Decision: PASS | REJECT" in text or "Decision: PASS" in text


def test_stage4_dispatch_calls_figure_skill_required():
    """Stage 4 producer task description must call the figure skill
    REQUIRED (not optional). Wording change from PR #58: 'may auto-REJECT'
    → 'auto-REJECT', 'no exceptions'."""
    from onemancompany.core import pipeline_engine
    import inspect
    src = inspect.getsource(pipeline_engine)
    assert 'load_skill("paper-framework-figure")' in src
    # Phrase must be assertive — no "may" / "could" weasels
    s4 = src.split('stage["id"] == 4')[1].split('elif stage["id"] == 5')[0]
    assert "auto-REJECT" in s4 and "no exceptions" in s4
