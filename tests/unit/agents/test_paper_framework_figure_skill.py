"""Structural tests for the paper-framework-figure SKILL.

nature-skills #4: before calling the image model, the skill must force a
one-paragraph "figure contract" (takeaway / central-vs-supporting /
specificity hook) so the argument is decided before the drawing — the
defence against generic-flowchart figures the critic auto-REJECTs."""
from __future__ import annotations

from pathlib import Path


SKILLS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "onemancompany"
    / "default_skills"
)
FIGURE_SKILL = SKILLS_ROOT / "paper-framework-figure" / "SKILL.md"


def _text() -> str:
    return FIGURE_SKILL.read_text(encoding="utf-8")


class TestFigureContractGate:
    def test_file_exists(self):
        assert FIGURE_SKILL.exists()

    def test_hard_gate_requires_figure_contract(self):
        flat = " ".join(_text().split())
        assert "figure contract" in flat
        # the three required contract fields
        assert "Takeaway" in flat
        assert "Central vs supporting" in flat
        assert "Specificity hook" in flat

    def test_contract_demands_argument_before_drawing(self):
        flat = " ".join(_text().split())
        assert "decide the argument before the drawing" in flat

    def test_contract_lives_inside_hard_gate(self):
        text = _text()
        gate = text.split("<HARD-GATE>")[1].split("</HARD-GATE>")[0]
        assert "figure contract" in gate
        # API key check still present and numbered after the contract item
        assert "OPENROUTER_API_KEY" in gate
