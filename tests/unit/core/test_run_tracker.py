"""Unit tests for the Stage 6 run-id tracker (core/run_tracker.py)."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from onemancompany.core import run_tracker


# ---------------------------------------------------------------------------
# _filter_for_project — substring match against ``run_command``
# ---------------------------------------------------------------------------

def test_filter_for_project_matches_omc_prefix():
    """A run whose ``run_command`` includes ``omc/<pid>/<iter>`` is claimed."""
    runs = [
        {
            "run_id": "run_a",
            "run_command": "cd omc/abc123def456/iter_001/upstream && python x.py",
        },
        {
            "run_id": "run_b",
            "run_command": "cd omc/zzz999/iter_001/upstream && python x.py",  # other project
        },
        {
            "run_id": "run_c",
            "run_command": "cd omc/abc123def456/iter_002/upstream && python y.py",  # other iter
        },
    ]
    matched = run_tracker._filter_for_project(runs, "abc123def456", "iter_001")
    assert [r["run_id"] for r in matched] == ["run_a"]


def test_filter_for_project_ignores_runs_missing_run_command():
    """A malformed run record (no ``run_command``) is silently skipped — we
    must not raise; the cron loop has to survive infra schema drift."""
    runs = [
        {"run_id": "run_a"},                       # missing run_command
        {"run_id": "run_b", "run_command": None},  # explicit None
        {"run_id": "run_c", "run_command": "cd omc/abc123/iter_001/foo && python z.py"},
    ]
    matched = run_tracker._filter_for_project(runs, "abc123", "iter_001")
    assert [r["run_id"] for r in matched] == ["run_c"]


# ---------------------------------------------------------------------------
# _summarise_run — reduce the full infra record to the fields we persist
# ---------------------------------------------------------------------------

def test_summarise_run_keeps_only_documented_fields():
    """The persisted shape is the contract the API endpoint relies on. We
    pin the keys so an infra-side schema change can't silently drop
    something the UI needs."""
    full = {
        "run_id": "run_x",
        "user_id": "alice",
        "project_id": "scaling-laws",
        "status": "running",
        "run_command": "cd omc/abc/iter_001/up && python a.py",
        "actual_cost": 1.23,
        "estimated_cost": 2.5,
        "created_at": "2026-06-01T12:00:00Z",
        "started_at": "2026-06-01T12:00:01Z",
        "finished_at": "",
        "error_message": "",
        "metrics": {"accuracy_direct": 0.4, "accuracy_cot": 1.0},
        "extra_field_infra_may_add": "ignored",
    }
    out = run_tracker._summarise_run(full)
    assert set(out.keys()) == {
        "status", "run_command", "actual_cost", "estimated_cost",
        "created_at", "started_at", "finished_at", "error_message", "metrics",
    }
    assert out["status"] == "running"
    assert out["metrics"]["accuracy_cot"] == 1.0


# ---------------------------------------------------------------------------
# _should_poll — phase/stage gate
# ---------------------------------------------------------------------------

def test_should_poll_returns_true_for_active_phases():
    eng = SimpleNamespace(current_stage=6, phase="producer_b",
                          state={"stage_started_at": {}})
    assert run_tracker._should_poll(eng) is True

    eng.phase = "producer"
    assert run_tracker._should_poll(eng) is True

    eng.phase = "critic"
    assert run_tracker._should_poll(eng) is True


def test_should_poll_skips_non_stage6_projects():
    eng = SimpleNamespace(current_stage=4, phase="producer", state={})
    assert run_tracker._should_poll(eng) is False


def test_should_poll_skips_old_done_projects():
    """``phase=done`` projects fall out of the poll cycle after 6 hours so
    we don't burn requests forever on completed projects."""
    from datetime import datetime, timedelta, timezone
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    eng = SimpleNamespace(current_stage=6, phase="done",
                          state={"stage_started_at": {"6": long_ago}})
    assert run_tracker._should_poll(eng) is False


# ---------------------------------------------------------------------------
# poll_active_projects — end-to-end behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_active_projects_updates_state_for_matching_runs(monkeypatch):
    """When the infra response includes a run for an active OMC project,
    that project's ``stage_6_runs`` map is populated with the summarised
    record."""
    saved = {}
    state = {"current_stage": 6, "stage_6_runs": {}}

    class _FakeEngine:
        current_stage = 6
        phase = "producer_b"

        def __init__(self):
            self.state = state

        def _iteration_id(self):
            return "iter_001"

        def _save(self):
            saved.update(self.state)

    fake = _FakeEngine()
    monkeypatch.setattr(run_tracker, "_active_pipelines", {"abc123": fake}, raising=False)
    # Inject _active_pipelines as a module-level attribute on pipeline_engine
    from onemancompany.core import pipeline_engine
    monkeypatch.setattr(pipeline_engine, "_active_pipelines", {"abc123": fake})

    monkeypatch.setattr(run_tracker, "_list_infra_runs", lambda limit=100: [
        {
            "run_id": "run_xyz",
            "run_command": "cd omc/abc123/iter_001/upstream && python a.py",
            "status": "succeeded",
            "actual_cost": 0.05,
            "created_at": "2026-06-01T12:00:00Z",
            "finished_at": "2026-06-01T12:00:30Z",
            "error_message": "",
            "metrics": {"accuracy_cot": 1.0},
        },
        {
            "run_id": "run_other",
            "run_command": "cd omc/different_pid/iter_001/up && python z.py",
            "status": "succeeded",
        },
    ])

    counts = await run_tracker.poll_active_projects()
    assert counts == {"abc123": 1}
    assert "run_xyz" in saved.get("stage_6_runs", {})
    assert "run_other" not in saved.get("stage_6_runs", {})
    assert saved["stage_6_runs"]["run_xyz"]["status"] == "succeeded"
    assert saved["stage_6_runs"]["run_xyz"]["actual_cost"] == 0.05


@pytest.mark.asyncio
async def test_poll_active_projects_no_active_returns_empty(monkeypatch):
    """When no project is in an active Stage 6 phase, the poller skips
    the infra call entirely."""
    from onemancompany.core import pipeline_engine
    monkeypatch.setattr(pipeline_engine, "_active_pipelines", {})

    called = {"infra": False}
    def _no_call(limit=100):
        called["infra"] = True
        return []
    monkeypatch.setattr(run_tracker, "_list_infra_runs", _no_call)

    counts = await run_tracker.poll_active_projects()
    assert counts == {}
    assert called["infra"] is False, "Should NOT call infra when no active projects"


@pytest.mark.asyncio
async def test_poll_active_projects_handles_empty_infra_response(monkeypatch):
    """Infra returning ``[]`` (network failure, empty session) leaves
    state untouched but still reports the project as 'seen' with 0 runs."""
    state = {"current_stage": 6, "stage_6_runs": {"existing": {"status": "running"}}}

    class _FakeEngine:
        current_stage = 6
        phase = "producer_b"

        def __init__(self):
            self.state = state
            self._saved = False

        def _iteration_id(self):
            return "iter_001"

        def _save(self):
            self._saved = True

    fake = _FakeEngine()
    from onemancompany.core import pipeline_engine
    monkeypatch.setattr(pipeline_engine, "_active_pipelines", {"abc": fake})
    monkeypatch.setattr(run_tracker, "_list_infra_runs", lambda limit=100: [])

    counts = await run_tracker.poll_active_projects()
    assert counts == {"abc": 0}
    # The existing run map is preserved when infra returns []; we don't
    # overwrite known state with no data.
    assert state["stage_6_runs"] == {"existing": {"status": "running"}}
    assert fake._saved is False
