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

def test_should_poll_state_returns_true_for_active_phases():
    for phase in ("producer", "producer_b", "critic", "gate"):
        state = {"current_stage": 6, "phase": phase}
        assert run_tracker._should_poll_state(state) is True, f"phase={phase}"


def test_should_poll_state_skips_non_stage6_projects():
    assert run_tracker._should_poll_state({"current_stage": 4, "phase": "producer"}) is False


def test_should_poll_state_skips_old_done_projects():
    """``phase=done`` projects fall out of the poll cycle after 6 hours so
    we don't burn requests forever on completed projects."""
    from datetime import datetime, timedelta, timezone
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    state = {"current_stage": 6, "phase": "done", "stage_started_at": {"6": long_ago}}
    assert run_tracker._should_poll_state(state) is False


# ---------------------------------------------------------------------------
# poll_active_projects — end-to-end behaviour
# ---------------------------------------------------------------------------

def _make_project_iter(tmp_path, pid: str, phase: str = "producer_b"):
    """Create a fake ``projects/<pid>/iterations/iter_001/pipeline_state.yaml``
    under ``tmp_path/projects/`` and return the iter_dir Path. Used by
    disk-walking tests."""
    import yaml
    iter_dir = tmp_path / "projects" / pid / "iterations" / "iter_001"
    iter_dir.mkdir(parents=True)
    state = {"current_stage": 6, "phase": phase, "stage_6_runs": {}}
    (iter_dir / "pipeline_state.yaml").write_text(yaml.safe_dump(state))
    return iter_dir


@pytest.mark.asyncio
async def test_poll_active_projects_updates_state_for_matching_runs(monkeypatch, tmp_path):
    """When the infra response includes a run for an active OMC project,
    that project's ``stage_6_runs`` map is persisted onto disk."""
    import yaml

    iter_dir = _make_project_iter(tmp_path, "abc123", phase="producer_b")
    monkeypatch.setattr("onemancompany.core.config.PROJECTS_DIR", tmp_path / "projects")

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

    on_disk = yaml.safe_load((iter_dir / "pipeline_state.yaml").read_text())
    assert "run_xyz" in on_disk["stage_6_runs"]
    assert "run_other" not in on_disk["stage_6_runs"]
    assert on_disk["stage_6_runs"]["run_xyz"]["status"] == "succeeded"
    assert on_disk["stage_6_runs"]["run_xyz"]["actual_cost"] == 0.05


@pytest.mark.asyncio
async def test_poll_active_projects_no_active_returns_empty(monkeypatch, tmp_path):
    """When the projects dir has no active Stage 6 iters, the poller
    skips the infra call entirely."""
    (tmp_path / "projects").mkdir()
    monkeypatch.setattr("onemancompany.core.config.PROJECTS_DIR", tmp_path / "projects")

    called = {"infra": False}
    def _no_call(limit=100):
        called["infra"] = True
        return []
    monkeypatch.setattr(run_tracker, "_list_infra_runs", _no_call)

    counts = await run_tracker.poll_active_projects()
    assert counts == {}
    assert called["infra"] is False, "Should NOT call infra when no active projects on disk"


@pytest.mark.asyncio
async def test_poll_active_projects_handles_empty_infra_response(monkeypatch, tmp_path):
    """Infra returning ``[]`` (network failure, empty session) leaves
    on-disk state untouched but still reports the project as seen with 0 runs."""
    import yaml

    iter_dir = _make_project_iter(tmp_path, "abc", phase="producer_b")
    # Seed with existing runs we expect NOT to be wiped.
    state = yaml.safe_load((iter_dir / "pipeline_state.yaml").read_text())
    state["stage_6_runs"] = {"existing": {"status": "running"}}
    (iter_dir / "pipeline_state.yaml").write_text(yaml.safe_dump(state))

    monkeypatch.setattr("onemancompany.core.config.PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(run_tracker, "_list_infra_runs", lambda limit=100: [])

    counts = await run_tracker.poll_active_projects()
    assert counts == {"abc": 0}
    on_disk = yaml.safe_load((iter_dir / "pipeline_state.yaml").read_text())
    assert on_disk["stage_6_runs"] == {"existing": {"status": "running"}}, (
        "Existing runs must be preserved when infra returns no data"
    )


@pytest.mark.asyncio
async def test_poll_active_projects_skips_underscore_prefixed_project_dirs(monkeypatch, tmp_path):
    """``_adhoc_ceo`` and other underscore-prefixed system dirs are not
    real projects and must not show up in poll targets."""
    _make_project_iter(tmp_path, "_adhoc_ceo", phase="producer_b")
    _make_project_iter(tmp_path, "real_pid_aaa", phase="producer_b")
    monkeypatch.setattr("onemancompany.core.config.PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(run_tracker, "_list_infra_runs", lambda limit=100: [])

    counts = await run_tracker.poll_active_projects()
    assert "_adhoc_ceo" not in counts
    assert "real_pid_aaa" in counts
