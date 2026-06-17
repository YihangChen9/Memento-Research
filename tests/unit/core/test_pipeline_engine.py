from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from onemancompany.core.task_tree import TaskTree, register_tree
from onemancompany.core import pipeline_engine as pe


@pytest.fixture(autouse=True)
def clear_pipeline_registry():
    pe._active_pipelines.clear()
    yield
    pe._active_pipelines.clear()


@pytest.fixture(autouse=True)
def _stub_stage3_aigraph(monkeypatch):
    """Stage 3 is the pipeline entry (Stages 1/2 were removed). Its
    ``_dispatch_producer`` fetches aigraph grounding over the network; keep that
    off the wire in unit tests so Stage 3 behaves as a plain producer dispatch
    (returns None -> LLM-producer fallback). Tests that exercise grounding
    monkeypatch ``_fetch_aigraph_idea_report`` themselves and override this."""
    monkeypatch.setattr(
        pe.PipelineEngine, "_fetch_aigraph_idea_report", lambda self: None, raising=False
    )


def _employee_config(name: str, skills: list[str]) -> SimpleNamespace:
    return SimpleNamespace(name=name, skills=skills)


def test_state_round_trip_and_registry_reload(tmp_path):
    assert pe._load_state(str(tmp_path)) == {}

    state = {"topic": "graph RAG", "current_stage": 4, "phase": "gate"}
    pe._save_state(str(tmp_path), state)

    assert pe._load_state(str(tmp_path)) == state
    assert pe.get_or_load_pipeline("missing", str(tmp_path / "empty")) is None

    engine = pe.get_or_load_pipeline("p1", str(tmp_path))
    assert engine is pe.get_pipeline("p1")
    assert engine.state["topic"] == state["topic"]
    assert engine.state["current_stage"] == state["current_stage"]
    assert engine.state["phase"] == state["phase"]
    assert engine.state["memory_retrievals"] == {}
    assert engine.state["memory_episodes"] == {}
    assert engine.state["memory_feedback"] == {}

    assert pe.get_or_load_pipeline("p1", str(tmp_path)) is engine


def test_find_employee_by_skill_uses_first_matching_config(monkeypatch):
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "00010": _employee_config("Writer", ["paper_writer"]),
            "00011": _employee_config("Reviewer", ["adversarial_review"]),
        },
    )

    assert pe._find_employee_by_skill("adversarial_review") == "00011"
    assert pe._find_employee_by_skill("missing") is None


def test_start_clamps_stage_uses_assignment_and_builds_context(tmp_path, monkeypatch):
    dispatched = []
    emitted = []

    def fake_dispatch(self, employee_id, description, title):
        dispatched.append((employee_id, description, title))
        self.state["active_node_id"] = "node-1"
        self.state["active_employee_id"] = employee_id

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", fake_dispatch)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: emitted.append((args, kwargs)))
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {"emp-9": _employee_config("Closer", ["peer_reviewer"])})

    engine = pe.PipelineEngine("p1", str(tmp_path), "causal discovery")
    engine.state["stage_results"] = {"1": "refined topic"}
    engine.start(
        start_stage=12,
        end_stage=0,
        prior_context="uploaded notes",
        stage_assignments={"9": "emp-9"},
    )

    assert engine.current_stage == 9
    assert engine.state["start_stage"] == 9
    assert engine.state["end_stage"] == 9
    assert dispatched[0][0] == "emp-9"
    assert dispatched[0][2] == "Stage 9: Self-Review"
    assert "uploaded notes" in dispatched[0][1]
    assert "refined topic" in dispatched[0][1]
    assert "stage9_peer_reviewer.md" in dispatched[0][1]
    assert emitted == [(("stage_start", 9), {"employee_name": "Closer", "employee_id": "emp-9"})]


def test_dispatch_producer_fails_when_no_employee(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_producer()

    assert engine.phase == "failed"


def test_dispatch_producer_with_feedback_uses_skill_lookup(tmp_path, monkeypatch):
    dispatched = []
    emitted = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-topic" if skill == "idea_generator" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: emitted.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_producer(feedback="tighten the framing")

    assert dispatched[0][0] == "emp-topic"
    assert "Feedback from previous review" in dispatched[0][1]
    assert "tighten the framing" in dispatched[0][1]
    assert emitted == [(("stage_start", 3), {"employee_name": "emp-topic", "employee_id": "emp-topic"})]


def test_queue_pending_feedback_appends_and_persists(tmp_path):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.queue_pending_feedback("first hint")
    engine.queue_pending_feedback("second hint")
    assert "first hint" in engine.state["pending_user_feedback"]
    assert "second hint" in engine.state["pending_user_feedback"]

    # Reload from disk → still there
    reloaded = pe._load_state(str(tmp_path))
    assert "first hint" in reloaded["pending_user_feedback"]
    assert "second hint" in reloaded["pending_user_feedback"]


def test_dispatch_producer_consumes_pending_user_feedback(tmp_path, monkeypatch):
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-topic")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.queue_pending_feedback("按意见整改")
    engine._dispatch_producer(feedback="critic says shorten")

    # Both critic feedback and queued CEO feedback land in the prompt.
    desc = dispatched[0][1]
    assert "shorten" in desc
    assert "按意见整改" in desc
    # Pending feedback is consumed after dispatch (single-use).
    assert engine.state.get("pending_user_feedback", "") == ""


def test_dispatch_producer_without_pending_feedback_unchanged(tmp_path, monkeypatch):
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-topic")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_producer()

    desc = dispatched[0][1]
    assert "Direct guidance from CEO" not in desc
    assert "pending_user_feedback" not in engine.state or engine.state.get("pending_user_feedback", "") == ""

def test_dispatch_producer_injects_research_memory_guidance(tmp_path, monkeypatch):
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-topic" if skill == "idea_generator" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)

    store = pe.ResearchMemoryStore("p1", str(tmp_path))
    store.record_stage_episode(
        topic="graph RAG",
        stage=pe.STAGES[0],
        producer_result="Refine graph RAG topic into a concrete benchmarkable claim.",
        critic_result="PASS confidence: 0.9. Clear scope and measurable criteria.",
        passed=True,
        confidence=0.9,
        retries=0,
        reward=0.9,
        outcome="critic_pass",
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "graph RAG")
    engine._dispatch_producer()

    assert "--- Retrieved Research Memory ---" in dispatched[0][1]
    assert "Useful prior memories" in dispatched[0][1]
    assert engine.state["memory_retrievals"]["3"]["ids"]


def test_dispatch_to_employee_uses_ea_child_as_parent_and_schedules(tmp_path, monkeypatch):
    scheduled = []

    class FakeManager:
        def schedule_node(self, employee_id, node_id, tree_path):
            scheduled.append(("schedule", employee_id, node_id, tree_path))

        def _schedule_next(self, employee_id):
            scheduled.append(("next", employee_id))

    import onemancompany.core.agent_loop as agent_loop

    monkeypatch.setattr(agent_loop, "employee_manager", FakeManager())

    tree = TaskTree("p1")
    root = tree.create_root("00001", "CEO request")
    ea_node = tree.add_child(root.id, "00004", "EA coordination", [])
    tree_path = tmp_path / "task_tree.yaml"
    register_tree(tree_path, tree)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_to_employee("00015", "do the work", "Stage 1")

    node = tree.get_node(engine.state["active_node_id"])
    assert node.parent_id == ea_node.id
    assert node.employee_id == "00015"
    assert node.title == "Stage 1"
    assert node.metadata["pipeline_managed"] is True
    assert scheduled[0] == ("schedule", "00015", node.id, str(tree_path))
    assert scheduled[1] == ("next", "00015")


def test_producer_completion_stores_result_and_dispatches_critic(tmp_path, monkeypatch):
    calls = []
    emitted = []

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_critic", lambda self, result: calls.append(result))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: emitted.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.on_task_complete("emp", "node", "producer output")

    assert engine.state["stage_results"]["3"] == "producer output"
    assert calls == ["producer output"]
    assert emitted == [(("stage_reviewing", 3), {})]


def test_critic_completion_pass_moves_to_gate(tmp_path, monkeypatch):
    critic_events = []
    stage_events = []
    gate_events = []

    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *args, **kwargs: critic_events.append((args, kwargs)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: stage_events.append((args, kwargs)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *args, **kwargs: gate_events.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {"3": "producer output"}
    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.82")

    assert engine.phase == "gate"
    assert engine.state["critic_result"].startswith("PASS")
    assert critic_events == [((3, "PASS\nConfidence Score: 0.82", True, 0.82), {})]
    assert stage_events == [(("stage_complete", 3), {"confidence": 0.82})]
    assert gate_events == [((3, 0.82), {})]


def test_critic_completion_records_research_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {"3": "producer output"}
    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.82")

    store = pe.ResearchMemoryStore("p1", str(tmp_path))
    records = store._read_records()
    assert len(records) == 1
    assert records[0]["outcome"] == "critic_pass"
    assert records[0]["reward"] == 0.82
    assert engine.state["memory_episodes"]["3"] == records[0]["id"]


def test_critic_reject_retries_with_feedback(tmp_path, monkeypatch):
    producer_feedback = []
    stage_events = []
    critic_events = []

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": producer_feedback.append(feedback))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: stage_events.append((args, kwargs)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *args, **kwargs: critic_events.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "critic"
    engine.on_task_complete("critic", "node", "REJECT\nconfidence: 0.41\nNeeds tighter scope")

    assert engine.state["retries"] == 1
    assert producer_feedback == ["REJECT\nconfidence: 0.41\nNeeds tighter scope"]
    assert stage_events == [(("stage_failed", 3), {"confidence": 0.41})]
    assert critic_events == [((3, "REJECT\nconfidence: 0.41\nNeeds tighter scope", False, 0.41), {})]


def test_critic_reject_exhausted_waits_for_ceo(tmp_path, monkeypatch):
    gate_events = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *args, **kwargs: gate_events.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "critic"
    engine.state["retries"] = pe.MAX_RETRIES
    engine.on_task_complete("critic", "node", "REJECT confidence: 0.2")

    assert engine.phase == "gate"
    assert gate_events == [((3, 0.2), {"exhausted": True})]


@pytest.mark.asyncio
async def test_auto_approve_refuses_exhausted_gate(tmp_path, monkeypatch):
    """auto_approve=True must NOT advance past a retries-exhausted gate.

    Regression for a failure observed in smoke testing: an upstream LLM
    error burned 3 retries on Stage 6a, the engine emitted an exhausted
    gate, and ``_auto_approve_gate`` then called ``on_ceo_approve("")``
    which advanced ``phase`` to ``done`` with empty ``stage_results``.
    The correct behavior is to mark ``phase=failed`` so downstream
    consumers see the project as terminated, not silently completed.
    """
    failed_events = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_pipeline_failed",
        lambda self, stage_id, reason: failed_events.append((stage_id, reason)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "gate"
    engine.state["auto_approve"] = True
    advanced = []
    monkeypatch.setattr(
        pe.PipelineEngine, "on_ceo_approve",
        lambda self, feedback="": advanced.append(feedback),
    )

    await engine._auto_approve_gate(stage_id=6, exhausted=True)

    assert advanced == [], "Auto-approve must not call on_ceo_approve when exhausted"
    assert engine.state["phase"] == "failed"
    assert engine.state["failure_reason"] == "stage_6_retries_exhausted"
    assert failed_events == [(6, "retries_exhausted")]


@pytest.mark.asyncio
async def test_auto_approve_still_advances_clean_gate(tmp_path, monkeypatch):
    """Clean (non-exhausted) gates must still auto-advance under auto_approve."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_failed", lambda self, *a, **kw: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "gate"
    engine.state["auto_approve"] = True
    approved = []
    monkeypatch.setattr(
        pe.PipelineEngine, "on_ceo_approve",
        lambda self, feedback="": approved.append(feedback),
    )

    await engine._auto_approve_gate(stage_id=2, exhausted=False)

    assert approved == [""], "Clean gate must still trigger on_ceo_approve"
    assert engine.state["phase"] == "gate", "Phase stays 'gate' until on_ceo_approve runs"
    assert "failure_reason" not in engine.state


@pytest.mark.asyncio
async def test_emit_pipeline_failed_publishes_payload(tmp_path, monkeypatch):
    """``_emit_pipeline_failed`` mirrors ``_emit_pipeline_complete`` and
    publishes a ``pipeline_failed`` event so frontend / archive consumers
    can close the project as terminal."""
    published = []

    async def fake_publish(event):
        published.append(event)

    monkeypatch.setattr(pe.event_bus, "publish", fake_publish)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["stage_results"] = {"1": "done", "2": "done"}
    engine._emit_pipeline_failed(stage_id=3, reason="retries_exhausted")
    await asyncio.sleep(0)

    assert len(published) == 1
    payload = published[0].payload
    assert payload["type"] == "pipeline_failed"
    assert payload["project_id"] == "p1"
    assert payload["stage"] == 3
    assert payload["reason"] == "retries_exhausted"
    assert payload["stages_completed"] == 2


# ===========================================================================
# Stage 6b long-running waiter (#93, #97)
# ===========================================================================


def test_canonical_runs_block_is_authoritative_over_prose():
    """Durable fix for the recurring report-format whack-a-mole (runs
    fa50cb183b3c table form, 9dd6160ea99b 'Validated run:' form): when the
    runner emits a machine-readable ```runs block, the engine parses ONLY
    that — deterministic, immune to whatever prose/label the LLM chose, and
    immune to stray run-ids in the Limitations narrative."""
    report = (
        "## Results\n\n"
        "- Validated run: `run_8e540235b830`\n- Status: succeeded\n\n"
        "## 5. Limitations\n"
        "First full run `run_1ea2dbecbae2` failed with OOM; cleanup "
        "`run_eaddfe8cadff` freed it.\n\n"
        "```runs\n"
        "run_8e540235b830  succeeded\n"
        "run_dace6992d448  succeeded\n"
        "```\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    assert pairs == [
        ("run_8e540235b830", "succeeded"),
        ("run_dace6992d448", "succeeded"),
    ], "the ```runs block must win; the OOM/cleanup run-ids in prose must NOT appear"
    assert pe.PipelineEngine._data_gate_stage6(report)[0] is True


def test_canonical_runs_block_skips_header_and_tolerates_pipes():
    """The block parser tolerates an optional header row and pipe/backtick
    decorations, and an empty block authoritatively means zero runs."""
    report = (
        "```runs\n"
        "| run_id | status |\n"
        "| run_aaa111bbb222 | `succeeded` |\n"
        "| run_ccc333ddd444 | still_running |\n"
        "```\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    assert pairs == [
        ("run_aaa111bbb222", "succeeded"),
        ("run_ccc333ddd444", "still_running"),
    ]
    # empty block = authoritative "no runs"
    assert pe.PipelineEngine._parse_runner_report_runs("```runs\n```\n") == []


def test_no_canonical_block_falls_back_to_prose_heuristic():
    """Backward compatibility: reports without a ```runs block still use the
    existing label/table heuristic (no regression for already-good reports)."""
    report = "- **run_id**: `run_d032e33e194a`\n- **status**: succeeded\n"
    assert pe.PipelineEngine._parse_runner_report_runs(report) == [
        ("run_d032e33e194a", "succeeded"),
    ]


def _make_committed_upstream(path, subject="Stage 6 adaptation: GSM benchmark"):
    """A clean upstream/ git repo with a base commit + an adaptation commit."""
    import os, subprocess
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    def g(*a):
        subprocess.run(["git", *a], cwd=str(path), env=env, check=True,
                       capture_output=True, text=True)
    path.mkdir(parents=True, exist_ok=True)
    g("init", "-q", "-b", "main")
    (path / "base.py").write_text("x = 1\n")
    g("add", "."); g("commit", "-q", "-m", "base")
    (path / "benchmark.py").write_text("# adapted\nprint('run')\n")
    g("add", "."); g("commit", "-q", "-m", subject)


def test_stage6a_stub_with_committed_adaptation_synthesizes_receipt(tmp_path, monkeypatch):
    """LIVE regression (run c49e8a914f5a): 6a committed the adaptation but its
    final turn stubbed before writing the receipt → hard-gate fail → exhausted.
    Now the engine synthesizes the receipt from git state and proceeds to 6b
    instead of re-running (and re-stubbing) full 6a."""
    dispatched_b, rerouted = [], []
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b",
                        lambda self: dispatched_b.append(True))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                        lambda self, feedback="": rerouted.append(feedback))
    _make_committed_upstream(tmp_path / "upstream")

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer"
    engine.on_task_complete("00103", "node6a", "Executed: bash")  # 6a stub

    receipt = tmp_path / "stage6_implementation_receipt.md"
    assert receipt.exists() and receipt.stat().st_size >= 200, "receipt must be synthesized"
    assert "adaptation_commit" in receipt.read_text()
    assert dispatched_b == [True], "must proceed to 6b"
    assert rerouted == [], "must NOT re-run full 6a"


def test_stage6a_stub_without_committed_work_still_fails(tmp_path, monkeypatch):
    """The salvage must not paper over a genuinely empty 6a: a stub with no
    committed adaptation (no upstream) does NOT synthesize a receipt — it
    re-runs 6a as before."""
    dispatched_b, rerouted = [], []
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b",
                        lambda self: dispatched_b.append(True))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                        lambda self, feedback="": rerouted.append(feedback))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer"
    engine.on_task_complete("00103", "node6a", "Executed: bash")

    assert not (tmp_path / "stage6_implementation_receipt.md").exists()
    assert dispatched_b == [], "no committed work → must not advance to 6b"
    assert len(rerouted) == 1, "must re-run 6a"


def test_producer_b_stub_with_ondisk_report_salvages_instead_of_rerouting(tmp_path, monkeypatch):
    """LIVE regression (run 9dd6160ea99b): the 6b runner wrote a full report
    with two `succeeded` runs but its submit_result was an "Executed: bash"
    stub. The stub branch rerouted to 6a and the finished experiment was
    discarded → retries exhausted → failed. Now: a 6b stub with a
    data-bearing on-disk report falls through to normal handling (critic),
    not a 6a reroute."""
    rerouted, critic = [], []
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                        lambda self, feedback="": rerouted.append(feedback))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_critic",
                        lambda self, r: critic.append(r))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    skill = engine._stage_def(6).get("skill", "")
    report = "Runs complete.\n\n```runs\nrun_aaa111bbb222 succeeded\n```\n"
    (tmp_path / f"stage6_{skill}.md").write_text(report, encoding="utf-8")

    engine.on_task_complete("00025", "node6b", "Executed: bash")  # stub submit_result

    assert rerouted == [], "must NOT reroute to 6a when the on-disk report has real runs"
    assert len(critic) == 1, "should proceed to the critic with the salvaged report"


def test_parse_runner_report_runs_extracts_id_status_pairs():
    """The parser handles the runner's various Markdown decorations:
    bold-bracketed labels, backtick-wrapped ids, plain text."""
    report = """
### T1 — Smoke run

- **run_id**: `run_d032e33e194a`
- **status**: succeeded

### T2 — Long full run

- run_id: run_45aa663ec237
- status: still_running

### T3 — Stalled

- **run_id**: `run_aa01`
- status: queued
"""
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    assert pairs == [
        ("run_d032e33e194a", "succeeded"),
        ("run_45aa663ec237", "still_running"),
        ("run_aa01", "queued"),
    ]


def test_parse_runner_report_runs_ignores_run_ids_inside_fenced_code():
    """The runner embeds RESULT_JSON inside a ```json fenced block; its
    internal ``run_id`` is the script's seed tag (e.g. ``smoke_seed42``)
    and must NOT be confused with a real infra run_id. Otherwise the
    engine would wait forever on a synthetic id that run_tracker can
    never observe in stage_6_runs."""
    report = """### T1 — Smoke run

- **run_id**: `run_real_infra_id_aa01`
- **status**: succeeded

**Parsed RESULT_JSON:**
```json
{
  "run_id": "smoke_seed42",
  "mode": "smoke",
  "accuracy_direct": 0.667,
  "status": "still_running"
}
```

(end of report)
"""
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    rids = [rid for rid, _ in pairs]
    assert "run_real_infra_id_aa01" in rids
    assert "smoke_seed42" not in rids, (
        "RESULT_JSON's internal run_id must not be treated as an infra run_id"
    )
    # And the JSON's misleading "status: still_running" must not produce
    # a phantom pending entry attached to the real run_id.
    assert not pe.PipelineEngine._runs_have_pending(pairs)


def test_parse_runner_report_runs_drops_placeholders():
    """``run_id: NOT AVAILABLE``-style placeholders must not produce
    spurious pending runs that would park the pipeline forever."""
    report = (
        "- run_id: NONE\n"
        "- status: blocked\n\n"
        "- run_id: missing\n"
        "- status: blocked\n\n"
        "- run_id: real_run_xyz\n"
        "- status: running\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    rids = [rid for rid, _ in pairs]
    assert "real_run_xyz" in rids
    assert all(rid.lower() not in {"none", "missing"} for rid in rids)


def test_parse_runner_report_runs_handles_vertical_markdown_table():
    """LIVE REGRESSION (run fa50cb183b3c): the runner wrote each field as a
    two-cell Markdown table row — ``| run_id | `run_x` |`` / ``| status |
    `succeeded` |`` — instead of the inline ``run_id: x`` form. The parser
    only understood the colon form, so a clean experiment with two
    ``succeeded`` runs parsed as ZERO runs → data-gate FAIL overrode the
    critic's PASS → the whole Stage 6 died at retries-exhausted."""
    report = (
        "| Field | Value |\n"
        "|-------|-------|\n"
        "| run_id | `run_94770b1537b6` |\n"
        "| status | `succeeded` |\n\n"
        "Parsed `RESULT_JSON`: {...}\n\n"
        "| Field | Value |\n"
        "|-------|-------|\n"
        "| run_id | `run_d07b5daba3a7` |\n"
        "| status | `succeeded` |\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    assert pairs == [
        ("run_94770b1537b6", "succeeded"),
        ("run_d07b5daba3a7", "succeeded"),
    ]
    assert pe.PipelineEngine._data_gate_stage6(report)[0] is True


def test_parse_runner_report_runs_ignores_horizontal_header_columns():
    """The pipe separator must NOT turn a horizontal summary header
    (``| run_id | status | cost |``) into a spurious run whose id is the
    next COLUMN NAME (``status``). Column-name words are denylisted."""
    report = (
        "| run_id | status | cost |\n"
        "|--------|--------|------|\n"
        "| run_realjob9 | succeeded | $0 |\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    rids = [rid for rid, _ in pairs]
    assert "status" not in rids and "cost" not in rids


def test_runs_have_pending_distinguishes_terminal_from_active():
    assert pe.PipelineEngine._runs_have_pending([("r1", "running")]) is True
    assert pe.PipelineEngine._runs_have_pending([("r1", "still_running")]) is True
    assert pe.PipelineEngine._runs_have_pending([("r1", "queued")]) is True
    assert pe.PipelineEngine._runs_have_pending([("r1", "succeeded")]) is False
    assert pe.PipelineEngine._runs_have_pending([("r1", "failed")]) is False
    # Mixed: any pending → True
    assert pe.PipelineEngine._runs_have_pending(
        [("r1", "succeeded"), ("r2", "running")]
    ) is True


def test_all_pending_terminal_returns_true_when_every_pending_is_terminal():
    runs_map = {
        "r1": {"status": "succeeded"},
        "r2": {"status": "failed"},
        "r3": {"status": "blocked"},
    }
    assert pe.PipelineEngine._all_pending_terminal(["r1", "r2", "r3"], runs_map) is True
    assert pe.PipelineEngine._all_pending_terminal([], runs_map) is True  # empty list


def test_all_pending_terminal_returns_false_when_any_still_running():
    runs_map = {
        "r1": {"status": "succeeded"},
        "r2": {"status": "running"},
    }
    assert pe.PipelineEngine._all_pending_terminal(["r1", "r2"], runs_map) is False


def test_all_pending_terminal_returns_false_when_run_missing_from_map():
    """Defensive: a pending run that hasn't shown up on infra yet (e.g.
    runner submitted but tracker hasn't polled yet) must NOT be treated
    as terminal — the engine should keep waiting."""
    runs_map = {"r1": {"status": "succeeded"}}
    assert pe.PipelineEngine._all_pending_terminal(["r1", "r2_not_yet"], runs_map) is False


def test_producer_b_complete_with_pending_runs_parks_in_waiting(tmp_path, monkeypatch):
    """When the 6b runner's report carries `status: still_running` for one
    or more run_ids, the engine must NOT dispatch the critic. Instead it
    enters ``producer_b_waiting`` with the pending run_ids persisted.
    Regression for #93 (long experiments REJECTED on 9-min poll budget)."""
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    report = (
        "## Tasks\n\n"
        "- run_id: run_long_a\n"
        "- status: still_running\n\n"
        "- run_id: run_long_b\n"
        "- status: running\n"
    )

    engine.on_task_complete("00025", "nodeB", report)

    assert dispatched_critic == [], "Critic must not be dispatched with pending runs"
    assert engine.state["phase"] == "producer_b_waiting"
    assert engine.state["pending_run_ids"] == ["run_long_a", "run_long_b"]
    assert "pending_waiting_started_at" in engine.state


def test_producer_b_waiting_parses_on_disk_report_not_submit_summary(tmp_path, monkeypatch):
    """REGRESSION (run 1a255f1aaf3d, Round-7 MF-BO): the 6b runner's
    submit_result was a prose summary with a Markdown TABLE
    (``| run_id | type | status |``) that the line-pair parser can't read,
    while the on-disk ``stage6_experimentalist.md`` carried perfectly
    parseable ``- run_id: ... / - status: running`` rows. The engine parsed
    only the submit_result → found no pending runs → dispatched the critic
    on a still-running pilot → procedural REJECT burned a retry.

    Same disease as #27's original gate bug: never parse critical state
    from the lossy submit summary when the canonical deliverable is on
    disk. The waiter must read the deliverable first."""
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"

    # On-disk canonical report: bullet rows, one still-running pilot.
    (tmp_path / "stage6_experimentalist.md").write_text(
        "# Stage 6 — Auto Experiment Results\n\n"
        "### T1 — Pilot\n"
        "- run_id: run_pilot_a\n"
        "- status: running\n\n"
        "### T0 — Smoke (validated by 6a)\n"
        "- run_id: run_smoke_b\n"
        "- status: succeeded\n",
        encoding="utf-8",
    )
    # submit_result: prose + table — the format that defeated the parser.
    summary = (
        "**Stage 6b — STILL_RUNNING**\n\n"
        "**Deliverable saved**: stage6_experimentalist.md (50 lines)\n\n"
        "| run_id | type | status |\n"
        "|--------|------|--------|\n"
        "| `run_pilot_a` | pilot | running |\n"
        "| `run_smoke_b` | smoke | succeeded |\n"
    )

    engine.on_task_complete("00025", "nodeB", summary)

    assert dispatched_critic == [], (
        "critic must NOT run on a still-running pilot — the on-disk report "
        "names a running run_id"
    )
    assert engine.state["phase"] == "producer_b_waiting"
    assert "run_pilot_a" in engine.state["pending_run_ids"]


def test_producer_b_complete_with_all_terminal_dispatches_critic(tmp_path, monkeypatch):
    """Fast path: when every run in the 6b report is terminal, the engine
    skips ``producer_b_waiting`` and dispatches the critic immediately
    (pre-#93 behavior preserved for short experiments)."""
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    report = (
        "- **run_id**: `smoke_a`\n"
        "- **status**: succeeded\n\n"
        "- **run_id**: `full_b`\n"
        "- **status**: succeeded\n"
    )

    engine.on_task_complete("00025", "nodeB", report)

    assert len(dispatched_critic) == 1, "Critic must dispatch when all runs terminal"
    assert engine.state["phase"] == "producer_b"  # critic dispatch flips it later
    assert "pending_run_ids" not in engine.state


def test_producer_b_complete_with_no_runs_at_all_dispatches_critic(tmp_path, monkeypatch):
    """Defensive: a blocked/budget-failed runner report with no run_ids at
    all should NOT park in waiting (no runs to wait on). Critic gets the
    honest BLOCKED report and decides."""
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    report = "BLOCKED: session_available=0.00, no runs submitted"

    engine.on_task_complete("00025", "nodeB", report)

    assert len(dispatched_critic) == 1
    assert "pending_run_ids" not in engine.state


def test_on_runs_all_terminal_advances_to_finalize(tmp_path, monkeypatch):
    """When all pending runs are terminal, on_runs_all_terminal must
    transition to ``producer_b_finalize`` and re-dispatch the runner."""
    finalize_dispatched = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b_finalize",
        lambda self: finalize_dispatched.append(True),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_waiting"
    engine.state["pending_run_ids"] = ["run_long_a", "run_long_b"]

    engine.on_runs_all_terminal()

    assert engine.state["phase"] == "producer_b_finalize"
    assert finalize_dispatched == [True]


def test_on_runs_all_terminal_idempotent_outside_waiting_phase(tmp_path, monkeypatch):
    """Defensive: a duplicate run_tracker callback (e.g. two ticks racing)
    must not re-dispatch the runner if the engine already advanced."""
    finalize_dispatched = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b_finalize",
        lambda self: finalize_dispatched.append(True),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_finalize"  # already advanced
    engine.state["pending_run_ids"] = ["run_long_a"]

    engine.on_runs_all_terminal()

    assert finalize_dispatched == [], "Must not re-dispatch outside waiting phase"


def test_producer_b_finalize_complete_dispatches_critic_and_clears_pending(tmp_path, monkeypatch):
    """After the runner re-dispatch in finalize mode completes, the engine
    must dispatch the critic on the final report AND clean up
    ``pending_run_ids`` from state so a subsequent retry doesn't see
    stale waiting bookkeeping."""
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_finalize"
    engine.state["pending_run_ids"] = ["run_long_a"]
    engine.state["pending_waiting_started_at"] = "2026-06-01T17:00:00Z"

    engine.on_task_complete("00025", "nodeB-final", "## Final report\n\n- run_id: run_long_a\n- status: succeeded")

    assert len(dispatched_critic) == 1
    assert "pending_run_ids" not in engine.state
    assert "pending_waiting_started_at" not in engine.state


def test_on_task_failed_in_finalize_redispatches_finalize_not_initial(tmp_path, monkeypatch):
    """If the LLM task for ``producer_b_finalize`` fails (e.g. transient
    503), the engine must re-dispatch the FINALIZE path, NOT the initial
    submit-and-run path — the runs are already terminal; re-submitting
    them would orphan the originals and double-charge."""
    initial_dispatches = []
    finalize_dispatches = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b",
        lambda self, feedback="": initial_dispatches.append(feedback),
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b_finalize",
        lambda self: finalize_dispatches.append(True),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_finalize"
    engine.state["pending_run_ids"] = ["run_a"]

    engine.on_task_failed("00025", "node-finalize", "503 Service Unavailable")

    assert finalize_dispatches == [True], "Must re-dispatch finalize, not initial submit"
    assert initial_dispatches == [], "Must NOT call _dispatch_producer_b (would re-submit)"


def test_stub_result_in_finalize_redispatches_finalize_not_initial(tmp_path, monkeypatch):
    """Same invariant for stub-result retry: a stub during finalize must
    re-dispatch finalize, not the initial submit task."""
    initial_dispatches = []
    finalize_dispatches = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b",
        lambda self, feedback="": initial_dispatches.append(feedback),
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b_finalize",
        lambda self: finalize_dispatches.append(True),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_finalize"
    engine.state["pending_run_ids"] = ["run_a"]

    engine.on_task_complete("00025", "node-finalize", "Executed: bash")

    assert finalize_dispatches == [True], "Stub retry in finalize must re-dispatch finalize"
    assert initial_dispatches == [], "Must NOT call _dispatch_producer_b on finalize stub"


# ===========================================================================
# Stage 6b waiter — review feedback regressions (PR #107 review by iamlilAJ)
# ===========================================================================


def test_runs_have_pending_treats_unknown_status_as_pending():
    """Fail-safe semantics: anything NOT in _RUN_TERMINAL_STATUSES is
    considered pending. Otherwise an LLM-phrased ``in_progress`` /
    ``executing`` / typo would fall through to terminal and re-introduce
    the #93 \"critic dispatched on still-running\" failure mode."""
    # Known terminal — not pending
    assert pe.PipelineEngine._runs_have_pending([("r1", "succeeded")]) is False
    assert pe.PipelineEngine._runs_have_pending([("r1", "failed")]) is False
    # Known pending — pending
    assert pe.PipelineEngine._runs_have_pending([("r1", "running")]) is True
    # Unknown free-form tokens — MUST be treated as pending
    assert pe.PipelineEngine._runs_have_pending([("r1", "in_progress")]) is True
    assert pe.PipelineEngine._runs_have_pending([("r1", "executing")]) is True
    assert pe.PipelineEngine._runs_have_pending([("r1", "unknown")]) is True
    # The fail-closed parser uses "unknown" as the no-status-found marker
    # — make sure that marker is pending so the engine keeps waiting
    # rather than firing the critic on an unverified run.


def test_pending_run_ids_from_uses_fail_safe_semantics():
    """``_pending_run_ids_from`` must use the same fail-safe semantics
    as ``_runs_have_pending`` (anything not terminal counts as pending)."""
    runs = [
        ("r_term", "succeeded"),
        ("r_running", "running"),
        ("r_weird", "in_progress"),
        ("r_unknown", "unknown"),
    ]
    pending = pe.PipelineEngine._pending_run_ids_from(runs)
    assert "r_term" not in pending
    assert set(pending) == {"r_running", "r_weird", "r_unknown"}


def test_parse_runner_report_pairs_status_within_run_block_not_globally():
    """Each run_id binds to a status that lives WITHIN its block (between
    this run_id's offset and the next run_id's offset). A status appearing
    BEFORE its run_id, or attached to a different run's block, must not
    leak across.

    Previously the parser walked status_hits in document order and could
    drop a ``still_running`` status if a sibling block's terminal status
    came first.
    """
    report = (
        "- run_id: r_first\n"
        "- status: succeeded\n\n"
        "- run_id: r_second\n"
        "- status: still_running\n\n"
        "- run_id: r_third_no_status\n"
        "(no status line for this block)\n"
    )
    pairs = pe.PipelineEngine._parse_runner_report_runs(report)
    by_rid = dict(pairs)
    assert by_rid["r_first"] == "succeeded"
    assert by_rid["r_second"] == "still_running", (
        "r_second's still_running must not be lost — block-bounded pairing required"
    )
    # r_third has no status in its block → fail-closed to "unknown"
    assert by_rid["r_third_no_status"] == "unknown"
    # And "unknown" makes the run pending under fail-safe semantics
    assert pe.PipelineEngine._runs_have_pending(pairs) is True


def test_strip_fenced_code_blocks_only_targets_json_fences():
    """A bash / python / unlabelled fence must NOT have its body blanked
    — otherwise an outer wrap or unbalanced fence would wipe legitimate
    ``- run_id: ...`` list-item lines, parser sees no runs, critic
    dispatched on \"no runs\" (a #93 regression class)."""
    text = (
        "preamble\n\n"
        "- run_id: r_real_outside_fence\n"
        "- status: still_running\n\n"
        "```bash\n"
        "RID=\"foobar\"\n"
        "- run_id: should_not_be_stripped\n"
        "```\n\n"
        "```json\n"
        '{"run_id": "smoke_seed42"}\n'
        "```\n"
    )
    stripped = pe.PipelineEngine._strip_fenced_code_blocks(text)
    # Bash fence's contents must survive
    assert "foobar" in stripped, "bash fence content must NOT be blanked"
    # json fence's contents must be blanked (so synthetic run_id doesn't leak)
    assert "smoke_seed42" not in stripped, "json fence must be blanked"
    # Legitimate list-item outside any fence must survive
    assert "r_real_outside_fence" in stripped


def test_strip_fenced_code_blocks_survives_unbalanced_outer_fence():
    """A whole-report outer fence (no closing triple-backtick) must NOT
    silently consume the rest of the document — only json info-string
    fences are blanked, so an unbalanced bash-opening fence just stays
    as-is."""
    text = (
        "```bash\n"  # opening only; never closed
        "- run_id: r_inside_unbalanced_bash\n"
        "- status: still_running\n"
    )
    stripped = pe.PipelineEngine._strip_fenced_code_blocks(text)
    # Content must survive — bash info-string is not in the strip list
    assert "r_inside_unbalanced_bash" in stripped


def test_run_tracker_active_phases_includes_producer_b_waiting():
    """REGRESSION (PR #107 review blocker): if producer_b_waiting is
    missing from _ACTIVE_PHASES, the cron filters out parked projects,
    stage_6_runs never refreshes, and on_runs_all_terminal is unreachable
    — every long-running experiment hangs forever, the exact failure
    mode this PR exists to fix."""
    from onemancompany.core import run_tracker
    assert "producer_b_waiting" in run_tracker._ACTIVE_PHASES
    assert "producer_b_finalize" in run_tracker._ACTIVE_PHASES


def test_should_poll_state_returns_true_for_producer_b_waiting():
    """Companion to the _ACTIVE_PHASES check: the gate function used by
    the disk-walker must also let producer_b_waiting through."""
    from onemancompany.core import run_tracker
    state = {"current_stage": 6, "phase": "producer_b_waiting"}
    assert run_tracker._should_poll_state(state) is True


def test_on_runs_wait_timeout_opens_exhausted_gate(tmp_path, monkeypatch):
    """When the engine sits in producer_b_waiting past the max-wait
    deadline, run_tracker calls on_runs_wait_timeout. The engine must
    open a gate with exhausted=True (which under auto_approve, via #106,
    transitions to phase=failed instead of silently advancing)."""
    gate_events = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_gate_event",
        lambda self, stage_id, confidence=None, exhausted=False:
            gate_events.append((stage_id, confidence, exhausted)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_waiting"
    engine.state["pending_run_ids"] = ["run_hung_a"]

    engine.on_runs_wait_timeout(wait_seconds=12 * 3600)

    assert engine.state["phase"] == "gate"
    assert engine.state["failure_reason"] == "stage_6_waiting_timeout_43200s"
    assert gate_events == [(6, None, True)], (
        "Must open an EXHAUSTED gate (so #106's auto-approve refusal kicks in)"
    )


def test_on_runs_wait_timeout_idempotent_outside_waiting(tmp_path, monkeypatch):
    """If run_tracker fires the deadline callback after the engine
    already advanced (race), the call must be a no-op."""
    gate_events = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_gate_event",
        lambda self, stage_id, confidence=None, exhausted=False:
            gate_events.append((stage_id, confidence, exhausted)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b_finalize"  # already advanced

    engine.on_runs_wait_timeout(wait_seconds=99999)

    assert engine.state["phase"] == "producer_b_finalize", "must not regress phase"
    assert gate_events == [], "must not emit a duplicate exhausted gate"


def test_producer_b_immediate_terminal_via_state_short_circuits_waiting(tmp_path, monkeypatch):
    """If the runner's report says still_running but run_tracker has
    already marked the runs terminal in stage_6_runs (race between the
    cron tick and the runner's completion event), the engine must
    short-circuit straight to finalize instead of waiting for a poll
    cycle that has nothing left to do."""
    finalize_dispatched = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer_b_finalize",
        lambda self: finalize_dispatched.append(True),
    )
    # Also stub critic dispatch so we can tell which path was taken
    dispatched_critic = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_critic",
        lambda self, r: dispatched_critic.append(r),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    # run_tracker already filled in terminal status before we got here
    engine.state["stage_6_runs"] = {
        "run_long_a": {"status": "succeeded"},
    }
    report = "- run_id: run_long_a\n- status: still_running"

    engine.on_task_complete("00025", "nodeB", report)

    assert finalize_dispatched == [True], "Must short-circuit to finalize, not park"
    assert dispatched_critic == [], "Must not dispatch critic — finalize will do that"


def test_dispatch_critic_without_critic_auto_passes(tmp_path, monkeypatch):
    stage_events = []
    gate_events = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: stage_events.append((args, kwargs)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *args, **kwargs: gate_events.append((args, kwargs)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_critic("producer output")

    assert engine.phase == "gate"
    assert stage_events == [(("stage_complete", 3), {"confidence": None})]
    assert gate_events == [((3, None), {})]


def test_dispatch_critic_sends_review_task_to_critic(tmp_path, monkeypatch):
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._dispatch_critic("producer output")

    assert engine.phase == "critic"
    assert dispatched[0][0] == "critic-1"
    assert "Gate Review: Stage 3" in dispatched[0][1]
    assert "--- Producer Output ---\nproducer output" in dispatched[0][1]
    assert dispatched[0][2] == "Gate Review: Stage 3"


def test_ceo_approval_revision_advance_and_complete(tmp_path, monkeypatch):
    producer_feedback = []
    completed = []

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": producer_feedback.append((self.current_stage, feedback)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_complete", lambda self: completed.append(self.project_id))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["end_stage"] = 5
    engine.state["retries"] = 2
    engine.state["critic_result"] = "old"
    # on_ceo_approve only acts at a gate (#157). In a live run the critic
    # returns the engine to "gate" between each CEO action; the mock doesn't,
    # so re-enter the gate before each approve to mirror that.
    engine.state["phase"] = "gate"

    engine.on_ceo_approve("please REVISE the method")
    assert engine.state["retries"] == 0
    assert producer_feedback == [(4, "please REVISE the method")]

    engine.state["phase"] = "gate"
    engine.on_ceo_approve()
    assert engine.current_stage == 5
    assert engine.state["critic_result"] is None
    assert producer_feedback[-1] == (5, "")

    engine.state["phase"] = "gate"
    engine.on_ceo_approve()
    assert engine.phase == "done"
    assert completed == ["p1"]

def test_ceo_revision_updates_research_memory_feedback(tmp_path, monkeypatch):
    producer_feedback = []

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": producer_feedback.append(feedback))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["stage_results"] = {"3": "producer output"}
    engine.state["phase"] = "gate"  # on_ceo_approve only acts at a gate (#157)
    memory_id = engine._record_stage_memory(
        pe.STAGES[0],
        producer_result="producer output",
        critic_result="PASS confidence: 0.8",
        passed=True,
        confidence=0.8,
        outcome="critic_pass",
    )

    engine.on_ceo_approve("please REVISE with stricter scope")

    store = pe.ResearchMemoryStore("p1", str(tmp_path))
    records = {record["id"]: record for record in store._read_records()}
    assert producer_feedback == ["please REVISE with stricter scope"]
    assert records[memory_id]["ceo_approved"] is False
    assert records[memory_id]["reward"] < 0.8
    assert engine.state["memory_feedback"]["3"]["episode_id"] == memory_id


def test_record_stage_memory_persists_phase_elapsed_seconds(tmp_path):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    memory_id = engine._record_stage_memory(
        pe.STAGES[0],
        producer_result="producer output",
        critic_result="PASS confidence: 0.9",
        passed=True,
        confidence=0.9,
        outcome="critic_pass",
        producer_elapsed_seconds=42.5,
        critic_elapsed_seconds=8.0,
    )

    store = pe.ResearchMemoryStore("p1", str(tmp_path))
    record = next(r for r in store._read_records() if r["id"] == memory_id)
    assert record["producer_elapsed_seconds"] == 42.5
    assert record["critic_elapsed_seconds"] == 8.0


def test_on_task_complete_updates_attempt_timing_and_records_it(tmp_path, monkeypatch):
    recorded = {}

    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda *a, **k: None)

    def fake_record(self, stage, **kwargs):
        recorded.update(kwargs)
        return "m-1"

    monkeypatch.setattr(pe.PipelineEngine, "_record_stage_memory", fake_record)

    now = {"t": 1000.0}

    def fake_time():
        return now["t"]

    monkeypatch.setattr(pe.time, "time", fake_time)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 3
    engine.state["phase"] = "producer"
    engine.state["active_task_started_at"] = 970.0
    engine.state["attempt_timing"] = {"producer_elapsed_seconds": 0.0, "critic_elapsed_seconds": None}
    engine.state["stage_results"] = {"3": "producer output"}

    now["t"] = 1000.0
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_critic", lambda self, _result: None)
    engine.on_task_complete("00006", "n1", "producer output")
    assert engine.state["attempt_timing"]["producer_elapsed_seconds"] == 30.0

    engine.state["phase"] = "critic"
    engine.state["active_task_started_at"] = 1000.0
    now["t"] = 1012.0
    engine.on_task_complete("00014", "n2", "PASS confidence: 0.7")

    assert recorded["producer_elapsed_seconds"] == 30.0
    assert recorded["critic_elapsed_seconds"] == 12.0


def test_critic_retry_persists_attempt_timing_before_reset(tmp_path, monkeypatch):
    recorded = {}

    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_parse_critic_pass", lambda *a, **k: False)
    monkeypatch.setattr(pe.PipelineEngine, "_parse_confidence", lambda *a, **k: 0.2)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": None)

    def fake_record(self, stage, **kwargs):
        recorded.update(kwargs)
        return "m-1"

    monkeypatch.setattr(pe.PipelineEngine, "_record_stage_memory", fake_record)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 3
    engine.state["phase"] = "critic"
    engine.state["retries"] = 0
    engine.state["stage_results"] = {"3": "producer output"}
    engine.state["attempt_timing"] = {"producer_elapsed_seconds": 41.0, "critic_elapsed_seconds": 9.0}

    engine.on_task_complete("00014", "n2", "REJECT confidence: 0.2")

    assert recorded["producer_elapsed_seconds"] == 41.0
    assert recorded["critic_elapsed_seconds"] == 9.0
    assert engine.state["attempt_timing"] == {"producer_elapsed_seconds": 0.0, "critic_elapsed_seconds": None}


def test_record_active_phase_elapsed_clamps_clock_jump_outliers(tmp_path, monkeypatch):
    now = {"t": 10_000.0}
    monkeypatch.setattr(pe.time, "time", lambda: now["t"])

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    monkeypatch.setattr(engine, "_stage_def", lambda stage_id=None: {"id": 2, "timeout_seconds": 1200})
    engine.state["active_task_started_at"] = 10.0
    engine.state["attempt_timing"] = {"producer_elapsed_seconds": 0.0, "critic_elapsed_seconds": None}

    engine._record_active_phase_elapsed("producer")

    assert engine.state["attempt_timing"]["producer_elapsed_seconds"] == 2400.0


@pytest.mark.parametrize("feedback,expect_revise", [
    # advance-with-comment chats that must NOT trigger a redo
    ("再补充一点细节", False),
    ("再讨论一下这个点", False),
    ("可以修改一下措辞", False),
    ("再加一个 baseline", False),
    # explicit redo triggers
    ("重新写 stage 4", True),
    ("重做这部分", True),
    ("please REVISE the methodology", True),
    ("Let's redo this stage", True),
    ("再写一遍 introduction", True),
])
def test_on_ceo_approve_revision_keyword_matching(tmp_path, monkeypatch, feedback, expect_revise):
    """Narrow keyword matcher: single-char '再' / ambiguous '修改' must not
    trigger a redo on otherwise benign CEO chat. Explicit multi-char redo
    triggers should fire."""
    redispatched = []

    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": redispatched.append((self.current_stage, feedback)))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_complete", lambda self: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["end_stage"] = 9
    engine.state["phase"] = "gate"  # guard requires gate phase (#157)
    initial_stage = engine.current_stage

    engine.on_ceo_approve(feedback)

    if expect_revise:
        # revise path: same stage, _dispatch_producer called with feedback
        assert redispatched and redispatched[-1][0] == initial_stage
        assert engine.state["retries"] == 0
    else:
        # advance path: stage advanced, no producer redispatch with feedback
        assert engine.current_stage == initial_stage + 1
        # _dispatch_producer is called on advance too (for the new stage) — feedback should be empty
        assert all(fb == "" for _, fb in redispatched), f"unexpected redispatch with feedback: {redispatched}"


def test_parse_critic_decision_and_confidence(tmp_path):
    # ``_parse_critic_pass`` is now an instance method (it reaches for the
    # on-disk gate-review file as a stub-recovery fallback) — needs an engine.
    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 4

    assert engine._parse_critic_pass("reject: weak evidence") is False
    assert engine._parse_critic_pass("pass: strong enough") is True
    # Default-REJECT on ambiguity (was: default PASS, the silent-auto-approve
    # loophole behind #60 / #63).
    assert engine._parse_critic_pass("looks fine") is False
    # Table-format verdict (#60 fix 4).
    assert engine._parse_critic_pass("| Decision | PASS |") is True
    assert engine._parse_critic_pass("| **Decision** | **REJECT** |") is False

    assert pe.PipelineEngine._parse_confidence("Confidence: 1.0") == 1.0
    assert pe.PipelineEngine._parse_confidence("no score") is None


def test_parse_critic_pass_stub_falls_back_to_disk(tmp_path):
    """When the critic submits a stub like ``"Executed: bash"``, parser must
    fall back to reading ``stage{N}_gate_review.md`` from disk and verdict
    against THAT content. Default to REJECT if neither yields a signal."""
    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 6

    # Case A: stub + on-disk gate review exists with a PASS verdict
    gate_review = tmp_path / "stage6_gate_review.md"
    gate_review.write_text("# Gate Review\n\n| Decision | PASS |\n\nConfidence 0.92.")
    assert engine._parse_critic_pass("Executed: bash") is True, (
        "Stub critic result + on-disk PASS should resolve to PASS"
    )

    # Case B: stub + on-disk gate review with REJECT
    gate_review.write_text("# Gate Review\n\n| Decision | REJECT |\n\nMissing run_ids.")
    assert engine._parse_critic_pass("Executed: bash") is False

    # Case C: stub + no on-disk file → default REJECT (safer than auto-PASS)
    gate_review.unlink()
    assert engine._parse_critic_pass("Executed: bash") is False, (
        "Stub critic result + no fallback file should default to REJECT (not PASS)"
    )


def test_parse_critic_pass_long_toolresult_stub_falls_back_to_disk(tmp_path):
    """REGRESSION (#19, caught by the 4→9 e2e, project 5232b74836ee): the
    critic wrote a perfect gate_review.md ('| Decision | PASS |') but its
    submit_result was a ~867-char tool-result echo:

        Executed: write
        write → {'status': 'ok', 'path': '.../stage6_gate_review.md', 'type': ...}

    This is LONGER than the 300-char stub threshold, so _is_stub_result
    didn't flag it, the on-disk fallback never fired, and the parser saw a
    long blob with no PASS/REJECT → defaulted to REJECT → 3 retries → the
    whole run died at Stage 6 even though the critic had PASSED.

    The verdict lives in the FILE; when the submit_result text carries no
    clear PASS/REJECT signal, the parser MUST consult the on-disk
    gate_review before defaulting to REJECT — regardless of stub length."""
    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 6

    # Neutral path text (no 'pass'/'reject' substring) so the parser can't
    # accidentally pattern-match a verdict out of the tool echo itself.
    long_toolresult_stub = (
        "Executed: write\n"
        "write → {'status': 'ok', 'path': '/work/stage6_gate_review.md', "
        "'type': 'file', 'bytes': 3125, 'note': '" + ("x" * 700) + "'}"
    )
    assert len(long_toolresult_stub) > 300  # NOT caught by the short-stub heuristic
    assert "PASS" not in long_toolresult_stub.upper()
    assert "REJECT" not in long_toolresult_stub.upper()

    gate_review = tmp_path / "stage6_gate_review.md"
    gate_review.write_text("# Gate Review\n\n| **Decision** | **PASS** |\n| Confidence | 0.58 |")
    assert engine._parse_critic_pass(long_toolresult_stub) is True, (
        "long tool-result stub + on-disk PASS must resolve to PASS via file fallback"
    )

    gate_review.write_text("# Gate Review\n\n| **Decision** | **REJECT** |\nNo data.")
    assert engine._parse_critic_pass(long_toolresult_stub) is False

    # A real conversational verdict in the text still wins without touching disk.
    gate_review.write_text("| Decision | PASS |")  # stale file says PASS...
    assert engine._parse_critic_pass("REJECT: the smoke run failed") is False, (
        "an explicit in-text verdict must take precedence over the on-disk file"
    )


def test_cancel_marks_pipeline_terminal_and_stops_retries(tmp_path, monkeypatch):
    """REGRESSION (R5-1, zombie 76ad6534ed86): /api/task/<pid>/abort cancelled
    the task-tree nodes but never told the pipeline engine — the engine saw
    the cancellation as an ordinary producer failure and RE-DISPATCHED,
    resurrecting the pipeline. Three aborts in a row could not kill it.

    ``cancel()`` must put the pipeline in a terminal phase so subsequent
    node-failure events are no-ops."""
    dispatched = []
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_producer",
        lambda self, feedback="": dispatched.append(feedback),
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None,
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer"
    engine.state["active_node_id"] = "node-6a"

    engine.cancel(reason="CEO abort")

    assert engine.phase == "failed", "cancel must reach a terminal phase"
    assert "cancel" in (engine.state.get("failure_reason") or "").lower()
    assert engine.state.get("active_node_id") is None

    # The cancelled node's failure event arrives AFTER the abort — must not
    # resurrect the pipeline.
    engine.on_task_failed("00103", "node-6a", "Cancelled by CEO (project aborted)")
    assert dispatched == [], "a cancelled pipeline must never re-dispatch"
    assert engine.phase == "failed"

    # Idempotent.
    engine.cancel(reason="again")
    assert engine.phase == "failed"


def test_cancel_noop_on_done_pipeline(tmp_path):
    """cancel() must not stomp a pipeline that already finished."""
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 9
    engine.state["phase"] = "done"

    engine.cancel(reason="late abort")

    assert engine.phase == "done", "a done pipeline stays done"


def test_parse_critic_pass_consults_latest_versioned_review(tmp_path):
    """REGRESSION (run e04df33b06bb, Round-3 GSM8K e2e): on a retry cycle the
    critic sees ``stage6_gate_review.md`` already exists (the previous
    cycle's REJECT) and writes its NEW verdict to
    ``stage6_gate_review_v2.md`` — PASS. Its submit_result was again a
    tool-echo stub, so the parser fell back to disk… and read the FIXED
    filename: the stale v1 REJECT. A scientifically successful run
    (CoT 87.04% vs direct 18.65% on full GSM8K, vLLM, 88 s wall-clock)
    died at retries-exhausted on a verdict from the previous cycle.

    The disk consult must read the LATEST ``stage{N}_gate_review*.md``
    (by mtime), not the fixed name."""
    import os
    import time

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 6

    stub = (
        "Executed: write\n"
        "write → {'status': 'ok', 'path': '/work/stage6_gate_review_v2.md', "
        "'type': 'create', 'note': '" + ("x" * 700) + "'}"
    )
    assert "PASS" not in stub.upper() and "REJECT" not in stub.upper()

    old = time.time() - 3600
    v1 = tmp_path / "stage6_gate_review.md"
    v1.write_text("# Gate Review\n\n| Decision | REJECT |\n\nNo usable data.")
    os.utime(v1, (old, old))
    v2 = tmp_path / "stage6_gate_review_v2.md"
    v2.write_text("# Gate Review — v2\n\n| Decision | PASS |\n\nReal data confirmed.")

    assert engine._parse_critic_pass(stub) is True, (
        "stale v1 REJECT must not shadow the newer v2 PASS"
    )

    # Reverse: newest verdict REJECT must win over an older PASS — no
    # cherry-picking a favorable old file.
    os.utime(v2, (old, old))
    v3 = tmp_path / "stage6_gate_review_v3.md"
    v3.write_text("# Gate Review — v3\n\n| Decision | REJECT |\n\nRegression found.")
    assert engine._parse_critic_pass(stub) is False

    # If the newest file is garbled (no verdict), fall back to the next
    # newest that has one.
    v4 = tmp_path / "stage6_gate_review_v4.md"
    v4.write_text("# Gate Review — v4\n\n(truncated mid-write, no decision row)")
    assert engine._parse_critic_pass(stub) is False, (
        "garbled newest file falls through to v3's REJECT"
    )


def test_cap_for_critic_trims_oversized_producer_output():
    """#62: critic's input must stay under a soft budget so late-stage
    runs don't blow Kimi-K2.6's 262K context window. Cap keeps head +
    tail with an explicit elision marker."""
    # Under budget: passed through unchanged
    short = "x" * 10_000
    assert pe.PipelineEngine._cap_for_critic(short, stage_id=4) == short
    # Empty → empty
    assert pe.PipelineEngine._cap_for_critic("", stage_id=4) == ""
    # Over budget: head + elision marker + tail
    head = "HEAD" + ("a" * 49_996)        # exactly 50K
    middle = "M" * 200_000                # 200K elided
    tail = ("z" * 24_996) + "TAIL"        # exactly 25K
    big = head + middle + tail
    out = pe.PipelineEngine._cap_for_critic(big, stage_id=6)
    assert out.startswith("HEAD"), "Head bytes must be preserved"
    assert out.endswith("TAIL"), "Tail bytes must be preserved"
    assert "elided" in out, "Elision marker must be present"
    assert len(out) < len(big), "Output must be shorter than input"
    # Total budget respected (head + tail + ~120-byte marker)
    assert len(out) <= 80_000 + 200, f"Capped output exceeded budget: {len(out)}"


def test_is_stub_result():
    """Stub detection — used by parser fallback and (future) producer
    stub-detection gates. ``"Executed: ..."``-style outputs come from
    the agent runtime falling back to tool-name summaries when the LLM
    returned no text content."""
    assert pe.PipelineEngine._is_stub_result("Executed: bash") is True
    assert pe.PipelineEngine._is_stub_result("Executed tools: write, read, bash") is True
    assert pe.PipelineEngine._is_stub_result("") is True
    assert pe.PipelineEngine._is_stub_result("# Gate Review\n\n## Decision\n\nPASS — 0.95 confidence.\n\nFull analysis follows... " + "x" * 350) is False
    # BEHAVIOUR CHANGE (R13-1, run df3fd56612e5): the old design treated a
    # long "Executed: ..." echo as legitimate output. An 852KB tool-echo of
    # /api/list_runs then polluted the waiter with 100 account-wide run_ids.
    # The prefix is the runtime's no-text fallback signature — it is a stub
    # at ANY length; the real deliverable lives on disk, never in an echo.
    long_executed = "Executed: bash\n" + "real captured output line\n" * 50
    assert pe.PipelineEngine._is_stub_result(long_executed) is True


def test_parse_confidence_handles_unparseable_match(monkeypatch):
    class BadMatch:
        def group(self, index):
            assert index == 1
            return "bad"

    import re

    monkeypatch.setattr(re, "search", lambda *args, **kwargs: BadMatch())

    assert pe.PipelineEngine._parse_confidence("confidence: bad") is None


def test_parse_confidence_handles_bold_and_table_decorations():
    """LIVE gap (run fa50cb183b3c): the real gate review wrote
    ``**Confidence**: **0.92**`` and ``| Confidence | 0.92 |`` — the old
    ``[:\\s]*`` separator couldn't cross the ``**``/``|`` markers, so the
    critic's stated confidence was lost (parsed as None)."""
    assert pe.PipelineEngine._parse_confidence("**Confidence**: **0.92**") == 0.92
    assert pe.PipelineEngine._parse_confidence("| Confidence | 0.92 |") == 0.92
    assert pe.PipelineEngine._parse_confidence("Confidence Score: 0.8") == 0.8
    # Plain prose with no value must still yield None (no false reading).
    assert pe.PipelineEngine._parse_confidence("confident in the approach") is None


def test_verdict_from_text_handles_table_and_backtick_verdicts():
    """Same decoration-blindness class as the data gates: a verdict written
    as a backtick-wrapped table cell, or labeled ``Verdict``/``Recommendation``
    rather than ``Decision``, must still parse."""
    V = pe.PipelineEngine._verdict_from_text
    assert V("| Decision | PASS |") is True
    assert V("| Verdict | `REJECT` |") is False
    assert V("| Recommendation | `PASS` |") is True
    assert V("**Decision**: **PASS**") is True
    # A per-dimension row must NOT be read as the overall verdict (#19), and
    # a bare prose mention stays ambiguous.
    assert V("Looks good, I'd pass this.") is None


def test_event_emitters_skip_when_no_running_loop(tmp_path):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["stage_results"] = {"1": "done"}

    engine._emit_critic_result(1, "REJECT", False)
    engine._emit_stage_event("stage_complete", 1, confidence=0.5)
    engine._emit_gate_event(1, 0.5)
    engine._emit_pipeline_complete()


@pytest.mark.asyncio
async def test_event_emitters_publish_payloads_in_running_loop(tmp_path, monkeypatch):
    published = []

    async def fake_publish(event):
        published.append(event)

    monkeypatch.setattr(pe.event_bus, "publish", fake_publish)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["stage_results"] = {"1": "done"}

    await engine._emit_async({"type": "direct"})
    engine._emit_critic_result(1, "PASS confidence: 0.7", True, 0.7)
    engine._emit_stage_event("stage_start", 1, employee_name="Analyst", employee_id="00015")
    engine._emit_gate_event(1, 0.7, exhausted=True)
    engine._emit_pipeline_complete()
    await asyncio.sleep(0)

    payloads = [event.payload for event in published]
    assert payloads[0] == {"type": "direct"}
    assert payloads[1]["type"] == "critic_result"
    assert payloads[1]["decision"] == "PASS"
    assert payloads[2]["type"] == "stage_start"
    assert payloads[2]["employee_name"] == "Analyst"
    assert payloads[3]["type"] == "breakpoint_hit"
    assert payloads[3]["retries_exhausted"] is True
    assert payloads[4] == {"type": "pipeline_complete", "project_id": "p1", "stages_completed": 1, "pipeline_managed": True}


def test_dispatch_producer_stage4_injects_pdf_url_reading_instructions(tmp_path, monkeypatch):
    """Stage 4 task description must instruct the methodology_designer to
    read any PDF files and fetch referenced URLs (arXiv etc.) before starting
    the debate. This grounds the methodology in real literature rather than
    recalled knowledge."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-meth" if skill == "methodology_designer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine._dispatch_producer()

    assert dispatched
    desc = dispatched[0][1]
    assert "read_pdf" in desc, "Stage 4 must instruct the producer to read any PDF files"
    assert "fetch" in desc or "web_search" in desc, (
        "Stage 4 must instruct the producer to fetch URLs from arXiv / literature survey"
    )
    assert "arxiv.org" in desc, "Stage 4 must mention arXiv as the primary literature source"


def test_dispatch_producer_stage4_injects_methodology_debate_skill_trigger(tmp_path, monkeypatch):
    """Stage 4 (Methodology Design) task description must instruct the
    methodology_designer to load the methodology-debate-convener skill
    before producing the deliverable. Other stages must not get this trigger."""
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-meth" if skill == "methodology_designer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine._dispatch_producer()

    assert dispatched, "producer must dispatch"
    desc = dispatched[0][1]
    assert 'load_skill("methodology-debate-convener")' in desc, (
        "Stage 4 task description must instruct the producer to load the convener skill"
    )
    # Preamble must describe the draft → debate → revise flow, not the
    # pre-#19 "synthesise transcript into methodology document" wording.
    assert "draft" in desc.lower() and "revise" in desc.lower(), (
        "Stage 4 trigger preamble must mention the draft → debate → revise flow"
    )


def test_dispatch_producer_stage4_requires_methodological_depth(tmp_path, monkeypatch):
    """Stage 4 dispatch must demand formal methodological depth — a formal
    Method section with equations, a pseudocode/Algorithm block, and an
    explicit Contributions/novelty statement. Audit M1: even the most
    rigorous historical methodology (a0aee5044ce2) had zero pseudocode, ~no
    real equations, and one novelty mention — the bar only graded experiment
    rigor, never the method's formalization."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-meth" if skill == "methodology_designer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine._dispatch_producer()

    desc = dispatched[0][1]
    low = desc.lower()
    assert "pseudocode" in low or "algorithm" in low, (
        "Stage 4 dispatch must require a pseudocode / Algorithm block"
    )
    assert "equation" in low or "formal" in low or "$" in desc, (
        "Stage 4 dispatch must require formal equations / mathematical formalization"
    )
    assert "contribution" in low or "novel" in low, (
        "Stage 4 dispatch must require an explicit contributions / novelty statement"
    )


def test_dispatch_producer_non_stage4_does_not_inject_debate_skill(tmp_path, monkeypatch):
    """Stages other than 4 must not contain the methodology-debate-convener trigger."""
    dispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-topic" if skill == "idea_generator" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 3
    engine._dispatch_producer()

    assert dispatched, "producer must dispatch"
    desc = dispatched[0][1]
    assert "methodology-debate-convener" not in desc, (
        "Non-Stage-4 stages must not carry the debate convener trigger"
    )


def test_dispatch_critic_stage4_injects_methodology_quality_critic_skill(tmp_path, monkeypatch):
    """Stage 4 critic dispatch must instruct the reviewer to load the
    methodology-quality-critic skill, which enforces CCF-A grade criteria.
    Other stages' critic dispatches must not get this directive."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine._dispatch_critic("draft methodology document")

    assert dispatched, "critic must be dispatched"
    desc = dispatched[0][1]
    assert 'load_skill("methodology-quality-critic")' in desc, (
        "Stage 4 critic description must instruct the reviewer to load the quality-critic skill"
    )


def test_dispatch_critic_non_stage4_does_not_inject_quality_critic(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5  # Experiment Design
    engine._dispatch_critic("experiment plan output")

    assert dispatched, "critic must be dispatched"
    desc = dispatched[0][1]
    assert "methodology-quality-critic" not in desc, (
        "Non-Stage-4 critic dispatch must not carry the methodology critic skill trigger"
    )


def test_dispatch_producer_stage5_injects_experiment_debate_skill_trigger(tmp_path, monkeypatch):
    """Stage 5 (Experiment Design) task description must instruct the
    producer to load the experiment-debate-convener skill before designing."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-exp" if skill == "experiment_designer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine._dispatch_producer()

    assert dispatched, "producer must dispatch"
    desc = dispatched[0][1]
    assert 'load_skill("experiment-debate-convener")' in desc, (
        "Stage 5 task description must instruct the producer to load the experiment convener skill"
    )


def test_dispatch_critic_stage5_injects_experiment_quality_critic_skill(tmp_path, monkeypatch):
    """Stage 5 critic dispatch must instruct the reviewer to load the
    experiment-quality-critic skill."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine._dispatch_critic("draft experiment plan")

    assert dispatched, "critic must be dispatched"
    desc = dispatched[0][1]
    assert 'load_skill("experiment-quality-critic")' in desc, (
        "Stage 5 critic description must instruct the reviewer to load the experiment quality-critic skill"
    )


def test_dispatch_critic_stage5_enforces_feasibility_first_gate(tmp_path, monkeypatch):
    """Stage 5 critic must reject designs that skip the feasibility tier or
    omit the figure manifest — the gate that makes the producer-side
    feasibility-first requirement (PR #169) actually binding."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine._dispatch_critic("draft experiment plan")

    desc = dispatched[0][1]
    low = desc.lower()
    assert "feasibility" in low, "Stage 5 critic must check for the feasibility tier"
    assert "figure manifest" in low, "Stage 5 critic must check for the figure manifest"
    assert "budget" in low, "Stage 5 critic must check the design is budget-bound"
    assert "reject" in low, "the checks must be REJECT-enforced, not advisory"


def test_dispatch_producer_stage5_trigger_not_in_stage4_or_other(tmp_path, monkeypatch):
    """Triggers should be stage-id-scoped — Stage 5 trigger must not appear
    in Stage 4 producer (which has its own methodology trigger) or Stage 3."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    for stage_id in (3, 4):
        dispatched.clear()
        engine = pe.PipelineEngine(f"p{stage_id}", str(tmp_path), "topic")
        engine.state["current_stage"] = stage_id
        engine._dispatch_producer()
        if dispatched:
            assert "experiment-debate-convener" not in dispatched[0][1]


# ---------------------------------------------------------------------------
# Stage 6 (Auto Experiment) — runner preference + experiment-infra trigger
# ---------------------------------------------------------------------------

def test_find_employee_for_stage_6_resolves_code_implementer(monkeypatch):
    """Stage 6's first dispatch maps to the code_implementer (Stage 6a).
    The experiment_runner is the *second* dispatch (Stage 6b) — see
    ``_find_stage_6b_employee``."""
    by_skill = {
        "code_implementer": "emp-coder-027",
        "experiment_runner": "emp-runner-025",
    }
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda s: by_skill.get(s))
    assert pe._find_employee_for_stage(6, "experimentalist") == "emp-coder-027"


def test_find_stage_6b_employee_prefers_runner_over_experimentalist(monkeypatch):
    """Stage 6b's runner resolution prefers the experiment_runner (real
    remote infra) over the experimentalist (simulator-only)."""
    by_skill = {
        "experiment_runner": "emp-runner-007",
        "experimentalist": "emp-sim-001",
    }
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda s: by_skill.get(s))
    assert pe._find_stage_6b_employee() == "emp-runner-007"


def test_find_stage_6b_employee_falls_back_to_experimentalist(monkeypatch):
    """No experiment_runner on roster — Stage 6b falls back to the
    experimentalist so the pipeline still runs (degraded, simulation only)."""
    by_skill = {"experimentalist": "emp-sim-001"}
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda s: by_skill.get(s))
    assert pe._find_stage_6b_employee() == "emp-sim-001"


def test_find_employee_for_stage_5_unchanged_no_runner_fallback(monkeypatch):
    """Runner fallback is Stage 6 only — Stage 5 must keep using the
    primary skill (experiment_designer) so we don't accidentally swap
    who writes the experiment plan."""
    by_skill = {
        "experiment_runner": "emp-runner-007",
        "experiment_designer": "emp-designer-006",
    }
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda s: by_skill.get(s))
    assert pe._find_employee_for_stage(5, "experiment_designer") == "emp-designer-006"


def test_dispatch_producer_stage6_injects_code_implementation_runbook_trigger(tmp_path, monkeypatch):
    """Stage 6's first producer dispatch is Stage 6a — the code implementer.
    Its description must instruct the agent to load the
    code-implementation-runbook (which carries the upstream-pin Phase 0)."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-coder-027" if skill == "code_implementer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_producer()

    assert dispatched, "Stage 6a producer must dispatch"
    desc = dispatched[0][1]
    assert 'load_skill("code-implementation-runbook")' in desc, (
        "Stage 6a task description must instruct the producer to load "
        "code-implementation-runbook (Phase 0 honours the upstream pin)"
    )
    assert "stage5_codebase_pin.md" in desc or "pin" in desc.lower(), (
        "Stage 6a task description must reference the Stage 5 codebase pin"
    )


def test_dispatch_producer_stage6_injects_stage4_methodology_as_required_input(tmp_path, monkeypatch):
    """Stage 6a dispatch must instruct the agent to read stage4_methodology_designer.md
    before loading the runbook. The Stage 4 doc is the immutable contract for IVs/DVs/
    metrics — without reading it, the pin-path self-check (Step 0.1.5 #6) cannot be
    performed, leading to silent parameter mismatches (e.g. run be7144a49333)."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-coder-027" if skill == "code_implementer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_producer()

    assert dispatched, "Stage 6a producer must dispatch"
    desc = dispatched[0][1]
    assert "stage4_methodology_designer.md" in desc, (
        "Stage 6a dispatch must instruct the agent to read stage4_methodology_designer.md "
        "as the immutable contract before any implementation work"
    )
    assert "stage5_experiment_designer.md" in desc, (
        "Stage 6a dispatch must also require reading stage5_experiment_designer.md "
        "for locked parameter values"
    )
    assert "immutable contract" in desc.lower() or "IVs" in desc or "metrics" in desc, (
        "Stage 6a dispatch must convey that Stage 4 is the parameter contract"
    )


def test_dispatch_producer_stage6_routes_to_code_implementer_employee(tmp_path, monkeypatch):
    """Stage 6's first dispatch resolves to the code_implementer
    (not the experiment_runner — that comes in the 6b second pass)."""
    dispatched = []
    monkeypatch.setattr(
        pe, "_find_employee_by_skill",
        lambda skill: {"code_implementer": "emp-coder",
                       "experiment_runner": "emp-runner"}.get(skill),
    )
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, emp_id, *rest: dispatched.append(emp_id))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_producer()

    assert dispatched == ["emp-coder"], (
        f"Expected first Stage 6 dispatch to code_implementer, got {dispatched}"
    )


def test_dispatch_producer_b_stage6_injects_execution_runbook_trigger(tmp_path, monkeypatch):
    """Stage 6b's producer dispatch must instruct the agent to load the
    experiment-execution-runbook and reference the implementation receipt
    from Stage 6a."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-runner-025" if skill == "experiment_runner" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_producer_b()

    assert dispatched, "Stage 6b producer must dispatch"
    emp_id, desc, title = dispatched[0]
    assert emp_id == "emp-runner-025"
    assert 'load_skill("experiment-execution-runbook")' in desc, (
        "Stage 6b task description must instruct the runner to load the execution runbook"
    )
    assert "stage6_implementation_receipt.md" in desc, (
        "Stage 6b must reference Stage 6a's implementation receipt"
    )


def test_stage6_phase_transitions_producer_to_producer_b_to_critic(tmp_path, monkeypatch):
    """On a successful Stage 6 run, on_task_complete must walk the
    producer → producer_b → critic transition (not jump straight to
    the critic on producer completion)."""
    transitions = []

    def _capture_phase(self, *a, **kw):
        transitions.append(self.state["phase"])

    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: {"code_implementer": "emp-coder",
                                       "experiment_runner": "emp-runner",
                                       pe.CRITIC_SKILL: "emp-critic"}.get(skill))
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: _capture_phase(self))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    # Satisfy the Stage 6a hard-gate: receipt file must exist (>= 200 bytes),
    # and no upstream/ git repo means the uncommitted-patches check is skipped.
    (tmp_path / "stage6_implementation_receipt.md").write_text(
        "# Receipt\n" + "x" * 250  # > 200-byte threshold
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6

    engine._dispatch_producer()                       # → producer (6a)
    engine.on_task_complete("emp-coder", "n1", "6a receipt")  # → producer_b
    engine.on_task_complete("emp-runner", "n2", "6b report")  # → critic

    assert transitions == ["producer", "producer_b", "critic"], (
        f"Expected producer → producer_b → critic, got {transitions}"
    )
    # 6a result is stored separately; 6b's report is the canonical stage 6 result
    assert engine.state["stage_6a_result"] == "6a receipt"
    assert engine.state["stage_results"]["6"] == "6b report"


def test_producer_stub_result_retries_instead_of_advancing(tmp_path, monkeypatch):
    """When a producer returns a stub like ``"Executed: bash"`` (agent
    runtime fallback when LLM produced no text), the engine MUST retry
    the producer with feedback rather than store the stub as the stage
    deliverable. Closes #60 fix #2.

    Old behaviour: stub → stored as stage_result → critic gets tool-name
    summary → ``_parse_critic_pass`` defaulted to PASS on ambiguity →
    NOT TESTED paper marches on."""
    feedbacks = []
    orig = pe.PipelineEngine._dispatch_producer
    def _capturing(self, feedback=""):
        feedbacks.append(feedback)
        return orig(self, feedback=feedback)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", _capturing)
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-meth")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 4  # any non-stage-6 stage; stub gate is universal

    engine._dispatch_producer()  # initial dispatch
    # Producer returns a stub
    engine.on_task_complete("emp", "n1", "Executed: bash")

    assert engine.state["retries"] == 1, "Stub result should bump retries, not advance"
    assert any("stub" in fb.lower() for fb in feedbacks), (
        f"Retry feedback must name the stub failure mode; got {feedbacks!r}"
    )
    # Stage result must NOT have been stored (otherwise critic would see it)
    assert "4" not in engine.state.get("stage_results", {}), (
        "Stub result must not pollute stage_results"
    )


def test_stage6a_hard_gate_retries_on_missing_receipt(tmp_path, monkeypatch):
    """If Stage 6a producer finishes WITHOUT producing
    stage6_implementation_receipt.md, the engine must retry the producer
    (with feedback explaining the gap) rather than silently advancing to
    Stage 6b — which would always BLOCK on missing receipt and burn a full
    6a → 6b → critic cycle. Closes #63's fix #4."""
    dispatched_phases = []
    feedbacks = []

    def _capture(self, *args, **kw):
        dispatched_phases.append(self.state["phase"])

    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: {"code_implementer": "emp-coder",
                                       "experiment_runner": "emp-runner",
                                       pe.CRITIC_SKILL: "emp-critic"}.get(skill))
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: _capture(self))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event",
                        lambda self, *args, **kwargs: None)
    # Capture the feedback string the retry receives
    orig_dispatch_producer = pe.PipelineEngine._dispatch_producer
    def _capturing_dispatch_producer(self, feedback=""):
        feedbacks.append(feedback)
        return orig_dispatch_producer(self, feedback=feedback)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", _capturing_dispatch_producer)

    # Do NOT create the receipt file — gate should fail
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6

    engine._dispatch_producer()  # initial 6a dispatch
    engine.on_task_complete("emp-coder", "n1", "incomplete 6a output")

    # After hard-gate failure, retry should re-dispatch producer (not advance)
    assert dispatched_phases[-1] == "producer", (
        f"Hard-gate failure should retry producer, got phases {dispatched_phases}"
    )
    # Retry count incremented
    assert engine.state["retries"] == 1
    # Feedback should mention the missing receipt
    assert any("stage6_implementation_receipt.md" in fb for fb in feedbacks), (
        f"Retry feedback must name the missing receipt; got {feedbacks!r}"
    )


def test_dispatch_critic_stage6_injects_evidence_grading(tmp_path, monkeypatch):
    """Stage 6 critic dispatch must instruct the reviewer to verify real
    run_ids + cost + log_tail — fabricated results are auto-REJECT."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_critic("Stage 6 producer report")

    assert dispatched, "Stage 6 critic must be dispatched"
    desc = dispatched[0][1]
    assert "run_id" in desc, (
        "Stage 6 critic prompt must require evidence — a real run_id"
    )
    assert "auto-REJECT" in desc or "fabricat" in desc.lower(), (
        "Stage 6 critic prompt must explicitly call out fabricated results"
    )


def test_stage6_trigger_not_in_stage4_or_stage5(tmp_path, monkeypatch):
    """Stage 6 triggers (code-impl-runbook + execution-runbook + run_id
    grading) must be stage-scoped — earlier stages must not carry them."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    for stage_id in (4, 5):
        dispatched.clear()
        engine = pe.PipelineEngine(f"p{stage_id}", str(tmp_path), "topic")
        engine.state["current_stage"] = stage_id
        engine._dispatch_producer()
        if dispatched:
            desc = dispatched[0][1]
            assert "experiment-execution-runbook" not in desc
            assert "code-implementation-runbook" not in desc


# ---------------------------------------------------------------------------
# on_task_failed — producer failure handling (PR #34)
# ---------------------------------------------------------------------------


def test_on_task_failed_critic_phase_auto_passes(tmp_path, monkeypatch):
    """A FAILED *critic* must not re-dispatch the producer (which would
    discard the existing producer output and double-bill tokens). Mirrors
    the "no critic employee found" branch in _dispatch_critic by
    auto-passing on the stored producer output."""
    on_pass_calls = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-meth")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_to_employee",
        lambda self, *args: pytest.fail("must not re-dispatch on critic failure"),
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_stage_event",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_gate_event",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_on_critic_pass",
        lambda self, result, confidence=None: on_pass_calls.append((result, confidence)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "critic"
    engine.state["stage_results"]["4"] = "producer output for stage 4"

    engine.on_task_failed("critic-emp", "node-x", "critic crashed: OOM")

    assert on_pass_calls == [("producer output for stage 4", None)]
    assert engine.state.get("retries", 0) == 0, "retries must not increment on critic failure"


def test_on_task_failed_retries_until_exhausted(tmp_path, monkeypatch):
    """A FAILED producer is treated like a critic REJECT: retry up to
    MAX_RETRIES, then hold the gate for CEO. Crucially, the failure must
    NOT fall through to vessel.py's legacy completion check (which would
    misdeclare the project done)."""
    dispatched = []
    emitted = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-meth")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_to_employee",
        lambda self, *args: dispatched.append(args),
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_stage_event",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pe.PipelineEngine, "_emit_gate_event",
        lambda self, *args, **kwargs: emitted.append(("gate", args, kwargs)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "producer"

    # First failure: retry 1
    engine.on_task_failed("emp-meth", "node-a", "TypeError: …")
    assert engine.state["retries"] == 1
    assert engine.state["phase"] == "producer"
    assert dispatched, "should re-dispatch producer"
    assert "TypeError" in dispatched[-1][1]

    # Second failure: retry 2
    engine.on_task_failed("emp-meth", "node-b", "OOM")
    assert engine.state["retries"] == 2

    # Third failure: retry 3
    engine.on_task_failed("emp-meth", "node-c", "timeout")
    assert engine.state["retries"] == 3

    # Fourth failure: retries exhausted → gate, no new dispatch
    redispatch_count_before = len(dispatched)
    engine.on_task_failed("emp-meth", "node-d", "still broken")
    assert engine.state["phase"] == "gate"
    assert len(dispatched) == redispatch_count_before, "must not re-dispatch when exhausted"
    assert any(kind == "gate" for kind, _, _ in emitted)


def test_emit_pipeline_complete_marks_ceo_root_finished(tmp_path, monkeypatch):
    """When the pipeline truly completes (after stage 9 / end_stage), the
    engine must close the CEO root so the UI's project-complete affordance
    fires here and only here. Previously the legacy EA-anchor heuristic
    in vessel.py closed the root after Stage 1, before the pipeline was
    actually done."""
    from onemancompany.core.task_tree import TaskTree, register_tree, get_tree
    from onemancompany.core.task_lifecycle import NodeType, TaskPhase
    from onemancompany.core.config import TASK_TREE_FILENAME

    tree = TaskTree(project_id="p1")
    root = tree.create_root("00001", "do research")
    root.node_type = NodeType.CEO_PROMPT
    root.set_status(TaskPhase.PROCESSING)
    tree_path = tmp_path / TASK_TREE_FILENAME
    tree.save(tree_path)
    register_tree(tree_path, tree)

    monkeypatch.setattr(pe.PipelineEngine, "_emit_async", lambda self, payload: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._emit_pipeline_complete()

    root2 = tree.get_node(tree.root_id)
    assert root2.status == TaskPhase.FINISHED.value


@pytest.mark.parametrize("starting_status", ["failed", "blocked", "cancelled"])
def test_emit_pipeline_complete_skips_finalize_on_illegal_source(tmp_path, monkeypatch, starting_status):
    """If the CEO root is in a state from which COMPLETED is illegal
    (FAILED / BLOCKED / CANCELLED), ``_mark_ceo_root_finished`` must
    log-and-bail instead of crashing with IllegalTransitionError. The
    pipeline_complete event still emits — the root simply isn't walked
    further."""
    from onemancompany.core.task_tree import TaskTree, register_tree
    from onemancompany.core.task_lifecycle import NodeType, TaskPhase
    from onemancompany.core.config import TASK_TREE_FILENAME

    tree = TaskTree(project_id="p1")
    root = tree.create_root("00001", "do research")
    root.node_type = NodeType.CEO_PROMPT
    # Walk through legal transitions to reach the requested terminal-ish state.
    if starting_status == "failed":
        root.set_status(TaskPhase.PROCESSING)
        root.set_status(TaskPhase.FAILED)
    elif starting_status == "blocked":
        root.set_status(TaskPhase.BLOCKED)
    elif starting_status == "cancelled":
        root.set_status(TaskPhase.CANCELLED)
    tree_path = tmp_path / TASK_TREE_FILENAME
    tree.save(tree_path)
    register_tree(tree_path, tree)

    monkeypatch.setattr(pe.PipelineEngine, "_emit_async", lambda self, payload: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._emit_pipeline_complete()  # must NOT raise IllegalTransitionError

    # Root stays in its original status — not coerced through illegal path.
    root_after = tree.get_node(tree.root_id)
    assert root_after.status == starting_status


def test_emit_pipeline_complete_idempotent_on_already_finished_root(tmp_path, monkeypatch):
    """If the CEO root is already FINISHED (e.g. via a prior call), the
    second emit must not raise or attempt illegal status transitions."""
    from onemancompany.core.task_tree import TaskTree, register_tree
    from onemancompany.core.task_lifecycle import NodeType, TaskPhase
    from onemancompany.core.config import TASK_TREE_FILENAME

    tree = TaskTree(project_id="p1")
    root = tree.create_root("00001", "do research")
    root.node_type = NodeType.CEO_PROMPT
    root.set_status(TaskPhase.PROCESSING)
    root.set_status(TaskPhase.COMPLETED)
    root.set_status(TaskPhase.ACCEPTED)
    root.set_status(TaskPhase.FINISHED)
    tree_path = tmp_path / TASK_TREE_FILENAME
    tree.save(tree_path)
    register_tree(tree_path, tree)

    monkeypatch.setattr(pe.PipelineEngine, "_emit_async", lambda self, payload: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine._emit_pipeline_complete()  # must not raise


# ---------------------------------------------------------------------------
# revert_to_stage — git-backed checkpoint + new instructions + re-run
# ---------------------------------------------------------------------------


def test_start_calls_ensure_initialized(tmp_path, monkeypatch):
    """Engine.start should auto-init a git repo in the workspace so
    subsequent commit_stage calls have somewhere to commit."""
    init_calls = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)

    from onemancompany.core import project_repo
    monkeypatch.setattr(
        project_repo, "ensure_initialized",
        lambda repo_dir, iteration: init_calls.append((repo_dir, iteration)),
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.start(start_stage=1, end_stage=3)

    assert init_calls, "engine.start must call project_repo.ensure_initialized"
    assert init_calls[0][0] == str(tmp_path)


def test_on_critic_pass_commits_and_tags_stage(tmp_path, monkeypatch):
    """After a stage passes, the engine commits the workspace and tags
    the commit so the user can later revert here."""
    commit_calls = []

    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *args, **kwargs: None)

    from onemancompany.core import project_repo
    def _fake_commit(repo_dir, iteration, stage, stage_name):
        commit_calls.append((repo_dir, iteration, stage, stage_name))
        return "deadbeef"
    monkeypatch.setattr(project_repo, "commit_stage", _fake_commit)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 3
    engine._on_critic_pass("stage 3 output", confidence=0.9)

    assert commit_calls, "critic-pass must trigger commit_stage"
    repo_dir, iteration, stage, stage_name = commit_calls[0]
    assert repo_dir == str(tmp_path)
    assert stage == 3
    assert "Idea Generation" in stage_name  # STAGES[2].name


@pytest.mark.asyncio
async def test_revert_to_stage_checkouts_branch_and_redispatches(tmp_path, monkeypatch):
    """revert_to_stage(N, instructions) at a gate should:
       1. Create a feature branch rooted at the previous stage's tag.
       2. Reload state from disk (checkout flipped the file).
       3. Set current_stage=N, phase=producer, retries=0.
       4. Queue the user's instructions for stage N's producer.
       5. Re-dispatch.
       6. NOT scrub the workspace (no active task, no partial writes).
    """
    checkout_calls = []
    redispatched = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(
        pe.PipelineEngine, "_dispatch_to_employee",
        lambda self, eid, desc, title: redispatched.append((eid, desc, title)),
    )
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *args, **kwargs: None)

    from onemancompany.core import project_repo
    def fake_checkout(repo_dir, iteration, stage, branch_name=None):
        checkout_calls.append((repo_dir, iteration, stage, branch_name))
        # Simulate the disk-side effect of a checkout: write a fresh
        # pipeline_state.yaml as if it had been restored from the prior tag.
        from pathlib import Path
        import yaml
        prior_state = {
            "topic": "topic",
            "current_stage": 2,
            "phase": "gate",
            "stage_results": {"1": "stage 1 result"},
            "retries": 0,
            "end_stage": 9,
        }
        (Path(repo_dir) / pe.STATE_FILENAME).write_text(yaml.safe_dump(prior_state))
        return branch_name or "feat-stage3-abcdef"
    monkeypatch.setattr(project_repo, "checkout_branch_from_stage", fake_checkout)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "gate"
    engine.state["retries"] = 2
    engine.state["stage_results"] = {
        "1": "...", "2": "...", "3": "...", "4": "...", "5": "...", "6": "...",
    }
    engine._save()

    branch = await engine.revert_to_stage(stage=3, instructions="please use H2O instead")

    assert checkout_calls, "checkout_branch_from_stage must be called"
    assert checkout_calls[0][2] == 3, "checkout target must be the stage being reverted to"
    assert branch == "feat-stage3-abcdef"

    # State was reloaded from disk (current_stage=2 came from disk), then
    # advanced to the revert target (3), with phase reset to producer.
    assert engine.state["current_stage"] == 3
    assert engine.state["phase"] == "producer"
    assert engine.state["retries"] == 0
    # Stale fields from the pre-revert state should be cleared.
    assert engine.state.get("critic_result") in (None, "")
    # Producer was re-dispatched, and the instructions reached its prompt
    # via the pending-feedback queue → _consume_pending_feedback drain.
    assert redispatched, "producer must be re-dispatched after revert"
    _eid, desc, _title = redispatched[0]
    assert "please use H2O instead" in desc, (
        "user's revert instructions must appear in the producer's prompt"
    )
    # Queue is drained after dispatch (single-use).
    assert engine.state.get("pending_user_feedback", "") == ""


def _stub_revert_environment(monkeypatch, *, restored_state: dict, branch: str = "feat-revert-xx"):
    """Common monkeypatches for the revert tests.

    Returns a dict whose keys record observable side effects: ``aborted``,
    ``waited`` (asyncio.Task handles whose .done() was awaited),
    ``discarded`` (repo paths scrubbed), ``checkout_calls``.
    """
    aborted: list[str] = []
    discarded: list[str] = []
    checkout_calls: list[tuple] = []

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda *a, **k: None)

    from onemancompany.core import project_repo, agent_loop

    class _FakeManager:
        _running_tasks: dict = {}
        def abort_employee(self, emp_id: str) -> int:
            aborted.append(emp_id)
            return 1
    monkeypatch.setattr(agent_loop, "employee_manager", _FakeManager())
    monkeypatch.setattr(project_repo, "discard_uncommitted_changes", lambda repo: discarded.append(repo))

    def fake_checkout(repo_dir, iteration, stage, branch_name=None):
        checkout_calls.append((repo_dir, iteration, stage, branch_name))
        from pathlib import Path
        import yaml
        (Path(repo_dir) / pe.STATE_FILENAME).write_text(yaml.safe_dump(restored_state))
        return branch_name or branch
    monkeypatch.setattr(project_repo, "checkout_branch_from_stage", fake_checkout)

    return {"aborted": aborted, "discarded": discarded, "checkout_calls": checkout_calls}


@pytest.mark.asyncio
async def test_revert_to_stage_cancels_active_task_and_discards_dirty_workspace(tmp_path, monkeypatch):
    """When a later stage is mid-flight, revert cancels the active task,
    discards uncommitted workspace changes from it, and proceeds with the
    checkout — the user no longer has to wait for the next gate."""
    sink = _stub_revert_environment(monkeypatch, restored_state={
        "topic": "t", "current_stage": 3, "phase": "gate",
        "stage_results": {}, "retries": 0, "end_stage": 9,
    }, branch="feat-stage3-deadbe")

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "producer"  # mid-flight
    engine.state["active_employee_id"] = "emp-007"
    engine.state["active_node_id"] = "node-abc"
    engine._save()

    branch = await engine.revert_to_stage(stage=3, instructions="redo with X")

    assert sink["aborted"] == ["emp-007"], "active employee's task must be cancelled before checkout"
    assert sink["discarded"] == [str(tmp_path)], "uncommitted workspace changes must be discarded mid-flight"
    assert sink["checkout_calls"], "checkout must run after cancel + discard"
    assert branch == "feat-stage3-deadbe"
    assert engine.state["current_stage"] == 3
    assert engine.state["phase"] == "producer"
    assert engine.state["active_node_id"] is None
    assert engine.state["active_employee_id"] is None


@pytest.mark.asyncio
async def test_revert_to_stage_cancels_critic_phase_task(tmp_path, monkeypatch):
    """phase=critic is also mid-flight — cancel path must run."""
    sink = _stub_revert_environment(monkeypatch, restored_state={
        "topic": "t", "current_stage": 3, "phase": "gate",
        "stage_results": {}, "retries": 0, "end_stage": 9,
    })

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "critic"
    engine.state["active_employee_id"] = "critic-emp"
    engine._save()

    await engine.revert_to_stage(stage=3, instructions="x")

    assert sink["aborted"] == ["critic-emp"], "critic-phase revert must cancel the critic task"
    assert sink["discarded"] == [str(tmp_path)], "critic-phase revert must scrub the workspace too"


@pytest.mark.asyncio
async def test_revert_at_gate_preserves_manual_workspace_edits(tmp_path, monkeypatch):
    """At a gate, no task is running, so revert must NOT call
    discard_uncommitted_changes — that would silently wipe any manual
    edits the user made between gates. The downstream checkout's
    DirtyWorkspaceError should be the loud signal instead."""
    sink = _stub_revert_environment(monkeypatch, restored_state={
        "topic": "t", "current_stage": 3, "phase": "gate",
        "stage_results": {}, "retries": 0, "end_stage": 9,
    })

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine.state["phase"] = "gate"
    engine.state["active_employee_id"] = "stale-id"  # leftover, not actually running
    engine._save()

    await engine.revert_to_stage(stage=3, instructions="x")

    assert sink["aborted"] == [], "no cancel at a gate"
    assert sink["discarded"] == [], "no destructive workspace scrub at a gate"
    assert sink["checkout_calls"], "checkout still runs"


@pytest.mark.asyncio
async def test_revert_to_stage_validates_stage_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda *a, **k: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine.state["phase"] = "gate"
    engine.state["end_stage"] = 9
    engine._save()

    with pytest.raises(ValueError):
        await engine.revert_to_stage(stage=0, instructions="x")
    with pytest.raises(ValueError):
        await engine.revert_to_stage(stage=10, instructions="x")


@pytest.mark.asyncio
async def test_revert_to_stage_refuses_when_no_employee_with_skill(tmp_path, monkeypatch):
    """Pre-flight check: if there's no agent that can run the producer
    for the target stage, refuse BEFORE touching git. Otherwise the
    user ends up on a new branch with corrupt state and no in-flight
    task — non-recoverable from the UI."""
    checkout_calls = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda *a, **k: None)
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: None)

    from onemancompany.core import project_repo
    monkeypatch.setattr(
        project_repo, "checkout_branch_from_stage",
        lambda *a, **k: checkout_calls.append(a) or "feat-stage3-xxxxxx",
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "gate"
    engine.state["end_stage"] = 9
    engine._save()

    with pytest.raises(pe.RevertNotAllowedError):
        await engine.revert_to_stage(stage=3, instructions="redo with X")

    assert not checkout_calls, "git checkout must not run when no agent can handle the stage"


@pytest.mark.asyncio
async def test_revert_to_stage_raises_when_checkout_loses_state_file(tmp_path, monkeypatch):
    """Critical defence: if for any reason the snapshot we checked out
    has no ``pipeline_state.yaml``, ``_load_state`` returns ``{}``. The
    OLD code silently kept the abandoned branch's state — corrupting
    the new branch. The fix raises explicitly instead."""
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp")
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda *a, **k: None)

    from onemancompany.core import project_repo
    def fake_checkout(repo_dir, iteration, stage, branch_name=None):
        # Simulate a checkout that did NOT restore pipeline_state.yaml
        # (e.g. the file was somehow gitignored, or the snapshot
        # predates the engine writing it).
        from pathlib import Path
        path = Path(repo_dir) / pe.STATE_FILENAME
        if path.exists():
            path.unlink()
        return "feat-stage3-xxxxxx"
    monkeypatch.setattr(project_repo, "checkout_branch_from_stage", fake_checkout)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    engine.state["phase"] = "gate"
    engine.state["end_stage"] = 9
    engine._save()

    with pytest.raises(pe.RevertNotAllowedError, match="pipeline_state.yaml"):
        await engine.revert_to_stage(stage=3, instructions="x")


@pytest.mark.parametrize("dirname,expected_prefix", [
    ("iter_001", "iter_001"),
    ("iter_042", "iter_042"),
    ("workspace", "iter_"),       # legacy/non-standard → synthetic
    ("p7506fc954142", "iter_"),
    ("", "iter_"),
])
def test_iteration_id_normalisation(tmp_path, dirname, expected_prefix):
    """Tag namespaces must not collide across projects with the same
    legacy basename. Standard ``iter_NNN`` dirs use the literal name;
    anything else gets a hashed synthetic id derived from the full path."""
    if dirname:
        proj_dir = tmp_path / dirname
        proj_dir.mkdir()
    else:
        proj_dir = tmp_path  # tmp_path basename is opaque hash → non-standard
    engine = pe.PipelineEngine("p1", str(proj_dir), "topic")
    iid = engine._iteration_id()
    assert iid.startswith(expected_prefix)
    if expected_prefix == "iter_":
        # Hashed: not the standard form, but stable for a given path.
        assert engine._iteration_id() == iid


# ---------------------------------------------------------------------------
# End-to-end with real git subprocess — defends against mocks-only blind spot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revert_real_git_restores_state_and_prunes_stage_results(tmp_path, monkeypatch):
    """Smoke test: no mocks on ``project_repo``. Verifies that:
      1. ``ensure_initialized`` actually creates a git repo.
      2. ``commit_stage`` actually commits + tags after a passed stage.
      3. ``revert_to_stage`` actually checks out a feat branch rooted at
         the previous stage's tag, restoring the workspace files.
      4. The reloaded state contains only ``stage_results`` from before
         the revert point.
    """
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()

    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-x")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda *a, **k: None)

    engine = pe.PipelineEngine("p1", str(iter_dir), "topic")
    engine.state["end_stage"] = 9
    engine._save()

    # Manually walk through two passed stages, emulating what
    # vessel.py + the producer/critic loop would do.
    (iter_dir / "stage3.md").write_text("v1 stage 3 output\n")
    engine.state["current_stage"] = 3
    engine.state["stage_results"] = {"3": "stage 3 result"}
    engine._save()
    # ensure_initialized + first commit happen on first _on_critic_pass.
    from onemancompany.core import project_repo
    project_repo.ensure_initialized(str(iter_dir), iteration="iter_001")
    engine._on_critic_pass("stage 3 result")

    (iter_dir / "stage4.md").write_text("v1 stage 4 output\n")
    engine.state["current_stage"] = 4
    engine.state["stage_results"] = {"3": "stage 3 result", "4": "stage 4 result"}
    engine._save()
    engine._on_critic_pass("stage 4 result")

    # Sanity: both files present on the main branch.
    assert (iter_dir / "stage3.md").exists()
    assert (iter_dir / "stage4.md").exists()

    # Now revert to stage 4. Expected: branch from iter_001/stage-3's
    # commit, so stage4.md disappears from the workspace.
    branch = await engine.revert_to_stage(stage=4, instructions="please rewrite stage 4 to use approach Y")

    assert branch.startswith("feat-stage4-"), branch
    assert (iter_dir / "stage3.md").exists(), "stage 3's output must survive the revert"
    assert not (iter_dir / "stage4.md").exists(), "stage 4's output must be checked-out away"

    # Reloaded state has only stage_results from BEFORE the revert.
    assert "3" in engine.state["stage_results"]
    assert "4" not in engine.state["stage_results"]
    assert engine.state["current_stage"] == 4
    assert engine.state["phase"] == "producer"
    assert engine.state["retries"] == 0


# ---------------------------------------------------------------------------
# Stage 7 (Result Analysis) — pre-registration contract + critic gating
# ---------------------------------------------------------------------------


def test_dispatch_producer_stage7_injects_result_analysis_runbook_trigger(tmp_path, monkeypatch):
    """Stage 7 producer task description must instruct the agent to load
    the result-analysis-runbook skill so the analyst obeys the
    pre-registration contract (no HARKing)."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine._dispatch_producer()

    assert dispatched, "Stage 7 producer must dispatch"
    desc = dispatched[0][1]
    assert 'load_skill("result-analysis-runbook")' in desc, (
        "Stage 7 task description must instruct the producer to load the "
        "result-analysis-runbook skill"
    )
    assert "pre-registration" in desc.lower() or "pre-registered" in desc.lower(), (
        "Stage 7 task description must reference the pre-registration "
        "contract from Stage 4/5"
    )
    assert "HARK" in desc, (
        "Stage 7 task description must call out the no-HARKing rule so "
        "the producer treats it as a hard contract"
    )


def test_dispatch_producer_stage7_requires_result_figures(tmp_path, monkeypatch):
    """Stage 7 producer must be instructed to GENERATE result figures
    (stage7_*.png) per the Stage 5 figure manifest via the result-figures
    skill. Audit B1: all 5 historical papers embedded only the Stage-4
    framework figure and ZERO result figures — because the critic checked
    for figures (D11) and a result-figures skill exists, but the producer
    was never told to render them. This wires the missing middle step."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine._dispatch_producer()

    desc = dispatched[0][1]
    low = desc.lower()
    assert "result-figures" in low or "stage7_" in low, (
        "Stage 7 dispatch must instruct generating result figures "
        "(result-figures skill / stage7_*.png)"
    )
    assert "figure manifest" in low or "manifest" in low, (
        "Stage 7 dispatch must tie figure generation to the Stage 5 figure manifest"
    )
    assert ".png" in low, (
        "Stage 7 dispatch must require concrete PNG figure outputs"
    )


def test_dispatch_producer_stage8_enforces_number_traceability(tmp_path, monkeypatch):
    """Stage 8 producer must be told the #44 number-traceability rule + run a
    self-check before submit. Run 32e19fc5f3e7 died at Stage 8 because the
    paper-writer fabricated high-precision stats not in the Stage 4-7 evidence
    and the deterministic gate rejected it across all retries."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 8
    engine._dispatch_producer()

    low = dispatched[0][1].lower()
    assert "stage7_result_analyst.md" in low or "result_json" in low, (
        "Stage 8 dispatch must name the evidence numbers must trace to"
    )
    assert "traceab" in low and ("self-check" in low or "self check" in low), (
        "Stage 8 dispatch must state the traceability rule AND mandate a self-check"
    )
    assert "fabricat" in low or "invent" in low or "recompute" in low, (
        "Stage 8 dispatch must forbid inventing/recomputing numbers"
    )


def test_dispatch_critic_stage7_injects_result_quality_critic(tmp_path, monkeypatch):
    """Stage 7 critic dispatch must instruct the reviewer to load the
    result-quality-critic runbook so HARKing is auto-REJECTED."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine._dispatch_critic("Stage 7 producer report")

    assert dispatched, "Stage 7 critic must be dispatched"
    desc = dispatched[0][1]
    assert 'load_skill("result-quality-critic")' in desc, (
        "Stage 7 critic prompt must instruct the reviewer to load the "
        "result-quality-critic runbook"
    )
    assert "HARK" in desc, (
        "Stage 7 critic prompt must explicitly call out HARKing as the "
        "primary failure mode"
    )
    assert "auto-REJECT" in desc, (
        "Stage 7 critic prompt must mention the auto-REJECT triggers "
        "(HARKing / fabrication / non-English)"
    )


def test_dispatch_critic_stage8_requires_references_figures_and_style(tmp_path, monkeypatch):
    """#45 (output-quality fix C): the Stage-8 critic prompt must require a
    resolvable References section (run a1df5c26f6ea shipped a paper with
    inline citations and NO References section), embedded stage7 figures,
    statistics style, and a headline-number spot-check."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 8
    engine._dispatch_critic("Stage 8 paper draft")

    assert dispatched, "Stage 8 critic must be dispatched"
    desc = dispatched[0][1]
    assert "References section" in desc
    assert "stage7_*.png" in desc, "must check the Stage-7 figures are embedded"
    assert "p < .001" in desc, "must enforce the statistics style rules"
    assert "stage7_result_analyst.md" in desc, "must spot-check headline numbers"


def test_dispatch_producer_stage7_not_in_other_stages(tmp_path, monkeypatch):
    """The Stage 7 trigger must be stage-scoped — Stages 3/4/5/6 producers
    must not carry the result-analysis-runbook trigger."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: f"emp-{skill}" if skill else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    for stage_id in (3, 4, 5, 6):
        dispatched.clear()
        engine = pe.PipelineEngine(f"p{stage_id}", str(tmp_path), "topic")
        engine.state["current_stage"] = stage_id
        engine._dispatch_producer()
        if dispatched:
            assert "result-analysis-runbook" not in dispatched[0][1], (
                f"Stage {stage_id} producer must not carry the Stage 7 "
                f"result-analysis-runbook trigger"
            )


def test_dispatch_critic_stage7_not_in_other_stages(tmp_path, monkeypatch):
    """The Stage 7 critic trigger must be stage-scoped — Stages 4/5/6
    critics must not carry the result-quality-critic trigger."""
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "critic-1" if skill == pe.CRITIC_SKILL else None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: dispatched.append(args))

    for stage_id in (4, 5, 6):
        dispatched.clear()
        engine = pe.PipelineEngine(f"p{stage_id}", str(tmp_path), "topic")
        engine.state["current_stage"] = stage_id
        engine._dispatch_critic(f"Stage {stage_id} producer report")
        if dispatched:
            assert "result-quality-critic" not in dispatched[0][1], (
                f"Stage {stage_id} critic must not carry the Stage 7 "
                f"result-quality-critic trigger"
            )



# ---------------------------------------------------------------------------
# paper-framework-figure dispatch wiring (Stage 4 final step + Stage 8 first step)
# ---------------------------------------------------------------------------

def test_stage4_desc_triggers_paper_framework_figure_after_methodology():
    """Stage 4 task description must tell the methodology agent to
    render the framework figure AFTER the methodology is written.
    Without this trigger, the bundled paper-framework-figure skill
    sits unused in the agent's skills/ dir."""
    from onemancompany.core import pipeline_engine
    import inspect
    src = inspect.getsource(pipeline_engine)
    assert 'load_skill("paper-framework-figure")' in src
    # Stage 4 specifically — appears in the Stage 4 branch (REQUIRED FINAL STEP)
    assert "REQUIRED FINAL STEP" in src
    assert "stage4_framework_figure.png" in src


def test_stage8_desc_reuses_stage4_figure_does_not_regenerate():
    """Stage 8 must REUSE stage4_framework_figure.png by reference, NOT
    call paper-framework-figure to regenerate it (which would burn API
    budget + produce a potentially inconsistent figure). The CCF-A
    section list is still required."""
    from onemancompany.core import pipeline_engine
    import inspect
    src = inspect.getsource(pipeline_engine)
    # Stage 8 branch must exist
    assert 'stage["id"] == 8' in src
    # Must reference the existing PNG by path
    assert "stage4_framework_figure.png" in src
    # Must explicitly forbid regeneration. Grab the Stage 8 desc block.
    after_marker = src.split('stage["id"] == 8', 1)[1]
    # Stage 8 is the last elif; the generic `desc += (\n            f"\nYour task` line
    # marks the end of stage-specific dispatching. Cut there.
    stage8 = after_marker.split('f"\\nYour task', 1)[0]
    assert ("Do NOT call" in stage8) or ("do NOT regenerate" in stage8.lower()), (
        "Stage 8 desc must explicitly forbid figure regeneration"
    )
    assert "Abstract" in stage8 and "Reproducibility" in stage8


# ---------------------------------------------------------------------------
# STAGE_TALENT_DEFAULTS — canonical default employee per stage from hire_list
# (PR #67, merged into the 6a/6b architecture)
# ---------------------------------------------------------------------------

def _talent_config(name: str, skills: list[str], talent_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, skills=skills, talent_id=talent_id)


def test_stage_talent_defaults_maps_each_stage_to_hire_list_talent():
    """Every pipeline stage (1..9) must have a canonical default talent
    drawn from company/hire_list.json so the producer is deterministic
    when multiple hired employees share the same skill."""
    import json
    from pathlib import Path

    expected_stages = {3, 4, 5, 6, 7, 8, 9}
    assert set(pe.STAGE_TALENT_DEFAULTS.keys()) == expected_stages

    hire_list_path = Path(pe.__file__).resolve().parents[3] / "company" / "hire_list.json"
    talent_ids = {e["talent_id"] for e in json.loads(hire_list_path.read_text())}
    for sid, tid in pe.STAGE_TALENT_DEFAULTS.items():
        assert tid in talent_ids, f"Stage {sid} default '{tid}' not in hire_list.json"


def test_frontend_stages_talent_ids_match_backend_defaults():
    """The frontend ``STAGES`` array in ``frontend/index.html`` declares a
    ``talent`` string per stage that the picker uses to surface the
    canonical agent name. It must stay aligned with backend
    ``STAGE_TALENT_DEFAULTS`` — if either side is edited without the
    other, the dropdown silently falls back to ``Auto`` for the drifted
    stage. Lock the mapping by parsing the HTML."""
    import re
    from pathlib import Path

    index_html = Path(pe.__file__).resolve().parents[3] / "frontend" / "index.html"
    src = index_html.read_text(encoding="utf-8")
    # Match e.g.  {id:4,name:'Methodology Design',talent:'methodology-designer',...}
    pattern = re.compile(r"\{id:(\d+),[^}]*talent:'([^']+)'")
    frontend = {int(sid): tid for sid, tid in pattern.findall(src)}

    assert frontend == pe.STAGE_TALENT_DEFAULTS, (
        "Frontend STAGES.talent ↔ backend STAGE_TALENT_DEFAULTS drifted.\n"
        f"  frontend: {frontend}\n"
        f"  backend:  {pe.STAGE_TALENT_DEFAULTS}\n"
        "If you change one, change the other (and re-verify the picker "
        "default labels)."
    )


def test_find_employee_by_talent_id_returns_matching_employee(monkeypatch):
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "00010": _talent_config("Topic A", ["topic_refiner"], talent_id="other"),
            "00011": _talent_config("Topic B", ["topic_refiner"], talent_id="topic-refiner"),
        },
    )
    assert pe._find_employee_by_talent_id("topic-refiner") == "00011"
    assert pe._find_employee_by_talent_id("missing") is None


def test_find_employee_for_stage_prefers_canonical_talent(monkeypatch):
    """Two employees both carry the stage's primary skill — the one hired
    from the canonical hire_list talent_id wins."""
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "emp-clone": _talent_config("Clone", ["idea_generator"], talent_id=""),
            "emp-canon": _talent_config("Canon", ["idea_generator"], talent_id="idea-generator"),
        },
    )
    assert pe._find_employee_for_stage(3, "idea_generator") == "emp-canon"


def test_find_employee_for_stage_falls_back_to_skill_when_no_canonical(monkeypatch):
    """No employee carries the canonical talent_id — fall back to the
    existing skill-based lookup so the pipeline still runs."""
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "emp-clone": _talent_config("Clone", ["topic_refiner"], talent_id=""),
        },
    )
    assert pe._find_employee_for_stage(1, "topic_refiner") == "emp-clone"


def test_find_employee_for_stage_6_code_implementer_preference_wins(monkeypatch):
    """Stage 6's initial dispatch is Stage 6a — a ``code_implementer``
    employee wins over both the canonical experimentalist talent AND any
    experiment_runner on the roster (the runner is reserved for Stage 6b,
    dispatched separately by ``on_task_complete``)."""
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "emp-canon": _talent_config("Sim", ["experimentalist"], talent_id="experimentalist"),
            "emp-runner": _talent_config("Runner", ["experiment_runner"], talent_id="experiment-runner"),
            "emp-coder": _talent_config("Coder", ["code_implementer"], talent_id="experiment-code-writer"),
        },
    )
    # Stage 6a — code_implementer wins
    assert pe._find_employee_for_stage(6, "experimentalist") == "emp-coder"
    # Stage 6b — experiment_runner wins (canonical "experimentalist" is the
    # last-resort fallback below it; runner skill is the primary preference)
    assert pe._find_stage_6b_employee() == "emp-runner"


def test_producer_b_stub_routes_to_6a_rebuild(tmp_path, monkeypatch):
    """UPDATED for #20: a stub return from Stage 6b (producer_b phase) routes
    back to 6a (``_dispatch_producer``) to REBUILD the code, NOT re-run the
    same runner on the same (usually broken) code. Re-running the runner on a
    stub just stubs again — observed 3× → total failure in run 3f644a5996bb.
    6a's completion re-dispatches a fresh 6b, so transient runner hiccups
    also recover."""
    redispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-runner" if skill == "experiment_runner" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee", lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                        lambda self, feedback="": redispatched.append(("a", feedback)))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b",
                        lambda self, feedback="": redispatched.append(("b", feedback)))

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"

    engine.on_task_complete("emp", "n1", "Executed: bash")

    assert redispatched and redispatched[0][0] == "a", (
        f"Stub at producer_b must route to 6a (rebuild), got {redispatched!r}"
    )
    assert "stub" in redispatched[0][1].lower()


def test_producer_stub_exhausted_opens_ceo_gate(tmp_path, monkeypatch):
    """When stub-result retries hit MAX_RETRIES, the stage holds at the
    CEO gate (rather than looping forever or auto-passing)."""
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-meth")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "producer"
    engine.state["retries"] = pe.MAX_RETRIES  # already exhausted

    engine.on_task_complete("emp", "n1", "Executed: bash")

    assert engine.state["phase"] == "gate", (
        f"Stub at exhausted retries must hold for CEO; got phase={engine.state['phase']!r}"
    )


def test_stage6a_hard_gate_exhausted_opens_ceo_gate(tmp_path, monkeypatch):
    """Hard-gate retries also cap at MAX_RETRIES — beyond that the stage
    holds at the CEO gate. Otherwise an LLM that keeps skipping Phase 5
    would loop indefinitely."""
    monkeypatch.setattr(pe, "_find_employee_by_skill",
                        lambda skill: "emp-coder" if skill == "code_implementer" else None)
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event",
                        lambda self, *args, **kwargs: None)

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer"
    engine.state["retries"] = pe.MAX_RETRIES  # already exhausted
    # NB: no receipt file → hard-gate would fire if retries allowed

    # Non-stub result so we reach the hard-gate (not the stub gate)
    engine.on_task_complete("emp-coder", "n1", "Real-but-incomplete output (no receipt, no commits)" * 10)

    assert engine.state["phase"] == "gate", (
        f"Hard-gate exhausted must hold for CEO; got phase={engine.state['phase']!r}"
    )


def test_stage3_uses_file_deliverable_when_present(tmp_path, monkeypatch):
    """Stage 3's actual deliverable is the literature-conflict-graph file
    on disk, not the agent's chat summary. When the file exists with the
    expected header, ``on_task_complete`` swaps the chat result for the
    file content so the critic sees the same thing the UI renders. (PR #67.)"""
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda skill: "emp-critic")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *args: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *args, **kwargs: None)

    # Stage 3 deliverable file with the expected header
    stage_skill = pe.STAGES[0]["skill"]  # stage 3 is the entry stage (index 0)
    (tmp_path / f"stage3_{stage_skill}.md").write_text(
        "# Selected Hypotheses\n\nH1: ...\nH2: ...\n",
        encoding="utf-8",
    )

    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 3
    engine.state["phase"] = "producer"

    engine.on_task_complete("emp", "n1", "quick summary from chat, not the full graph")

    # The file content (with the header) — not the chat summary — was stored
    stored = engine.state["stage_results"]["3"]
    assert "# Selected Hypotheses" in stored
    assert "H1:" in stored


def test_find_stage_6b_falls_back_to_canonical_experimentalist(monkeypatch):
    """When no experiment_runner is hired, Stage 6b falls back to the
    canonical ``experimentalist`` talent_id before any random
    experimentalist-skilled employee."""
    monkeypatch.setattr(
        pe,
        "load_employee_configs",
        lambda: {
            "emp-clone": _talent_config("Clone", ["experimentalist"], talent_id=""),
            "emp-canon": _talent_config("Canon", ["experimentalist"], talent_id="experimentalist"),
        },
    )
    assert pe._find_stage_6b_employee() == "emp-canon"


# ---------------------------------------------------------------------------
# Stage 8 paper-writer output-format branches
# ---------------------------------------------------------------------------

def _setup_stage8(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(pe, "_find_employee_by_skill", lambda s: "emp-pw")
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *a: dispatched.append(a))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *a, **k: None)
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 8
    return engine, dispatched


def test_stage8_dispatch_default_markdown(tmp_path, monkeypatch):
    engine, dispatched = _setup_stage8(tmp_path, monkeypatch)
    engine._dispatch_producer()
    desc = dispatched[0][1]
    # Output-format directive line is "output_format=markdown" with no
    # trailing " venue=…" appended — that branch only fires for latex/both.
    assert "output_format=markdown\n" in desc
    assert "stage4_framework_figure.png" in desc


def test_stage8_dispatch_latex_uses_default_venue(tmp_path, monkeypatch):
    engine, dispatched = _setup_stage8(tmp_path, monkeypatch)
    engine.state["paper_config"] = {"output_format": "latex"}
    engine._dispatch_producer()
    desc = dispatched[0][1]
    assert "output_format=latex" in desc
    assert "venue=iclr2026" in desc


def test_stage8_dispatch_both_with_explicit_venue(tmp_path, monkeypatch):
    engine, dispatched = _setup_stage8(tmp_path, monkeypatch)
    engine.state["paper_config"] = {"output_format": "both", "venue": "neurips2025"}
    engine._dispatch_producer()
    desc = dispatched[0][1]
    assert "output_format=both" in desc
    assert "venue=neurips2025" in desc


def test_stage8_dispatch_docx_skips_venue(tmp_path, monkeypatch):
    engine, dispatched = _setup_stage8(tmp_path, monkeypatch)
    engine.state["paper_config"] = {"output_format": "docx", "venue": "iclr2026"}
    engine._dispatch_producer()
    desc = dispatched[0][1]
    # docx skips the venue branch — even if venue is set in paper_config,
    # the rendered directive line must end without a " venue=…" suffix.
    assert "output_format=docx\n" in desc


# ---------------------------------------------------------------------------
# _auto_approve_gate — unattended mode
# ---------------------------------------------------------------------------

def test_auto_approve_gate_advances_when_phase_is_gate(tmp_path, monkeypatch):
    import asyncio
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "gate"
    called = []
    monkeypatch.setattr(engine, "on_ceo_approve", lambda txt: called.append(txt))
    asyncio.run(engine._auto_approve_gate(stage_id=3, exhausted=False))
    assert called == [""]


def test_auto_approve_gate_no_op_when_phase_left_gate(tmp_path, monkeypatch):
    import asyncio
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["phase"] = "producer"
    called = []
    monkeypatch.setattr(engine, "on_ceo_approve", lambda txt: called.append(txt))
    asyncio.run(engine._auto_approve_gate(stage_id=3, exhausted=True))
    assert called == []


# ---------------------------------------------------------------------------
# Memory-store exception paths in _retrieve_memory_guidance /
# _record_stage_memory / _apply_ceo_memory_feedback. The pipeline must keep
# running when the research-memory layer fails; it must NOT propagate.
# ---------------------------------------------------------------------------

class _BoomStore:
    """Memory store stand-in that raises on every method."""
    def retrieve_stage_guidance(self, **kw): raise RuntimeError("retrieve boom")
    def record_stage_episode(self, **kw): raise RuntimeError("record boom")
    def apply_ceo_feedback(self, **kw): raise RuntimeError("feedback boom")


def test_retrieve_memory_guidance_swallows_store_errors(tmp_path, monkeypatch):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    monkeypatch.setattr(engine, "_memory_store", lambda: _BoomStore())
    assert engine._retrieve_memory_guidance({"id": 1, "name": "x", "skill": "y"}, "ctx") == ""


def test_record_stage_memory_swallows_store_errors(tmp_path, monkeypatch):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    monkeypatch.setattr(engine, "_memory_store", lambda: _BoomStore())
    out = engine._record_stage_memory(
        {"id": 1, "name": "x", "skill": "y"},
        producer_result="p", critic_result="c",
        passed=True, confidence=0.5, outcome="critic_pass",
    )
    assert out is None


def test_apply_ceo_memory_feedback_swallows_store_errors(tmp_path, monkeypatch):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    # Seed an episode id so apply_ceo_memory_feedback reaches the try block.
    engine.state["memory_episodes"] = {"1": "ep-1"}
    monkeypatch.setattr(engine, "_memory_store", lambda: _BoomStore())
    # Should not raise.
    engine._apply_ceo_memory_feedback({"id": 1}, "feedback", approved=True)


# ---------------------------------------------------------------------------
# _dispatch_producer_b — guard branches
# ---------------------------------------------------------------------------

def test_dispatch_producer_b_rejects_non_stage_6(tmp_path, monkeypatch):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 5
    # Should log error and return without changing phase.
    engine._dispatch_producer_b()
    assert engine.state["phase"] == "producer"  # untouched


def test_dispatch_producer_b_marks_failed_when_no_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "_find_stage_6b_employee", lambda: None)
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine._dispatch_producer_b()
    assert engine.state["phase"] == "failed"


def test_find_employee_by_talent_id_returns_none_for_empty(monkeypatch):
    monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
    assert pe._find_employee_by_talent_id("") is None


def test_queue_pending_feedback_ignores_empty_text(tmp_path):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.queue_pending_feedback("")
    engine.queue_pending_feedback("   \n  ")
    assert engine.state.get("pending_user_feedback", "") == ""


def test_queue_pending_feedback_appends_to_existing(tmp_path):
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.queue_pending_feedback("first")
    engine.queue_pending_feedback("second")
    assert "first" in engine.state["pending_user_feedback"]
    assert "second" in engine.state["pending_user_feedback"]


def test_on_task_failed_unexpected_phase_is_ignored(tmp_path):
    """on_task_failed in gate/done phases shouldn't change state — just
    log and return. Covers the defensive guard added after a race report."""
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 4
    engine.state["phase"] = "gate"
    before = dict(engine.state)
    engine.on_task_failed("emp", "node", "boom")
    # State must be unchanged.
    assert engine.state["phase"] == before["phase"]
    assert engine.state["retries"] == before["retries"]


def test_dispatch_producer_b_injects_feedback_and_user_feedback(tmp_path, monkeypatch):
    dispatched = []
    monkeypatch.setattr(pe, "_find_stage_6b_employee", lambda: "emp-runner")
    monkeypatch.setattr(pe, "load_employee_configs",
                        lambda: {"emp-runner": _employee_config("Runner", ["experiment_runner"])})
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                        lambda self, *a: dispatched.append(a))
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                        lambda self, *a, **k: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["pending_user_feedback"] = "ceo direct guidance"
    engine._dispatch_producer_b(feedback="critic notes")

    assert dispatched, "Stage 6b must dispatch"
    desc = dispatched[0][1]
    assert "critic notes" in desc
    assert "ceo direct guidance" in desc
    # Pending user feedback is consumed.
    assert engine.state.get("pending_user_feedback", "") == ""


# ===========================================================================
# #27 Hard data gate — deterministic post-critic check (closes #96 A+C, #94)
# ===========================================================================


def test_data_gate_stage6_fails_when_no_run_succeeded():
    """Stage 6 data gate: a runner report whose every run is non-succeeded
    (still_running / blocked / failed) has no usable experimental data —
    the gate must FAIL even if the critic graded the report well-written."""
    report = (
        "## Tasks executed\n\n"
        "- run_id: run_a\n- status: still_running\n\n"
        "- run_id: run_b\n- status: blocked\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(6, report)
    assert ok is False
    assert "succeed" in reason.lower() or "run" in reason.lower()


def test_data_gate_stage6_passes_when_one_run_succeeded():
    """Stage 6 gate passes as soon as one run is succeeded, even if others
    are still_running (the long-running waiter already finalized the rest)."""
    report = (
        "- run_id: run_a\n- status: succeeded\n\n"
        "- run_id: run_b\n- status: still_running\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(6, report)
    assert ok is True, reason


def test_data_gate_stage7_fails_when_all_hypotheses_not_tested():
    """Stage 7 gate: if every hypothesis row is NOT TESTED, there is no
    confirmatory result — the gate FAILs regardless of how polished the
    analysis prose is (the 70cb46f4e26a INCONCLUSIVE-paper failure mode)."""
    report = (
        "## Hypotheses\n\n"
        "### H1\nDecision: NOT TESTED\n\n"
        "### H2\nDecision: NOT TESTED\n\n"
        "### H3\nDecision: NOT TESTED\n\n"
        "Overall: INCONCLUSIVE_DUE_TO_COVERAGE\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(7, report)
    assert ok is False
    assert "tested" in reason.lower() or "hypothes" in reason.lower()


def test_data_gate_stage7_passes_with_one_supported_hypothesis():
    """One SUPPORTED/REJECTED hypothesis with a real decision is enough to
    clear the Stage 7 gate."""
    report = (
        "### H1\nTest result: t=3.2 (95% CI [1.1, 5.3]), effect size: 0.6\n"
        "Decision: SUPPORTED\n\n"
        "### H2\nDecision: NOT TESTED\n\n"
        "Overall: PARTIALLY CONFIRMED\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(7, report)
    assert ok is True, reason


def test_data_gate_stage7_reads_horizontal_results_table():
    """Sibling of the Stage-6 table bug (run fa50cb183b3c): the result-analyst
    commonly writes one row per hypothesis in a Markdown table with a Decision
    column. The gate must read that column — not only inline ``Decision:`` —
    so a valid analysis isn't falsely failed for formatting."""
    report = (
        "## Results\n\n"
        "| Hypothesis | Decision | p-value | effect |\n"
        "|---|---|---|---|\n"
        "| H1 | SUPPORTED | 0.01 | 0.6 |\n"
        "| H2 | NOT SUPPORTED | 0.20 | 0.1 |\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(7, report)
    assert ok is True, reason


def test_data_gate_stage7_reads_vertical_decision_table():
    """The vertical key-value layout ``| Decision | SUPPORTED |`` must parse,
    while a horizontal HEADER (Decision as a middle column) must NOT capture
    the next column name as a verdict."""
    report = (
        "## H1\n| Field | Value |\n|---|---|\n| Decision | `SUPPORTED` |\n| p | 0.01 |\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(7, report)
    assert ok is True, reason


def test_data_gate_stage7_table_still_blocks_all_no_data():
    """The table reader must not WEAKEN the gate: an all-NOT-TESTED/BLOCKED
    results table still fails (no experiment ran)."""
    report = (
        "| Hypothesis | Verdict |\n|---|---|\n"
        "| H1 | NOT TESTED |\n| H2 | BLOCKED |\n"
    )
    ok, reason = pe.PipelineEngine._data_gate(7, report)
    assert ok is False
    assert "no-data" in reason.lower() or "no experiment" in reason.lower()


def test_data_gate_stage7_ignores_prose_supported():
    """A stray "supported" in prose (no decision label, no table) must NOT
    pass the gate — the column reader only reads designated cells."""
    report = "The direction is well supported by prior work, but we ran nothing.\n"
    ok, _ = pe.PipelineEngine._data_gate(7, report)
    assert ok is False


def test_data_gate_blocks_stage6_advance_when_critic_passed_but_no_data(tmp_path, monkeypatch):
    """The flagship #27 behavior: critic votes PASS on a well-written Stage 6
    report, but every run is still_running (no real data). The data gate must
    override the PASS — the stage does NOT advance to gate; it routes to the
    critic-reject retry path (which #106 then auto-fails on exhaustion)."""
    redispatched = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": redispatched.append(feedback))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b", lambda self, feedback="": redispatched.append(feedback))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "6": "- run_id: run_a\n- status: still_running\n",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")

    # Did NOT advance to gate — the gate override turned PASS into a reject.
    assert engine.phase != "gate", f"must not advance; phase={engine.phase}"
    assert engine.state["retries"] == 1
    assert redispatched, "must re-dispatch (reject path)"
    assert "DATA_GATE_FAIL" in redispatched[0]


def test_data_gate_allows_stage6_advance_when_data_present(tmp_path, monkeypatch):
    """Control: a Stage 6 report WITH a succeeded run passes the gate and
    advances to the CEO gate exactly as before (no regression)."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "6": "- run_id: run_a\n- status: succeeded\n",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")

    assert getattr(engine, "_passed", False) is True, "must call _on_critic_pass when data present"


def test_data_gate_stage8_fails_when_upstream_stage7_has_no_data(tmp_path, monkeypatch):
    """Stage 8 (paper writing) must not advance if the upstream Stage 7
    result analysis had no confirmatory data — the paper would be written
    about nothing (#96 Failure C)."""
    redispatched = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": redispatched.append(feedback))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 8
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "7": "### H1\nDecision: NOT TESTED\n\n### H2\nDecision: NOT TESTED\n",
        "8": "# Paper\n\nSome well-written abstract and intro.",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.98")

    assert engine.phase != "gate", f"must not advance; phase={engine.phase}"
    assert redispatched, "must re-dispatch (reject path)"
    assert "Stage 7" in redispatched[0]


def test_stage8_numbers_gate_includes_upstream_audit_artifacts(tmp_path):
    """#44 evidence must include upstream/ audit/result artifacts. The
    diagnostic numbers (coef L2-norms, positive-control, per-fold accuracy)
    live in audit.jsonl, NOT in RESULT_JSON — run fbb851a3d131 shipped a
    fabricated coef-norm 0.284 (real 2.34 in audit.jsonl) that the gate missed
    because audit.jsonl was not in its evidence set. Now it is."""
    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    skill7 = engine._stage_def(7).get("skill", "")
    (tmp_path / f"stage7_{skill7}.md").write_text(
        "### H1\nDecision: SUPPORTED\nacc_scaled_mean 0.9789, acc_raw_mean 0.9508\n",
        encoding="utf-8")
    up = tmp_path / "upstream"; up.mkdir()
    (up / "audit.jsonl").write_text(
        '{"fold":0,"coef_l2_raw":2.34,"coef_l2_scaled":3.68}\n', encoding="utf-8")
    skill8 = engine._stage_def(8).get("skill", "")
    paper8 = tmp_path / f"stage8_{skill8}.md"

    # (a) cites the REAL audit values → traceable → pass
    paper8.write_text(
        "Raw coef L2-norm 2.34, scaled 3.68. Accuracy 0.9789 vs 0.9508.\n", encoding="utf-8")
    ok, _ = engine._stage_data_gate(8, paper8.read_text())
    assert ok, "values present in upstream/audit.jsonl must be traceable now"

    # (b) cites FABRICATED diagnostics absent from every artifact → reject
    paper8.write_text(
        "Raw coef L2-norm 0.284, scaled 2.87. Accuracy 0.9789 vs 0.9508.\n", encoding="utf-8")
    ok2, reason = engine._stage_data_gate(8, paper8.read_text())
    assert not ok2 and "traceable" in reason.lower(), (
        "fabricated high-precision diagnostics must be caught now that "
        f"audit.jsonl is in the evidence set (got ok={ok2}, reason={reason!r})"
    )


def test_data_gate_stage8_passes_when_upstream_stage7_has_data(tmp_path, monkeypatch):
    """Control: Stage 8 advances when Stage 7 carried a real tested hypothesis."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 8
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "7": "### H1\nTest result: t=3.2\nDecision: SUPPORTED\n",
        "8": "# Paper\n\nResults show H1 supported.",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")

    assert getattr(engine, "_passed", False) is True


class TestPaperNumbersGate:
    """#44 (output-quality fix B): every high-precision statistic in the
    Stage-8 paper must be traceable to the upstream Stage 4-7 evidence.
    Number fabrication is the worst paper failure mode and the Stage 8
    critic only spot-checks; this is the deterministic backstop."""

    def test_traceable_numbers_pass_including_percent_scale_and_rounding(self):
        paper = (
            "Direct-512 accuracy is 16.38% and CoT-512 is 85.97%, a paired "
            "difference of 69.60 percentage points (Cohen's h = 1.5406)."
        )
        upstream = (
            'RESULT_JSON: {"accuracy_direct": 0.16376, "accuracy_cot": 0.859742}\n'
            "paired diff: 0.695982\nCohen's h: 1.5406\n"
        )
        ok, reason = pe.PipelineEngine._paper_numbers_gate(paper, upstream)
        assert ok, reason

    def test_fabricated_statistics_fail_and_are_named(self):
        paper = (
            "Accuracy reached 87.42% with an effect size of 2.3145, while "
            "the baseline scored 12.99%."
        )
        upstream = 'RESULT_JSON: {"accuracy_cot": 0.8597, "accuracy_direct": 0.1638}'
        ok, reason = pe.PipelineEngine._paper_numbers_gate(paper, upstream)
        assert not ok
        assert "87.42" in reason and "2.3145" in reason

    def test_years_sections_counts_and_low_precision_ignored(self):
        """Citation years, section refs, sample sizes, token budgets and
        p < 0.001 style values must never trip the gate."""
        paper = (
            "Following [Wei et al., 2022] and Section 5.1, we evaluate all "
            "1,319 problems with max_tokens 512 (p < 0.001, alpha 0.05)."
        )
        ok, reason = pe.PipelineEngine._paper_numbers_gate(paper, "no numbers here")
        assert ok, reason

    def test_single_orphan_tolerated_but_reported(self):
        """One unmatched high-precision number could be a legitimate derived
        value (e.g. an author-computed ratio) — tolerated, but named in the
        reason so the critic can eyeball it. Two or more → fail."""
        paper = "CoT reaches 85.97%; the speedup factor is 58.38x."
        upstream = '"accuracy_cot": 0.8597'
        ok, reason = pe.PipelineEngine._paper_numbers_gate(paper, upstream)
        assert ok
        assert "58.38" in reason

    def test_stage8_gate_rejects_fabricated_paper(self, tmp_path, monkeypatch):
        """Wire-level: critic PASS on a paper with ≥2 fabricated statistics
        must flip to the reject path, naming the orphans in the feedback."""
        redispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": redispatched.append(feedback))

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 8
        engine.state["phase"] = "critic"
        engine.state["stage_results"] = {
            "7": "### H1\nDecision: SUPPORTED\naccuracy_cot 0.8597, accuracy_direct 0.1638\n",
            "8": "# Paper\n\nAccuracy hit 91.23% with effect size 3.1415.",
        }

        engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")

        assert redispatched, "fabricated numbers must trigger the reject path"
        assert "91.23" in redispatched[0]

    def test_stage8_gate_passes_traceable_paper(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 8
        engine.state["phase"] = "critic"
        engine.state["stage_results"] = {
            "7": "### H1\nDecision: SUPPORTED\naccuracy_cot 0.8597, accuracy_direct 0.1638\n",
            "8": "# Paper\n\nCoT reaches 85.97% versus 16.38% for direct.",
        }

        engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")

        assert getattr(engine, "_passed", False) is True


def test_stub_detects_giant_tool_echo(tmp_path):
    """REGRESSION (R13-1, run df3fd56612e5): the 6b runner ended with an
    852KB tool-echo ('Executed: bash\\nbash → {full /api/list_runs dump}').
    The <300-char heuristic missed it, the waiter's fallback parse then
    swallowed 100 account-wide run_ids as pending and the pipeline parked
    on 99 ghosts. The 'Executed: ' prefix IS the runtime's fallback
    signature — flag it regardless of length."""
    giant = "Executed: bash\nbash → {'stdout': '" + ("x" * 900_000) + "'}"
    assert pe.PipelineEngine._is_stub_result(giant) is True
    assert pe.PipelineEngine._is_stub_result("Executed tools: write, read" + "y" * 5000) is True
    # Real reports stay non-stub.
    real = "# Stage 6 report\n\n- run_id: run_a\n- status: succeeded\n" + "analysis " * 100
    assert pe.PipelineEngine._is_stub_result(real) is False


def test_parking_scopes_implausible_pending_to_known_project_runs(tmp_path, monkeypatch):
    """REGRESSION (R13-1): when the parsed pending list is implausibly large
    (an account-wide dump leaked into the parse), scope it to the runs the
    tracker already attributes to THIS project instead of parking on the
    whole account's history."""
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_critic", lambda self, r: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"
    # Tracker already attributed ONE running run to this project.
    engine.state["stage_6_runs"] = {"run_mine": {"status": "running"}}

    rows = "\n".join(
        f"- run_id: run_ghost_{i}\n- status: running" for i in range(40)
    )
    report = f"## Runs\n- run_id: run_mine\n- status: running\n{rows}\n"

    engine.on_task_complete("00025", "nodeB", report)

    assert engine.state["phase"] == "producer_b_waiting"
    assert engine.state["pending_run_ids"] == ["run_mine"], (
        "an implausibly large parse must be scoped to the tracker-known "
        "project runs, not 41 account-wide ids"
    )


def test_parking_small_pending_kept_even_without_tracker_map(tmp_path, monkeypatch):
    """Normal path: a fresh park lists its own few runs before the tracker's
    first poll — keep them verbatim."""
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_critic", lambda self, r: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"

    engine.on_task_complete(
        "00025", "nodeB",
        "- run_id: run_a\n- status: running\n- run_id: run_b\n- status: still_running\n",
    )

    assert engine.state["pending_run_ids"] == ["run_a", "run_b"]


class TestStage9RevisionLoop:
    """#46 (output-quality fix D): the Stage-9 peer review's actionable
    verdict must be CONSUMED, not archived. When the review says
    MINOR/MAJOR REVISION, the paper-writer gets exactly ONE bounded
    revision pass, then Stage 9 re-reviews. ACCEPT (or an already-used
    revision budget) completes as before."""

    def _engine(self, tmp_path, monkeypatch, review_verdict: str, revised: bool = False):
        rec = {"revision": [], "producer": [], "passed": []}
        monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_paper_revision",
                            lambda self: rec["revision"].append(True))
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": rec["producer"].append(feedback))
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass",
                            lambda self, result, confidence=None: rec["passed"].append(result))
        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 9
        engine.state["phase"] = "critic"
        if revised:
            engine.state["paper_revised"] = True
        engine.state["stage_results"] = {
            # Real data upstream — the revision loop only makes sense for
            # papers about actual results (#94 clamp interplay).
            "6": "- run_id: run_x\n- status: succeeded\n",
            "7": "### H1\nDecision: SUPPORTED\n",
            "8": "# Paper v1",
            "9": f"# Peer Review\n\n## Verdict\n| **Decision** | **{review_verdict}** |\n\nComments: tighten §5.",
        }
        return engine, rec

    def test_minor_revision_triggers_one_revision_pass(self, tmp_path, monkeypatch):
        engine, rec = self._engine(tmp_path, monkeypatch, "MINOR REVISION")
        engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")
        assert rec["revision"] == [True], "review asked for revision — must dispatch the writer"
        assert rec["passed"] == [], "must NOT complete before the revision pass"

    def test_accept_completes_without_revision(self, tmp_path, monkeypatch):
        engine, rec = self._engine(tmp_path, monkeypatch, "ACCEPT")
        engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")
        assert rec["revision"] == []
        assert len(rec["passed"]) == 1

    def test_revision_budget_is_one(self, tmp_path, monkeypatch):
        """Second time around (paper already revised) MUST complete even if
        the re-review still says MINOR REVISION — bounded loop."""
        engine, rec = self._engine(tmp_path, monkeypatch, "MINOR REVISION", revised=True)
        engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")
        assert rec["revision"] == []
        assert len(rec["passed"]) == 1

    def test_revision_complete_rereviews_stage9(self, tmp_path, monkeypatch):
        """When the writer's revision lands, the revised paper replaces
        stage_results['8'] and Stage 9 is re-dispatched for a fresh review."""
        rec = {"producer": []}
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": rec["producer"].append(feedback))
        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 9
        engine.state["phase"] = "paper_revision"
        engine.state["stage_results"] = {"8": "# Paper v1", "9": "old review"}

        engine.on_task_complete("00111", "node-rev", "# Paper v2 (revised per review)")

        assert engine.state["paper_revised"] is True
        assert engine.state["stage_results"]["8"] == "# Paper v2 (revised per review)"
        assert rec["producer"], "stage 9 must be re-dispatched to review the revision"
        assert engine.state["phase"] == "producer"

    def test_revision_task_failure_degrades_to_complete(self, tmp_path, monkeypatch):
        """A failed/cancelled revision task must never strand the pipeline —
        degrade to completing with the original paper."""
        rec = {"passed": []}
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass",
                            lambda self, result, confidence=None: rec["passed"].append(result))
        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 9
        engine.state["phase"] = "paper_revision"
        engine.state["stage_results"] = {"8": "# Paper v1", "9": "review"}

        engine.on_task_failed("00111", "node-rev", "503")

        assert engine.state["paper_revised"] is True, "budget consumed — no infinite retry"
        assert len(rec["passed"]) == 1, "must complete with the original paper"


def test_stage9_verdict_clamped_to_major_revision_when_no_data(tmp_path, monkeypatch):
    """Stage 9 (self-review) is terminal — it can't be blocked/retried like
    6/7/8. Instead, when the pipeline has no real experimental data, an
    acceptance-class verdict must be CLAMPED to MAJOR REVISION. A paper that
    never ran its experiments cannot be ACCEPT (#94)."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 9
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "6": "- run_id: run_a\n- status: blocked\n",      # no data
        "7": "### H1\nDecision: NOT TESTED\n",             # no data
        "9": "## Review\n\nVerdict: ACCEPT\n\nStrong methodology.",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")

    clamped = engine.state["stage_results"]["9"]
    assert "ACCEPT" not in clamped.split("clamp")[0] or "MAJOR REVISION" in clamped, clamped
    assert "MAJOR REVISION" in clamped, "verdict must be clamped to MAJOR REVISION"
    # Stage 9 still completes (terminal) — not routed to reject/retry.
    assert getattr(engine, "_passed", False) is True


def test_stage9_verdict_preserved_when_data_present(tmp_path, monkeypatch):
    """Control: with real data upstream, Stage 9's ACCEPT verdict is left
    untouched."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 9
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {
        "6": "- run_id: run_a\n- status: succeeded\n",
        "7": "### H1\nDecision: SUPPORTED\n",
        "9": "## Review\n\nVerdict: ACCEPT\n\nStrong methodology.",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.9")

    assert "Verdict: ACCEPT" in engine.state["stage_results"]["9"]
    assert "MAJOR REVISION" not in engine.state["stage_results"]["9"]
    assert getattr(engine, "_passed", False) is True


@pytest.mark.asyncio
async def test_data_gate_fail_exhausts_to_autofail_under_auto_approve(tmp_path, monkeypatch):
    """End-to-end reuse: a Stage 6 data-gate fail routes through the existing
    critic-reject retry path; after MAX_RETRIES it opens an exhausted gate,
    which under auto_approve (the #106 fix) marks the pipeline failed rather
    than silently advancing an empty stage."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b", lambda self, feedback="": None)
    failed = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_failed", lambda self, sid, reason: failed.append((sid, reason)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "critic"
    engine.state["auto_approve"] = True
    engine.state["retries"] = pe.MAX_RETRIES  # already at the cap
    engine.state["stage_results"] = {"6": "- run_id: run_a\n- status: still_running\n"}

    # Critic says PASS, but gate fails → reject → exhausted (retries==MAX) → gate
    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")
    assert engine.phase == "gate"

    # auto-approve fires on the exhausted gate → #106 refuses → failed
    await engine._auto_approve_gate(stage_id=6, exhausted=True)
    assert engine.state["phase"] == "failed"
    assert failed == [(6, "retries_exhausted")]


def test_data_gate_reads_ondisk_deliverable_not_submit_result_summary(tmp_path, monkeypatch):
    """REGRESSION (caught by the 4→9 e2e, project 538372e68022): the runner's
    submit_result is a PROSE SUMMARY ("Smoke run: `run_x` — succeeded") whose
    run_id/status share a line — _parse_runner_report_runs finds 0 runs in it.
    The canonical evidence is the on-disk stage6_experimentalist.md (proper
    `- run_id:` / `- status:` lines). The gate MUST read the file, else it
    wrongly fails a healthy Stage 6 and (via #106) kills the whole run."""
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_gate_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: setattr(self, "_passed", True))

    # On-disk canonical evidence file with real, parseable runs.
    (tmp_path / "stage6_experimentalist.md").write_text(
        "### T2 — Smoke\n- run_id: run_914f8929b927\n- status: succeeded\n\n"
        "### T3 — Full\n- run_id: run_131741e1d009\n- status: succeeded\n"
    )

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "critic"
    # The stored deliverable is the runner's PROSE summary — does NOT parse.
    engine.state["stage_results"] = {
        "6": "The deliverable stage6_experimentalist.md exists. "
             "Smoke run: `run_914f8929b927` — `succeeded`, $0.00.",
    }

    engine.on_task_complete("critic", "node", "PASS\nConfidence Score: 0.95")

    assert getattr(engine, "_passed", False) is True, (
        "gate must read the on-disk report (which has real succeeded runs), "
        "not the unparseable submit_result summary"
    )


def test_parse_critic_pass_ignores_incidental_reject_word(tmp_path):
    """REGRESSION (#19 facet 2, caught by 4→9 run 6d188381963c at Stage 7):
    the critic's submit_result was a clear PASS ('Gate Review Complete —
    PASS (0.95)') but contained the rubric header 'Auto-REJECT trigger
    check:'. The old bare-substring `if 'REJECT' in text` matched that
    incidental word and returned REJECT, killing a PASSED stage.

    A verdict must come from a LABELED/structured signal (Decision: X,
    Verdict: X, Gate Review ... — X, | Decision | X |), not any stray
    occurrence of the words pass/reject."""
    engine = pe.PipelineEngine("p", str(tmp_path), "topic")
    engine.state["current_stage"] = 7

    real_submit_result = (
        "**Stage 7 Gate Review Complete — PASS (0.95 confidence).**\n\n"
        "| D9 | Reproducibility | PASS |\n"
        "| D10 | Language & Style | PASS |\n\n"
        "**Auto-REJECT trigger check:**\n"
        "- (a) HARKing: none detected\n"
        "- (b) fabricated data: none\n"
    )
    assert engine._parse_critic_pass(real_submit_result) is True, (
        "an incidental 'Auto-REJECT' rubric mention must not flip a PASS to REJECT"
    )

    # And a genuinely labeled REJECT still parses as REJECT.
    assert engine._parse_critic_pass(
        "Gate Review Complete — REJECT. The smoke run produced no data."
    ) is False


def test_data_gate_stage7_accepts_not_supported_null_result(tmp_path):
    """REGRESSION (#27 Stage 7 gate too narrow, caught by 4→9 run
    6d188381963c): a hypothesis decided 'NOT SUPPORTED — Δ=0.00 < 0.20' IS a
    tested result with a real statistic (a null result), not 'no data'. The
    gate's job is to block 'no experiment ran', not 'effect not found'. The
    old regex only knew SUPPORTED/REJECTED/CONFIRMED/NOT TESTED and missed
    'NOT SUPPORTED' + the 'Decision: PASS' manipulation-check style → wrongly
    reported 'no decision lines'."""
    report = (
        "## G1 — Scaling criterion\n"
        "**Decision:** **NOT SUPPORTED** — Δ = 0.00 < 0.20. Both at 100% accuracy.\n\n"
        "## Manipulation check\n"
        "**Decision:** **PASS** — cot_trace_nonempty_ratio = 1.0.\n\n"
        "**No hypotheses are NOT TESTED.** All checks have full coverage.\n"
        "**Overall verdict:** PARTIALLY CONFIRMED.\n"
    )
    ok, reason = pe.PipelineEngine._data_gate_stage7(report)
    assert ok is True, f"NOT SUPPORTED is a tested null result — gate must pass. reason={reason}"


def test_data_gate_stage7_still_fails_pure_not_tested(tmp_path):
    """Control: a report where EVERY hypothesis is genuinely NOT TESTED /
    INCONCLUSIVE_DUE_TO_COVERAGE (no experiment ran) must still FAIL."""
    report = (
        "### H1\n**Decision:** NOT TESTED — Stage 6 collected 0 observations.\n\n"
        "### H2\n**Decision:** NOT TESTED.\n\n"
        "**Overall verdict:** INCONCLUSIVE_DUE_TO_COVERAGE.\n"
    )
    ok, reason = pe.PipelineEngine._data_gate_stage7(report)
    assert ok is False


def test_producer_b_stub_routes_to_6a_rebuild_not_runner_rerun(tmp_path, monkeypatch):
    """REGRESSION (#20, caught by 4→9 run 3f644a5996bb): the 6b runner
    thrashed on a broken entrypoint ('--output-dir' the 6a code didn't
    accept), burned its step budget, and returned a 14-char stub. Re-running
    the SAME runner on the SAME broken code just stubs again (observed 3×) →
    exhausted → whole run dies.

    A producer_b stub means the runner couldn't produce a usable report —
    usually because the experiment code/entrypoint is broken. Route the retry
    back to 6a (rebuild code) rather than re-running the runner."""
    to_6a = []
    to_6b = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, feedback="": to_6a.append(feedback))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer_b", lambda self, feedback="": to_6b.append(feedback))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "producer_b"

    engine.on_task_complete("00025", "node", "Executed: bash")  # 14-char runner stub

    assert to_6a, "producer_b stub must route to 6a (rebuild), not re-run the runner"
    assert to_6b == [], "must NOT re-dispatch the same runner on a stub"
    assert engine.state["retries"] == 1
    # feedback should hint the code/entrypoint is the suspect
    assert "entrypoint" in to_6a[0].lower() or "runner" in to_6a[0].lower()


# ===========================================================================
# Result-driven loop (#40): result-reviewer routes back to 4/5/6 or advances
# ===========================================================================


def test_parse_result_route_advance():
    """Reviewer says the result is sound → advance (no revert target)."""
    txt = "RESULT REVIEW\nReasonableness: REASONABLE\nAction: ADVANCE\nReason: solid."
    action, target, reason = pe.PipelineEngine._parse_result_route(txt)
    assert action == "advance"
    assert target is None


def test_parse_result_route_revert_to_code():
    """Reviewer judges it a code bug → revert to Stage 6, with the stage it gave."""
    txt = ("RESULT REVIEW\nReasonableness: UNREASONABLE\nAction: REVERT\n"
           "Revert to stage: 6\nReason: direct accuracy 0.0 across all problems — "
           "extraction is almost certainly broken.")
    action, target, reason = pe.PipelineEngine._parse_result_route(txt)
    assert action == "revert"
    assert target == 6
    assert "extraction" in reason.lower()


def test_parse_result_route_revert_to_methodology():
    """Reviewer judges the design too weak → revert to Stage 4."""
    txt = ("Action: REVERT\nRevert to stage: 4\n"
           "Reason: n=7 with no baseline cannot support the causal claim.")
    action, target, reason = pe.PipelineEngine._parse_result_route(txt)
    assert action == "revert"
    assert target == 4


def test_parse_result_route_table_form_and_aliases():
    """Tolerate decorated / aliased phrasings the LLM may emit."""
    assert pe.PipelineEngine._parse_result_route(
        "| **Decision** | **REVERT** |\n| Revert-to-stage | 5 |")[1] == 5
    assert pe.PipelineEngine._parse_result_route(
        "verdict: advance — results look reasonable")[0] == "advance"


def test_parse_result_route_ambiguous_defaults_to_advance():
    """If the reviewer says REVERT but gives no parseable target stage, do NOT
    guess a stage — treat as advance (don't loop on a malformed review)."""
    action, target, reason = pe.PipelineEngine._parse_result_route(
        "Action: REVERT\nReason: something feels off but I won't say where")
    assert action == "advance"
    assert target is None


def test_parse_result_route_rejects_out_of_range_target():
    """Only 4/5/6 are valid revert targets for the result loop."""
    action, target, _ = pe.PipelineEngine._parse_result_route(
        "Action: REVERT\nRevert to stage: 2")
    assert action == "advance"  # 2 is not a valid result-loop target → no-op


def test_result_review_advance_proceeds_to_next_stage(tmp_path, monkeypatch):
    """result_review verdict ADVANCE → normal advance (via _on_critic_pass)."""
    advanced = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, result, confidence=None: advanced.append(result))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "result_review"
    engine.state["stage_results"] = {"7": "analysis"}

    engine.on_task_complete("critic", "n", "Action: ADVANCE\nReason: sound result.")

    assert advanced == ["analysis"], "ADVANCE must proceed via _on_critic_pass"


def test_result_review_revert_schedules_revert_and_counts_loop(tmp_path, monkeypatch):
    """REVERT to 6 → schedule revert_to_stage(6) and bump that stage's loop count."""
    reverts = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, *a, **k: reverts.append("ADVANCED"))
    monkeypatch.setattr(pe.PipelineEngine, "_schedule_result_revert",
                        lambda self, stage, reason: reverts.append((stage, reason)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "result_review"
    engine.state["stage_results"] = {"7": "analysis"}

    engine.on_task_complete("critic", "n",
        "Action: REVERT\nRevert to stage: 6\nReason: accuracy 0.0 — extraction broken.")

    assert reverts and reverts[0][0] == 6, f"must schedule revert to 6, got {reverts}"
    assert engine.state["result_loops"]["6"] == 1
    assert "ADVANCED" not in reverts


def test_result_review_revert_budget_exhausted_advances_with_caveat(tmp_path, monkeypatch):
    """When a target's loop budget is spent, stop looping → advance instead."""
    calls = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, *a, **k: calls.append("ADVANCED"))
    monkeypatch.setattr(pe.PipelineEngine, "_schedule_result_revert",
                        lambda self, stage, reason: calls.append(("REVERT", stage)))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "result_review"
    engine.state["stage_results"] = {"7": "analysis"}
    engine.state["result_loops"] = {"6": pe.MAX_RESULT_LOOPS}  # already exhausted

    engine.on_task_complete("critic", "n",
        "Action: REVERT\nRevert to stage: 6\nReason: still looks broken.")

    assert calls == ["ADVANCED"], f"exhausted budget must advance, got {calls}"


def test_stage7_pass_dispatches_result_reviewer_not_straight_to_paper(tmp_path, monkeypatch):
    """Stage 7 critic PASS must go through the result-reviewer (phase
    result_review), NOT advance straight to Stage 8."""
    reviewer = []
    advanced = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_record_stage_memory", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_stage_data_gate", lambda self, sid, d: (True, ""))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_result_reviewer", lambda self, conf=None: reviewer.append(True))
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, *a, **k: advanced.append(True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 7
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {"7": "### H1\nDecision: SUPPORTED\n"}

    engine.on_task_complete("critic", "n", "PASS\nConfidence: 0.9")

    assert reviewer == [True], "Stage 7 pass must dispatch the result-reviewer"
    assert advanced == [], "Stage 7 pass must NOT call _on_critic_pass directly"


def test_stage6_pass_still_advances_normally_not_via_reviewer(tmp_path, monkeypatch):
    """Control: only Stage 7 triggers the result-reviewer. Stage 6 pass keeps
    its normal advance path."""
    reviewer = []
    advanced = []
    monkeypatch.setattr(pe.PipelineEngine, "_emit_critic_result", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_record_stage_memory", lambda self, *a, **k: None)
    monkeypatch.setattr(pe.PipelineEngine, "_stage_data_gate", lambda self, sid, d: (True, ""))
    monkeypatch.setattr(pe.PipelineEngine, "_dispatch_result_reviewer", lambda self, conf=None: reviewer.append(True))
    monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass", lambda self, *a, **k: advanced.append(True))

    engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
    engine.state["current_stage"] = 6
    engine.state["phase"] = "critic"
    engine.state["stage_results"] = {"6": "- run_id: r1\n- status: succeeded\n"}

    engine.on_task_complete("critic", "n", "PASS\nConfidence: 0.9")

    assert advanced == [True] and reviewer == [], "Stage 6 keeps normal advance"


# ---------------------------------------------------------------------------
# Issue #159 — Concurrency cap
# ---------------------------------------------------------------------------

class TestConcurrencyCap:
    """Admission control: excess pipeline starts queue instead of dispatch."""

    def _make_engine(self, tmp_path, pid, phase="producer"):
        eng = pe.PipelineEngine(pid, str(tmp_path / pid), "topic")
        eng.state["phase"] = phase
        pe._active_pipelines[pid] = eng
        return eng

    def teardown_method(self, _):
        pe._active_pipelines.clear()

    def test_start_queues_when_at_cap(self, tmp_path, monkeypatch):
        """A new start() must set phase=queued when executing == cap."""
        monkeypatch.setattr(pe, "MAX_CONCURRENT_PIPELINE_RUNS", 1)
        dispatched = []
        monkeypatch.setattr(pe, "_find_employee_for_stage", lambda sid, skill: "emp-1")
        monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                            lambda self, *a: dispatched.append(self.project_id))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                            lambda self, *a, **kw: None)
        monkeypatch.setattr(pe.PipelineEngine, "_ensure_memory_state", lambda self: None)
        monkeypatch.setattr(pe.PipelineEngine, "_ensure_timing_state", lambda self: None)

        # One pipeline already executing
        self._make_engine(tmp_path, "existing", phase="producer")

        # New pipeline — should queue, not dispatch
        new_eng = pe.PipelineEngine("new-p", str(tmp_path / "new-p"), "topic")
        pe._active_pipelines["new-p"] = new_eng
        monkeypatch.setattr(pe.PipelineEngine, "_iteration_id", lambda self: "iter_001")
        import onemancompany.core.project_repo as _pr
        monkeypatch.setattr(_pr, "ensure_initialized", lambda *a, **kw: None)
        new_eng.start()

        assert new_eng.state["phase"] == "queued", "Should be queued when at cap"
        assert "new-p" not in dispatched, "Should NOT dispatch when queued"

    def test_start_dispatches_when_below_cap(self, tmp_path, monkeypatch):
        """start() dispatches immediately when executing < cap."""
        monkeypatch.setattr(pe, "MAX_CONCURRENT_PIPELINE_RUNS", 2)
        dispatched = []
        monkeypatch.setattr(pe, "_find_employee_for_stage", lambda sid, skill: "emp-1")
        monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                            lambda self, *a: dispatched.append(self.project_id))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                            lambda self, *a, **kw: None)
        monkeypatch.setattr(pe.PipelineEngine, "_ensure_memory_state", lambda self: None)
        monkeypatch.setattr(pe.PipelineEngine, "_ensure_timing_state", lambda self: None)

        # One executing, cap is 2 — room for one more
        self._make_engine(tmp_path, "existing", phase="producer")

        new_eng = pe.PipelineEngine("new-p", str(tmp_path / "new-p"), "topic")
        pe._active_pipelines["new-p"] = new_eng
        monkeypatch.setattr(pe.PipelineEngine, "_iteration_id", lambda self: "iter_001")
        import onemancompany.core.project_repo as _pr
        monkeypatch.setattr(_pr, "ensure_initialized", lambda *a, **kw: None)
        new_eng.start()

        assert "new-p" in dispatched, "Should dispatch when below cap"
        assert new_eng.state["phase"] != "queued"

    def test_dequeue_next_pipeline_promotes_oldest(self, tmp_path, monkeypatch):
        """dequeue_next_pipeline() picks the queued engine with the earliest
        queue_requested_at and dispatches it."""
        dispatched = []
        monkeypatch.setattr(pe, "_find_employee_for_stage", lambda sid, skill: "emp-1")
        monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                            lambda self, *a: dispatched.append(self.project_id))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                            lambda self, *a, **kw: None)

        older = self._make_engine(tmp_path, "older", phase="queued")
        older.state["queue_requested_at"] = "1000.0"
        newer = self._make_engine(tmp_path, "newer", phase="queued")
        newer.state["queue_requested_at"] = "2000.0"

        result = pe.dequeue_next_pipeline()

        assert result is True
        assert dispatched == ["older"], "Should promote the oldest queued pipeline"

    def test_on_became_terminal_evicts_and_dequeues(self, tmp_path, monkeypatch):
        """_on_became_terminal() evicts the finished engine and promotes the
        next queued pipeline."""
        dispatched = []
        monkeypatch.setattr(pe, "_find_employee_for_stage", lambda sid, skill: "emp-1")
        monkeypatch.setattr(pe, "load_employee_configs", lambda: {})
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_to_employee",
                            lambda self, *a: dispatched.append(self.project_id))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_stage_event",
                            lambda self, *a, **kw: None)

        done_eng = self._make_engine(tmp_path, "done-p", phase="done")
        queued_eng = self._make_engine(tmp_path, "queued-p", phase="queued")
        queued_eng.state["queue_requested_at"] = "1000.0"

        done_eng._on_became_terminal()

        assert "done-p" not in pe._active_pipelines, "Terminal engine must be evicted"
        assert dispatched == ["queued-p"], "Queued pipeline must be promoted"

    def test_producer_b_waiting_does_not_count_toward_cap(self, tmp_path, monkeypatch):
        """producer_b_waiting is excluded from the executing count so a
        long-running remote experiment does not permanently block new starts."""
        monkeypatch.setattr(pe, "MAX_CONCURRENT_PIPELINE_RUNS", 1)
        # One pipeline in producer_b_waiting — should NOT count
        self._make_engine(tmp_path, "waiting-p", phase="producer_b_waiting")

        count = pe._count_executing()
        assert count == 0, "producer_b_waiting must not count toward the cap"


# ---------------------------------------------------------------------------
# C: feasibility -> full execution split (research_phase state machine)
# ---------------------------------------------------------------------------

class TestFeasibilityPhaseSplit:
    """staged=True runs a cheap feasibility study first; a positive go/no-go
    signal at Stage 7 promotes to the full study (research_phase flips,
    pipeline returns to Stage 5) instead of going straight to the paper.
    Default (staged=False) is unchanged — zero regression."""

    def teardown_method(self, _):
        pe._active_pipelines.clear()

    def test_default_start_is_full_phase(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_iteration_id", lambda self: "iter_001")
        import onemancompany.core.project_repo as _pr
        monkeypatch.setattr(_pr, "ensure_initialized", lambda *a, **k: None)
        eng = pe.PipelineEngine("p1", str(tmp_path), "topic")
        eng.start()
        assert eng.state.get("research_phase", "full") == "full", "default must be full phase"

    def test_staged_start_sets_feasibility_phase(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer", lambda self, **k: None)
        monkeypatch.setattr(pe.PipelineEngine, "_iteration_id", lambda self: "iter_001")
        import onemancompany.core.project_repo as _pr
        monkeypatch.setattr(_pr, "ensure_initialized", lambda *a, **k: None)
        eng = pe.PipelineEngine("p1", str(tmp_path), "topic")
        eng.start(staged=True)
        assert eng.state["research_phase"] == "feasibility"

    def test_feasibility_advance_promotes_to_full_not_paper(self, tmp_path, monkeypatch):
        """ADVANCE at Stage 7 while in feasibility phase must promote to the
        full study (research_phase=full, current_stage=5, re-dispatch) and NOT
        call _on_critic_pass (which would go to Stage 8)."""
        promoted = []
        paper = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, **k: promoted.append((self.current_stage, self.state.get("research_phase"))))
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass",
                            lambda self, *a, **k: paper.append(True))
        monkeypatch.setattr(pe.PipelineEngine, "_parse_result_route",
                            lambda self, r: ("advance", None, ""))

        eng = pe.PipelineEngine("p1", str(tmp_path), "topic")
        eng.state["current_stage"] = 7
        eng.state["phase"] = "result_review"
        eng.state["research_phase"] = "feasibility"
        eng.state["stage_results"] = {"7": "signal found, effect large"}

        eng.on_task_complete("rr", "n", "ADVANCE")

        assert eng.state["research_phase"] == "full", "must flip to full study"
        assert eng.current_stage == 5, "must return to Stage 5 to design the full study"
        assert promoted and promoted[-1] == (5, "full"), "must re-dispatch Stage 5 in full phase"
        assert paper == [], "must NOT proceed to Stage 8 on the feasibility run"

    def test_full_advance_proceeds_to_paper(self, tmp_path, monkeypatch):
        """ADVANCE in full phase proceeds to Stage 8 normally (no promotion)."""
        paper = []
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass",
                            lambda self, *a, **k: paper.append(True))
        monkeypatch.setattr(pe.PipelineEngine, "_parse_result_route",
                            lambda self, r: ("advance", None, ""))

        eng = pe.PipelineEngine("p1", str(tmp_path), "topic")
        eng.state["current_stage"] = 7
        eng.state["phase"] = "result_review"
        eng.state["research_phase"] = "full"
        eng.state["stage_results"] = {"7": "confirmatory result"}

        eng.on_task_complete("rr", "n", "ADVANCE")

        assert paper == [True], "full phase ADVANCE must proceed to Stage 8"

    def test_non_staged_advance_proceeds_to_paper(self, tmp_path, monkeypatch):
        """No research_phase set (legacy / non-staged) behaves as full."""
        paper = []
        monkeypatch.setattr(pe.PipelineEngine, "_on_critic_pass",
                            lambda self, *a, **k: paper.append(True))
        monkeypatch.setattr(pe.PipelineEngine, "_parse_result_route",
                            lambda self, r: ("advance", None, ""))

        eng = pe.PipelineEngine("p1", str(tmp_path), "topic")
        eng.state["current_stage"] = 7
        eng.state["phase"] = "result_review"
        eng.state["stage_results"] = {"7": "result"}

        eng.on_task_complete("rr", "n", "ADVANCE")

        assert paper == [True], "non-staged ADVANCE must proceed to Stage 8"
# Issue #157 — on_ceo_approve idempotency guard
# ---------------------------------------------------------------------------

class TestOnCeoApproveIdempotencyGuard:
    """on_ceo_approve must be a no-op when the pipeline is not at a gate."""

    def test_approve_at_gate_advances(self, tmp_path, monkeypatch):
        """Normal path: approve at gate phase advances to next stage."""
        dispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": dispatched.append(self.current_stage))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_complete", lambda self: None)

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 7
        engine.state["end_stage"] = 9
        engine.state["phase"] = "gate"

        engine.on_ceo_approve()

        assert engine.current_stage == 8, "Should advance to stage 8"
        assert dispatched == [8], "Should dispatch stage 8 producer"

    def test_approve_during_producer_is_noop(self, tmp_path, monkeypatch):
        """Duplicate approve while producer is running must not advance (#157)."""
        dispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": dispatched.append(self.current_stage))

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 8
        engine.state["end_stage"] = 9
        engine.state["phase"] = "producer"  # Stage 8 already running

        engine.on_ceo_approve()

        assert engine.current_stage == 8, "Stage must not advance"
        assert dispatched == [], "No producer dispatch must happen"
        assert engine.phase == "producer", "Phase must remain producer"

    def test_approve_during_critic_is_noop(self, tmp_path, monkeypatch):
        """Approve while critic is reviewing must be ignored."""
        dispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": dispatched.append(self.current_stage))

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 7
        engine.state["phase"] = "critic"

        engine.on_ceo_approve()

        assert dispatched == [] and engine.current_stage == 7

    def test_approve_on_done_is_noop(self, tmp_path, monkeypatch):
        """Approve on a finished pipeline must not raise or advance."""
        dispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": dispatched.append(True))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_complete", lambda self: None)

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 9
        engine.state["phase"] = "done"

        engine.on_ceo_approve()  # must not raise

        assert dispatched == []

    def test_duplicate_approve_does_not_skip_stage(self, tmp_path, monkeypatch):
        """Regression for #157: two approve signals for stage 7 must not skip
        stage 8 (paper). First approve at gate advances; second while stage 8
        producer runs must be a no-op."""
        dispatched = []
        monkeypatch.setattr(pe.PipelineEngine, "_dispatch_producer",
                            lambda self, feedback="": dispatched.append(self.current_stage))
        monkeypatch.setattr(pe.PipelineEngine, "_emit_pipeline_complete", lambda self: None)

        engine = pe.PipelineEngine("p1", str(tmp_path), "topic")
        engine.state["current_stage"] = 7
        engine.state["end_stage"] = 9
        engine.state["phase"] = "gate"

        # First approve — valid, advances to stage 8
        engine.on_ceo_approve()
        assert engine.current_stage == 8
        assert dispatched == [8]

        # Simulate the real state machine: _dispatch_producer sets phase=producer.
        # The mock doesn't do this, so set it explicitly to reflect reality.
        engine.state["phase"] = "producer"

        # Second stale approve must be ignored
        engine.on_ceo_approve()

        assert engine.current_stage == 8, "Stage 8 must still be running"
        assert dispatched == [8], "No second dispatch must have happened"
