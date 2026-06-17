"""
pipeline_engine.py — Deterministic state machine for the 9-stage research pipeline.

Replaces LLM-driven orchestration (EA/Research Director reading SOP).
The pipeline engine controls stage sequencing, critic dispatch, and CEO gates.
LLM agents only do research work within a stage — they never decide "what's next."

Runs on top of OMC: uses employee_manager.schedule_node() to dispatch tasks,
task tree for node management, and WebSocket events for frontend updates.
"""

from __future__ import annotations

import re
import time
import yaml
from pathlib import Path
from loguru import logger

from onemancompany.core.events import event_bus, CompanyEvent, EventType
from onemancompany.core.config import SYSTEM_AGENT
from onemancompany.core.config import load_employee_configs
from onemancompany.core.research_memory import ResearchMemoryStore

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

# Stages 1 (topic_refiner) and 2 (literature_surveyor) were removed: aigraph
# grounds Stage 3 over the arxiv corpus (which subsumes the literature survey),
# and the raw keyword topic feeds Stage 3 directly (no separate refinement). The
# remaining stage ids are kept as 3..9 so all stage-specific handling keyed on
# the stage-specific handling for ids 4 / 6 / 8 stays valid; lookups are by id,
# not list position.
STAGES = [
    {"id": 3, "skill": "idea_generator",        "name": "Idea Generation"},
    {"id": 4, "skill": "methodology_designer",  "name": "Methodology Design"},
    {"id": 5, "skill": "experiment_designer",   "name": "Experiment Design"},
    {"id": 6, "skill": "experimentalist",       "name": "Auto Experiment"},
    {"id": 7, "skill": "result_analyst",        "name": "Result Analysis"},
    {"id": 8, "skill": "paper_writer",          "name": "Paper Generation"},
    {"id": 9, "skill": "peer_reviewer",         "name": "Self-Review"},
]
FIRST_STAGE_ID = STAGES[0]["id"]   # 3 — the pipeline entry (aigraph-grounded ideation)
LAST_STAGE_ID = STAGES[-1]["id"]   # 9

CRITIC_SKILL = "adversarial_review"
MAX_RETRIES = 3

# Result-driven loop (#40): max times the result-reviewer may send the
# pipeline back to a given earlier stage (4/5/6) before we stop looping and
# proceed with the best result (written up honestly with its limitations).
# Per-target so a code-fix loop and a redesign loop don't share a budget.
MAX_RESULT_LOOPS = 2

# Concurrency cap (#159): max number of pipelines that may be actively
# dispatching LLM work at the same time. Excess starts are queued
# (phase="queued") and dequeued FIFO when a slot frees. Override with
# OMC_MAX_CONCURRENT_RUNS env var. producer_b_waiting does NOT count
# toward the cap — the remote infra job is running without using a local
# LLM slot.
import os as _os
MAX_CONCURRENT_PIPELINE_RUNS: int = int(_os.getenv("OMC_MAX_CONCURRENT_RUNS", "2"))

# Phases that consume a local LLM slot and therefore count toward the cap.
_EXECUTING_PHASES = frozenset({
    "producer", "producer_b", "producer_b_finalize", "critic", "gate", "paper_revision",
})

# Canonical default employee per stage, sourced from company/hire_list.json.
# When multiple hired employees share the same skill, the one originating
# from the canonical talent_id wins. Falls back to skill-based lookup if
# the canonical talent is not on the roster.
STAGE_TALENT_DEFAULTS = {
    # Stages 1 (topic-refiner) + 2 (literature-surveyor) removed — aigraph grounds
    # Stage 3 over the arxiv corpus and the keyword feeds it directly.
    3: "idea-generator",
    4: "methodology-designer",
    5: "experiment-designer",
    6: "experimentalist",
    7: "result-analyst",
    8: "paper-writer",
    9: "paper-reviewer",
}

# Iteration identifier used in git tag names (``<iteration>/stage-<N>``).
# The literal directory name (e.g. ``iter_001``) is fine — git tag names
# allow underscores. Centralised here so the engine and project_repo
# agree on tag format.
_DEFAULT_ITERATION = "iter_001"


class RevertNotAllowedError(Exception):
    """Raised when ``revert_to_stage`` is called in a phase that would
    clobber in-flight work. Only ``gate`` and ``done`` are safe."""

# Tag for pipeline-managed nodes so vessel can identify them
PIPELINE_NODE_TAG = "pipeline_managed"

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

STATE_FILENAME = "pipeline_state.yaml"


def _load_state(project_dir: str) -> dict:
    path = Path(project_dir) / STATE_FILENAME
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Startup watchdog — replay missing on_task_complete / on_task_failed events
# that vanished into a backend crash, completion-consumer timeout, or EA /
# pipeline_engine desync (see issue #82 for the production stall this fixes).
# ---------------------------------------------------------------------------

# Terminal pipeline phases — never re-fire stage events.
_WATCHDOG_TERMINAL_PHASES = frozenset({"done", "failed"})

# Phases owned by ``run_tracker``'s remote-run poller, NOT by the OMC
# task-node lifecycle. During ``producer_b_waiting`` the Stage 6b runner
# node is already COMPLETED on disk — the runner submitted an interim
# report and exited within its task budget — but the experiment is still
# executing on remote infra. The disk-scanning watchdogs must leave this
# phase alone: ``recover_stalled_pipelines`` would otherwise replay
# ``on_task_complete`` on that completed node (no-op at best; at worst it
# races with run_tracker flipping to ``producer_b_finalize`` and re-fires
# the stale 6b result as a finalize completion, double-dispatching the
# critic), and ``detect_stuck_pipelines`` would surface a spurious
# PIPELINE_STUCK that conflicts with run_tracker's own
# ``on_runs_wait_timeout`` deadline handling (#30).
_WATCHDOG_RUN_TRACKER_PHASES = frozenset({"producer_b_waiting"})

# Task-tree node statuses the watchdog treats as "stage producer finished
# successfully — engine should advance". COMPLETED is the canonical state
# right after submit_result; ACCEPTED / FINISHED are reached after the EA
# wraps up and would normally trigger the same engine advance.
_WATCHDOG_COMPLETED_STATUSES = frozenset({"completed", "accepted", "finished"})

# Task-tree node statuses that mean the producer is done but failed —
# engine.on_task_failed should retry / fail the stage.
_WATCHDOG_FAILED_STATUSES = frozenset({"failed", "cancelled"})


# Pipelines whose ``pipeline_state.yaml`` has not been written for at least
# this many seconds, AND whose producer node is still in-flight on disk
# (so :func:`recover_stalled_pipelines` cannot auto-resolve it), are
# surfaced as ``PIPELINE_STUCK`` events for user intervention. Producers
# typically finish in well under 30 minutes; >1 h with zero state change
# means something silent went wrong (issue #82, PR 3).
PIPELINE_STUCK_THRESHOLD_SECONDS = 3600


def detect_stuck_pipelines(projects_root) -> list[dict]:
    """Scan every project iteration for pipelines that are silently stuck
    and beyond :func:`recover_stalled_pipelines`'s reach.

    A pipeline is "stuck" when ALL of the following hold:
      - ``pipeline_state.yaml`` has an ``active_node_id`` and a
        non-terminal ``phase``,
      - the active node is still PROCESSING / PENDING / etc. on disk
        (so the recovery watchdog has nothing to replay),
      - the state file has not been written for at least
        :data:`PIPELINE_STUCK_THRESHOLD_SECONDS`.

    Returns descriptors used by the lifespan to publish a
    :data:`EventType.PIPELINE_STUCK` event the user can act on.
    """
    import time as _time
    import yaml as _yaml
    from pathlib import Path as _Path

    root = _Path(projects_root)
    if not root.exists():
        return []

    now = _time.time()
    stuck: list[dict] = []
    for state_path in root.glob("*/iterations/*/pipeline_state.yaml"):
        try:
            state = _yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[stuck-detect] failed to read {}: {}", state_path, exc)
            continue

        active_node_id = state.get("active_node_id")
        phase = state.get("phase", "")
        if (
            not active_node_id
            or phase in _WATCHDOG_TERMINAL_PHASES
            or phase in _WATCHDOG_RUN_TRACKER_PHASES
        ):
            continue

        try:
            mtime = state_path.stat().st_mtime
        except OSError as exc:
            logger.warning("[stuck-detect] stat failed for {}: {}", state_path, exc)
            continue
        stale_seconds = now - mtime
        if stale_seconds < PIPELINE_STUCK_THRESHOLD_SECONDS:
            continue

        iter_dir = state_path.parent
        project_id = iter_dir.parents[1].name
        tree_path = iter_dir / "task_tree.yaml"
        if not tree_path.exists():
            continue

        try:
            tree_doc = _yaml.safe_load(tree_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[stuck-detect] failed to read {}: {}", tree_path, exc)
            continue

        node = (tree_doc.get("nodes") or {}).get(active_node_id)
        if not node:
            continue

        node_status = str(node.get("status", "")).lower()
        # Resolved-on-disk cases are recover_stalled_pipelines's job —
        # flagging them here would emit a spurious event that gets
        # resolved milliseconds later by the replay path.
        if node_status in (_WATCHDOG_COMPLETED_STATUSES | _WATCHDOG_FAILED_STATUSES):
            continue

        stuck.append({
            "project_id": project_id,
            "current_stage": state.get("current_stage"),
            "phase": phase,
            "active_node_id": active_node_id,
            "stale_seconds": int(stale_seconds),
        })
    return stuck


def recover_stalled_pipelines(projects_root) -> int:
    """Scan every project iteration under ``projects_root`` for a
    ``pipeline_state.yaml`` whose ``active_node_id`` points at a task tree
    node that has already resolved on disk, and re-fire the missing
    ``on_task_complete`` / ``on_task_failed`` event into the pipeline
    engine. Returns the number of stalled pipelines recovered.

    Idempotent: a second call after the first one ran finds nothing to do
    (active_node_id will be cleared by the engine handlers).
    """
    import yaml as _yaml
    from pathlib import Path as _Path

    root = _Path(projects_root)
    if not root.exists():
        return 0

    recovered = 0
    for state_path in root.glob("*/iterations/*/pipeline_state.yaml"):
        try:
            state = _yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[watchdog] failed to read {}: {}", state_path, exc)
            continue

        active_node_id = state.get("active_node_id")
        phase = state.get("phase", "")
        if (
            not active_node_id
            or phase in _WATCHDOG_TERMINAL_PHASES
            or phase in _WATCHDOG_RUN_TRACKER_PHASES
        ):
            continue

        # Resolve project_id and project_dir from the layout
        # ``<root>/<project_id>/iterations/<iter>/pipeline_state.yaml``.
        iter_dir = state_path.parent
        project_id = iter_dir.parents[1].name
        project_dir = str(iter_dir)

        tree_path = iter_dir / "task_tree.yaml"
        if not tree_path.exists():
            logger.warning(
                "[watchdog] task_tree.yaml missing for project={} iter={}, skipping",
                project_id, iter_dir.name,
            )
            continue

        try:
            tree_doc = _yaml.safe_load(tree_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[watchdog] failed to read {}: {}", tree_path, exc)
            continue

        node = (tree_doc.get("nodes") or {}).get(active_node_id)
        if not node:
            logger.warning(
                "[watchdog] active_node {} not found in tree for project={}, skipping",
                active_node_id, project_id,
            )
            continue

        node_status = str(node.get("status", "")).lower()
        if node_status not in (_WATCHDOG_COMPLETED_STATUSES | _WATCHDOG_FAILED_STATUSES):
            # Node is still in flight (pending / processing / holding / blocked).
            # Engine is right to keep waiting.
            continue

        engine = get_or_load_pipeline(project_id, project_dir)
        if engine is None:
            logger.warning(
                "[watchdog] could not load pipeline for project={}, skipping",
                project_id,
            )
            continue

        employee_id = str(node.get("employee_id", ""))
        result = str(node.get("result", "") or "")

        try:
            if node_status in _WATCHDOG_COMPLETED_STATUSES:
                logger.warning(
                    "[watchdog] project={} stage={} phase={} active_node={} is {} on disk "
                    "but pipeline engine still believes it's in flight — replaying "
                    "on_task_complete",
                    project_id, state.get("current_stage"), phase,
                    active_node_id, node_status,
                )
                engine.on_task_complete(employee_id, active_node_id, result)
            else:
                logger.warning(
                    "[watchdog] project={} stage={} phase={} active_node={} is {} on disk "
                    "but pipeline engine still believes it's in flight — replaying "
                    "on_task_failed",
                    project_id, state.get("current_stage"), phase,
                    active_node_id, node_status,
                )
                engine.on_task_failed(employee_id, active_node_id, result or "stalled, recovered by watchdog")
            recovered += 1
        except Exception as exc:
            logger.exception(
                "[watchdog] replay failed for project={} node={}: {}",
                project_id, active_node_id, exc,
            )

    return recovered


def _save_state(project_dir: str, state: dict):
    path = Path(project_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(state, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Employee lookup
# ---------------------------------------------------------------------------

def _find_employee_by_skill(skill: str) -> str | None:
    """Find the first employee whose skills list contains the given skill."""
    configs = load_employee_configs()
    for emp_id, cfg in configs.items():
        if skill in cfg.skills:
            return emp_id
    return None


def _find_employee_by_talent_id(talent_id: str) -> str | None:
    """Find the first employee whose ``talent_id`` matches.

    ``talent_id`` is the hire_list.json identifier carried forward by
    ``execute_hire`` so the pipeline can route each stage to the canonical
    default talent rather than any arbitrary employee that happens to
    share the same skill.
    """
    if not talent_id:
        return None
    configs = load_employee_configs()
    for emp_id, cfg in configs.items():
        if getattr(cfg, "talent_id", "") == talent_id:
            return emp_id
    return None


def _find_employee_for_stage(stage_id: int, primary_skill: str) -> str | None:
    """Resolve the producer employee for a stage with stage-specific fallbacks.

    Resolution order:
      1. Stage 6 only: a ``code_implementer`` employee (Stage 6a). The
         two-step Stage 6 producer flow writes the experiment code (6a)
         then executes it on remote infra (6b — see
         :func:`_find_stage_6b_employee`). The initial producer dispatch
         maps to 6a; 6b is dispatched by ``on_task_complete``.
      2. The canonical hire_list talent for the stage
         (see ``STAGE_TALENT_DEFAULTS``).
      3. Any employee whose skills include ``primary_skill``.
    """
    if stage_id == 6:
        coder = _find_employee_by_skill("code_implementer")
        if coder:
            return coder
        # Fall through to canonical/skill lookup so single-employee fixtures
        # still find SOMETHING when no dedicated code_implementer is hired.
    canonical = _find_employee_by_talent_id(STAGE_TALENT_DEFAULTS.get(stage_id, ""))
    if canonical:
        return canonical
    return _find_employee_by_skill(primary_skill)


def _find_stage_6b_employee() -> str | None:
    """Resolve the Stage 6b runner employee.

    Order: ``experiment_runner`` skill (real remote-infra runner) →
    canonical ``experimentalist`` talent_id (PR #67's hire_list mapping) →
    any ``experimentalist`` skill (last-resort simulated-report fallback).
    """
    runner = _find_employee_by_skill("experiment_runner")
    if runner:
        return runner
    canonical = _find_employee_by_talent_id(STAGE_TALENT_DEFAULTS.get(6, "experimentalist"))
    if canonical:
        return canonical
    return _find_employee_by_skill("experimentalist")


# ---------------------------------------------------------------------------
# In-memory registry of active pipelines
# ---------------------------------------------------------------------------

_active_pipelines: dict[str, "PipelineEngine"] = {}  # project_id → engine


def get_pipeline(project_id: str) -> "PipelineEngine | None":
    return _active_pipelines.get(project_id)


def get_or_load_pipeline(project_id: str, project_dir: str) -> "PipelineEngine | None":
    """Get from memory or reload from disk state."""
    if project_id in _active_pipelines:
        return _active_pipelines[project_id]
    state = _load_state(project_dir)
    if not state:
        return None
    engine = PipelineEngine(project_id, project_dir, state.get("topic", ""))
    engine.state = state
    engine._ensure_memory_state()
    engine._ensure_timing_state()
    _active_pipelines[project_id] = engine
    return engine


def _count_executing() -> int:
    """Count pipelines in _active_pipelines that hold a local LLM slot."""
    return sum(
        1 for eng in _active_pipelines.values()
        if eng.state.get("phase") in _EXECUTING_PHASES
    )


def dequeue_next_pipeline() -> bool:
    """Promote the oldest queued pipeline to executing. Returns True if one
    was found. Called whenever a pipeline becomes terminal so the next
    queued run can start immediately without waiting for the next poll.
    """
    queued = [
        eng for eng in _active_pipelines.values()
        if eng.state.get("phase") == "queued"
    ]
    if not queued:
        return False
    # FIFO by queue_requested_at timestamp; fall back to project_id for stability.
    next_eng = min(
        queued,
        key=lambda e: (e.state.get("queue_requested_at") or "", e.project_id),
    )
    logger.info(
        "[PIPELINE] Dequeuing {} (was queued at {})",
        next_eng.project_id, next_eng.state.get("queue_requested_at", "?"),
    )
    next_eng.state["phase"] = "producer"
    next_eng._save()
    next_eng._dispatch_producer()
    return True


# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------

class PipelineEngine:
    """Deterministic state machine for the research pipeline.

    Phases per stage:
        producer → critic → gate → (next stage or done)

    The engine dispatches tasks via OMC's task tree + employee_manager.
    It never calls an LLM itself.
    """

    def __init__(self, project_id: str, project_dir: str, topic: str):
        self.project_id = project_id
        self.project_dir = project_dir
        self.topic = topic
        self.state: dict = {
            "topic": topic,
            "current_stage": FIRST_STAGE_ID,
            "start_stage": FIRST_STAGE_ID,
            "end_stage": LAST_STAGE_ID,
            "prior_context": "",
            "stage_assignments": {},  # stage_id (str) → employee_id override
            "phase": "producer",  # producer | producer_b | producer_b_waiting | producer_b_finalize | critic | gate | done | failed
            "retries": 0,
            "stage_results": {},
            "critic_result": None,
            "active_node_id": None,  # current task node being executed
            "active_employee_id": None,
            "active_task_started_at": None,
            "attempt_timing": {
                "producer_elapsed_seconds": None,
                "critic_elapsed_seconds": None,
            },
            "memory_retrievals": {},
            "memory_episodes": {},
            "memory_feedback": {},
        }
        _active_pipelines[project_id] = self

    @property
    def current_stage(self) -> int:
        return self.state.get("current_stage", 1)

    @property
    def phase(self) -> str:
        return self.state.get("phase", "producer")

    def _save(self):
        self._ensure_memory_state()
        self._ensure_timing_state()
        _save_state(self.project_dir, self.state)

    def _ensure_memory_state(self):
        self.state.setdefault("memory_retrievals", {})
        self.state.setdefault("memory_episodes", {})
        self.state.setdefault("memory_feedback", {})

    def _ensure_timing_state(self):
        self.state.setdefault("active_task_started_at", None)
        timing = self.state.setdefault("attempt_timing", {})
        if not isinstance(timing, dict):
            timing = {}
            self.state["attempt_timing"] = timing
        timing.setdefault("producer_elapsed_seconds", None)
        timing.setdefault("critic_elapsed_seconds", None)

    def _stage_def(self, stage_id: int = None) -> dict:
        sid = stage_id or self.current_stage
        return next((s for s in STAGES if s["id"] == sid), {})

    def _memory_store(self) -> ResearchMemoryStore:
        return ResearchMemoryStore(self.project_id, self.project_dir)

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_to_employee(self, employee_id: str, description: str, title: str):
        """Create a task node in the tree and schedule it for the employee."""
        from onemancompany.core.task_tree import get_tree, save_tree_async
        from onemancompany.core.config import TASK_TREE_FILENAME
        from onemancompany.core.agent_loop import employee_manager

        tree_path = str(Path(self.project_dir) / TASK_TREE_FILENAME)
        tree = get_tree(Path(tree_path), project_id=self.project_id)

        # Find parent node (the root or EA node)
        root = tree.get_node(tree.root_id) if tree.root_id else None
        parent_id = tree.root_id
        # If root has an EA child, use that as parent
        if root:
            for child in tree.get_active_children(root.id):
                if child.employee_id in ("00004", "00002"):
                    parent_id = child.id
                    break

        node = tree.add_child(
            parent_id=parent_id,
            employee_id=employee_id,
            description=description,
            acceptance_criteria=[],
        )
        node.title = title
        node.project_id = self.project_id
        node.project_dir = self.project_dir
        # Tag so vessel knows this is pipeline-managed
        if not hasattr(node, 'metadata'):
            node.metadata = {}
        node.metadata = {**(node.metadata or {}), "pipeline_managed": True}

        save_tree_async(tree_path)

        self.state["active_node_id"] = node.id
        self.state["active_employee_id"] = employee_id
        self.state["active_task_started_at"] = time.time()
        self._save()

        # Start each pipeline stage with a FRESH Claude conversation. The
        # daemon otherwise resumes one session per (employee, project) and
        # accumulates history across every stage it touches — the critic
        # reviews all 9 stages, so its resumed history blows past the model
        # context window (observed: 623K tokens > 262K limit, Stage 6 critic
        # failed → empty deliverable). Pipeline tasks pass full context in
        # the prompt, so resumed history is pure overhead.
        try:
            from onemancompany.core.claude_session import reset_session
            reset_session(employee_id, self.project_id)
        except Exception as _e:  # best-effort; never block dispatch
            logger.debug("[PIPELINE] reset_session skipped for {}: {}", employee_id, _e)

        employee_manager.schedule_node(employee_id, node.id, tree_path)
        employee_manager._schedule_next(employee_id)

        logger.info(
            "[PIPELINE] Dispatched {} to employee {} (stage={}, phase={})",
            title, employee_id, self.current_stage, self.phase,
        )

    def _build_context(self) -> str:
        """Build cumulative context from prior context + all previous stage results."""
        parts = [f"Research topic: {self.topic}\n"]
        prior = self.state.get("prior_context", "")
        if prior:
            parts.append(f"--- Prior Context (uploaded files) ---\n{prior}\n")
        for sid in sorted(self.state.get("stage_results", {}).keys(), key=int):
            stage_def = self._stage_def(int(sid))
            result = self.state["stage_results"][sid]
            parts.append(f"--- Stage {sid}: {stage_def.get('name', '')} ---\n{result}\n")
        return "\n".join(parts)

    def _retrieve_memory_guidance(self, stage: dict, context: str, feedback: str = "") -> str:
        """Retrieve MemRL-style prior lessons for the current stage."""
        try:
            retrieved = self._memory_store().retrieve_stage_guidance(
                topic=self.topic,
                stage=stage,
                context=context,
                feedback=feedback,
            )
        except Exception as exc:
            logger.warning("[PIPELINE] Research memory retrieval failed: {}", exc)
            return ""

        self._ensure_memory_state()
        self.state["memory_retrievals"][str(stage["id"])] = {
            "ids": retrieved.memory_ids,
            "query": retrieved.query,
            "simmax": retrieved.simmax,
        }
        if retrieved.memory_ids:
            logger.info(
                "[PIPELINE] Retrieved {} research memories for stage {}",
                len(retrieved.memory_ids), stage["id"],
            )
        return retrieved.guidance

    def _record_stage_memory(
        self,
        stage: dict,
        *,
        producer_result: str,
        critic_result: str,
        passed: bool,
        confidence: float | None,
        outcome: str,
        producer_elapsed_seconds: float | None = None,
        critic_elapsed_seconds: float | None = None,
    ) -> str | None:
        self._ensure_memory_state()
        stage_key = str(stage["id"])
        retrieved_ids = self.state.get("memory_retrievals", {}).get(stage_key, {}).get("ids", [])
        reward = self._critic_reward(
            passed=passed,
            confidence=confidence,
            retries=self.state.get("retries", 0),
            exhausted=outcome == "critic_reject_exhausted",
        )
        try:
            memory_id = self._memory_store().record_stage_episode(
                topic=self.topic,
                stage=stage,
                producer_result=producer_result,
                critic_result=critic_result,
                passed=passed,
                confidence=confidence,
                retries=self.state.get("retries", 0),
                reward=reward,
                retrieved_memory_ids=retrieved_ids,
                outcome=outcome,
                producer_elapsed_seconds=producer_elapsed_seconds,
                critic_elapsed_seconds=critic_elapsed_seconds,
            )
        except Exception as exc:
            logger.warning("[PIPELINE] Research memory write failed: {}", exc)
            return None

        self.state["memory_episodes"][stage_key] = memory_id
        logger.info(
            "[PIPELINE] Recorded research memory {} for stage {} (reward={:.2f})",
            memory_id, stage["id"], reward,
        )
        return memory_id

    def _apply_ceo_memory_feedback(self, stage: dict, feedback: str, approved: bool) -> None:
        self._ensure_memory_state()
        stage_key = str(stage["id"])
        episode_id = self.state.get("memory_episodes", {}).get(stage_key)
        retrieved_ids = self.state.get("memory_retrievals", {}).get(stage_key, {}).get("ids", [])
        if not episode_id and not retrieved_ids:
            return
        try:
            update = self._memory_store().apply_ceo_feedback(
                episode_id=episode_id,
                retrieved_memory_ids=retrieved_ids,
                feedback=feedback,
                approved=approved,
            )
        except Exception as exc:
            logger.warning("[PIPELINE] Research memory CEO feedback update failed: {}", exc)
            return
        self.state["memory_feedback"][stage_key] = update
        self._save()

    @staticmethod
    def _critic_reward(
        *,
        passed: bool,
        confidence: float | None,
        retries: int,
        exhausted: bool = False,
    ) -> float:
        if exhausted:
            return -1.0
        if passed:
            base = confidence if confidence is not None else 0.7
            return max(-1.0, min(1.0, float(base) - (0.15 * int(retries))))
        miss = 1.0 - float(confidence if confidence is not None else 0.0)
        return max(-1.0, min(1.0, -max(0.35, miss)))

    # ------------------------------------------------------------------
    # Public API — called by routes.py and vessel.py
    # ------------------------------------------------------------------

    def _iteration_id(self) -> str:
        """Identifier used in git tag names. Standard layout is
        ``.../iterations/iter_NNN``; we use the basename directly so
        multi-iteration projects keep their tag namespaces separate.

        For legacy / non-standard layouts where the basename doesn't
        match ``iter_\\d+``, we hash the full project_dir into a stable
        synthetic id to avoid cross-iteration tag collisions (which
        would silently overwrite each other under ``tag -f``).
        """
        name = Path(self.project_dir).name
        if name and re.match(r"^iter_\d+$", name):
            return name
        if not name:
            return _DEFAULT_ITERATION
        # Non-standard dir name. Derive a stable synthetic id from the
        # path so different projects with the same basename don't collide.
        import hashlib
        digest = hashlib.sha1(self.project_dir.encode("utf-8")).hexdigest()[:8]
        logger.debug(
            "[PIPELINE] Non-standard project dir basename {!r}; using synthetic iteration id iter_{}",
            name, digest,
        )
        return f"iter_{digest}"

    def start(self, start_stage: int = FIRST_STAGE_ID, end_stage: int = LAST_STAGE_ID, prior_context: str = "", stage_assignments: dict = None, auto_approve: bool = False, paper_config: dict = None, staged: bool = False):
        """Begin the pipeline from the given stage.

        ``auto_approve`` (headless/unattended mode): when True, every CEO gate
        is advanced automatically — the pipeline runs end-to-end with no human
        confirmation. Used for background full-auto runs.

        ``paper_config`` (Stage 8 only): {"output_format": "markdown"|"latex"|"docx"|"both",
        "venue": "iclr2026"|"neurips2026"}. Read only when dispatching Stage 8 —
        earlier stages never see it. Persisted into pipeline_state.yaml so a
        revert to Stage 8 reuses the same target format.

        ``staged`` (#feasibility-first, C): when True the pipeline runs a cheap
        Tier-1 feasibility study FIRST (research_phase="feasibility"); a positive
        go/no-go signal at the Stage 7 result-review promotes it to the full
        study (research_phase flips to "full", pipeline returns to Stage 5 to
        design the rigorous experiment). Default False → research_phase="full"
        and behaviour is unchanged (zero regression).
        """
        self.state["current_stage"] = max(FIRST_STAGE_ID, min(start_stage, LAST_STAGE_ID))
        self.state["start_stage"] = self.state["current_stage"]
        self.state["end_stage"] = max(self.state["current_stage"], min(end_stage, LAST_STAGE_ID))
        self.state["prior_context"] = prior_context
        self.state["stage_assignments"] = stage_assignments or {}
        self.state["auto_approve"] = bool(auto_approve)
        self.state["paper_config"] = paper_config or {}
        self.state["research_phase"] = "feasibility" if staged else "full"
        self.state["phase"] = "producer"
        self.state["retries"] = 0
        self._save()
        # Auto-init the workspace as a git repo so per-stage commits and
        # later revert-to-here ops have somewhere to land. Idempotent —
        # existing repos are left alone.
        from onemancompany.core import project_repo
        try:
            project_repo.ensure_initialized(self.project_dir, iteration=self._iteration_id())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("[PIPELINE] project_repo init failed for {}: {}", self.project_dir, exc)
        logger.info("[PIPELINE] Starting from stage {} to stage {}", self.state["current_stage"], self.state["end_stage"])
        # Admission control (#159): if we are already at the concurrency cap,
        # park this run as "queued" instead of dispatching. It will be promoted
        # by dequeue_next_pipeline() when a running pipeline becomes terminal.
        # Exclude self from the count — our own phase is already "producer" at
        # this point but we haven't dispatched yet, so we're not truly executing.
        executing = sum(
            1 for pid, eng in _active_pipelines.items()
            if pid != self.project_id and eng.state.get("phase") in _EXECUTING_PHASES
        )
        if executing >= MAX_CONCURRENT_PIPELINE_RUNS:
            import time as _time
            self.state["phase"] = "queued"
            self.state["queue_requested_at"] = str(_time.time())
            self._save()
            logger.info(
                "[PIPELINE] Queued {} (cap={}, executing={})",
                self.project_id, MAX_CONCURRENT_PIPELINE_RUNS, executing,
            )
            return
        self._dispatch_producer()

    def queue_pending_feedback(self, text: str) -> None:
        """Buffer CEO/user feedback to inject into the next producer dispatch.

        Called when the CEO sends a chat message while the pipeline is mid-flight
        (producer/critic running, or auto-retrying after a REJECT). The pipeline
        is not at a gate, so we cannot call ``on_ceo_approve`` — but the user's
        guidance is valuable for the next producer iteration. The buffered text
        is consumed on the next ``_dispatch_producer`` call.
        """
        text = (text or "").strip()
        if not text:
            return
        pending = self.state.get("pending_user_feedback", "")
        self.state["pending_user_feedback"] = (pending + "\n\n" + text) if pending else text
        self._save()
        logger.info(
            "[PIPELINE] Queued CEO feedback (len={}) at stage {} phase {}",
            len(text), self.current_stage, self.phase,
        )

    def _consume_pending_feedback(self) -> str:
        text = self.state.get("pending_user_feedback", "")
        if text:
            self.state["pending_user_feedback"] = ""
            self._save()
        return text

    @staticmethod
    def _extract_aigraph_markdown(result) -> str:
        """Pull the markdown out of an aigraph tool result, which may be a
        markdown string or a JSON object carrying ``ideas_markdown`` /
        ``markdown`` / ``report`` / ``result`` (research_ideas returns the
        markdown wrapped in a JSON envelope alongside its stats)."""
        import json as _json
        d = None
        if isinstance(result, dict):
            d = result
        elif isinstance(result, str):
            s = result.strip()
            if s.startswith("{"):
                try:
                    d = _json.loads(s)
                except Exception:  # noqa: BLE001
                    return result
            else:
                return result
        else:
            return str(result)
        for key in ("ideas_markdown", "markdown", "report", "result"):
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return _json.dumps(d, ensure_ascii=False)

    def _save_stage3_graph_artifacts(self, bundle) -> None:
        """Persist the aigraph planet-graph artifacts to the workspace:
        ``stage3_conflict_graph.html`` (self-contained D3 page, openable in a
        browser) and ``stage3_conflict_graph.json`` (the {nodes, edges} data).
        Best-effort; never raises."""
        try:
            pdir = Path(self.project_dir)
            if getattr(bundle, "graph_html", ""):
                (pdir / "stage3_conflict_graph.html").write_text(
                    bundle.graph_html, encoding="utf-8"
                )
            if getattr(bundle, "graph", None):
                import json as _json
                (pdir / "stage3_conflict_graph.json").write_text(
                    _json.dumps(bundle.graph, ensure_ascii=False), encoding="utf-8"
                )
            if bundle.dashboard_url or bundle.graph_url:
                logger.info(
                    "[aigraph] Stage 3 planet graph: dashboard={} graph={}",
                    bundle.dashboard_url or "-", bundle.graph_url or "-",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[aigraph] could not save Stage 3 graph artifacts: {}", exc)

    def _stage3_markdown_with_trace(self, bundle) -> str:
        """Append an aigraph provenance footer (source/run/coverage/dashboard
        links) to the report so the Stage 3 deliverable is self-documenting."""
        md = bundle.markdown
        try:
            lines = [
                "", "<!-- aigraph provenance -->",
                f"<!-- source={bundle.source} status={bundle.status} run_id={bundle.run_id} "
                f"coverage={bundle.n_matched}/{bundle.n_total} top_relevance={bundle.top_relevance} -->",
            ]
            if bundle.dashboard_url:
                lines.append(f"<!-- dashboard: {bundle.dashboard_url} -->")
            if bundle.graph_url:
                lines.append(f"<!-- graph: {bundle.graph_url} -->")
            if (Path(self.project_dir) / "stage3_conflict_graph.html").exists():
                lines.append("<!-- planet graph saved: stage3_conflict_graph.html -->")
            return md.rstrip() + "\n" + "\n".join(lines) + "\n"
        except Exception:  # noqa: BLE001
            return md

    def _fetch_aigraph_idea_report(self) -> str | None:
        """Deterministically fetch the aigraph (LCG) idea report for Stage 3.

        Primary path: a direct, registration-INDEPENDENT MCP call
        (``aigraph_grounding.fetch_idea_report`` → aigraph's ``get_idea_report``
        over the live streamable-http MCP session). This is the reliable path
        because it does NOT depend on ``aigraph_research_ideas`` being registered
        as an asset tool — that registration is the retired #132 startup binding,
        so it is usually absent, which previously left Stage 3 silently
        ungrounded. The MCP call runs in a dedicated thread's own event loop so
        it is safe inside uvicorn's running loop (no anyio cancel-scope crash).

        Fallback path: the registered ``aigraph_research_ideas`` asset tool, for
        environments that still wire the MCP tools. Both return the markdown
        report (0-LLM, arxiv-grounded), or None if unreachable/empty. Never
        raises."""
        # --- primary: one-shot research_e2e bundle (report + planet graph) ---
        try:
            from onemancompany.agents.aigraph_grounding import fetch_stage3_bundle
            b = fetch_stage3_bundle(self.topic)
            if b.ok and b.markdown and "arxiv:" in b.markdown:
                self._save_stage3_graph_artifacts(b)
                logger.info(
                    "[PIPELINE] Stage 3 grounded via aigraph {} (status={}): {} chars, "
                    "coverage={} {}/{}, graph_html={}b",
                    b.source, b.status, len(b.markdown), b.strength or "?",
                    b.n_matched, b.n_total, len(b.graph_html),
                )
                return self._stage3_markdown_with_trace(b)
            if b.ok and b.is_weak and b.markdown:
                self._save_stage3_graph_artifacts(b)
                logger.warning(
                    "[PIPELINE] aigraph weak corpus coverage for topic={!r} "
                    "(matched={}); Stage 3 grounding sparse",
                    (self.topic or "")[:80], b.n_matched,
                )
                return self._stage3_markdown_with_trace(b)
            if not b.ok:
                logger.debug(
                    "[PIPELINE] aigraph bundle fetch failed ({}); trying registered tool", b.error,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[PIPELINE] aigraph bundle unavailable ({}); trying registered tool", exc,
            )
        # --- fallback: the registered aigraph_research_ideas asset tool ---
        try:
            from onemancompany.core.tool_registry import tool_registry
            tool = tool_registry.get_tool("aigraph_research_ideas")
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PIPELINE] aigraph tool lookup failed: {}", exc)
            return None
        if tool is None:
            logger.debug(
                "[PIPELINE] aigraph_research_ideas not registered and direct MCP "
                "unavailable; Stage 3 ungrounded"
            )
            return None
        try:
            result = tool.invoke({
                "topic": self.topic,
                "min_ideas": self.state.get("aigraph_min_ideas", 8),
                "reuse": self.state.get("aigraph_reuse", True),
                "as_markdown": True,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PIPELINE] aigraph_research_ideas invocation failed: {}", exc)
            return None
        report = self._extract_aigraph_markdown(result)
        if not report or len(report) < 500:
            logger.warning(
                "[PIPELINE] aigraph_research_ideas returned empty/stub ({} chars); "
                "Stage 3 left to agent fallback", len(report or ""))
            return None
        logger.info("[PIPELINE] aigraph grounding fetched for Stage 3: {} chars", len(report))
        return report

    def _ensure_stage3_grounded(self, deliverable: "Path", result: str) -> str:
        """Guarantee the Stage 3 deliverable is arxiv-grounded (augmentation backstop).

        If the producer dropped the injected grounding (its deliverable carries
        no ``arxiv:`` citations) but the engine fetched a grounded report in the
        pre-step (``aigraph_grounding.md``), append the verbatim report so Stage 3
        is grounded regardless of producer behavior — the deterministic backstop
        for the LLM ignoring the injected grounding (#130). Never raises.
        """
        try:
            grounding_path = Path(self.project_dir) / "aigraph_grounding.md"
            if not grounding_path.exists():
                return result
            grounding = grounding_path.read_text(encoding="utf-8").strip()
            if "arxiv:" not in grounding:
                return result  # weak/empty grounding — nothing worth grafting
            current = result or ""
            if "arxiv:" in current and "# Selected Hypotheses" in current:
                return result  # producer preserved the grounding — nothing to do
            logger.warning(
                "[aigraph] Stage 3 deliverable was not grounded — grafting the "
                "deterministic aigraph report back in (producer dropped it)"
            )
            prefix = (current.rstrip() + "\n\n") if current.strip() else ""
            merged = (
                prefix
                + "<!-- aigraph grounding grafted deterministically by the pipeline -->\n\n"
                + grounding
                + "\n"
            )
            try:
                deliverable.write_text(merged, encoding="utf-8")
            except OSError as exc:
                logger.debug("[aigraph] could not rewrite Stage 3 deliverable: {}", exc)
            return merged
        except Exception as exc:  # noqa: BLE001
            logger.debug("[aigraph] ensure-grounded step skipped: {}", exc)
            return result

    def _dispatch_producer(self, feedback: str = ""):
        """Dispatch the current stage's producer. Uses user assignment if set."""
        if self.phase != "critic":
            self._reset_attempt_timing()
        stage = self._stage_def()
        # Check if user assigned a specific employee to this stage
        assignments = self.state.get("stage_assignments", {})
        assigned = assignments.get(str(stage["id"]))
        employee_id = assigned if assigned else _find_employee_for_stage(stage["id"], stage["skill"])
        if not employee_id:
            logger.error("[PIPELINE] No employee with skill '{}' for stage {}", stage["skill"], stage["id"])
            self.state["phase"] = "failed"
            self._save()
            self._on_became_terminal()
            return

        context = self._build_context()
        memory_guidance = self._retrieve_memory_guidance(stage, context, feedback)
        desc = (
            f"Stage {stage['id']}: {stage['name']}\n\n"
            f"{context}\n"
        )
        if memory_guidance:
            desc += f"\n--- Retrieved Research Memory ---\n{memory_guidance}\n"
        if feedback:
            desc += f"\nFeedback from previous review:\n{feedback}\n"
        user_feedback = self._consume_pending_feedback()
        if user_feedback:
            desc += f"\nDirect guidance from CEO (received during the previous attempt):\n{user_feedback}\n"
        # Stage 3 (Idea Generation) is HARD-CODED to the aigraph (LCG) report.
        # The producer LLM was observed ignoring even the verbatim-injected
        # report and fabricating #claim-N citations from the Stage-2 survey, so
        # "let the LLM generate ideas" is removed for Stage 3: the engine fetches
        # the report (reliable 0-LLM call), writes it as the deliverable, and
        # goes straight to the critic — no producer LLM, no chance to fabricate.
        # Falls back to the LLM only if aigraph is unreachable.
        if stage["id"] == 3:
            aigraph_report = self._fetch_aigraph_idea_report()
            if aigraph_report:
                # AUGMENTATION: the engine deterministically fetched the
                # arxiv-grounded report. Persist it for the producer-complete
                # backstop, inject it verbatim, and let the producer synthesise
                # ONE runnable pilot on top. The producer cannot fabricate the
                # grounding away — _ensure_stage3_grounded grafts the verbatim
                # report back into the deliverable if the producer drops it (#130).
                try:
                    (Path(self.project_dir) / "aigraph_grounding.md").write_text(
                        aigraph_report, encoding="utf-8"
                    )
                except OSError as exc:
                    logger.debug("[aigraph] could not persist Stage 3 grounding: {}", exc)
                logger.info(
                    "[PIPELINE] Stage 3 grounded deterministically ({} chars); producer "
                    "will synthesise a pilot on top (augmentation)", len(aigraph_report),
                )
                desc += (
                    "\n## AIGRAPH GROUNDING (verbatim, authoritative — do NOT hand-write or fabricate)\n"
                    "The engine already fetched the arxiv-grounded Selected Hypotheses report "
                    "below. Your ONLY job is to SYNTHESISE exactly ONE focused, "
                    "single-GPU-runnable pilot hypothesis on top:\n"
                    "  ## Primary Pilot Hypothesis\n"
                    "  state H0/H1, ONE metric, a named dataset, sample size N, and the decoding "
                    "setting — citing the arxiv claim ids (arxiv:NNNN.NNNNN#cNN) drawn from the "
                    "report.\n"
                    "Then PRESERVE the full report verbatim BELOW your pilot in "
                    "stage3_idea_generator.md (it contains the '# Selected Hypotheses' section "
                    "the downstream conflict graph and critic read). Do NOT invent citations or "
                    "replace the grounded content.\n\n"
                    f"{aigraph_report}\n"
                )
            else:
                desc += (
                    "\n## Stage 3 grounding (FALLBACK — aigraph unavailable)\n"
                    "The aigraph report could not be fetched by the engine. Call "
                    "`aigraph_get_idea_report(topic=<the refined topic>, "
                    "run='arxiv-reasoning-v0.7-540p-thaw1', k=8, kind='creator')` yourself "
                    "and write its output verbatim. If it is still unavailable, write "
                    "'FALLBACK: agent-generated (ungrounded)' in the header and cite ZERO "
                    "claim IDs — do NOT fabricate citations.\n"
                )
        # Stage 4 (Methodology Design) must run a multi-agent debate before
        # writing the methodology. The convener skill is the runbook.
        if stage["id"] == 4:
            desc += (
                "\n## READING RESEARCH MATERIALS\n"
                "Before starting, gather all available research context:\n"
                "- If the project workspace contains any PDF files (*.pdf), use the "
                "`read_pdf` tool to extract their content — these may be reference papers "
                "or prior work that should inform the methodology.\n"
                "- If the Stage 2 (Literature Survey) output references specific paper URLs "
                "or DOIs, use the `fetch` or `web_search` tool to retrieve their abstracts "
                "or full text from arXiv/Semantic Scholar. This grounds the methodology in "
                "real published work rather than recalled knowledge.\n"
                "- URLs in the format `https://arxiv.org/abs/...` can be fetched directly "
                "to get the abstract; `https://arxiv.org/pdf/...` for the full paper.\n\n"
                "## REQUIRED FIRST STEP\n"
                'After reading available materials, call load_skill("methodology-debate-convener") '
                "and follow the runbook exactly. It walks you through the full "
                "draft → debate → revise flow: assemble a diverse team, write a v1 "
                "methodology draft, convene a debate that critiques the draft, save "
                "the transcript, and revise v1 into a CCF-A-grade final methodology "
                "(8 sections, English only). Do not skip any phase.\n"
                "\n## REQUIRED FINAL STEP\n"
                'After the final methodology document is saved, call load_skill("paper-framework-figure") '
                "and follow that runbook to render a CCF-A-grade framework figure "
                "via nano banana (google/gemini-2.5-flash-image on OpenRouter). The "
                "skill walks you through synthesising the 4-section work summary "
                "(背景 / 问题和难点 / 创新点 / 具体的技术路线) from the prior stages, "
                "calling the image API with the correct 'Generate ONE image' wrapper, "
                "saving the PNG as stage4_framework_figure.png, and embedding it in "
                "stage4_methodology_designer.md with a numbered Figure caption. The "
                "Stage 4 critic checks D10 (Framework Figure) as a hard gate — "
                "missing or generic figure = auto-REJECT. Every CCF-A methodology "
                "ships with one, no exceptions.\n"
                "\n## METHODOLOGICAL DEPTH (required — a methodology is not just an experiment plan)\n"
                "The final methodology MUST formalize the METHOD itself, not only "
                "how it will be tested. Include all three:\n"
                "- **Formal definition with equations**: state the method/objective "
                "in real mathematics — the objective/loss function, the key "
                "quantities, and any derivation or complexity claim — using LaTeX "
                "math ($...$ inline, $$...$$ for displayed equations). Decorative "
                "notation is not enough; the reader must be able to reimplement the "
                "method from the equations.\n"
                "- **A pseudocode / Algorithm block**: give at least one numbered "
                "Algorithm environment (Input / Output / numbered steps) for the "
                "core procedure, so the method is unambiguous and implementable.\n"
                "- **An explicit Contributions / Novelty statement**: a short "
                "section naming what is NEW versus prior work (why this is a "
                "contribution, not a trivial recombination), grounded in the "
                "Stage 2/3 claim IDs. The Stage 4 critic grades methodological "
                "depth (D11 Method Formalization + D12 Contribution) — a "
                "methodology with no equations, no pseudocode, or no novelty "
                "claim is shallow and will be rejected.\n"
            )
        # Stage 5 (Experiment Design) mirrors the Stage 4 flow: draft → debate
        # → revise → coordination (assignments table). The experiment convener
        # skill is the runbook.
        elif stage["id"] == 5:
            desc += (
                "\n## REQUIRED FIRST STEP\n"
                'Before doing anything else, call load_skill("experiment-debate-convener") '
                "and follow the runbook exactly. It walks you through reading the Stage 4 "
                "methodology, drafting an initial experiment plan, debating it with the "
                "team, revising it into a CCF-A-grade experiment plan, and producing a "
                "coordination assignments table for Stage 6 execution. Do not write the "
                "experiment plan directly without convening the debate first.\n"
            )
        # Stage 6a (Code Implementation) is the FIRST of Stage 6's two
        # sequential producers. It clones the upstream codebase named in
        # stage5_codebase_pin.md, applies the patches the pin lists, and
        # produces stage6_implementation_receipt.md naming the runnable
        # entrypoint. Stage 6b (the experiment_runner) reads that receipt
        # and submits the actual runs — see _dispatch_producer_b().
        elif stage["id"] == 6:
            desc += (
                "\n## REQUIRED INPUTS — READ BEFORE ANYTHING ELSE\n"
                "Read these three files from the project workspace before loading "
                "the runbook or touching any code:\n"
                "  read('stage4_methodology_designer.md')  # immutable contract: "
                "IVs, DVs, evaluation metrics, statistical tests — every hardcoded "
                "value in your implementation must match this document exactly\n"
                "  read('stage5_experiment_designer.md')   # locked parameter "
                "values: seeds, sample sizes, decoding params, n_conditions\n"
                "  read('stage5_codebase_pin.md')          # upstream repo + "
                "adaptation surface\n"
                "\n## REQUIRED IMPLEMENTATION STEP\n"
                'Then call load_skill("code-implementation-runbook") '
                "and follow it. This is Stage 6a (Implementation). The runbook's "
                "Phase 0 walks you through cloning the upstream repo at the pinned "
                "commit, running the upstream test suite on a clean checkout, "
                "applying only the patches the pin's Adaptation surface table "
                "lists, and re-running the tests. Phase 5 produces "
                "stage6_implementation_receipt.md naming the runnable entrypoint "
                "command. ADAPT, do not REWRITE — the upstream pin exists precisely "
                "to avoid from-scratch code. The exception path (NO USABLE UPSTREAM "
                "FOUND in the pin) is allowed but triggers extra critic scrutiny. "
                "The Stage 6b runner depends on your receipt; do not skip it.\n"
            )
        # Stage 7 (Result Analysis) reads the Stage 4 methodology, the
        # Stage 5 experiment plan + assignments, and the Stage 6
        # experimentalist report, then produces a confirmatory analysis
        # that obeys the pre-registered tests and labels every claim as
        # confirmatory or exploratory. HARKing is auto-REJECTED.
        elif stage["id"] == 7:
            desc += (
                "\n## REQUIRED FIRST STEP\n"
                'Before doing anything else, call load_skill("result-analysis-runbook") '
                "and follow it. The runbook tells you how to reconstruct the "
                "pre-registration contract from Stage 4/5, map Stage 6 evidence "
                "onto each hypothesis, run only the pre-registered statistical "
                "tests with effect sizes + 95% CIs, run the manipulation and "
                "falsification checks, and cap the overall verdict at whatever "
                "coverage Stage 6 actually delivered. Do not invent new tests, "
                "do not substitute metrics, do not HARK.\n"
                "\n## REQUIRED — GENERATE RESULT FIGURES (do not skip)\n"
                "After the statistics are computed, render the result figures "
                "the Stage 5 FIGURE MANIFEST specifies. For each manifest row "
                'call load_skill("result-figures") and follow it to plot the '
                "RESULT_JSON field named in the manifest, saving each as a "
                "`stage7_<name>.png` in the project workspace and embedding it "
                "in the report with a numbered caption "
                "(`![Figure N: ...](stage7_<name>.png)`). The figure's values "
                "MUST match the Section-3 confirmatory tables. If Stage 5 did "
                "not lock a manifest, still produce at least one figure of the "
                "primary metric/effect with its CI. The Stage 7 critic grades "
                "D11 Result Figures — a report with confirmatory numbers but no "
                "`stage7_*.png` is incomplete. (Historical gap B1: every prior "
                "paper shipped with only the Stage-4 framework figure and zero "
                "result figures because this step was never required.)\n"
            )
        # Stage 8 (Paper Generation) renders the CCF-A paper from Stage 4
        # methodology + Stage 5 plan + Stage 6 run + Stage 7 results. A
        # framework figure (Figure 1) is non-negotiable for CCF-A venues.
        elif stage["id"] == 8:
            desc += (
                "\n## REQUIRED FIRST STEP — REUSE THE STAGE 4 FRAMEWORK FIGURE\n"
                "Stage 4 already rendered the framework figure as "
                "`stage4_framework_figure.png` in this same iteration directory. "
                "Do NOT call `paper-framework-figure` again — do NOT regenerate via "
                "nano banana — it would burn API budget and produce a different "
                "(potentially inconsistent) figure. Instead, embed the existing "
                "PNG as the paper's Figure 1 by including a line of the form\n"
                "\n    ![Figure 1. <one-paragraph caption naming every box/arrow shown>]"
                "(stage4_framework_figure.png)\n"
                "\nin the Methodology section (or the Introduction, whichever you "
                "reference first). The caption must NAME every component the figure "
                "actually shows (Stage 1: Prompt-Format Control, Stage 2: Gated Routing, "
                "Stage 3: Adaptive Budgeting, Stage 4: Evaluation & Gatekeeping, "
                "plus the Unified Evaluation row and the Shared Controls row) — no "
                "'see above', no vague pronouns. The Stage 8 critic checks D-FIG "
                "(Figure 1 embedded + named) as a hard gate.\n"
                "\nWrite stage8_paper_writer.md with the standard CCF-A sections "
                "(Abstract, Introduction, Related Work, Methodology, Experimental "
                "Setup, Results, Discussion, Limitations, Conclusion, "
                "Reproducibility, References). Preserve all LaTeX notation "
                "($...$, $$...$$) from Stage 4 verbatim.\n"
                "\n## NUMBER DISCIPLINE — every statistic must be traceable (hard gate)\n"
                "A deterministic gate (#44) REJECTS the paper when ≥2 high-precision "
                "numbers cannot be traced to the evidence, and run 32e19fc5f3e7 died "
                "here. A 'high-precision' number is a decimal with ≥2 decimal places "
                "AND ≥3 significant digits (e.g. 0.80, 85.97, 1.5406; integers, "
                "1-decimal values like 5.1, and p<0.001/alpha 0.05 are exempt).\n"
                "Rules — follow exactly:\n"
                "  - Every such number MUST appear (allowing rounding and "
                "fraction↔percent scaling, e.g. 0.80↔80.0) in stage7_result_analyst.md "
                "or a Stage 6 RESULT_JSON. Copy them; do NOT invent, recompute, or "
                "'estimate' values.\n"
                "  - Do NOT introduce a derived statistic (CI, effect size, %% "
                "improvement, mean) that Stage 7 did not already compute. If you want "
                "to report one and it is not in Stage 7, you cannot fabricate it here — "
                "state only what Stage 7 measured.\n"
                "  - **Self-check before submit_result**: scan your paper for every "
                "decimal with ≥2 places; for each, confirm the same value (±rounding, "
                "±100×) is present in stage7_result_analyst.md or RESULT_JSON; delete "
                "or rephrase any that is not. A paper that fails this gate is rejected "
                "regardless of how well it reads.\n"
            )
            _paper_cfg = self.state.get("paper_config") or {}
            _fmt = (_paper_cfg.get("output_format") or "markdown").strip().lower()
            _venue = (_paper_cfg.get("venue") or "").strip().lower()
            desc += (
                "\n## OUTPUT FORMAT DIRECTIVE\n"
                f"output_format={_fmt}"
            )
            if _fmt in ("latex", "both"):
                desc += f" venue={_venue or 'iclr2026'}"
            desc += (
                "\n(Parse this directive per skills/paper_writer/SKILL.md Step 4. "
                "For latex/both, call fetch_latex_template(venue=..., dest_dir=<workspace>/stage8_paper) "
                "and overwrite main.tex with your synthesised content. "
                "For docx, call render_docx. For markdown, write stage8_paper_writer.md as usual.)\n"
            )
        desc += (
            f"\nYour task: produce the deliverable for this stage. "
            f"Write your output to a file named stage{stage['id']}_{stage['skill']}.md "
            f"in the project workspace using the write() tool. "
            f"Then call submit_result() with a summary."
        )

        self.state["phase"] = "producer"
        self._save()
        self._dispatch_to_employee(employee_id, desc, f"Stage {stage['id']}: {stage['name']}")
        # Resolve employee name for frontend display
        emp_name = employee_id
        configs = load_employee_configs()
        if employee_id in configs:
            emp_name = configs[employee_id].name
        self._emit_stage_event("stage_start", stage["id"], employee_name=emp_name, employee_id=employee_id)

    def _stage6_infra_paths(self):
        """Locate the experiment-infra skill's scripts dir + config (#156).

        Env-first (``EXPERIMENT_INFRA_SCRIPTS`` / ``STAGE6_INFRA_CONFIG``), else a
        best-effort glob under the data root. Returns ``(scripts_dict, config_path)``;
        ``scripts`` is {} if not found (the caller then degrades to the agent runner).
        Nothing experiment-specific is baked in — same philosophy as #152.
        """
        import os as _os
        from onemancompany.agents import stage6_infra as s6
        scripts_dir = _os.environ.get("EXPERIMENT_INFRA_SCRIPTS")
        if not scripts_dir:
            try:
                p = Path(self.project_dir)
                for anc in [p, *p.parents]:
                    hits = list(anc.glob("**/skills/experiment-infra/scripts"))
                    if hits:
                        scripts_dir = str(hits[0])
                        break
                    if (anc / "company").is_dir():
                        break
            except Exception:  # noqa: BLE001
                scripts_dir = None
        scripts = s6.find_infra_scripts(scripts_dir)
        config = _os.environ.get("STAGE6_INFRA_CONFIG", "")
        if not config and scripts_dir:
            cand = Path(scripts_dir).parent / "assets" / "base.conf.json"
            if cand.exists():
                config = str(cand)
        return scripts, config

    def _try_deterministic_stage6_submit(self):
        """#156: the engine submits the Stage-6 experiment itself, parameterised
        from the 6a receipt + the experiment-infra skill's OWN scripts (not
        hardcoded). Returns ``{"run_ids": [...]}`` on success, else None to fall
        back to the agent runner. Never raises."""
        try:
            from onemancompany.agents import stage6_infra as s6
            receipt_path = Path(self.project_dir) / "stage6_implementation_receipt.md"
            if not receipt_path.exists():
                return None
            receipt = s6.parse_receipt(
                receipt_path.read_text(encoding="utf-8", errors="replace"),
                project_id=self.project_id, iteration=self._iteration_id() or "iter_001",
            )
            if not receipt.ok:
                logger.debug("[PIPELINE] #156: no runnable entrypoint in receipt; using agent runner")
                return None
            scripts, config = self._stage6_infra_paths()
            if not (scripts and config):
                logger.debug("[PIPELINE] #156: experiment-infra scripts/config not found; using agent runner")
                return None
            res = s6.submit(receipt, scripts, config, kind="smoke")
            if not res.ok:
                logger.warning(
                    "[PIPELINE] #156: deterministic Stage-6 submit unavailable ({}); using agent runner",
                    res.error,
                )
                return None
            logger.info("[PIPELINE] #156: Stage-6 smoke submitted by engine — run_id={}", res.run_id)
            return {"run_ids": [res.run_id]}
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PIPELINE] #156: deterministic submit errored ({}); using agent runner", exc)
            return None

    def _dispatch_producer_b(self, feedback: str = ""):
        """Dispatch Stage 6b — the experiment runner. Runs after Stage 6a
        (code_implementer) produces stage6_implementation_receipt.md.

        #156: the engine first tries a DETERMINISTIC submit (parameterised from
        the 6a receipt + the experiment-infra skill scripts) and parks in
        ``producer_b_waiting`` so the existing run_tracker → finalize machinery
        handles polling + the report from a REAL run. Only if infra is
        unavailable does it fall back to the (unreliable) agent runner.
        """
        stage = self._stage_def()
        if stage["id"] != 6:
            logger.error("[PIPELINE] _dispatch_producer_b called for non-Stage-6 stage {}", stage["id"])
            return

        # --- #156: deterministic engine-driven submit (preferred) ---
        sub = self._try_deterministic_stage6_submit()
        if sub and sub.get("run_ids"):
            from datetime import datetime, timezone
            self.state["phase"] = "producer_b_waiting"
            self.state["pending_run_ids"] = sub["run_ids"]
            self.state["pending_waiting_started_at"] = datetime.now(timezone.utc).isoformat()
            runs = self.state.setdefault("stage_6_runs", {})
            for rid in sub["run_ids"]:
                runs.setdefault(rid, {"status": "running"})
            self._save()
            logger.info(
                "[PIPELINE] Stage 6b submitted deterministically (#156): {} — parked in "
                "producer_b_waiting; run_tracker will poll to terminal.", sub["run_ids"],
            )
            self._emit_stage_event("stage_waiting", stage["id"])
            return

        # --- fallback: the agent runner (unchanged) ---
        employee_id = _find_stage_6b_employee()
        if not employee_id:
            logger.error("[PIPELINE] No experiment_runner employee for Stage 6b")
            self.state["phase"] = "failed"
            self._save()
            self._on_became_terminal()
            return

        context = self._build_context()
        desc = (
            f"Stage 6b: Auto Experiment Execution\n\n"
            f"Stage 6a (code implementer) has produced stage6_implementation_receipt.md "
            f"in the project workspace naming the runnable entrypoint. Your job is to "
            f"submit the run(s) to remote infra and capture evidence.\n\n"
            f"{context}\n"
        )
        if feedback:
            desc += f"\nFeedback from previous review:\n{feedback}\n"
        user_feedback = self._consume_pending_feedback()
        if user_feedback:
            desc += f"\nDirect guidance from CEO (received during the previous attempt):\n{user_feedback}\n"
        desc += (
            "\n## REQUIRED FIRST STEP\n"
            'Before doing anything else, call load_skill("experiment-execution-runbook") '
            "and follow it. The runbook tells you how to read "
            "stage6_implementation_receipt.md (Stage 6a's output) and "
            "stage5_assignments.md, then submit smoke-then-full runs via "
            'load_skill("experiment-infra") for each `experiment_runner` row. '
            "Non-runner rows are noted as deferred. Do not fabricate or "
            "simulate results — if a remote submit is required but "
            "credentials are missing, report the failure.\n"
        )
        desc += (
            f"\nYour task: write stage6_experimentalist.md (the evidence "
            f"report) and call submit_result() with a summary referencing it."
        )

        self.state["phase"] = "producer_b"
        self._save()
        self._dispatch_to_employee(employee_id, desc, f"Stage 6b: Auto Experiment Execution")
        emp_name = employee_id
        configs = load_employee_configs()
        if employee_id in configs:
            emp_name = configs[employee_id].name
        self._emit_stage_event("stage_start", stage["id"], employee_name=emp_name, employee_id=employee_id)

    def _dispatch_critic(self, producer_result: str):
        """Dispatch the adversarial critic to review the producer's output."""
        stage = self._stage_def()
        critic_id = _find_employee_by_skill(CRITIC_SKILL)
        if not critic_id:
            logger.warning("[PIPELINE] No critic employee found, auto-passing stage {}", stage["id"])
            self._record_stage_memory(
                stage,
                producer_result=producer_result,
                critic_result="No adversarial critic was available; auto-pass.",
                passed=True,
                confidence=None,
                outcome="auto_pass",
                producer_elapsed_seconds=self.state.get("attempt_timing", {}).get("producer_elapsed_seconds"),
                critic_elapsed_seconds=self.state.get("attempt_timing", {}).get("critic_elapsed_seconds"),
            )
            self._on_critic_pass(producer_result)
            return

        desc = (
            f"Gate Review: Stage {stage['id']} ({stage['name']})\n\n"
            f"Review the following output and provide:\n"
            f"1. A confidence score (0.0 to 1.0)\n"
            f"2. A PASS or REJECT decision\n"
            f"3. Specific reasoning\n\n"
            f"If REJECT, explain exactly what needs to be improved.\n\n"
        )
        # Stage 3 (Idea Generation) is produced by the aigraph
        # literature-conflict-graph tool, so its deliverable is an aigraph idea
        # report — NOT a hand-written idea-generation document. Grade on that.
        if stage["id"] == 3:
            desc += (
                "## STAGE 3 IS A LITERATURE-CONFLICT-GRAPH (aigraph) DELIVERABLE\n"
                "The Stage 3 deliverable is generated by aigraph from its frozen "
                "literature corpus — NOT hand-written prose. It comes in one of "
                "two valid shapes, both correct and intentional:\n"
                "  (a) a `# Ideas — <topic>` report: a summary line citing "
                "N idea(s) from N papers / N claims, then `## <n>. <title> "
                "_(…, conf=…)_` idea items, each with a `**TL;DR.**`, a concrete "
                "mechanism, and arxiv-grounded evidence; or\n"
                "  (b) a `# Selected Hypotheses` report with `### Anomaly a… —` "
                "and `### h… —` / `### a…#cr… —` hypothesis items citing claim IDs.\n"
                "PASS when: a topic / ideas heading is present AND the report "
                "contains at least one grounded idea or hypothesis with corpus "
                "evidence (arxiv references or claim IDs). Do NOT require the "
                "generic idea-generation sections (evaluation architecture, "
                "method pseudocode, risk tables, outcome scenarios) — those "
                "belong to Stages 4–5, not here. Only REJECT if the report is "
                "empty, has zero ideas/hypotheses, or says it found no matches "
                "for the topic.\n\n"
            )
        # Stage 4 (Methodology Design) is graded against a CCF-A quality
        # checklist. Load the runbook first so the critic applies the same
        # bar an ICML/NeurIPS reviewer would.
        elif stage["id"] == 4:
            desc += (
                "## REQUIRED FIRST STEP\n"
                'Before reading the producer output, call '
                'load_skill("methodology-quality-critic") and follow that '
                "runbook to grade the methodology against CCF-A criteria "
                "(formalism, algorithmic detail, statistical rigor, "
                "reproducibility, threats-to-validity depth, citation of the "
                "debate transcript). Reject confidently when any required "
                "section is shallow or missing.\n\n"
                "## METHODOLOGICAL-DEPTH GATE (D11/D12 — REJECT if shallow)\n"
                "Beyond experiment rigor, verify the methodology formalizes the "
                "METHOD itself and REJECT when:\n"
                "  - D11 Method Formalization: no real equations (objective/loss "
                "function, key quantities, derivation/complexity in LaTeX) OR no "
                "pseudocode/Algorithm block for the core procedure. Decorative "
                "notation alone fails. (A pure comparison study with no proposed "
                "method is exempt from pseudocode but must still formalize its "
                "estimands/metrics.)\n"
                "  - D12 Contribution & Novelty: no explicit, specific statement "
                "of what is NEW vs prior work (grounded in Stage 2/3 claim IDs); "
                "boilerplate 'first to…' without naming the filled gap fails.\n"
            )
        elif stage["id"] == 5:
            desc += (
                "## REQUIRED FIRST STEP\n"
                'Before reading the producer output, call '
                'load_skill("experiment-quality-critic") and follow that '
                "runbook to grade the experiment plan and coordination "
                "assignments against CCF-A criteria (operational procedure, "
                "sample-size/power math, pre-registration spec, failure-mode "
                "mitigations, reproducibility, debate citation, and a fully "
                "populated assignments table). Reject confidently when any "
                "required section is shallow or missing.\n\n"
                "## FEASIBILITY-FIRST GATE (REJECT if any fails)\n"
                "The plan must be designed in two explicitly-labelled tiers "
                "(see the Stage 5 producer contract). Verify and REJECT when:\n"
                "  - There is NO Tier-1 FEASIBILITY experiment — a cheap, "
                "minimal go/no-go check (simple/ready-made dataset, minimal "
                "config, one/few seeds, one primary metric) that runs FIRST "
                "with an explicit signal/effect-size decision rule for "
                "advancing to the full study.\n"
                "  - The Tier-2 FULL study omits any of: a field-standard "
                "dataset survey, baseline layers (sanity/naive/SOTA/"
                "ablated-self), one ablation per claimed contribution, a "
                "hyperparameter plan (locked/swept/sensitivity), or seeds "
                ">= 3 reporting the median.\n"
                "  - There is NO locked FIGURE MANIFEST — a table naming each "
                "figure/table, the claim it answers, the RESULT_JSON field it "
                "draws from, and the plot type. Without it Stage 6 cannot know "
                "what to dump and Stage 7 cannot know what to plot.\n"
                "  - The design is not BUDGET-bound: no total compute estimate, "
                "or a design that plainly cannot finish within budget (the "
                "run-a0aee5044ce2 failure — 26 models pre-registered, 2 "
                "trained, ablations BLOCKED, paper rested on n=1).\n\n"
            )
        # Stage 6 critic checks that the Auto Experiment report is grounded
        # in real run_ids (not fabricated), that every assignments-table row
        # is accounted for (executed or explicitly deferred), and that any
        # remote runs report status + cost + a log_tail excerpt.
        elif stage["id"] == 6:
            desc += (
                "## REQUIRED FIRST STEP\n"
                "Grade the Stage 6 report by asking:\n"
                "  - Is every row of stage5_assignments.md addressed?\n"
                "  - For rows tagged `experiment_runner`, is there a real "
                "run_id, a terminal status, an actual_cost, and a log_tail "
                "excerpt? Fabricated/simulated results when a runner was "
                "available are an auto-REJECT.\n"
                "  - For rows deferred to non-runner assignees, is the "
                "deferral explicit (not silent)?\n"
                "  - Does the aggregate summary tally total tasks, "
                "successes, failures, and total cost?\n"
                "Reject when the report claims success without a verifiable "
                "run_id.\n\n"
            )
        # Stage 7 critic enforces the pre-registration contract: every
        # confirmatory claim in Stage 7 must trace back to a Stage 4/5
        # pre-registered test and a real Stage 6 run_id. HARKing is an
        # explicit auto-REJECT trigger.
        elif stage["id"] == 7:
            desc += (
                "## REQUIRED FIRST STEP\n"
                'Before reading the producer output, call '
                'load_skill("result-quality-critic") and follow that '
                "runbook to grade Stage 7 against the immutable Stage 4/5 "
                "pre-registration contract and the actual Stage 6 evidence. "
                "Three auto-REJECT triggers: (a) any test in Stage 7 "
                "confirmatory section not present verbatim in Stage 4/5 "
                "(HARKing); (b) any confirmatory claim without a real "
                "Stage 6 run_id (fabrication); (c) non-English document.\n\n"
            )
        # Stage 8 critic grades the PAPER itself: structural completeness
        # (References is the historically-skipped section — run a1df5c26f6ea
        # shipped without one), citation resolution, figure embedding, and
        # statistics style. The deterministic numbers gate (#44) then
        # backstops statistic traceability after a PASS.
        elif stage["id"] == 8:
            desc += (
                "## REQUIRED FIRST STEP\n"
                "Grade the Stage 8 paper by asking:\n"
                "  - Does a References section exist with full entries, and "
                "does EVERY inline [Author, Year] citation resolve to an "
                "entry (and every entry get cited at least once in the "
                "body)? A paper with inline citations but NO References "
                "section is an auto-REJECT.\n"
                "  - Are the Stage 7 result figures (stage7_*.png, when they "
                "exist in the workspace) embedded in the Results section "
                "with captions?\n"
                "  - Statistics style: prose statistics at 3-4 significant "
                "figures (full precision in tables only); p-values as "
                "'p < .001' never 'p = 0.0'; no pipeline internals "
                "('RESULT_JSON', 'log_tail', 'the runner', run ids outside "
                "the reproducibility appendix) leaking into the prose.\n"
                "  - Spot-check at least 5 headline numbers against "
                "stage7_result_analyst.md — any mismatch is an auto-REJECT.\n\n"
            )
        # Cap producer output sent to critic (#62): full cumulative context
        # has been observed to blow Kimi-K2.6's 262K-token window (993K input
        # in late-stage runs), causing ContextWindowExceededError and the
        # critic to silently auto-pass on its stored stub. We keep the head
        # (where summaries / decisions usually live) plus the tail (where
        # spec-tables / receipts live), with an explicit elision marker so
        # the critic knows we trimmed.
        producer_excerpt = self._cap_for_critic(producer_result, stage_id=stage["id"])
        desc += f"--- Producer Output ---\n{producer_excerpt}\n"

        self.state["phase"] = "critic"
        self._save()
        self._dispatch_to_employee(critic_id, desc, f"Gate Review: Stage {stage['id']}")

    def _dispatch_result_reviewer(self, confidence: float = None):
        """Result-driven loop (#40): after Stage 7 passes its critic + data
        gate, ask a reviewer whether the experiment RESULT is scientifically
        sound — and if not, which earlier stage to loop back to. The reviewer
        gives the target; the engine executes it.

        Falls back to advancing (normal Stage 7 → 8) if no reviewer employee
        exists, so the loop is additive and never strands the pipeline.
        """
        reviewer_id = _find_employee_by_skill(CRITIC_SKILL)
        if not reviewer_id:
            logger.info("[PIPELINE] No reviewer for result-loop; advancing Stage 7 → 8")
            self._on_critic_pass(self.state["stage_results"].get("7", ""), confidence)
            return

        loops = self.state.get("result_loops", {}) or {}
        results = self.state.get("stage_results", {}) or {}
        stage6 = self._read_stage_deliverable(6, fallback=results.get("6", ""))
        stage7 = self._read_stage_deliverable(7, fallback=results.get("7", ""))
        ctx = self._cap_for_critic(
            f"--- Stage 6 experiment evidence ---\n{stage6}\n\n"
            f"--- Stage 7 result analysis ---\n{stage7}\n",
            stage_id=7,
        )
        desc = (
            "Result Review (Stage 7 → routing decision)\n\n"
            "Stage 7 passed its quality critic. Before writing the paper, judge "
            "whether the experiment's RESULT is scientifically SOUND and worth "
            "writing up — not whether the report is well-formatted.\n\n"
            "Decide ONE of:\n"
            "  - ADVANCE — the result is sound and supports a real finding; proceed to the paper.\n"
            "  - REVERT to stage 6 (code) — the numbers look like an implementation bug, "
            "not science: e.g. accuracy exactly 0% or 100% across all conditions, "
            "below-random accuracy, extraction yield < 100%, NaN/identical outputs.\n"
            "  - REVERT to stage 5 (experiment design) — the result is real but the design "
            "is too weak to conclude: n below the power analysis, confidence intervals too "
            "wide, missing baseline/control, no variance estimate (single seed).\n"
            "  - REVERT to stage 4 (methodology) — a conceptual flaw: wrong hypothesis, "
            "uncontrolled confound, the claim cannot be tested by this design.\n\n"
            f"Loop budget already used per stage: {loops} (max {MAX_RESULT_LOOPS} each — "
            "if a target is exhausted, prefer ADVANCE and note the limitation).\n\n"
            "Output EXACTLY these lines:\n"
            "Action: ADVANCE | REVERT\n"
            "Revert to stage: <4|5|6>   (only if REVERT)\n"
            "Reason: <one paragraph naming the specific metric/design flaw>\n\n"
            f"{ctx}"
        )
        self.state["phase"] = "result_review"
        self._save()
        self._dispatch_to_employee(reviewer_id, desc, "Result Review: Stage 7")

    # Verdict patterns the Stage-9 peer review uses to request changes.
    _REVISION_VERDICT_RE = re.compile(
        r"(?:verdict|recommendation|decision)\b[^\n]{0,40}?(minor|major)\s+revision",
        re.IGNORECASE,
    )

    def _review_requests_revision(self) -> bool:
        """True iff the Stage-9 peer review's own verdict asks for a
        MINOR/MAJOR REVISION (read from the canonical deliverable)."""
        review = self._read_stage_deliverable(
            9, fallback=(self.state.get("stage_results") or {}).get("9", "")
        )
        return bool(self._REVISION_VERDICT_RE.search(review or ""))

    def _dispatch_paper_revision(self) -> None:
        """Bounded revision loop (#46): give the paper-writer ONE pass to
        address the Stage-9 review, then Stage 9 re-reviews. Falls back to
        completing normally if no paper-writer exists — additive, never
        strands the pipeline."""
        writer_id = _find_employee_by_skill("paper_writer")
        if not writer_id:
            logger.info("[PIPELINE] No paper_writer for revision loop; completing as-is")
            self.state["paper_revised"] = True
            self._save()
            self._on_critic_pass(self.state["stage_results"].get("9", ""), None)
            return
        results = self.state.get("stage_results", {}) or {}
        review = self._read_stage_deliverable(9, fallback=results.get("9", ""))
        paper = self._read_stage_deliverable(8, fallback=results.get("8", ""))
        ctx = self._cap_for_critic(
            f"--- Peer review (address every actionable point) ---\n{review}\n\n"
            f"--- Current paper ---\n{paper}\n",
            stage_id=8,
        )
        desc = (
            "Paper Revision (one bounded pass)\n\n"
            "The Stage-9 peer review of your paper requested revisions. Address "
            "EVERY actionable comment, keeping all paper_writer rules in force "
            "(References resolution, statistics style, figure embeds, claim "
            "traceability — numbers may only come from the Stage 4-7 evidence). "
            "Update stage8_paper_writer.md IN PLACE in the project workspace and "
            "submit_result with a summary of what changed per comment. Do NOT "
            "argue with the review in the paper; fix or explicitly scope-note.\n\n"
            f"{ctx}"
        )
        self.state["phase"] = "paper_revision"
        self._save()
        logger.info("[PIPELINE] Stage-9 review requested revisions — dispatching paper-writer (1 bounded pass)")
        self._dispatch_to_employee(writer_id, desc, "Paper Revision (post-review)")

    # Soft cap on bytes sent to the critic as ``producer_output``. 80 KB
    # ≈ 20K tokens, comfortably under the smaller-window critic models
    # (Kimi-K2.6: 262K, MiniMax-M2.7: 128K-ish, Claude-Sonnet: 200K) while
    # leaving headroom for the system prompt + critic runbook + tool spec
    # which together routinely cost another 20-40 KB.
    _CRITIC_BUDGET_BYTES = 80_000
    _CRITIC_HEAD_BYTES = 50_000
    _CRITIC_TAIL_BYTES = 25_000

    @classmethod
    def _cap_for_critic(cls, producer_result: str, stage_id: int) -> str:
        """Trim ``producer_result`` to fit the critic's context budget.

        Strategy: keep head (decisions, summaries) + tail (tables, receipts);
        elide the middle with an explicit marker naming how many bytes were
        dropped. Stage 6's runner-report and Stage 8's paper are the typical
        offenders past the budget."""
        if not producer_result or len(producer_result) <= cls._CRITIC_BUDGET_BYTES:
            return producer_result
        head = producer_result[: cls._CRITIC_HEAD_BYTES]
        tail = producer_result[-cls._CRITIC_TAIL_BYTES :]
        elided = len(producer_result) - cls._CRITIC_HEAD_BYTES - cls._CRITIC_TAIL_BYTES
        logger.info(
            "[PIPELINE] Stage {} producer output trimmed for critic: {} bytes → head {} + tail {} + elided {} bytes",
            stage_id, len(producer_result), cls._CRITIC_HEAD_BYTES, cls._CRITIC_TAIL_BYTES, elided,
        )
        return (
            head
            + f"\n\n--- [ {elided:,} bytes elided from middle for critic context budget; "
            f"head {cls._CRITIC_HEAD_BYTES:,}B + tail {cls._CRITIC_TAIL_BYTES:,}B retained ] ---\n\n"
            + tail
        )

    def on_task_complete(self, employee_id: str, node_id: str, result: str):
        """Called by vessel when a pipeline-managed task completes."""
        self._record_active_phase_elapsed(self.phase)
        if self.phase in ("producer", "producer_b", "producer_b_finalize"):
            stage = self._stage_def()
            # Stub-result gate (#60 fix 2): if the producer returned a
            # placeholder like ``"Executed: bash"`` (the agent runtime's
            # fallback when the LLM produced no text content), treat it
            # as producer failure and retry with explicit feedback — do
            # NOT store as the stage deliverable, where the critic would
            # see only tool names and (under the old default-PASS parser)
            # silently advance. Closes #60 fix #2 / #63 fix #4.
            is_stub = self._is_stub_result(result)
            # Stage 6b salvage: the runner often does the work (writes a full
            # stage6_experimentalist.md with terminal runs) but its FINAL turn
            # returns an empty "Executed: bash" stub (run 9dd6160ea99b: a 7.6KB
            # report with two `succeeded` runs, yet submit_result was a stub).
            # Rerouting to 6a then discards a finished experiment. If the
            # on-disk report has data-bearing runs, treat the report as the
            # result and fall through to the normal producer_b handling
            # (mirrors the critic's on-disk gate_review fallback).
            if is_stub and self.phase == "producer_b":
                ondisk = self._read_stage_deliverable(6, fallback="")
                if ondisk and self._data_gate_stage6(ondisk)[0]:
                    logger.warning(
                        "[PIPELINE] Stage 6b submit_result was a stub, but the on-disk "
                        "report has data-bearing runs — using the report instead of "
                        "rerouting to 6a."
                    )
                    result = ondisk
                    is_stub = False
            # Stage 6a salvage: a 6a stub that nonetheless committed the
            # adaptation should NOT re-run full 6a (which just re-stubs). Fall
            # through to the hard-gate, which synthesizes the missing receipt
            # from git state and proceeds to 6b (run c49e8a914f5a).
            if is_stub and self.phase == "producer" and self.current_stage == 6:
                if self._synthesize_stage6_receipt():
                    logger.warning(
                        "[PIPELINE] Stage 6a stub but adaptation is committed — "
                        "synthesized the receipt; proceeding via the hard-gate."
                    )
                    is_stub = False
            if is_stub:
                feedback = (
                    f"Your submit_result was a stub: {result.strip()[:200]!r}. "
                    "This happens when the agent runtime falls back to summarising tool names "
                    "because your final response had no text content. You must produce a "
                    "non-trivial deliverable (write the actual file, then submit_result with a "
                    "summary referencing it). Re-run the full task; do not stop at tool calls."
                )
                retries = self.state.get("retries", 0)
                if retries < MAX_RETRIES:
                    self.state["retries"] = retries + 1
                    self._save()
                    logger.warning(
                        "[PIPELINE] Stage {} {} produced a stub ({} chars) — retry {}/{}",
                        stage["id"], self.phase, len(result or ""), retries + 1, MAX_RETRIES,
                    )
                    self._emit_stage_event("stage_failed", stage["id"])
                    if self.phase == "producer_b_finalize":
                        # Stub during finalize: runs are still terminal, we
                        # just need a written report. Re-dispatch the same
                        # finalize task (with feedback prepended), NOT the
                        # initial submit-and-run path — otherwise the
                        # runner would try to re-submit completed runs.
                        self._dispatch_producer_b_finalize()
                    elif self.phase == "producer_b":
                        # #20: a 6b-runner stub means the runner couldn't
                        # produce a usable report — almost always because the
                        # experiment code/entrypoint is broken (e.g. the
                        # runnable command in the receipt hits an argparse /
                        # import error), so the runner thrashes on re-submits
                        # and runs out of steps. Re-running the SAME runner on
                        # the SAME broken code just stubs again (observed 3× in
                        # run 3f644a5996bb → total failure). Route back to 6a to
                        # rebuild the code; 6a's completion re-dispatches a
                        # fresh 6b, so a transient runner hiccup also recovers.
                        rebuild_feedback = (
                            "Stage 6b runner produced no usable report (stub result), "
                            "which usually means the experiment could not be run cleanly — "
                            "e.g. the runnable command in stage6_implementation_receipt.md "
                            "hits an argparse/import/attribute error, or the entrypoint flags "
                            "don't match what benchmark.py actually accepts. Re-verify the "
                            "EXACT runnable entrypoint works (run it locally / dry-run the "
                            "args), fix any CLI/import mismatch, re-commit, and rewrite the "
                            "receipt with the corrected command.\n\n" + feedback
                        )
                        self._dispatch_producer(feedback=rebuild_feedback)
                    else:
                        self._dispatch_producer(feedback=feedback)
                    return
                logger.warning(
                    "[PIPELINE] Stage {} {} stub-result exhausted retries — holding for CEO",
                    stage["id"], self.phase,
                )
                self.state["phase"] = "gate"
                self._save()
                self._emit_gate_event(stage["id"], confidence=None, exhausted=True)
                return

        if self.phase == "producer":
            stage = self._stage_def()
            # Stage 6 has a 2-step producer: 6a (code_implementer) then 6b
            # (experiment_runner). The first dispatch maps to 6a; on
            # completion we hand off to 6b instead of going straight to
            # the critic. 6a's submit_result is informational only — the
            # canonical stage 6 deliverable is 6b's runner report.
            if stage["id"] == 6:
                # Hard-gate: 6a must have produced stage6_implementation_receipt.md
                # naming the runnable entrypoint, and the upstream/ patches must
                # be committed (so the runner can push a clean diff). The runbook
                # says these are mandatory, but LLMs frequently skip them after
                # writing the patch files — burning a 6a → 6b → critic cycle
                # that always ends BLOCKED. Catch it here.
                receipt_path = Path(self.project_dir) / "stage6_implementation_receipt.md"
                upstream_dir = Path(self.project_dir) / "upstream"
                # Salvage: if the adaptation is committed but the receipt is
                # missing (implementer stubbed before Phase 5), synthesize it
                # from git state rather than burning a retry that re-stubs.
                self._synthesize_stage6_receipt()
                missing = []
                if not receipt_path.exists() or receipt_path.stat().st_size < 200:
                    missing.append("stage6_implementation_receipt.md (must exist and be non-trivial)")
                if upstream_dir.exists() and (upstream_dir / ".git").exists():
                    # Check for uncommitted changes — patches should be in a commit.
                    import subprocess
                    try:
                        dirty = subprocess.run(
                            ["git", "status", "--short"],
                            cwd=str(upstream_dir), capture_output=True, text=True, timeout=10,
                        ).stdout.strip()
                        if dirty:
                            missing.append(
                                f"uncommitted patches in upstream/ (git status shows:\n{dirty[:300]}\n"
                                f"— run `cd upstream && git add -A && git commit -m 'Stage 6 adaptation'` before submit_result)"
                            )
                    except (subprocess.SubprocessError, OSError) as exc:
                        # Don't block on git failures — the receipt check is the
                        # primary gate; an uncheckable git tree at most under-reports
                        # missing-commit, not over-reports.
                        logger.debug(
                            "[PIPELINE] Stage 6a hard-gate git status probe failed: {} — skipping uncommitted-patches check",
                            exc,
                        )

                if missing:
                    feedback = (
                        "Stage 6a hard-gate FAILED. You wrote code but did not finalize Phase 5+6:\n\n"
                        + "\n".join(f"  - {m}" for m in missing)
                        + "\n\nGo back and complete: (1) commit the upstream/ patches as ONE commit, "
                        "(2) push them to remote via fast_push_code.sh (Phase 4), "
                        "(3) write stage6_implementation_receipt.md (Phase 5 template — at minimum: "
                        "pin status header, file list with line counts, runnable entrypoint command), "
                        "(4) call submit_result. Read the receipt back from disk to verify before submit. "
                        "Patches without a receipt are invisible to the 6b runner — your work is lost."
                    )
                    retries = self.state.get("retries", 0)
                    if retries < MAX_RETRIES:
                        self.state["retries"] = retries + 1
                        self._save()
                        logger.warning(
                            "[PIPELINE] Stage 6a hard-gate FAILED ({} missing) — retry {}/{}",
                            len(missing), retries + 1, MAX_RETRIES,
                        )
                        self._emit_stage_event("stage_failed", stage["id"])
                        self._dispatch_producer(feedback=feedback)
                        return
                    # Exhausted: still surface as a producer fail, hold for CEO.
                    logger.warning("[PIPELINE] Stage 6a hard-gate exhausted after {} retries", MAX_RETRIES)
                    self.state["phase"] = "gate"
                    self._save()
                    self._emit_gate_event(stage["id"], confidence=None, exhausted=True)
                    return

                self.state["stage_6a_result"] = result
                self._save()
                logger.info("[PIPELINE] Stage 6a (code impl) complete, dispatching Stage 6b (runner)")
                self._dispatch_producer_b()
                return

            # Stage 3 (literature-conflict-graph) deliverable is the FILE the
            # aigraph tool writes (``# Selected Hypotheses`` report). The agent's
            # chat result is often just a summary, which the UI can't render as
            # a conflict graph — so prefer the file content as the stage result
            # (the critic reads the file too, keeping them consistent).
            if stage["id"] == 3:
                deliverable = Path(self.project_dir) / f"stage3_{stage['skill']}.md"
                try:
                    if deliverable.exists():
                        file_text = deliverable.read_text(encoding="utf-8").strip()
                        if "# Selected Hypotheses" in file_text:
                            result = file_text
                except Exception as e:
                    logger.debug("[PIPELINE] Stage 3 file-content fallback failed: {}", e)
                # Augmentation backstop: if the producer dropped the injected
                # grounding, graft the deterministic aigraph report back in (#130).
                result = self._ensure_stage3_grounded(deliverable, result)

            # Producer finished → store result, dispatch critic
            self.state["stage_results"][str(stage["id"])] = result
            self._save()
            logger.info("[PIPELINE] Stage {} producer complete, dispatching critic", stage["id"])
            self._emit_stage_event("stage_reviewing", stage["id"])
            self._dispatch_critic(result)

        elif self.phase == "producer_b":
            # Stage 6b runner finished. Two paths from here:
            #
            # Fast path (smoke / short experiments): the runner polled all
            # submitted runs to terminal inside its own task budget. The
            # report has no `status: still_running` rows → store result,
            # dispatch critic immediately (matches pre-#93 behavior).
            #
            # Long-running path (#93): the runner submitted real
            # experiments that exceed its agent-task time budget. The
            # report carries `status: still_running` for one or more
            # run_ids. Park in ``producer_b_waiting``; the
            # ``run_tracker`` cron polls infra every 30 s and triggers
            # ``on_runs_all_terminal`` when every pending run reaches
            # terminal. Engine then re-dispatches the runner to write
            # the FINAL report from the now-terminal run data.
            stage = self._stage_def()
            self.state["stage_results"][str(stage["id"])] = result
            # Parse run statuses from the CANONICAL on-disk report first
            # (stage6_experimentalist.md), not the submit_result summary —
            # the summary is lossy prose (run 1a255f1aaf3d wrote the runs
            # as a Markdown table the line-pair parser can't read, so a
            # still-running pilot went to the critic instead of parking).
            # Same principle as the #27 gate fix. Fall back to the summary
            # only when the file yields nothing.
            report_text = self._read_stage_deliverable(stage["id"], fallback=result)
            runs = self._parse_runner_report_runs(report_text)
            if not runs and report_text is not result:
                runs = self._parse_runner_report_runs(result)
            pending = self._pending_run_ids_from(runs)
            # R13-1 plausibility scope: a real experiment pends a handful of
            # runs; dozens means an account-wide listing leaked into the
            # parse (e.g. a tool-echo of /api/list_runs). Scope to the runs
            # the tracker already attributes to THIS project (its map is
            # keyed by the omc/<pid>/<iter> run_command needle).
            if len(pending) > 8:
                known = self.state.get("stage_6_runs") or {}
                scoped = [r for r in pending if r in known]
                if not scoped:
                    scoped = self._pending_run_ids_from([
                        (rid, str((info or {}).get("status", "unknown")))
                        for rid, info in known.items()
                    ])
                logger.warning(
                    "[PIPELINE] Stage 6b parse yielded {} pending run_ids — "
                    "implausible; scoped to {} tracker-known project run(s)",
                    len(pending), len(scoped),
                )
                pending = scoped
            if pending:
                from datetime import datetime, timezone
                self.state["phase"] = "producer_b_waiting"
                self.state["pending_run_ids"] = pending
                self.state["pending_waiting_started_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                logger.info(
                    "[PIPELINE] Stage 6b parked in producer_b_waiting: {} run_ids still active "
                    "({}). run_tracker will fire on_runs_all_terminal when all reach terminal.",
                    len(pending), ", ".join(pending[:5]) + ("..." if len(pending) > 5 else ""),
                )
                self._emit_stage_event("stage_waiting", stage["id"])
                # Edge case: run_tracker already updated stage_6_runs before
                # we got here (e.g. the runner's report was stale by the time
                # the engine processed it). Check now so we don't wait
                # uselessly for a poll cycle that has nothing left to do.
                if self._all_pending_terminal(pending, self.state.get("stage_6_runs", {}) or {}):
                    self.on_runs_all_terminal()
                return
            self._save()
            logger.info("[PIPELINE] Stage 6b (runner) complete, dispatching critic")
            self._emit_stage_event("stage_reviewing", stage["id"])
            self._dispatch_critic(result)

        elif self.phase == "producer_b_finalize":
            # Finalization re-dispatch (after producer_b_waiting). The
            # runner has re-read the now-terminal run metrics and produced
            # the FINAL stage6_experimentalist.md. Clean up the pending-run
            # bookkeeping and proceed to critic.
            stage = self._stage_def()
            self.state["stage_results"][str(stage["id"])] = result
            self.state.pop("pending_run_ids", None)
            self.state.pop("pending_waiting_started_at", None)
            self._save()
            logger.info("[PIPELINE] Stage 6b finalize complete, dispatching critic")
            self._emit_stage_event("stage_reviewing", stage["id"])
            self._dispatch_critic(result)

        elif self.phase == "result_review":
            # Result-driven loop (#40). The result-reviewer judged whether the
            # Stage 7 RESULT (not its report) is scientifically sound and, if
            # not, named the stage to loop back to (4 methodology / 5 design /
            # 6 code). Route on its verdict; the LLM gives the target, the
            # engine executes it (reusing revert_to_stage).
            self.state["result_review_result"] = result
            action, target, reason = self._parse_result_route(result)
            if action == "revert" and target is not None:
                loops = self.state.setdefault("result_loops", {})
                used = int(loops.get(str(target), 0))
                if used < MAX_RESULT_LOOPS:
                    loops[str(target)] = used + 1
                    self._save()
                    logger.warning(
                        "[PIPELINE] Result-loop: reverting to stage {} ({}/{}). Reason: {}",
                        target, used + 1, MAX_RESULT_LOOPS, reason[:200],
                    )
                    self._schedule_result_revert(target, reason)
                    return
                logger.warning(
                    "[PIPELINE] Result-loop budget for stage {} exhausted ({}x) — "
                    "proceeding to paper with documented limitation. Reason: {}",
                    target, MAX_RESULT_LOOPS, reason[:200],
                )
            # Feasibility-first promotion (C): if this was the cheap Tier-1
            # feasibility study and the result-reviewer says the signal is
            # sound (ADVANCE), do NOT go to the paper — promote to the full
            # study. Flip research_phase, return to Stage 5 to design the
            # rigorous experiment with the feasibility results as context.
            if self.state.get("research_phase") == "feasibility":
                self._promote_to_full_study(result)
                return
            # ADVANCE, or revert-budget exhausted → proceed to Stage 8 as if
            # Stage 7 passed normally (the analysis is already stored).
            stage7_result = (self.state.get("stage_results") or {}).get("7", result)
            self._on_critic_pass(stage7_result, confidence=None)

        elif self.phase == "paper_revision":
            # Bounded revision loop (#46): the paper-writer revised the paper
            # per the Stage-9 review. Store the revision as the new Stage 8
            # deliverable, consume the one-revision budget, and send Stage 9
            # back for a fresh review of the revised paper.
            self.state["stage_results"]["8"] = result
            self.state["paper_revised"] = True
            self.state["phase"] = "producer"
            self._save()
            logger.info(
                "[PIPELINE] Paper revision complete — re-dispatching Stage 9 "
                "to review the revised paper",
            )
            self._emit_stage_event("stage_started", 9)
            self._dispatch_producer()

        elif self.phase == "critic":
            # Critic finished → parse decision
            self.state["critic_result"] = result
            self._save()
            is_pass = self._parse_critic_pass(result)
            confidence = self._parse_confidence(result)

            stage = self._stage_def()

            # #27 Hard data gate: a deterministic check that runs even when
            # the critic voted PASS, and overrides it. The critic grades
            # report *quality*; this gate verifies *real data exists* in the
            # upstream artifact. The LLM cannot vote its way past it. Closes
            # the #96/#94 failure mode where every critic accepted an honest
            # "I couldn't run / NOT TESTED" report and the empty stage
            # advanced to an INCONCLUSIVE paper.
            if is_pass:
                # Gate the PRODUCER's deliverable, not the critic's verdict
                # text — the data lives in the stage's own report. Prefer the
                # on-disk stage{N}_{skill}.md (the canonical evidence) over the
                # stored submit_result, which is often a prose summary that
                # points at the file without carrying parseable run/hypothesis
                # lines (regression caught by the 4→9 e2e).
                stored = (self.state.get("stage_results") or {}).get(str(stage["id"]), "")
                producer_deliverable = self._read_stage_deliverable(stage["id"], fallback=stored)
                if stage["id"] == 9:
                    # Stage 9 is terminal: it can't be blocked/retried like
                    # 6/7/8. When the pipeline has no real data, clamp an
                    # acceptance-class verdict to MAJOR REVISION instead of
                    # letting an unearned ACCEPT stand (#94). The stage still
                    # "passes" (the review itself is valid) — only the verdict
                    # is bounded by the existence of results.
                    if not self._pipeline_has_real_data():
                        clamped, changed = self._clamp_review_verdict(producer_deliverable)
                        if changed:
                            logger.warning(
                                "[PIPELINE] Stage 9 verdict clamped to MAJOR REVISION "
                                "— pipeline has no experimental data (#94)"
                            )
                            self.state["stage_results"]["9"] = clamped
                            self._save()
                else:
                    data_ok, gate_reason = self._stage_data_gate(stage["id"], producer_deliverable)
                    if not data_ok:
                        logger.warning(
                            "[PIPELINE] Stage {} data-gate FAIL despite critic PASS: {}",
                            stage["id"], gate_reason,
                        )
                        is_pass = False
                        result = (
                            f"DATA_GATE_FAIL: {gate_reason}\n\n"
                            "The deliverable was graded PASS for quality but contains no "
                            "real experimental data, so the stage cannot advance. Produce "
                            "a deliverable backed by actual run data.\n\n"
                            f"--- original critic verdict ---\n{result}"
                        )

            # Emit critic result to frontend so it shows in the stage card
            self._emit_critic_result(stage["id"], result, is_pass, confidence)

            if is_pass:
                self._record_stage_memory(
                    stage,
                    producer_result=self.state["stage_results"].get(str(stage["id"]), ""),
                    critic_result=result,
                    passed=True,
                    confidence=confidence,
                    outcome="critic_pass",
                    producer_elapsed_seconds=self.state.get("attempt_timing", {}).get("producer_elapsed_seconds"),
                    critic_elapsed_seconds=self.state.get("attempt_timing", {}).get("critic_elapsed_seconds"),
                )
                self._save()
                logger.info("[PIPELINE] Stage {} PASSED (confidence={})", stage["id"], confidence)
                # Result-driven loop (#40): when Stage 7 (Result Analysis)
                # passes its critic + data gate, don't advance straight to the
                # paper. First ask the result-reviewer whether the RESULT is
                # scientifically sound; it may route back to 4/5/6. Only Stage
                # 7 triggers this (it's where real metrics first exist).
                if stage["id"] == 7:
                    self._dispatch_result_reviewer(confidence)
                # Bounded revision loop (#46): when the Stage-9 peer review
                # itself asks for MINOR/MAJOR REVISION, the paper-writer gets
                # exactly ONE revision pass and Stage 9 re-reviews — review
                # comments are consumed, not archived. ACCEPT (or an
                # already-spent budget) completes as before.
                elif (
                    stage["id"] == 9
                    and not self.state.get("paper_revised")
                    # A clamped no-data MAJOR REVISION (#94) is not fixable by
                    # rewriting prose — revision only makes sense for papers
                    # about real results.
                    and self._pipeline_has_real_data()
                    and self._review_requests_revision()
                ):
                    self._dispatch_paper_revision()
                else:
                    self._on_critic_pass(self.state["stage_results"].get(str(stage["id"]), ""), confidence)
            else:
                retries = self.state.get("retries", 0)
                if retries < MAX_RETRIES:
                    self._record_stage_memory(
                        stage,
                        producer_result=self.state["stage_results"].get(str(stage["id"]), ""),
                        critic_result=result,
                        passed=False,
                        confidence=confidence,
                        outcome="critic_reject_retry",
                        producer_elapsed_seconds=self.state.get("attempt_timing", {}).get("producer_elapsed_seconds"),
                        critic_elapsed_seconds=self.state.get("attempt_timing", {}).get("critic_elapsed_seconds"),
                    )
                    self.state["retries"] = retries + 1
                    self._save()
                    logger.info("[PIPELINE] Stage {} REJECTED (retry {}/{})", stage["id"], retries + 1, MAX_RETRIES)
                    self._emit_stage_event("stage_failed", stage["id"], confidence=confidence)
                    self._reset_attempt_timing()
                    self._dispatch_producer(feedback=result)
                else:
                    self._record_stage_memory(
                        stage,
                        producer_result=self.state["stage_results"].get(str(stage["id"]), ""),
                        critic_result=result,
                        passed=False,
                        confidence=confidence,
                        outcome="critic_reject_exhausted",
                        producer_elapsed_seconds=self.state.get("attempt_timing", {}).get("producer_elapsed_seconds"),
                        critic_elapsed_seconds=self.state.get("attempt_timing", {}).get("critic_elapsed_seconds"),
                    )
                    logger.warning("[PIPELINE] Stage {} exhausted retries, holding for CEO", stage["id"])
                    self.state["phase"] = "gate"
                    self._save()
                    self._emit_gate_event(stage["id"], confidence, exhausted=True)

    def on_task_failed(self, employee_id: str, node_id: str, result: str):
        """Called by vessel when a pipeline-managed task fails (the agent
        threw, timed out, or otherwise produced no usable output).

        Branches on the current phase:

        * ``producer`` failure → retry the producer with the failure
          context as feedback (up to ``MAX_RETRIES``), then open the CEO
          gate. Symmetric with a critic REJECT.

        * ``critic`` failure → auto-pass the stage using the already-stored
          producer output. Mirrors the "no critic employee found" branch
          in ``_dispatch_critic``. Re-running the producer would discard
          its existing output and double-bill tokens for a problem that
          isn't the producer's.

        Without this hook a failed pipeline node would fall through to
        vessel.py's legacy completion check, which would mistake the
        first-completed stage anchor for an EA orchestrator and declare
        the project complete.
        """
        self._record_active_phase_elapsed(self.phase)
        stage = self._stage_def()
        current_phase = self.phase

        if current_phase == "paper_revision":
            # Bounded revision loop (#46): a failed revision task must never
            # strand the pipeline — consume the budget and complete with the
            # ORIGINAL paper (the review feedback is preserved on disk).
            logger.warning(
                "[PIPELINE] Paper-revision task failed ({}) — completing with the "
                "original paper", (result or "")[:120],
            )
            self.state["paper_revised"] = True
            self._save()
            self._on_critic_pass(self.state.get("stage_results", {}).get("9", ""), None)
            return

        if current_phase == "critic":
            stored = self.state.get("stage_results", {}).get(str(stage["id"]), "")
            logger.warning(
                "[PIPELINE] Stage {} critic FAILED — auto-passing on stored producer output (len={})",
                stage["id"], len(stored),
            )
            self._on_critic_pass(stored, confidence=None)
            return

        if current_phase not in ("producer", "producer_b", "producer_b_finalize"):
            # Should not happen — gate/done/failed/waiting phases mean no
            # task is in flight (the waiter is driven by run_tracker, not
            # by a dispatched LLM task).
            logger.warning(
                "[PIPELINE] on_task_failed called in unexpected phase {} (stage {}); ignoring",
                current_phase, stage["id"],
            )
            return

        truncated = (result or "(no output)").strip()[:600]
        # Differentiate 6a vs 6b vs 6b-finalize failures so the retry
        # feedback + re-dispatch target the right sub-phase.
        if current_phase == "producer_b_finalize":
            failure_feedback = (
                f"Stage 6b FINAL REPORT task failed without producing a deliverable. "
                f"The submitted runs already reached terminal status — just "
                f"re-fetch each run_id's evidence via fast_query_exp_status.sh "
                f"and write the report. Failure context:\n{truncated}"
            )
        elif current_phase == "producer_b":
            failure_feedback = (
                f"Stage 6b runner failed without producing a deliverable. "
                f"Failure context:\n{truncated}"
            )
        else:
            failure_feedback = (
                f"Producer for Stage {stage['id']} ({stage['name']}) failed without producing a deliverable. "
                f"Failure context:\n{truncated}"
            )
        retries = self.state.get("retries", 0)
        if retries < MAX_RETRIES:
            self.state["retries"] = retries + 1
            self.state["phase"] = current_phase
            self._save()
            logger.warning(
                "[PIPELINE] Stage {} {} FAILED (retry {}/{}) — re-dispatching",
                stage["id"], current_phase, retries + 1, MAX_RETRIES,
            )
            self._emit_stage_event("stage_failed", stage["id"])
            if current_phase == "producer_b_finalize":
                # Finalize failure → re-dispatch the same finalize task,
                # NOT the initial submit-and-run task. Runs are terminal;
                # we just need the report.
                self._dispatch_producer_b_finalize()
            elif current_phase == "producer_b":
                self._dispatch_producer_b(feedback=failure_feedback)
            else:
                self._dispatch_producer(feedback=failure_feedback)
        else:
            logger.error(
                "[PIPELINE] Stage {} exhausted retries after producer failure — holding for CEO",
                stage["id"],
            )
            self.state["phase"] = "gate"
            self._save()
            self._emit_gate_event(stage["id"], confidence=None, exhausted=True)

    def _on_critic_pass(self, result: str, confidence: float = None):
        """Critic passed → hold for CEO gate."""
        stage = self._stage_def()
        self.state["phase"] = "gate"
        self._save()
        # Commit the workspace as the canonical checkpoint for this stage
        # before opening the gate. This is the quiescent moment: producer
        # and critic are both finished, nothing else is writing files.
        # The tag ``<iteration>/stage-<N>`` lets the user later revert
        # here to redo this stage with new instructions.
        from onemancompany.core import project_repo
        try:
            project_repo.commit_stage(
                self.project_dir,
                iteration=self._iteration_id(),
                stage=stage["id"],
                stage_name=stage["name"],
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "[PIPELINE] commit_stage failed at stage {}: {}", stage["id"], exc,
            )
        self._emit_stage_event("stage_complete", stage["id"], confidence=confidence)
        self._emit_gate_event(stage["id"], confidence)

    async def _cancel_active_task_and_wait(self, *, timeout: float = 5.0) -> None:
        """Cancel the engine's active producer/critic task and wait for it
        to actually terminate.

        ``asyncio.Task.cancel()`` is non-blocking — it schedules cancellation;
        the task only stops on its next ``await``. If we returned right after
        calling cancel and proceeded to ``git reset --hard``, the cancelled
        producer could still land a ``write()`` between our reset and the
        checkout. We grab the task handle *before* calling
        ``abort_employee`` (which pops it from ``_running_tasks``), then
        ``await`` it with a timeout so the cancellation has actually
        propagated through ``_run_task``'s finally block.
        """
        if self.phase in ("gate", "done", "failed"):
            return
        emp_id = self.state.get("active_employee_id")
        if not emp_id:
            return

        from onemancompany.core.agent_loop import employee_manager

        # Capture the task handle before abort_employee pops it.
        running = employee_manager._running_tasks.get(emp_id)
        try:
            cancelled = employee_manager.abort_employee(emp_id)
            logger.info(
                "[PIPELINE] Cancelled {} active task(s) for employee {} before revert",
                cancelled, emp_id,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "[PIPELINE] abort_employee({}) failed during revert: {}",
                emp_id, exc,
            )

        # Wait for the producer's await chain to unwind. We swallow
        # CancelledError (expected) and the task's own task-side
        # exceptions (they're already logged by _run_task's finally
        # block) — what matters here is that the task has finished, so
        # no further file writes can land.
        if running is not None and not running.done():
            import asyncio
            try:
                await asyncio.wait_for(asyncio.shield(running), timeout=timeout)
            except asyncio.CancelledError:
                # Expected: that's what abort_employee() asked for. Task has
                # finished unwinding, which is the post-condition we needed.
                logger.debug("[PIPELINE] Cancelled task for {} terminated cleanly", emp_id)
            except asyncio.TimeoutError:
                logger.warning(
                    "[PIPELINE] Producer for {} did not stop within {}s; "
                    "proceeding with revert anyway", emp_id, timeout,
                )
            except Exception as exc:
                logger.debug(
                    "[PIPELINE] Cancelled task raised {} during teardown (ignored)", exc,
                )

        self.state["active_node_id"] = None
        self.state["active_employee_id"] = None

    def _promote_to_full_study(self, feasibility_result: str) -> None:
        """Feasibility-first promotion (C). The Tier-1 feasibility study
        produced a positive go/no-go signal, so promote to the full study:
        flip research_phase to "full", return to Stage 5 to design the
        rigorous experiment (baselines, ablations, full benchmark), and carry
        the feasibility findings forward as prior context. Stage 6/7 then
        re-run on the full design before the pipeline reaches the paper.
        """
        self.state["research_phase"] = "full"
        self.state["feasibility_result"] = feasibility_result
        # Carry the feasibility findings into the full-study design.
        prior = self.state.get("prior_context", "") or ""
        self.state["prior_context"] = (
            prior + "\n\n## FEASIBILITY RESULTS (Tier 1 — go signal confirmed)\n"
            "The cheap feasibility study below showed signal; you are now "
            "designing the FULL rigorous study (field-standard benchmark, "
            "baseline layers, one ablation per contribution, seeds >= 3, "
            "figure manifest). Use these findings to size the full study "
            "(effect size -> power -> sample size) and to choose baselines:\n\n"
            + (feasibility_result or "")
        ).strip()
        # Reset the per-target revert budget for the fresh full run.
        self.state["result_loops"] = {}
        self.state["current_stage"] = 5
        self.state["retries"] = 0
        self.state["critic_result"] = None
        self.state["phase"] = "producer"
        self._save()
        logger.info(
            "[PIPELINE] Feasibility signal confirmed (project={}) — promoting "
            "to FULL study, returning to Stage 5 for rigorous design",
            self.project_id,
        )
        self._emit_stage_event("stage_started", 5)
        self._dispatch_producer()

    # ------------------------------------------------------------------
    # Public API — revert to a previous stage with new instructions
    # ------------------------------------------------------------------

    def _schedule_result_revert(self, stage: int, reason: str) -> None:
        """Schedule an async ``revert_to_stage`` from the (sync) result-review
        completion handler. The reviewer's reason becomes the producer's
        instructions so the re-run targets exactly what was unsound.

        Mirrors how ``_emit_gate_event`` schedules ``_auto_approve_gate`` —
        we're inside a sync vessel callback with a running loop.
        """
        import asyncio
        instructions = (
            f"[Result-driven loop] The experiment result was judged unsound; "
            f"this stage is being re-run to fix it. Reviewer's reason:\n{reason}"
        )

        async def _do_revert():
            try:
                # R10-1 (run 59429240245f): the workspace legitimately holds
                # post-stage artifacts (runner report, routing decision) that
                # were never stage-committed; checkout_branch_from_stage's
                # DirtyWorkspaceError guard would block the revert. Checkpoint
                # them first — also preserves the forensics in git history.
                try:
                    from onemancompany.core import project_repo
                    project_repo.commit_pending(
                        self.project_dir,
                        message=f"Result-loop checkpoint before revert to stage {stage}",
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "[PIPELINE] pre-revert checkpoint failed: {} — attempting revert anyway",
                        exc,
                    )
                await self.revert_to_stage(stage=stage, instructions=instructions)
            except Exception as exc:  # noqa: BLE001 — never crash the callback
                logger.warning(
                    "[PIPELINE] result-loop revert_to_stage({}) failed: {} — "
                    "advancing instead", stage, exc,
                )
                self._on_critic_pass(
                    (self.state.get("stage_results") or {}).get("7", ""), confidence=None,
                )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_revert())
        except RuntimeError:
            # No running loop (sync test / off-loop) — fall back to advancing
            # so we never strand the pipeline. Tests stub this method out.
            logger.debug("[PIPELINE] no running loop for result revert; advancing")
            self._on_critic_pass(
                (self.state.get("stage_results") or {}).get("7", ""), confidence=None,
            )

    async def revert_to_stage(
        self, *, stage: int, instructions: str, branch_name: str | None = None,
    ) -> str:
        """Create a feature branch rooted at stage ``stage - 1``'s tag,
        switch the workspace to it, queue the user's instructions for
        the stage's producer, and re-dispatch.

        Returns the (possibly auto-generated) branch name so callers can
        surface it to the user.

        Raises ``ValueError`` for out-of-range stage numbers.

        Behaviour when a later stage is mid-flight (phase ∈ {producer,
        critic}): the active task is cancelled and any uncommitted
        workspace changes from that cancelled task are discarded before
        the checkout. Callers don't need to wait for a gate — the engine
        handles the cleanup so reverts work at any point in the pipeline.

        The semantics deliberately keep the *current branch's* tags and
        commits intact — reverting forks; it does not destroy history.
        """
        end = self.state.get("end_stage", 9)
        if not (1 <= stage <= end):
            raise ValueError(
                f"revert_to_stage: stage must be in [1, {end}]; got {stage}"
            )

        instructions = (instructions or "").strip()

        # Validate dispatchability BEFORE touching git or cancelling
        # tasks. The whole revert operation should be either fully
        # successful or fully a no-op; otherwise we leave the user on a
        # new branch with corrupt state and no in-flight task.
        # ``stage_assignments`` honours user overrides; otherwise the
        # engine resolves by skill from ``employee_configs``.
        stage_def = self._stage_def(stage)
        assignments = self.state.get("stage_assignments", {})
        assigned = assignments.get(str(stage_def["id"]))
        employee_id = assigned if assigned else _find_employee_by_skill(stage_def["skill"])
        if not employee_id:
            raise RevertNotAllowedError(
                f"Cannot revert to stage {stage}: no employee with skill "
                f"'{stage_def['skill']}' is available to run the producer."
            )

        # Cancel any in-flight producer/critic task before we touch git
        # and wait for it to actually stop (cancel() alone is non-blocking).
        # The cancelled task may have written partial output to the
        # workspace; ``discard_uncommitted_changes`` below scrubs that so
        # ``checkout_branch_from_stage``'s DirtyWorkspaceError guard
        # passes.
        was_mid_flight = self.phase in (
            "producer", "producer_b", "producer_b_waiting", "producer_b_finalize", "critic",
        )
        if was_mid_flight:
            await self._cancel_active_task_and_wait()

        from onemancompany.core import project_repo
        # Only scrub the workspace when we just cancelled a task. At
        # gate/done the workspace should already be clean (the previous
        # stage's commit_stage left it that way), and an unconditional
        # ``git reset --hard`` here would silently destroy any manual
        # edits the user made between gates. Let
        # ``checkout_branch_from_stage`` raise ``DirtyWorkspaceError``
        # loudly in that case.
        if was_mid_flight:
            project_repo.discard_uncommitted_changes(self.project_dir)
        new_branch = project_repo.checkout_branch_from_stage(
            self.project_dir,
            iteration=self._iteration_id(),
            stage=stage,
            branch_name=branch_name,
        )

        # The checkout flipped pipeline_state.yaml back to its previous
        # snapshot. Reload from disk; refuse to proceed if the snapshot
        # somehow lacks a state file (would silently retain the abandoned
        # branch's state otherwise — corrupting the new branch on the
        # next ``_save``).
        loaded = _load_state(self.project_dir)
        if not loaded:
            raise RevertNotAllowedError(
                f"Reverted to branch '{new_branch}' but the checkout did "
                f"not restore a pipeline_state.yaml. Workspace may be "
                f"corrupt — investigate before retrying."
            )
        self.state = loaded
        self.state["current_stage"] = stage
        self.state["phase"] = "producer"
        self.state["retries"] = 0
        self.state["critic_result"] = None
        self.state["active_node_id"] = None
        self.state["active_employee_id"] = None
        # Drop any stage results at or beyond the revert point — they
        # belong to the abandoned branch and would mislead the producer's
        # context-building.
        sr = self.state.get("stage_results", {})
        self.state["stage_results"] = {
            sid: result for sid, result in sr.items() if int(sid) < stage
        }
        # Queue the user's instructions; _dispatch_producer consumes them
        # via _consume_pending_feedback and prepends them to the prompt.
        if instructions:
            self.state["pending_user_feedback"] = instructions
        self._save()

        logger.info(
            "[PIPELINE] Reverted to stage {} on branch '{}' with {} chars of instructions",
            stage, new_branch, len(instructions),
        )
        self._dispatch_producer()
        return new_branch

    # Keywords that trigger a *full re-dispatch* of the current stage from
    # scratch (retries=0). Kept narrow on purpose: every CEO chat at the
    # gate flows through this matcher (since task_followup now routes
    # gate-phase feedback here), so any false positive silently undoes the
    # stage and confuses the user. Single-character triggers like "再" or
    # ambiguous edits like "修改" are excluded — they appear in legitimate
    # advance-with-comment chats ("再补充一点", "可以修改一下措辞") that
    # should NOT trigger a redo.
    _REVISION_KEYWORDS = (
        "REVISION", "REVISE", "RE-RUN", "REDO",
        "重新",  # "重新跑", "重新写", "重新做"
        "重做", "重写", "重跑",
        "再来一遍", "再做一遍", "再写一遍", "再跑一遍",
    )

    def on_ceo_approve(self, feedback: str = ""):
        """CEO approved the current stage. Advance or re-run."""
        # Idempotency guard (#157): only act when the pipeline is actually
        # waiting at a gate. A stale / duplicate approve (retried HTTP call,
        # race between manual and auto-approver, or a breakpoint resume for a
        # stage that already advanced) must not advance a stage that is already
        # executing — doing so skips the running producer entirely.
        if self.phase != "gate":
            logger.info(
                "[PIPELINE] on_ceo_approve ignored — not at gate "
                "(phase={}, stage={})", self.phase, self.current_stage,
            )
            return

        stage = self._stage_def()

        if feedback and any(kw in feedback.upper() for kw in self._REVISION_KEYWORDS):
            # CEO wants revision
            logger.info("[PIPELINE] CEO requested revision for stage {}", stage["id"])
            self._apply_ceo_memory_feedback(stage, feedback, approved=False)
            self.state["retries"] = 0
            self._dispatch_producer(feedback=feedback)
            return

        self._apply_ceo_memory_feedback(stage, feedback, approved=True)

        # Advance to next stage
        end = self.state.get("end_stage", 9)
        if self.current_stage < end:
            next_stage = self.current_stage + 1
            self.state["current_stage"] = next_stage
            self.state["retries"] = 0
            self.state["critic_result"] = None
            self._save()
            logger.info("[PIPELINE] Advancing to stage {}", next_stage)
            self._dispatch_producer()
        else:
            self.state["phase"] = "done"
            self._save()
            logger.info("[PIPELINE] Pipeline complete!")
            self._emit_pipeline_complete()

    # ------------------------------------------------------------------
    # Critic result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stub_result(result: str) -> bool:
        """A producer/critic ``result`` is a stub when the agent runtime fell
        back to synthesising a summary from tool names (no real text content).
        These show up as ``"Executed: bash"`` / ``"Executed tools: write, read"``
        — see ``agents/base.py:_synthesize_fallback``. Stubs that pass to the
        critic look like work but contain no analysis, and the critic's own
        stub return then defaults to PASS by the old parse logic — closing
        the silent-empty-stage loop #60 / #63 describe.

        The ``Executed: `` / ``Executed tools: `` prefix IS the runtime's
        fallback signature — flag it REGARDLESS of length. R13-1 (run
        df3fd56612e5): an 852KB tool-echo (a full /api/list_runs dump)
        sailed past the old <300-char heuristic, the waiter's fallback
        parse swallowed 100 account-wide run_ids as pending, and the
        pipeline parked on 99 ghosts."""
        if not result:
            return True
        stripped = result.strip()
        if stripped.startswith("Executed: ") or stripped.startswith("Executed tools: "):
            return True
        return False

    # ------------------------------------------------------------------
    # Stage 6b long-running waiter (#93, #97)
    # ------------------------------------------------------------------

    # Status tokens emitted by the experiment-execution-runbook. Terminal
    # means the run has a final outcome and the engine can advance to the
    # critic; pending means the run is still active on infra and the engine
    # should park itself in ``producer_b_waiting`` until run_tracker flips
    # all pending runs to terminal.
    _RUN_TERMINAL_STATUSES = ("succeeded", "failed", "rejected", "blocked", "cancelled")
    _RUN_PENDING_STATUSES = ("running", "still_running", "queued", "pending", "submitted")

    _RUN_ID_RE = re.compile(
        # Tolerates Markdown decorations: ``run_id:``, ``**run_id**:``,
        # ``- **run_id**: `run_x```, etc. The ``\W*?`` between ``id`` and
        # the separator absorbs trailing ``**``/``\`` markers; the ``[`'\"*]*``
        # after it absorbs leading quote / backtick wrappers.
        # The separator class includes ``|`` so the two-cell Markdown table
        # form ``| run_id | `run_x` |`` also matches (run fa50cb183b3c wrote
        # the report this way; the colon-only regex parsed ZERO runs and a
        # clean experiment died at the data gate).
        r"run[_\s-]*id\b\W*?[:=|]\s*[`'\"*]*([A-Za-z][A-Za-z0-9_.\-]{4,})",
        re.IGNORECASE,
    )
    _STATUS_LINE_RE = re.compile(
        # Tolerates plain ``status:``, decorated ``- **status**:``, AND the
        # ``| status | `succeeded` |`` table-cell form (``|`` separator).
        # ``\W*?`` (lazy non-word chars) absorbs optional Markdown bold/italic
        # markers without requiring them.
        r"^\s*(?:[-*]\s*)?\W*?status\b\W*?[:=|]\s*[`'\"*]*([a-z_\-]+)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Info-strings whose fenced contents are known to embed a synthetic
    # ``run_id`` field (script seed-tag, not an infra job id). Only fences
    # with these info-strings get blanked — leaving the rest of the
    # report's content intact handles whole-report outer fences, unbalanced
    # fences, and other arbitrary code blocks gracefully.
    _FENCE_INFO_STRINGS_TO_STRIP = ("json", "result_json", "json5", "jsonc")

    @classmethod
    def _strip_fenced_code_blocks(cls, text: str) -> str:
        """Replace the content of JSON-ish fenced code blocks with
        whitespace, preserving character offsets so other regexes still
        align with the original document positions.

        Stage 6b reports embed RESULT_JSON inside a `````json`` block,
        and that JSON often has its own ``\"run_id\": \"smoke_seed42\"`` field
        — the script's internal seed-tag, NOT an infra run_id the engine
        should wait on. Targeting only ``json`` / ``RESULT_JSON``
        info-strings eliminates that false-positive class without blanking
        arbitrary unrelated fences (e.g. a wholedocument outer fence, a
        ``bash`` code example, an unbalanced fence that would otherwise
        consume the rest of the document — all real cases that would
        wipe legitimate ``- run_id: ...`` entries and silently regress
        into the #93 \"critic dispatched on no runs\" behavior).
        """
        if "```" not in text:
            return text
        out = []
        in_strippable_fence = False
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if in_strippable_fence:
                    # Closing the strippable fence.
                    in_strippable_fence = False
                else:
                    # Opening: check the info-string (text after ```).
                    info = stripped[3:].strip().lower()
                    # Strip ``json``, ``RESULT_JSON``, etc.; leave bash,
                    # python, plain ```, etc. alone.
                    if info in cls._FENCE_INFO_STRINGS_TO_STRIP or info.startswith("result_json"):
                        in_strippable_fence = True
                out.append(line)  # keep the fence delimiter either way
                continue
            if in_strippable_fence:
                # Blank out the line but preserve its length so character
                # offsets stay valid for the original document.
                out.append(" " * (len(line) - 1) + ("\n" if line.endswith("\n") else ""))
            else:
                out.append(line)
        return "".join(out)

    # Machine-readable runs block the runner is instructed to emit:
    #   ```runs
    #   run_8e540235b830  succeeded
    #   run_dace6992d448  succeeded
    #   ```
    # When present it is AUTHORITATIVE — parsed deterministically and immune
    # to whatever prose/label/table the LLM wrapped the run around (the
    # recurring report-format whack-a-mole: fa50cb183b3c used a vertical
    # table, 9dd6160ea99b used "- Validated run: `x`"). Only this block is
    # read, so stray run-ids in the Limitations narrative (failed/cleanup
    # runs) cannot become phantom-pending runs.
    _CANONICAL_RUNS_FENCE_RE = re.compile(
        r"```+[ \t]*runs[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
    )

    @classmethod
    def _parse_canonical_runs_block(cls, report: str) -> list[tuple[str, str]] | None:
        """Parse the runner's ```runs block. Returns the (run_id, status)
        list when a block is present (possibly empty — authoritative "no
        runs"), or ``None`` when there is no such block so the caller falls
        back to the prose heuristic. One run per line: ``<run_id> <status>``
        with optional ``|`` / ``=`` / ``:`` separators and markdown ticks."""
        m = cls._CANONICAL_RUNS_FENCE_RE.search(report or "")
        if m is None:
            return None
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in m.group(1).splitlines():
            line = raw.strip()
            if not line or set(line) <= set("-:| "):  # blank or table separator
                continue
            toks = [
                t.strip("`*'\" ")
                for t in re.split(r"[\s|=:]+", line.strip("|").strip())
                if t.strip("`*'\" ")
            ]
            if len(toks) < 2:
                continue
            rid, status = toks[0], toks[1].lower()
            if rid.lower() in {"run_id", "run", "id", "runid", "status"}:  # header row
                continue
            if rid in seen:
                continue
            seen.add(rid)
            out.append((rid, status))
        return out

    @classmethod
    def _parse_runner_report_runs(cls, report: str) -> list[tuple[str, str]]:
        """Extract ``[(run_id, status), ...]`` pairs from a 6b runner report.

        Prefers the authoritative ```runs block when present; otherwise walks
        the report top-to-bottom and pairs each ``run_id:`` line with
        the next ``status:`` line that follows it within the same run-block.
        Tolerates the various Markdown decorations the runner uses (e.g.
        backtick-quoted, asterisk-bold, plain ``run_id: run_x`` styles).

        Filters fenced code blocks first so the RESULT_JSON's internal
        ``run_id`` (the script's seed tag, not an infra job id) is not
        confused with a real infra run_id.

        Returns ``[]`` if no run_ids are found — caller treats this as "no
        runs to wait on" (e.g. budget BLOCKED report with no submitted runs).
        """
        if not report:
            return []
        # Authoritative machine-readable block wins over the prose heuristic.
        canonical = cls._parse_canonical_runs_block(report)
        if canonical is not None:
            return canonical
        scan_text = cls._strip_fenced_code_blocks(report)
        run_id_hits = [
            (m.start(), m.group(1))
            for m in cls._RUN_ID_RE.finditer(scan_text)
            # Denylist: placeholders + Markdown table COLUMN NAMES. With the
            # ``|`` separator now allowed, a horizontal header
            # ``| run_id | status | cost |`` would otherwise capture the next
            # column name ("status") as a bogus run_id.
            if m.group(1).lower() not in {
                "run_id", "rid", "none", "null", "missing", "n_a", "n/a",
                "status", "cost", "gpu", "value", "field", "metric",
                "result", "notes", "command", "actual", "estimated",
            }
        ]
        status_hits = [
            (m.start(), m.group(1).lower())
            for m in cls._STATUS_LINE_RE.finditer(scan_text)
        ]
        if not run_id_hits:
            return []
        # Bind each run_id to a status that lives **within its block** —
        # the block is bounded by this run_id's offset on the low side and
        # the NEXT run_id's offset (or end-of-document) on the high side.
        # Fail-closed: a run_id with no status in its block is paired with
        # ``"unknown"``, which the fail-safe ``_runs_have_pending`` will
        # then treat as pending (engine keeps waiting rather than silently
        # firing the critic on an unverified run).
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for i, (offset, rid) in enumerate(run_id_hits):
            if rid in seen:
                continue
            seen.add(rid)
            block_end = run_id_hits[i + 1][0] if i + 1 < len(run_id_hits) else len(scan_text)
            paired_status = "unknown"  # fail-closed default
            for s_off, s_val in status_hits:
                if offset < s_off < block_end:
                    paired_status = s_val
                    break
            out.append((rid, paired_status))
        return out

    @classmethod
    def _runs_have_pending(cls, runs: list[tuple[str, str]]) -> bool:
        """True iff any run in the report is still in a non-terminal state.

        **Fail-safe semantics**: a status is considered pending unless it
        appears in ``_RUN_TERMINAL_STATUSES``. An unknown / free-form
        status the LLM phrased differently (e.g. ``in_progress``,
        ``executing``, ``submitted``) is treated as pending so the engine
        keeps waiting rather than dispatching the critic on a possibly
        still-running experiment (#93 regression class).
        """
        return any(status not in cls._RUN_TERMINAL_STATUSES for _rid, status in runs)

    @classmethod
    def _pending_run_ids_from(cls, runs: list[tuple[str, str]]) -> list[str]:
        """Same fail-safe semantics as ``_runs_have_pending``: a run_id is
        considered pending unless its status is explicitly terminal."""
        return [rid for rid, status in runs if status not in cls._RUN_TERMINAL_STATUSES]

    @classmethod
    def _all_pending_terminal(cls, pending_run_ids: list[str], stage_6_runs: dict) -> bool:
        """True iff every entry in ``pending_run_ids`` has a terminal status
        on the engine's ``stage_6_runs`` map. Empty pending list is treated
        as "already done" so callers can fall through cleanly.
        """
        if not pending_run_ids:
            return True
        if not isinstance(stage_6_runs, dict):
            return False
        for rid in pending_run_ids:
            entry = stage_6_runs.get(rid)
            if not isinstance(entry, dict):
                return False
            if entry.get("status") not in cls._RUN_TERMINAL_STATUSES:
                return False
        return True

    # ------------------------------------------------------------------
    # #27 Hard data gate — deterministic post-critic data-existence check
    # ------------------------------------------------------------------
    # Runs AFTER the critic votes PASS and can override it. The critic
    # grades report *quality*; this gate checks *real data exists*. The
    # LLM cannot vote its way past it (closes #96 A+C, #94).

    # A Stage 6 run only counts as real data if it actually finished.
    _RUN_DATA_OK_STATUSES = ("succeeded", "partial_success")

    @classmethod
    def _data_gate(cls, stage_id: int, result: str) -> tuple[bool, str]:
        """Deterministic check that a stage's own deliverable (``result``)
        contains real experimental data. Pure / classmethod — depends only
        on the report text. Returns ``(ok, reason)``. ``ok=True`` for any
        stage without an own-result rule (1-5), so the gate is a no-op there.
        """
        if stage_id == 6:
            return cls._data_gate_stage6(result)
        if stage_id == 7:
            return cls._data_gate_stage7(result)
        return True, ""

    def _read_stage_deliverable(self, stage_id: int, fallback: str = "") -> str:
        """Return the canonical on-disk deliverable for a stage, falling back
        to ``fallback`` (usually the stored ``stage_results`` entry).

        The producer's ``submit_result`` is often a prose *summary* that
        points at the file ("see stage6_experimentalist.md") and does not
        itself carry the machine-parseable evidence (run_id/status lines,
        hypothesis decisions). The real artifact is the ``stage{N}_{skill}.md``
        file the producer wrote. The data gate must inspect that file, not
        the summary — otherwise it gates the wrong text. Mirrors the existing
        Stage 3 file-content fallback.
        """
        try:
            stage_def = self._stage_def(stage_id)
            path = Path(self.project_dir) / f"stage{stage_id}_{stage_def.get('skill', '')}.md"
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except (OSError, ValueError) as exc:
            logger.debug("[PIPELINE] could not read stage {} deliverable: {}", stage_id, exc)
        return fallback

    def _synthesize_stage6_receipt(self) -> bool:
        """6a salvage: if the implementer committed a Stage-6 adaptation to
        ``upstream/`` but left no receipt — its final turn returned an empty
        "Executed: bash" stub before writing Phase 5 (run c49e8a914f5a) — the
        whole stage dies at the 6a hard-gate even though the code is done.

        Reconstruct a minimal receipt from git state so the 6b runner can
        proceed. Returns True iff a non-trivial receipt now exists. Refuses
        (returns False) when there is no committed adaptation to salvage —
        a genuinely empty 6a must still fail.
        """
        receipt = Path(self.project_dir) / "stage6_implementation_receipt.md"
        try:
            if receipt.exists() and receipt.stat().st_size >= 200:
                return True
        except OSError:
            return False
        upstream = Path(self.project_dir) / "upstream"
        if not (upstream / ".git").exists():
            return False
        import subprocess

        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=str(upstream),
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()

        try:
            dirty = _git("status", "--short")
            head = _git("rev-parse", "HEAD")
            subject = _git("log", "-1", "--format=%s")
            diffstat = _git("show", "--stat", "--format=%h %s", "HEAD")
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("[PIPELINE] 6a receipt synthesis git probe failed: {}", exc)
            return False
        # Salvage only a CLEAN tree carrying an adaptation commit (not the bare
        # pinned checkout) — otherwise there is no committed work to record.
        if dirty or not head:
            return False
        if "adapt" not in subject.lower() and "stage 6" not in subject.lower():
            return False
        body = (
            "# Stage 6a — Implementation Receipt (engine-synthesized)\n\n"
            "> ⚠️ Auto-generated from git state: the implementer committed the\n"
            "> Stage 6 adaptation to `upstream/` but its final turn returned no\n"
            "> text (stub) before writing this receipt. Reconstructed so the\n"
            "> Stage 6b runner can proceed instead of discarding finished code.\n\n"
            "## 0. Pin status\n"
            f"- `adaptation_commit`: `{head[:12]} {subject}`\n\n"
            "## Changed files (git show --stat HEAD)\n\n"
            "```\n" + (diffstat or "(unavailable)") + "\n```\n\n"
            "## Runnable entrypoint\n"
            "- Not captured (implementer stubbed). The runner derives the run\n"
            "  command from `stage5_experiment_designer.md` / "
            "`stage5_codebase_pin.md`.\n"
        )
        try:
            receipt.write_text(body, encoding="utf-8")
            return receipt.stat().st_size >= 200
        except OSError as exc:
            logger.debug("[PIPELINE] 6a receipt synthesis write failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # #44 — paper-numbers gate: every high-precision statistic in the
    # Stage-8 paper must be traceable to the Stage 4-7 evidence. Number
    # fabrication is the worst paper failure mode and the LLM critic only
    # spot-checks; this is the deterministic backstop.
    #
    # Claim class: decimals with ≥2 decimal places and ≥3 significant
    # digits (85.97, 1.5406, 69.60). Deliberately EXCLUDED to avoid false
    # positives: integers (years, n, token budgets, section refs),
    # 1-decimal values (5.1), and low-significance values (p < 0.001,
    # alpha 0.05). Matching allows the writer's rounding (tolerance =
    # half-ULP at the written precision) and fraction↔percent scaling.
    # ------------------------------------------------------------------
    # Trailing guard: block a following digit or ``.digit`` (version strings
    # like 0.11.0) but NOT sentence punctuation — "…effect size 3.1415." must
    # still be captured.
    _PAPER_CLAIM_RE = re.compile(r"(?<![\w.])(\d{1,4}\.\d{2,6})(?!\d)(?!\.\d)")
    _UPSTREAM_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

    @staticmethod
    def _sig_digits(tok: str) -> int:
        return len(tok.replace(".", "").lstrip("0"))

    @classmethod
    def _paper_numbers_gate(cls, paper_text: str, upstream_text: str) -> tuple[bool, str]:
        """Return ``(ok, reason)``. ``ok=False`` only when ≥2 high-precision
        statistics in the paper match nothing upstream — a single orphan is
        tolerated (could be a legitimate author-derived value) but named in
        the reason for the critic to eyeball."""
        claims: list[str] = []
        seen: set[str] = set()
        for m in cls._PAPER_CLAIM_RE.finditer(paper_text or ""):
            tok = m.group(1)
            if tok not in seen and cls._sig_digits(tok) >= 3:
                seen.add(tok)
                claims.append(tok)
        if not claims:
            return True, ""
        upstream_vals = [float(t) for t in cls._UPSTREAM_NUM_RE.findall(upstream_text or "")]
        unmatched: list[str] = []
        for tok in claims:
            d = float(tok)
            dp = len(tok.split(".")[1])
            tol = 0.5 * 10 ** (-dp) + 1e-9
            if not any(
                abs(v - d) <= tol or abs(v * 100 - d) <= tol or abs(v / 100 - d) <= tol
                for v in upstream_vals
            ):
                unmatched.append(tok)
        if len(unmatched) >= 2:
            return False, (
                "paper contains statistics not traceable to the Stage 4-7 "
                f"evidence: {', '.join(unmatched)} — every reported number must "
                "come from stage7_result_analyst.md / the Stage 6 RESULT_JSON "
                "(or state its derivation next to a traceable source)"
            )
        if unmatched:
            return True, f"1 untraceable statistic tolerated (verify manually): {unmatched[0]}"
        return True, ""

    def _stage_data_gate(self, stage_id: int, result: str) -> tuple[bool, str]:
        """Instance-level gate orchestration used by ``on_task_complete``.

        Delegates to the pure ``_data_gate`` for stages whose rule depends
        only on their own ``result`` (6, 7). Stage 8 has no own-result rule
        but requires the UPSTREAM Stage 7 artifact to have carried real data
        — otherwise the paper-writer is writing about nothing. We enforce
        that by re-running the Stage 7 gate against the stored Stage 7
        result (#96 Failure C). Stage 8 additionally passes the
        paper-numbers gate (#44): high-precision statistics must be
        traceable to the Stage 4-7 evidence.
        """
        ok, reason = self._data_gate(stage_id, result)
        if not ok:
            return ok, reason
        if stage_id == 8:
            results = self.state.get("stage_results") or {}
            stage7 = self._read_stage_deliverable(7, fallback=results.get("7", ""))
            s7_ok, s7_reason = self._data_gate_stage7(stage7)
            if not s7_ok:
                return False, f"upstream Stage 7 has no confirmatory data ({s7_reason})"
            # #44 numbers gate — paper vs the union of upstream evidence.
            paper = self._read_stage_deliverable(8, fallback=result)
            upstream_parts = [stage7]
            for sid in (4, 5, 6):
                upstream_parts.append(
                    self._read_stage_deliverable(sid, fallback=results.get(str(sid), ""))
                )
            receipt = Path(self.project_dir) / "stage6_implementation_receipt.md"
            try:
                upstream_parts.append(receipt.read_text(encoding="utf-8"))
            except OSError:
                logger.debug("[PIPELINE] stage6 receipt not yet present — skipping for numbers gate")
            nums_ok, nums_reason = self._paper_numbers_gate(paper, "\n".join(upstream_parts))
            if not nums_ok:
                return False, nums_reason
            if nums_reason:
                logger.info("[PIPELINE] paper-numbers gate: {}", nums_reason)
        return True, ""

    def _pipeline_has_real_data(self) -> bool:
        """True iff the pipeline actually produced experimental data:
        Stage 6 had ≥1 succeeded run AND Stage 7 had ≥1 tested hypothesis.
        Reads the on-disk deliverables (canonical evidence), not the
        submit_result summaries. Used by the Stage 9 verdict clamp (#94)."""
        results = self.state.get("stage_results") or {}
        s6 = self._read_stage_deliverable(6, fallback=results.get("6", ""))
        s7 = self._read_stage_deliverable(7, fallback=results.get("7", ""))
        s6_ok, _ = self._data_gate_stage6(s6)
        s7_ok, _ = self._data_gate_stage7(s7)
        return s6_ok and s7_ok

    # Stage 9 review verdicts that imply the paper is acceptable. When no
    # experimental data exists, none of these may stand — clamp to
    # MAJOR REVISION. REJECT / MAJOR REVISION are already non-acceptance,
    # so they pass through unchanged.
    _ACCEPTANCE_VERDICT_RE = re.compile(
        r"((?:verdict|recommendation|decision)\s*[:=]\s*\**\s*)"
        r"(weak\s+accept|accept|minor\s+revision)",
        re.IGNORECASE,
    )

    @classmethod
    def _clamp_review_verdict(cls, review_text: str) -> tuple[str, bool]:
        """Rewrite an acceptance-class verdict to MAJOR REVISION. Returns
        ``(new_text, changed)``. Idempotent: if no acceptance verdict is
        present (already REJECT / MAJOR REVISION / not found), returns the
        text unchanged with ``changed=False``."""
        if not review_text:
            return review_text, False
        m = cls._ACCEPTANCE_VERDICT_RE.search(review_text)
        if not m:
            return review_text, False
        new_text = cls._ACCEPTANCE_VERDICT_RE.sub(
            lambda mm: f"{mm.group(1)}MAJOR REVISION", review_text, count=1
        )
        new_text += (
            "\n\n> [engine data-gate clamp] Verdict downgraded to MAJOR REVISION: "
            "the pipeline produced no real experimental data (no succeeded Stage 6 "
            "run / no tested Stage 7 hypothesis), so the paper cannot be accepted "
            "regardless of presentation quality (#94)."
        )
        return new_text, True

    @classmethod
    def _data_gate_stage6(cls, result: str) -> tuple[bool, str]:
        """Stage 6 passes only if ≥1 submitted run reached a data-bearing
        terminal status (succeeded / partial_success)."""
        runs = cls._parse_runner_report_runs(result)
        if not runs:
            return False, "no run_ids found in runner report — zero experiments submitted"
        ok_runs = [rid for rid, status in runs if status in cls._RUN_DATA_OK_STATUSES]
        if not ok_runs:
            statuses = ", ".join(f"{rid}={status}" for rid, status in runs[:5])
            return False, f"no run succeeded ({statuses})"
        return True, ""

    # Capture the value after a ``Decision:`` / ``Verdict:`` label (the
    # leading word(s), e.g. "NOT SUPPORTED", "PASS", "NOT TESTED",
    # "INCONCLUSIVE_DUE_TO_COVERAGE"). ``[\w ]+`` grabs word chars + spaces;
    # it stops at a dash / period / digit-start, which is where the
    # explanatory clause begins.
    _HYPOTHESIS_DECISION_RE = re.compile(
        r"(?:decision|verdict)\b\W*?[:：]\s*\*{0,2}\s*([A-Za-z][\w ]*)",
        re.IGNORECASE,
    )
    # Vertical key-value table row: ``| Decision | `SUPPORTED` |``. Anchored
    # to the FIRST cell (``^\|``) so a horizontal header
    # ``| H1 | Decision | p-value |`` (where "Decision" is a middle column,
    # not a row label) does NOT capture the next column name. The Stage 6
    # data-gate had this exact table-blindness (run fa50cb183b3c) — Stage 7
    # has the same gate, so harden it the same way.
    _STAGE7_VERTICAL_DECISION_RE = re.compile(
        r"^\s*\|\s*(?:decision|verdict|outcome|conclusion)\b\s*\|\s*"
        r"[`*'\"]*([A-Za-z][\w ]*?)[`*'\"]*\s*\|",
        re.IGNORECASE | re.MULTILINE,
    )
    # Header cell naming the decision column in a horizontal results table
    # (one row per hypothesis). Used to locate which column holds the verdict.
    _STAGE7_DECISION_COL_RE = re.compile(
        r"\b(?:decision|verdict|outcome|conclusion|result)\b", re.IGNORECASE
    )
    # Decisions that mean "the experiment did NOT produce data for this
    # hypothesis". Everything else (SUPPORTED, NOT SUPPORTED, REJECTED,
    # CONFIRMED, PASS, FAIL, INCONCLUSIVE, …) means a test actually ran —
    # including null results, which ARE data. Normalised: upper, underscores
    # and runs of spaces collapsed to single spaces.
    _STAGE7_NO_DATA_DECISIONS = (
        "NOT TESTED",
        "INCONCLUSIVE DUE TO COVERAGE",
        "BLOCKED",
        "NO DATA",
    )

    @staticmethod
    def _normalise_decision(raw: str) -> str:
        return re.sub(r"[\s_]+", " ", raw).strip().upper()

    @classmethod
    def _stage7_table_decisions(cls, text: str) -> list[str]:
        """Extract decision cells from a horizontal Markdown results table
        whose header names a Decision/Verdict/Outcome column (the common
        Stage-7 layout: one row per hypothesis). Reads ONLY the designated
        column — no prose scanning — so it cannot false-positive on a stray
        "supported" in body text."""
        out: list[str] = []
        lines = (text or "").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.count("|") >= 2 and cls._STAGE7_DECISION_COL_RE.search(line):
                header = [c.strip() for c in line.strip().strip("|").split("|")]
                col_idx = next(
                    (j for j, c in enumerate(header) if cls._STAGE7_DECISION_COL_RE.search(c)),
                    None,
                )
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                # A real table header is followed by a |---|---| separator row.
                if col_idx is not None and set(nxt.replace("|", "").strip()) <= set("-: "):
                    j = i + 2
                    while j < len(lines) and lines[j].count("|") >= 2:
                        row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                        if col_idx < len(row) and row[col_idx]:
                            cell = row[col_idx].strip("`*'\" ")
                            if cell:
                                out.append(cls._normalise_decision(cell))
                        j += 1
                    i = j
                    continue
            i += 1
        return out

    @classmethod
    def _data_gate_stage7(cls, result: str) -> tuple[bool, str]:
        """Stage 7 passes if ≥1 hypothesis/check has a decision indicating a
        test actually ran. The gate blocks "no experiment ran" (every
        decision is NOT TESTED / INCONCLUSIVE_DUE_TO_COVERAGE / BLOCKED), NOT
        "effect not found" — a ``NOT SUPPORTED`` / null result IS data and
        must pass (#27 Stage 7 false-positive caught by the 4→9 e2e).

        Decisions are collected from three layouts so a valid analysis is not
        falsely failed for formatting: inline ``Decision: X``, the vertical
        row form ``| Decision | X |``, and a horizontal results table with a
        Decision/Verdict column (run fa50cb183b3c showed the Stage 6 twin of
        this gate dying on a table-formatted report)."""
        decisions = [
            cls._normalise_decision(m.group(1))
            for m in cls._HYPOTHESIS_DECISION_RE.finditer(result or "")
        ]
        decisions += [
            cls._normalise_decision(m.group(1))
            for m in cls._STAGE7_VERTICAL_DECISION_RE.finditer(result or "")
        ]
        decisions += cls._stage7_table_decisions(result or "")
        if not decisions:
            return False, "no hypothesis/check decision lines found in result analysis"
        tested = [d for d in decisions if d not in cls._STAGE7_NO_DATA_DECISIONS]
        if not tested:
            return False, (
                f"all {len(decisions)} decisions are no-data markers "
                f"({', '.join(sorted(set(decisions)))}) — no experiment ran"
            )
        return True, ""

    def on_runs_wait_timeout(self, wait_seconds: int) -> None:
        """Called by ``run_tracker`` when a project has been parked in
        ``producer_b_waiting`` past the configured max-wait deadline
        without all pending runs reaching terminal status.

        Treat it as a producer failure so the existing exhausted-retries
        path opens a CEO gate (or, under auto_approve, marks the pipeline
        failed via the #106 fix). The on-disk ``pending_run_ids`` are
        preserved for forensics — the CEO can read which runs were
        still active when the deadline tripped.

        Idempotent: only acts when ``phase == "producer_b_waiting"``.
        """
        if self.phase != "producer_b_waiting":
            return
        pending = self.state.get("pending_run_ids") or []
        logger.warning(
            "[PIPELINE] Stage 6b max-wait timeout ({}s) — {} pending runs still active. "
            "Marking pipeline failed; CEO can inspect pending_run_ids for forensics.",
            wait_seconds, len(pending),
        )
        stage = self._stage_def()
        self.state["phase"] = "gate"
        self.state["failure_reason"] = f"stage_6_waiting_timeout_{wait_seconds}s"
        self._save()
        # Same gate-open path the rest of the engine uses on exhaustion.
        # Under auto_approve, #106's _auto_approve_gate flips this to
        # phase=failed; under interactive mode, the CEO sees an exhausted
        # gate with a failure_reason explaining the timeout.
        self._emit_gate_event(stage["id"], confidence=None, exhausted=True)

    def on_runs_all_terminal(self) -> None:
        """Called by ``run_tracker`` (or self-checked on each tick) when every
        ``pending_run_ids`` entry on this engine has reached a terminal status
        in ``stage_6_runs``.

        Transitions the engine out of ``producer_b_waiting`` by re-dispatching
        the runner with a "your runs finished — write the FINAL report"
        instruction. The runner re-fetches log_tails + metrics via the
        infra status endpoint and produces a complete experimentalist
        report, which then flows to the critic via the normal path.

        Idempotent: only acts when ``phase == "producer_b_waiting"``.
        """
        if self.phase != "producer_b_waiting":
            return
        pending = self.state.get("pending_run_ids") or []
        logger.info(
            "[PIPELINE] Stage 6b waiting → finalize: {} run_ids now terminal",
            len(pending),
        )
        self.state["phase"] = "producer_b_finalize"
        self._save()
        self._dispatch_producer_b_finalize()

    def _dispatch_producer_b_finalize(self) -> None:
        """Re-dispatch the experiment runner to write the FINAL Stage 6b
        report now that all submitted runs have reached terminal status.

        Uses the same employee + skill as the initial 6b dispatch; the
        runbook reads the ``pending_run_ids`` (carried via task description)
        and fetches each run's final metrics via ``fast_query_exp_status.sh``.
        """
        stage = self._stage_def()
        if stage["id"] != 6:
            logger.error(
                "[PIPELINE] _dispatch_producer_b_finalize called for non-Stage-6 stage {}",
                stage["id"],
            )
            return
        employee_id = _find_stage_6b_employee()
        if not employee_id:
            logger.error("[PIPELINE] No experiment_runner employee for Stage 6b finalize")
            self.state["phase"] = "failed"
            self._save()
            self._on_became_terminal()
            return

        pending = self.state.get("pending_run_ids") or []
        stage_6_runs = self.state.get("stage_6_runs", {}) or {}
        digest_lines = []
        for rid in pending:
            entry = stage_6_runs.get(rid, {}) if isinstance(stage_6_runs, dict) else {}
            digest_lines.append(
                f"  - {rid}: status={entry.get('status','?')} "
                f"cost={entry.get('actual_cost','?')} "
                f"finished_at={entry.get('finished_at','?')}"
            )
        digest = "\n".join(digest_lines) if digest_lines else "  (no pending run_ids recorded)"

        desc = (
            "Stage 6b: FINAL REPORT (runs are now terminal)\n\n"
            "You previously submitted experiments to remote infra and exited "
            "early so the engine could wait for them. All your runs have now "
            "reached terminal status. Read each run's final evidence and "
            "write the FINAL stage6_experimentalist.md.\n\n"
            f"Pending run_ids (now terminal, snapshot from run_tracker):\n{digest}\n\n"
        )
        user_feedback = self._consume_pending_feedback()
        if user_feedback:
            desc += (
                f"Direct guidance from CEO (received during the waiting "
                f"window — apply this when writing the report):\n{user_feedback}\n\n"
            )
        desc += (
            "## REQUIRED FIRST STEP\n"
            'Call load_skill("experiment-execution-runbook") and jump to '
            "Step 3 (write the report). For each run_id above, run "
            "`fast_query_exp_status.sh <run_id>` ONCE to capture the final "
            "log_tail / actual_cost / metrics, then write the canonical "
            "stage6_experimentalist.md. Do NOT re-submit; the runs are done.\n\n"
            "After writing the file, call submit_result() referencing it."
        )

        self._dispatch_to_employee(employee_id, desc, "Stage 6b: Final Report")
        emp_name = employee_id
        configs = load_employee_configs()
        if employee_id in configs:
            emp_name = configs[employee_id].name
        self._emit_stage_event("stage_start", stage["id"], employee_name=emp_name, employee_id=employee_id)

    # A verdict is only recognised from a LABELED / structured signal — never
    # an incidental occurrence of "pass"/"reject" (e.g. the rubric header
    # "Auto-REJECT trigger check", or a per-dimension "| D10 | … | PASS |"
    # row). #19 facet 2: a stray "Auto-REJECT" in an otherwise-PASS review
    # was flipping the verdict to REJECT and killing a healthy stage.
    _VERDICT_LABEL_RE = re.compile(
        r"(?:decision|verdict|recommendation|gate\s+review[^\n:：—–-]*)"
        r"\s*[:：—–-]\s*\*{0,2}\s*(PASS|REJECT)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _verdict_from_text(cls, text: str) -> bool | None:
        """Extract PASS (True) / REJECT (False) / ambiguous (None) from a
        critic text blob, using only LABELED/structured signals:
        - table ``| Decision | PASS |`` (markdown emphasis stripped),
        - labeled ``Decision: PASS`` / ``Verdict: REJECT`` /
          ``Gate Review Complete — PASS``,
        - a leading conversational ``PASS: …`` / ``REJECT: …``.

        A bare mention of "pass"/"reject" elsewhere does NOT count — that
        was the #19 false-verdict source (e.g. "Auto-REJECT trigger check")."""
        if not text:
            return None
        # #138: strip Markdown emphasis FIRST so a bolded label parses. The
        # label regex needs ``decision`` immediately followed by the ``:``
        # separator, but ``- **Decision**: PASS`` has ``**`` between them, so
        # the verdict was read as ambiguous → false REJECT → retries exhausted.
        # Stripping ``*``/``_`` also normalises ``Decision: **PASS**``.
        text = re.sub(r"[*_]", "", text)
        upper = text.upper()
        # Table-format ``| Decision | PASS |`` / ``| **Verdict** | `REJECT` |``.
        # Collapse pipes / emphasis / backticks so a backtick-wrapped cell
        # value parses, and accept Verdict/Recommendation labels too — not
        # only "Decision" (same decoration-blindness class as the data gates).
        compact = re.sub(r"[\s|*_`]+", " ", upper)
        for label in ("DECISION", "VERDICT", "RECOMMENDATION"):
            if f" {label} PASS " in compact:
                return True
            if f" {label} REJECT " in compact:
                return False
        # Labeled verdict anywhere in the text.
        m = cls._VERDICT_LABEL_RE.search(text)
        if m:
            return m.group(1).upper() == "PASS"
        # Leading conversational verdict (``pass: …`` / ``reject: …``).
        head = text.lstrip().lstrip("*# ").upper()
        if head.startswith("PASS"):
            return True
        if head.startswith("REJECT"):
            return False
        return None

    # Result-driven loop (#40): the result-reviewer routes the pipeline back
    # to an earlier stage when the experiment's RESULT (not its report) is
    # unsound. Only these stages are valid revert targets:
    #   4 = methodology (conceptual: wrong hypothesis / missing control)
    #   5 = experiment design (scale: n / seeds / power / baselines)
    #   6 = code (implementation: extraction broken, 0%/NaN, etc.)
    _RESULT_LOOP_TARGETS = (4, 5, 6)
    _RESULT_REVERT_RE = re.compile(
        r"revert[\s_-]*to[\s_-]*stage\b\W*?[:=|]\s*\*{0,2}\s*([4-6])",
        re.IGNORECASE,
    )

    @classmethod
    def _parse_result_route(cls, text: str) -> tuple[str, "int | None", str]:
        """Parse the result-reviewer's routing verdict.

        Returns ``(action, target_stage, reason)``:
        - ``("advance", None, reason)`` — result is sound, proceed to Stage 8.
        - ``("revert", N, reason)`` — N ∈ {4,5,6}; loop back to that stage.

        Fail-safe: a REVERT with no parseable in-range target stage degrades
        to ``advance`` — we never guess which stage to loop back to, and we
        don't loop on a malformed review."""
        t = text or ""
        # Match a line whose label is exactly "reason" (not "reasonableness").
        rm = re.search(r"^\s*\**\s*reason\s*\**\s*[:：—-]\s*(.+)$", t,
                       re.IGNORECASE | re.MULTILINE)
        reason = rm.group(1).strip() if rm else t.strip()[:300]

        m = cls._RESULT_REVERT_RE.search(t)
        target = int(m.group(1)) if m else None

        upper = t.upper()
        compact = re.sub(r"[\s|*_]+", " ", upper)
        wants_revert = (" ACTION REVERT " in f" {compact} "
                        or " DECISION REVERT " in compact
                        or "REVERT TO STAGE" in compact)

        if wants_revert and target in cls._RESULT_LOOP_TARGETS:
            return "revert", target, reason
        # Anything else — explicit ADVANCE, ambiguous, or out-of-range target.
        return "advance", None, reason

    def _parse_critic_pass(self, result: str) -> bool:
        """Parse the critic's PASS/REJECT verdict.

        Robustness (#60 fix 4 / #63 fix 4 / #19):
        - An explicit verdict in ``result`` (the critic's submit_result text)
          wins — table-format ``| Decision | PASS |`` or conversational.
        - Otherwise, the verdict is AMBIGUOUS. This happens when the
          submit_result was a stub (``"Executed: bash"``) OR a long
          tool-result echo (``"Executed: write\\nwrite → {'path': ...}"``,
          which slips past the short-stub heuristic — #19, seen in the 4→9
          e2e). In BOTH cases the real verdict lives in the on-disk
          ``stage{N}_gate_review.md`` the critic was told to write — consult
          it before giving up.
        - Default to REJECT only when neither the text nor the file yields a
          signal. (The old default-to-PASS branch was the #60/#63
          auto-approve-empty-stage loophole.)
        """
        verdict = self._verdict_from_text(result or "")
        if verdict is not None:
            return verdict

        # Ambiguous submit_result → the verdict is in the file on disk.
        # The critic VERSIONS its reviews on retry cycles (it finds
        # stage{N}_gate_review.md from the previous cycle and writes
        # _v2 / _v3 instead of overwriting), so consulting the fixed name
        # reads the PREVIOUS cycle's verdict — run e04df33b06bb's PASS-v2
        # was shadowed by the stale v1 REJECT and a successful experiment
        # died at retries-exhausted. Consult newest → oldest; the first
        # file with a parseable verdict wins.
        stage_id = self.current_stage
        candidates = sorted(
            # #138: the critic actually writes ``gate_review_stage{N}.md``;
            # the old glob only matched the inverted ``stage{N}_gate_review*.md``
            # so the on-disk PASS was never found → false REJECT. Match BOTH
            # (newest by mtime wins, across both naming conventions).
            [
                *Path(self.project_dir).glob(f"gate_review_stage{stage_id}*.md"),
                *Path(self.project_dir).glob(f"stage{stage_id}_gate_review*.md"),
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for gate_review in candidates:
            try:
                file_text = gate_review.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("[PIPELINE] Failed to read {}: {}", gate_review.name, exc)
                continue
            file_verdict = self._verdict_from_text(file_text)
            if file_verdict is not None:
                logger.info(
                    "[PIPELINE] Critic submit_result had no verdict — resolved {} "
                    "from {} ({} bytes, newest of {} review file(s))",
                    "PASS" if file_verdict else "REJECT", gate_review.name,
                    len(file_text), len(candidates),
                )
                return file_verdict

        # Neither text nor any review file yielded a verdict → safer default
        # is REJECT.
        logger.warning(
            "[PIPELINE] Critic verdict ambiguous (no PASS/REJECT in submit_result "
            "or any gate_review_stage{0}*.md / stage{0}_gate_review*.md) — "
            "defaulting to REJECT", stage_id,
        )
        return False

    @staticmethod
    def _parse_confidence(result: str) -> float | None:
        import re
        # Match "confidence: 0.72", "Confidence Score: 0.8", and the decorated
        # forms the critic actually uses — ``**Confidence**: **0.92**`` and the
        # table cell ``| Confidence | 0.92 |``. ``\W*?`` (lazy non-word chars)
        # absorbs the ``**`` / ``|`` / ``:`` markers between the label and the
        # number without requiring them (run fa50cb183b3c lost a real 0.92).
        m = re.search(r'confidence(?:\s+score)?\W*?([01]\.?\d*)', result, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError as exc:
                logger.debug("Unable to parse confidence value '{}': {}", m.group(1), exc)
        return None

    # ------------------------------------------------------------------
    # Event emission (WebSocket → frontend)
    # ------------------------------------------------------------------

    async def _emit_async(self, payload: dict):
        await event_bus.publish(CompanyEvent(
            type=EventType.STATE_SNAPSHOT,
            payload=payload,
            agent=SYSTEM_AGENT,
        ))

    def _emit_critic_result(self, stage_id: int, critic_text: str, is_pass: bool, confidence: float = None):
        """Emit critic review result so frontend can display it in the stage card."""
        import asyncio
        payload = {
            "type": "critic_result",
            "stage": stage_id,
            "stage_name": self._stage_def(stage_id).get("name", ""),
            "project_id": self.project_id,
            "pipeline_managed": True,
            "decision": "PASS" if is_pass else "REJECT",
            "confidence": confidence,
            "text": critic_text,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
        except RuntimeError as exc:
            logger.debug("Skipping critic_result event; no running event loop: {}", exc)

    def _emit_stage_event(self, event_type: str, stage_id: int, confidence: float = None, employee_name: str = "", employee_id: str = ""):
        """Emit stage lifecycle events for the frontend."""
        import asyncio
        payload = {
            "type": event_type,
            "stage": stage_id,
            "stage_name": self._stage_def(stage_id).get("name", ""),
            "employee_name": employee_name,
            "employee_id": employee_id,
            "project_id": self.project_id,
            "pipeline_managed": True,
        }
        if confidence is not None:
            payload["confidence"] = confidence
        if event_type == "stage_start":
            hint = self._stage_duration_hint(stage_id)
            if hint:
                payload["eta"] = hint
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
        except RuntimeError as exc:
            logger.debug("Skipping stage event; no running event loop: {}", exc)

    def _reset_attempt_timing(self) -> None:
        self._ensure_timing_state()
        self.state["attempt_timing"] = {
            "producer_elapsed_seconds": 0.0,
            "critic_elapsed_seconds": None,
        }

    def _record_active_phase_elapsed(self, phase: str) -> None:
        self._ensure_timing_state()
        started_at = self.state.get("active_task_started_at")
        self.state["active_task_started_at"] = None
        if started_at is None:
            return
        try:
            elapsed = max(0.0, float(time.time()) - float(started_at))
        except (TypeError, ValueError):
            return
        try:
            timeout_seconds = float(self._stage_def().get("timeout_seconds", 0) or 0)
        except Exception:
            timeout_seconds = 0
        if timeout_seconds <= 0:
            timeout_seconds = 3600.0
        elapsed = min(elapsed, timeout_seconds * 2.0)

        timing = self.state.get("attempt_timing", {})
        if phase in ("producer", "producer_b"):
            previous = timing.get("producer_elapsed_seconds")
            timing["producer_elapsed_seconds"] = float(previous or 0.0) + elapsed
        elif phase == "critic":
            timing["critic_elapsed_seconds"] = elapsed

    def _stage_duration_hint(self, stage_id: int) -> dict | None:
        try:
            stats = self._memory_store().summarize_stage_durations(limit_per_stage=30)
        except Exception as exc:
            logger.debug("[PIPELINE] Stage duration stats unavailable: {}", exc)
            return None
        stage = stats.get(str(stage_id))
        if not stage:
            return None
        return {
            "samples": int(stage.get("samples", 0) or 0),
            "window_size": int(stage.get("window_size", 0) or 0),
            "total": stage.get("total", {}),
            "producer": stage.get("producer", {}),
            "critic": stage.get("critic", {}),
            "retry_rate": float(stage.get("retry_rate", 0.0) or 0.0),
        }

    def _emit_gate_event(self, stage_id: int, confidence: float = None, exhausted: bool = False):
        """Emit breakpoint/gate event for frontend to show approval dialog."""
        import asyncio
        payload = {
            "type": "breakpoint_hit",
            "stage": stage_id,
            "stage_name": self._stage_def(stage_id).get("name", ""),
            "project_id": self.project_id,
            "confidence": confidence,
            "retries_exhausted": exhausted,
            "message": f"Stage {stage_id} complete. Waiting for your approval.",
            "pipeline_managed": True,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
            # Headless/unattended mode: advance the gate automatically so the
            # pipeline runs end-to-end with no human confirmation. Covers BOTH
            # gate openings (clean PASS and retries-exhausted) since both land
            # here. A human would otherwise click "approve" in the UI.
            if self.state.get("auto_approve"):
                loop.create_task(self._auto_approve_gate(stage_id, exhausted))
        except RuntimeError as exc:
            logger.debug("Skipping gate event; no running event loop: {}", exc)

    async def _auto_approve_gate(self, stage_id: int, exhausted: bool):
        """Unattended-mode gate advance: behaves like a CEO clicking approve
        on a clean PASS, but **refuses to approve an exhausted-retries gate**.

        Exhausted gates land here when the stage failed all its retries (stub
        results, hard-gate misses, critic REJECTs, or producer crashes). Those
        runs have no usable deliverable; advancing would mask the failure as
        ``phase=done`` with empty ``stage_results``. Mark the pipeline as
        ``failed`` instead — a human CEO can still POST /api/ceo/approve
        explicitly to override if they have an out-of-band recovery plan.
        """
        import asyncio
        await asyncio.sleep(0)  # let the gate event flush first
        if self.phase != "gate":
            return
        if exhausted:
            logger.warning(
                "[PIPELINE] AUTO-APPROVE refused for exhausted gate at stage {} "
                "— marking pipeline failed (manual /api/ceo/approve still available)",
                stage_id,
            )
            self.state["phase"] = "failed"
            self.state["failure_reason"] = f"stage_{stage_id}_retries_exhausted"
            self._save()
            self._emit_pipeline_failed(stage_id, "retries_exhausted")
            return
        logger.info(
            "[PIPELINE] AUTO-APPROVE (unattended): advancing gate at stage {}",
            stage_id,
        )
        self.on_ceo_approve("")

    def cancel(self, reason: str = "cancelled by user") -> None:
        """Terminally cancel this pipeline (R5-1).

        ``/api/task/<pid>/abort`` cancels the task-tree nodes, but without
        a terminal engine phase the cancelled node's failure event is
        indistinguishable from an ordinary producer crash — the engine
        retried and the pipeline resurrected itself (zombie 76ad6534ed86
        survived three aborts). Idempotent; no-op on already-terminal
        pipelines so a late abort cannot stomp a ``done`` result.
        """
        if self.phase in ("done", "failed"):
            return
        stage_id = self.current_stage
        logger.warning(
            "[PIPELINE] Cancelled at stage {} phase {} (project={}): {}",
            stage_id, self.phase, self.project_id, reason,
        )
        self.state["phase"] = "failed"
        self.state["failure_reason"] = f"cancelled: {reason}"
        self.state["active_node_id"] = None
        self._save()
        try:
            self._emit_pipeline_failed(stage_id, f"cancelled: {reason}")
        except Exception as exc:  # noqa: BLE001 — cancellation must never raise
            logger.warning("[PIPELINE] cancel(): failed-event emit raised: {}", exc)

    def _on_became_terminal(self) -> None:
        """Evict self from the in-memory cache and promote the next queued
        pipeline (#159). Called from every terminal transition so slots free
        promptly without waiting for a periodic poll.
        """
        _active_pipelines.pop(self.project_id, None)
        dequeue_next_pipeline()

    def _emit_pipeline_failed(self, stage_id: int, reason: str):
        """Mirror of ``_emit_pipeline_complete`` for the failed terminal state.

        Fires when auto-approve refuses an exhausted gate, so frontend /
        archive consumers see the project closed as failed rather than
        silently hanging at ``phase=gate``.
        """
        import asyncio
        payload = {
            "type": "pipeline_failed",
            "project_id": self.project_id,
            "stage": stage_id,
            "reason": reason,
            "stages_completed": len(self.state.get("stage_results", {})),
            "pipeline_managed": True,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
        except RuntimeError as exc:
            logger.debug("Skipping pipeline failed event; no running event loop: {}", exc)
        self._on_became_terminal()

    def _emit_pipeline_complete(self):
        import asyncio
        # Close the CEO root in the task tree so the UI's
        # "project complete" affordance fires HERE — at the end of the
        # pipeline — instead of at the legacy EA-anchor completion point
        # (which mis-fired after Stage 1).
        self._mark_ceo_root_finished()
        payload = {
            "type": "pipeline_complete",
            "project_id": self.project_id,
            "stages_completed": len(self.state.get("stage_results", {})),
            "pipeline_managed": True,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
        except RuntimeError as exc:
            logger.debug("Skipping pipeline complete event; no running event loop: {}", exc)
        self._on_became_terminal()

    def _mark_ceo_root_finished(self) -> None:
        """On pipeline completion, walk the CEO root through legal status
        transitions to FINISHED so downstream consumers (project archive,
        frontend completion banner) see the project as closed.

        Status transitions are validated against ``VALID_TRANSITIONS`` per
        step. If any step is illegal (e.g. the root is in BLOCKED/FAILED/
        CANCELLED), the method logs a warning and bails — the caller
        should not assume the root reached FINISHED.
        """
        from onemancompany.core.task_tree import get_tree
        from onemancompany.core.task_lifecycle import (
            NodeType, TaskPhase, can_transition,
        )
        from onemancompany.core.config import TASK_TREE_FILENAME

        tree_path = Path(self.project_dir) / TASK_TREE_FILENAME
        if not tree_path.exists():
            return
        try:
            tree = get_tree(tree_path, project_id=self.project_id)
            root = tree.get_node(tree.root_id) if tree.root_id else None
            if not root or root.node_type != NodeType.CEO_PROMPT:
                return
            if root.status == TaskPhase.FINISHED.value:
                return  # already terminal — idempotent

            # FAILED/CANCELLED roots are explicitly out of scope: the project
            # was marked failed/cancelled elsewhere (e.g. vessel root-failed
            # path), so finalizing it as a completed pipeline would
            # contradict that decision. Walking FAILED → PROCESSING → ...
            # → FINISHED is technically legal under VALID_TRANSITIONS, but
            # semantically wrong; refuse explicitly.
            if root.status in (TaskPhase.FAILED.value, TaskPhase.CANCELLED.value):
                logger.warning(
                    "[PIPELINE] Refusing to finalize CEO root {} from {} — pipeline completion conflicts with terminal failure/cancellation",
                    root.id, root.status,
                )
                return

            # Walk PROCESSING → COMPLETED → ACCEPTED → FINISHED, validating
            # each step. Skip steps the node is already past.
            target_chain = [
                TaskPhase.PROCESSING,
                TaskPhase.COMPLETED,
                TaskPhase.ACCEPTED,
                TaskPhase.FINISHED,
            ]
            for target in target_chain:
                if root.status == target.value:
                    continue
                current = TaskPhase(root.status)
                if not can_transition(current, target):
                    logger.warning(
                        "[PIPELINE] Cannot finalize CEO root {}: illegal transition {} → {} (skipping rest)",
                        root.id, current.value, target.value,
                    )
                    return
                root.set_status(target)

            # Synchronous save here on purpose: pipeline completion is a
            # rare, ordering-critical event. Async save would let external
            # readers see a stale tree between the in-memory mutation and
            # the background flush.
            tree.save(tree_path)
            logger.info(
                "[PIPELINE] Marked CEO root {} → FINISHED on pipeline completion",
                root.id,
            )
        except Exception as exc:  # pragma: no cover — defensive logging
            logger.warning("[PIPELINE] Failed to finalize CEO root on completion: {}", exc)
