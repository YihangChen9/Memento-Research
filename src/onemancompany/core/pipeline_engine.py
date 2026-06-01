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

STAGES = [
    {"id": 1, "skill": "topic_refiner",        "name": "Topic Refinement"},
    {"id": 2, "skill": "literature_surveyor",   "name": "Literature Survey"},
    {"id": 3, "skill": "idea_generator",        "name": "Idea Generation"},
    {"id": 4, "skill": "methodology_designer",  "name": "Methodology Design"},
    {"id": 5, "skill": "experiment_designer",   "name": "Experiment Design"},
    {"id": 6, "skill": "experimentalist",       "name": "Auto Experiment"},
    {"id": 7, "skill": "result_analyst",        "name": "Result Analysis"},
    {"id": 8, "skill": "paper_writer",          "name": "Paper Generation"},
    {"id": 9, "skill": "peer_reviewer",         "name": "Self-Review"},
]

CRITIC_SKILL = "adversarial_review"
MAX_RETRIES = 3

# Canonical default employee per stage, sourced from company/hire_list.json.
# When multiple hired employees share the same skill, the one originating
# from the canonical talent_id wins. Falls back to skill-based lookup if
# the canonical talent is not on the roster.
STAGE_TALENT_DEFAULTS = {
    1: "topic-refiner",
    2: "literature-surveyor",
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
    _active_pipelines[project_id] = engine
    return engine


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
            "current_stage": 1,
            "start_stage": 1,
            "end_stage": 9,
            "prior_context": "",
            "stage_assignments": {},  # stage_id (str) → employee_id override
            "phase": "producer",  # producer | producer_b | critic | gate | done | failed
            "retries": 0,
            "stage_results": {},
            "critic_result": None,
            "active_node_id": None,  # current task node being executed
            "active_employee_id": None,
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
        _save_state(self.project_dir, self.state)

    def _ensure_memory_state(self):
        self.state.setdefault("memory_retrievals", {})
        self.state.setdefault("memory_episodes", {})
        self.state.setdefault("memory_feedback", {})

    def _stage_def(self, stage_id: int = None) -> dict:
        sid = stage_id or self.current_stage
        return STAGES[sid - 1] if 1 <= sid <= 9 else {}

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

    def start(self, start_stage: int = 1, end_stage: int = 9, prior_context: str = "", stage_assignments: dict = None, auto_approve: bool = False):
        """Begin the pipeline from the given stage.

        ``auto_approve`` (headless/unattended mode): when True, every CEO gate
        is advanced automatically — the pipeline runs end-to-end with no human
        confirmation. Used for background full-auto runs.
        """
        self.state["current_stage"] = max(1, min(start_stage, 9))
        self.state["start_stage"] = self.state["current_stage"]
        self.state["end_stage"] = max(self.state["current_stage"], min(end_stage, 9))
        self.state["prior_context"] = prior_context
        self.state["stage_assignments"] = stage_assignments or {}
        self.state["auto_approve"] = bool(auto_approve)
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

    def _dispatch_producer(self, feedback: str = ""):
        """Dispatch the current stage's producer. Uses user assignment if set."""
        stage = self._stage_def()
        # Check if user assigned a specific employee to this stage
        assignments = self.state.get("stage_assignments", {})
        assigned = assignments.get(str(stage["id"]))
        employee_id = assigned if assigned else _find_employee_for_stage(stage["id"], stage["skill"])
        if not employee_id:
            logger.error("[PIPELINE] No employee with skill '{}' for stage {}", stage["skill"], stage["id"])
            self.state["phase"] = "failed"
            self._save()
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
        # Stage 4 (Methodology Design) must run a multi-agent debate before
        # writing the methodology. The convener skill is the runbook.
        if stage["id"] == 4:
            desc += (
                "\n## REQUIRED FIRST STEP\n"
                'Before doing anything else, call load_skill("methodology-debate-convener") '
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
                "\n## REQUIRED FIRST STEP\n"
                'Before doing anything else, call load_skill("code-implementation-runbook") '
                "and follow it. This is Stage 6a (Implementation). The runbook's "
                "Phase 0 walks you through reading stage5_codebase_pin.md, cloning "
                "the upstream repo at the pinned commit, running the upstream test "
                "suite on a clean checkout, applying only the patches the pin's "
                "Adaptation surface table lists, and re-running the tests. Phase 5 "
                "produces stage6_implementation_receipt.md naming the runnable "
                "entrypoint command. ADAPT, do not REWRITE — the upstream pin "
                "exists precisely to avoid from-scratch code. The exception path "
                "(NO USABLE UPSTREAM FOUND in the pin) is allowed but triggers "
                "extra critic scrutiny. The Stage 6b runner depends on your "
                "receipt; do not skip it.\n"
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

    def _dispatch_producer_b(self, feedback: str = ""):
        """Dispatch Stage 6b — the experiment runner. Runs after Stage 6a
        (code_implementer) produces stage6_implementation_receipt.md.
        Only called for stage 6; mirrors _dispatch_producer but targets
        the runner skill and uses the experiment-execution-runbook.
        """
        stage = self._stage_def()
        if stage["id"] != 6:
            logger.error("[PIPELINE] _dispatch_producer_b called for non-Stage-6 stage {}", stage["id"])
            return
        employee_id = _find_stage_6b_employee()
        if not employee_id:
            logger.error("[PIPELINE] No experiment_runner employee for Stage 6b")
            self.state["phase"] = "failed"
            self._save()
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
        # Stage 3 (Idea Generation) is produced by the literature-conflict-graph
        # (aigraph) tool, so its deliverable is a `# Selected Hypotheses` report —
        # NOT the generic idea-generation document. Grade it on that basis.
        if stage["id"] == 3:
            desc += (
                "## STAGE 3 IS A LITERATURE-CONFLICT-GRAPH DELIVERABLE\n"
                "The Stage 3 deliverable is generated by the aigraph "
                "literature-conflict-graph tool. Its expected shape is a "
                "`# Stage 3: Idea Generation — <topic>` heading followed by a "
                "`# Selected Hypotheses` report with `### Anomaly a… —` and "
                "hypothesis items grounded in real claim citations. Hypothesis "
                "items appear as `### h… —` (critic / conflict-explanation ideas) "
                "or `### a…#cr… —` (creator / new-method-proposal ideas); BOTH "
                "are valid. This format is correct and intentional — it is NOT "
                "hand-written prose.\n"
                "PASS when: a topic heading is present, there is a "
                "`# Selected Hypotheses` section, and it contains at least one "
                "hypothesis (`### h…` or `### a…#cr…`) citing claim IDs. Do NOT require the "
                "generic idea-generation sections (evaluation architecture, "
                "method pseudocode, risk tables, outcome scenarios) — those "
                "belong to Stages 4–5, not here. Only REJECT if the report is "
                "empty, has zero hypotheses, or says `_No matches for topic_`.\n\n"
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
        if self.phase in ("producer", "producer_b"):
            stage = self._stage_def()
            # Stub-result gate (#60 fix 2): if the producer returned a
            # placeholder like ``"Executed: bash"`` (the agent runtime's
            # fallback when the LLM produced no text content), treat it
            # as producer failure and retry with explicit feedback — do
            # NOT store as the stage deliverable, where the critic would
            # see only tool names and (under the old default-PASS parser)
            # silently advance. Closes #60 fix #2 / #63 fix #4.
            if self._is_stub_result(result):
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
                    if self.phase == "producer_b":
                        self._dispatch_producer_b(feedback=feedback)
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

            # Producer finished → store result, dispatch critic
            self.state["stage_results"][str(stage["id"])] = result
            self._save()
            logger.info("[PIPELINE] Stage {} producer complete, dispatching critic", stage["id"])
            self._emit_stage_event("stage_reviewing", stage["id"])
            self._dispatch_critic(result)

        elif self.phase == "producer_b":
            # Stage 6b runner finished → store as canonical Stage 6 result,
            # dispatch critic.
            stage = self._stage_def()
            self.state["stage_results"][str(stage["id"])] = result
            self._save()
            logger.info("[PIPELINE] Stage 6b (runner) complete, dispatching critic")
            self._emit_stage_event("stage_reviewing", stage["id"])
            self._dispatch_critic(result)

        elif self.phase == "critic":
            # Critic finished → parse decision
            self.state["critic_result"] = result
            self._save()
            is_pass = self._parse_critic_pass(result)
            confidence = self._parse_confidence(result)

            stage = self._stage_def()

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
                )
                self._save()
                logger.info("[PIPELINE] Stage {} PASSED (confidence={})", stage["id"], confidence)
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
                    )
                    self.state["retries"] = retries + 1
                    self._save()
                    logger.info("[PIPELINE] Stage {} REJECTED (retry {}/{})", stage["id"], retries + 1, MAX_RETRIES)
                    self._emit_stage_event("stage_failed", stage["id"], confidence=confidence)
                    self._dispatch_producer(feedback=result)
                else:
                    self._record_stage_memory(
                        stage,
                        producer_result=self.state["stage_results"].get(str(stage["id"]), ""),
                        critic_result=result,
                        passed=False,
                        confidence=confidence,
                        outcome="critic_reject_exhausted",
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
        stage = self._stage_def()
        current_phase = self.phase

        if current_phase == "critic":
            stored = self.state.get("stage_results", {}).get(str(stage["id"]), "")
            logger.warning(
                "[PIPELINE] Stage {} critic FAILED — auto-passing on stored producer output (len={})",
                stage["id"], len(stored),
            )
            self._on_critic_pass(stored, confidence=None)
            return

        if current_phase not in ("producer", "producer_b"):
            # Should not happen — gate/done/failed phases mean no task is in flight.
            logger.warning(
                "[PIPELINE] on_task_failed called in unexpected phase {} (stage {}); ignoring",
                current_phase, stage["id"],
            )
            return

        truncated = (result or "(no output)").strip()[:600]
        # Differentiate 6a vs 6b failures and ROUTE smoke/impl-bug 6b failures
        # back to 6a (#60 fix 3). Heuristic: if the failure text shows
        # signals that the runner's environment is healthy but the
        # implementation is broken (smoke crash / RESULT_JSON missing /
        # accuracy=0 / KeyError in user code), the right action is to
        # re-dispatch 6a with feedback — retrying 6b on a broken impl
        # just burns infra cycles. Otherwise (infra timeout, 401, etc.)
        # keep the current behaviour: re-dispatch 6b.
        route_target = current_phase
        if current_phase == "producer_b" and self._failure_indicates_impl_bug(result):
            route_target = "producer"
            failure_feedback = (
                "Stage 6b reported a failure that looks like a Stage 6a implementation bug, "
                f"not an infra issue. Re-dispatching Stage 6a to fix it. Failure context:\n{truncated}\n\n"
                "Common impl-bug signals: smoke run crashed, RESULT_JSON missing, accuracy=0, "
                "truncation>50%, KeyError in user code. Fix the code, commit, re-push, re-write receipt."
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
            self.state["phase"] = route_target
            self._save()
            logger.warning(
                "[PIPELINE] Stage {} {} FAILED (retry {}/{}) — re-dispatching {}",
                stage["id"], current_phase, retries + 1, MAX_RETRIES,
                "as 6a (impl-bug route-back, #60 fix 3)" if route_target != current_phase else route_target,
            )
            self._emit_stage_event("stage_failed", stage["id"])
            if route_target == "producer_b":
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

    # ------------------------------------------------------------------
    # Public API — revert to a previous stage with new instructions
    # ------------------------------------------------------------------

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
        stage_def = STAGES[stage - 1]
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
        was_mid_flight = self.phase in ("producer", "producer_b", "critic")
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

    # Signals in a Stage 6b failure result that point to an impl bug in 6a,
    # not a transient infra issue (#60 fix 3). When any of these appears,
    # ``on_task_failed`` routes back to 6a so the next retry rebuilds /
    # commits / re-pushes code rather than re-running the same broken impl
    # on the runner. Order: most-specific first so log messages stay tight.
    _IMPL_BUG_SIGNALS = (
        "SMOKE_FAIL",
        "QUALITY_FAIL",
        "QUALITY_FAIL_ACCURACY_ZERO",
        "QUALITY_FAIL_TRUNCATED",
        "blocked_smoke_failure",
        "blocked_smoke_invalid",
        "NO_RESULT_JSON",
        "NO_METRICS",
        "RESULT_JSON missing",
        "RESULT_JSON: missing",
        "KeyError",
        "TypeError",
        "ImportError when running",
        "ModuleNotFoundError when running",
        "accuracy=0",
        "accuracy: 0",
        "truncation_rate=1",
    )

    @classmethod
    def _failure_indicates_impl_bug(cls, result: str) -> bool:
        """Inspect a Stage 6b failure payload for signals that the bug
        lives in Stage 6a's code, not in the runner's infra. Used by
        ``on_task_failed`` to route the next retry back to 6a (#60 fix 3).

        False-positives are cheaper than false-negatives here: a stray
        keyword in an infra error → wasted 6a retry (~10 min Claude) but
        the hard-gate + critic will keep the code honest. A miss →
        endless 6b retries on broken impl, which is the failure mode
        Anjie reported.
        """
        if not result:
            return False
        text = result if isinstance(result, str) else str(result)
        return any(sig in text for sig in cls._IMPL_BUG_SIGNALS)

    @staticmethod
    def _is_stub_result(result: str) -> bool:
        """A producer/critic ``result`` is a stub when the agent runtime fell
        back to synthesising a summary from tool names (no real text content).
        These show up as ``"Executed: bash"`` / ``"Executed tools: write, read"``
        — see ``agents/base.py:_synthesize_fallback``. Stubs that pass to the
        critic look like work but contain no analysis, and the critic's own
        stub return then defaults to PASS by the old parse logic — closing
        the silent-empty-stage loop #60 / #63 describe.

        Threshold (~300 chars) is empirical: real CCF-A producer outputs are
        kilobytes; stubs are typically 14-200 chars."""
        if not result:
            return True
        stripped = result.strip()
        if len(stripped) < 300 and (
            stripped.startswith("Executed: ")
            or stripped.startswith("Executed tools: ")
        ):
            return True
        return False

    def _parse_critic_pass(self, result: str) -> bool:
        """Parse the critic's PASS/REJECT verdict.

        Robustness improvements (#60 fix 4 / #63 fix 4):
        - If the critic's ``submit_result`` was a stub (e.g. ``"Executed: bash"``
          from a context-window-truncated review), fall back to reading the
          on-disk ``stage{N}_gate_review.md`` the critic was told to write.
        - Support table-format verdicts (``| Decision | PASS |``) as well as
          the conversational ``"Decision: PASS"`` form.
        - Default to REJECT on ambiguity (no PASS, no REJECT signal). The old
          default-to-PASS branch was the auto-approve-empty-stage loophole.
        """
        text = result or ""
        # Stub → reach for the gate-review file on disk.
        if self._is_stub_result(text):
            stage_id = self.current_stage
            gate_review = Path(self.project_dir) / f"stage{stage_id}_gate_review.md"
            if gate_review.exists():
                try:
                    text = gate_review.read_text(encoding="utf-8")
                    logger.info(
                        "[PIPELINE] Critic submit_result was a stub — falling back to {} ({} bytes)",
                        gate_review.name, len(text),
                    )
                except OSError as exc:
                    logger.warning("[PIPELINE] Failed to read {}: {}", gate_review.name, exc)
                    text = result  # restore original
            else:
                logger.warning(
                    "[PIPELINE] Critic submit_result is a stub and no {} file found on disk — defaulting to REJECT",
                    gate_review.name,
                )

        upper = text.upper()
        # Explicit table-format ``| Decision | PASS |`` and ``| **Decision** | **PASS** |``
        # — strip markdown emphasis before checking.
        compact = re.sub(r"[\s|*_]+", " ", upper)
        if " DECISION PASS " in compact:
            return True
        if " DECISION REJECT " in compact:
            return False
        # Conversational form: prefer the FIRST explicit signal.
        if "REJECT" in upper:
            return False
        if "PASS" in upper:
            return True
        # Ambiguous → safer default is REJECT (auto-pass on empty stages was
        # the #63 / #60 root cause).
        logger.warning(
            "[PIPELINE] Critic verdict ambiguous (no PASS/REJECT signal, len={}) — defaulting to REJECT",
            len(text),
        )
        return False

    @staticmethod
    def _parse_confidence(result: str) -> float | None:
        import re
        # Match patterns like "confidence: 0.72" or "Confidence Score: 0.8".
        m = re.search(r'confidence(?:\s+score)?[:\s]*([01]\.?\d*)', result, re.IGNORECASE)
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
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_async(payload))
        except RuntimeError as exc:
            logger.debug("Skipping stage event; no running event loop: {}", exc)

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
        """Unattended-mode gate advance: behaves like a CEO clicking approve."""
        import asyncio
        await asyncio.sleep(0)  # let the gate event flush first
        if self.phase != "gate":
            return
        logger.info(
            "[PIPELINE] AUTO-APPROVE (unattended): advancing gate at stage {} "
            "(exhausted={})", stage_id, exhausted,
        )
        self.on_ceo_approve("")

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
