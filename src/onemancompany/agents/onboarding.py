"""Employee onboarding — code-driven hire flow.

Standalone functions for creating employees, setting up profiles,
copying talent assets, generating nicknames, and registering agent loops.
Called by routes.py (talent market hire) and hr_agent.py (_apply_results).
"""

from __future__ import annotations

import importlib.util
import json as _json
import random
import shutil
import subprocess
from pathlib import Path

import yaml

from loguru import logger

from onemancompany.core.config import (
    AGENT_DIR_NAME,
    DEFAULT_TOOL_PERMISSIONS,
    DEFAULT_TOOL_PERMISSIONS_FALLBACK,
    DEFAULT_DEPARTMENT,
    EMPLOYEES_DIR,
    HR_ID,
    open_utf,
    MANIFEST_FILENAME,
    MANIFEST_YAML_FILENAME,
    PROFILE_FILENAME,
    PROMPTS_DIR_NAME,
    PROJECT_YAML_FILENAME,
    ROLE_DEPARTMENT_MAP,
    SOUL_FILENAME,
    STATUS_IDLE,
    TALENT_PERSONA_FILENAME,
    TOOL_YAML_FILENAME,
    TOOLS_DIR,
    LAUNCH_SH_FILENAME,
    VESSEL_DIR_NAME,
    VESSEL_YAML_FILENAME,
    WORKSPACE_DIR_NAME,
    EmployeeConfig,
    ensure_employee_dir,
    settings,
    read_text_utf,
    write_text_utf,
)
from onemancompany.core import store as _store
from onemancompany.core.models import EventType, HostingMode
from onemancompany.core.events import CompanyEvent, event_bus
from onemancompany.core.layout import (
    compute_layout,
    get_next_desk_for_department,
    persist_all_desk_positions,
)
from onemancompany.core.state import Employee, company_state, make_title

# ---------------------------------------------------------------------------
# Single-file constants — filenames used during onboarding/talent installation
# ---------------------------------------------------------------------------
SKILL_FILENAME = "SKILL.md"
CLAUDE_MD_FILENAME = "CLAUDE.md"
CONNECTION_JSON_FILENAME = "connection.json"
LAUNCH_SCRIPT = LAUNCH_SH_FILENAME
HEARTBEAT_SCRIPT = "heartbeat.sh"

# Default skills injected for every new employee
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "default_skills"
_DEFAULT_SKILL_NAMES = ["task_lifecycle"]
# EA-only skills injected during founding team setup
_EA_SKILL_NAMES = ["project-brainstorming"]
# Profile-skill → required-runbook mapping.
# When an employee carries a skill listed here, the corresponding runbook(s)
# from default_skills/ are injected into their skills/ directory so that
# load_skill(<runbook>) can resolve at runtime. This is the SSOT for the
# pattern; adding a new convener-style skill is one-line dict edit.
#
# Skills whose runbook ships with the talent itself (cloned from the
# Talent Market) are NOT listed here — the talent-clone step in
# `add_or_update_employee` already copies them. Currently five talents
# follow this pattern (resolved via talent_market.onboard() → clone):
#   - methodology_designer → https://github.com/YihangChen9/methodology-designer
#   - experiment_designer  → https://github.com/YihangChen9/experiment-designer
#   - result_analyst       → https://github.com/YihangChen9/result-analyst
#   - code_implementer     → https://github.com/YihangChen9/experiment-team
#   - experiment_runner    → https://github.com/YihangChen9/experiment-team
# Only runbooks that still live in `default_skills/` and need to be
# injected onto talents that don't bundle them belong in this map. Right
# now only the adversarial-reviewer critic stack (used by the
# `adversarial_review` employee) is still in `default_skills/`.
_SKILL_REQUIRED_RUNBOOKS: dict[str, list[str]] = {
    "adversarial_review": [
        # Shared grading vocabulary loaded first by each critic via
        # load_skill("ccf-review-core") — single source of truth for the
        # grounding rule, Findings schema, confidence scale, and decision
        # conventions. Must be injected alongside the critics so the
        # load_skill call resolves.
        "ccf-review-core",
        "methodology-quality-critic",
        "experiment-quality-critic",
        "result-quality-critic",
    ],
    # Stage 4 (methodology) renders the headline framework figure via
    # nano banana. Stage 8 (paper) REUSES the Stage 4 PNG — it does not
    # regenerate (per the Stage 8 dispatch desc in pipeline_engine.py
    # which hard-pins "Do NOT call paper-framework-figure again").
    # So only methodology_designer needs the runbook injected;
    # paper_writer just references stage4_framework_figure.png by path.
    "methodology_designer": ["paper-framework-figure"],
}


# ---------------------------------------------------------------------------
# Nickname generation
# ---------------------------------------------------------------------------

def _get_existing_nicknames() -> set[str]:
    """Collect all nicknames in use by current and ex-employees."""
    from onemancompany.core.store import load_all_employees, load_ex_employees
    nicknames: set[str] = set()
    for edata in load_all_employees().values():
        nn = edata.get("nickname", "")
        if nn:
            nicknames.add(nn)
    for edata in load_ex_employees().values():
        nn = edata.get("nickname", "")
        if nn:
            nicknames.add(nn)
    return nicknames


_NICKNAMES_FILE = Path(__file__).resolve().parents[3] / "company" / "human_resource" / "nicknames.txt"


def _load_nickname_pool() -> list[str]:
    """Load the wuxia nickname pool from company/human_resource/nicknames.txt.

    Reads from disk every call (磁盘即唯一真相源). The file is small (~1000 lines)
    so there is no measurable overhead.
    """
    # Runtime data dir takes priority (user may have customised it there)
    from onemancompany.core.config import DATA_ROOT
    runtime_file = DATA_ROOT / "company" / "human_resource" / "nicknames.txt"
    src = runtime_file if runtime_file.exists() else _NICKNAMES_FILE

    if src.exists():
        pool = [
            line.strip() for line in read_text_utf(src).splitlines() if line.strip()
        ]
        logger.debug("Loaded {} nicknames from {}", len(pool), src)
        return pool

    logger.warning("Nickname file not found at {}; using built-in fallback", src)
    return []


def _pick_nickname(char_count: int, existing: set[str]) -> str:
    """Pick a random wuxia nickname from the pool, avoiding collisions."""
    import random
    pool = [n for n in _load_nickname_pool() if len(n) == char_count and n not in existing]
    if pool:
        return random.choice(pool)
    # Exhausted pool — generate from random wuxia chars
    wuxia_parts = "风云雷电霜雪星月剑刀枪棍龙虎鹤凤松竹梅兰"
    for _ in range(50):
        candidate = "".join(random.choices(wuxia_parts, k=char_count))
        if candidate not in existing:
            return candidate
    return ""


async def generate_nickname(name: str, role: str, is_founding: bool = False) -> str:
    """Assign a wuxia-themed Chinese nickname (花名) for an employee.

    Founding employees (level 4) get 3-character nicknames.
    Normal employees (level 1-3) get 2-character nicknames.
    All nicknames must be unique across all current and ex-employees.

    Nicknames are drawn from company/human_resource/nicknames.txt (replaceable
    by the user). No LLM calls — instant and reliable.
    """
    char_count = 3 if is_founding else 2
    existing = _get_existing_nicknames()
    nickname = _pick_nickname(char_count, existing)
    if nickname:
        logger.info("Assigned nickname '{}' to {} (from pool)", nickname, name)
    else:
        logger.warning("Could not find a unique {}-char nickname for {}", char_count, name)
    return nickname


# ---------------------------------------------------------------------------
# Tool user registration (allowed_users in company/assets/tools/*/tool.yaml)
# ---------------------------------------------------------------------------

def _update_tool_allowed_users(tool_name: str, employee_id: str, *, add: bool) -> None:
    """Add or remove *employee_id* from a central tool's ``allowed_users`` list.

    Employee-brought tools are personal — only the owning employee may use them.
    This function maintains the whitelist in ``tool.yaml``.
    """
    tool_yaml = TOOLS_DIR / tool_name / TOOL_YAML_FILENAME
    if not tool_yaml.exists():
        logger.warning("_update_tool_allowed_users: tool.yaml not found for '{}', skipping", tool_name)
        return
    with open_utf(tool_yaml) as f:
        data = yaml.safe_load(f) or {}
    allowed: list = data.get("allowed_users", [])
    if add:
        if employee_id not in allowed:
            allowed.append(employee_id)
    else:
        if employee_id in allowed:
            allowed.remove(employee_id)
    data["allowed_users"] = allowed
    with open_utf(tool_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def register_tool_user(tool_name: str, employee_id: str) -> None:
    """Grant *employee_id* access to a central LangChain tool."""
    _update_tool_allowed_users(tool_name, employee_id, add=True)


def _ensure_talent_tool_yaml(tool_dir: Path, tool_name: str, talent_dir_name: str) -> None:
    """Generate ``tool.yaml`` for a talent-bundled tool if it's missing.

    Talent packages ship tool dirs with ``manifest.yaml`` (type: python) but no
    ``tool.yaml`` — yet both ``register_tool_user`` (which writes
    ``allowed_users``) and ``tool_registry.load_asset_tools`` (which reads
    ``source_talent`` to decide whether scope enforcement applies) key off
    ``tool.yaml``. Without this step the tool silently registers as a company-
    wide asset visible to every employee (issue #130).

    Mirrors the ``tool.yaml`` emitted by ``install_talent_functions`` for the
    ``functions/`` flow — same ``source_talent`` field, same ``langchain_module``
    convention. ``allowed_users`` is initialised empty here; the per-employee
    grant is appended later by ``register_tool_user``.

    Idempotent: if ``tool.yaml`` already exists (operator-curated tool, or a
    second talent that brought the same subdir), this is a no-op so the
    existing whitelist isn't trampled.
    """
    tool_yaml = tool_dir / TOOL_YAML_FILENAME
    if tool_yaml.exists():
        return
    tool_meta = {
        "id": tool_name,
        "name": tool_name,
        "type": "langchain_module",
        "added_by": f"talent:{talent_dir_name}",
        "source_talent": talent_dir_name,
        "allowed_users": [],
    }
    try:
        with open_utf(tool_yaml, "w") as f:
            yaml.dump(tool_meta, f, default_flow_style=False, allow_unicode=True)
    except OSError as exc:
        # Don't abort onboarding — the tool is still importable; only the
        # scope-enforcement field is missing. Loud-log so the gap surfaces.
        logger.warning(
            "_ensure_talent_tool_yaml: failed to write {}: {} — tool will register as "
            "company-wide asset until tool.yaml is added", tool_yaml, exc,
        )


def unregister_tool_user(tool_name: str, employee_id: str) -> None:
    """Revoke *employee_id*'s access to a central LangChain tool."""
    _update_tool_allowed_users(tool_name, employee_id, add=False)


# ---------------------------------------------------------------------------
# Talent function installation
# ---------------------------------------------------------------------------

def _validate_tool_module(py_path) -> bool:
    """Dry-run import a .py file and check it contains at least one BaseTool instance."""
    from langchain_core.tools import BaseTool

    try:
        spec = importlib.util.spec_from_file_location(
            f"_validate_{py_path.stem}", str(py_path)
        )
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            if isinstance(getattr(mod, attr_name), BaseTool):
                return True
        logger.warning("No BaseTool instances found in %s", py_path)
        return False
    except Exception as exc:
        logger.warning("Failed to validate tool module %s: %s", py_path, exc)
        return False


def install_talent_functions(talent_dir: Path, emp_dir, employee_id: str) -> list[str]:
    """Install talent-brought functions into the central tool registry.

    Reads ``talent_dir/functions/manifest.yaml``, validates each
    declared .py module, copies it to ``company/assets/tools/{name}/``,
    generates ``tool.yaml``, and registers the employee as a user.

    Returns a list of successfully installed function names.
    """
    fn_dir = talent_dir / "functions"
    fn_manifest_path = fn_dir / MANIFEST_YAML_FILENAME
    if not fn_manifest_path.exists():
        return []

    with open_utf(fn_manifest_path) as f:
        raw = yaml.safe_load(f) or {}

    declarations = raw.get("functions", [])
    if not declarations:
        return []

    installed: list[str] = []
    for decl in declarations:
        name = decl.get("name", "")
        if not name:
            continue
        description = decl.get("description", "")
        scope = decl.get("scope", "personal")

        py_src = fn_dir / f"{name}.py"
        if not py_src.exists():
            logger.warning(
                "Function %s declared in %s but %s not found — skipping",
                name, fn_manifest_path, py_src,
            )
            continue

        # Validate the module contains at least one BaseTool
        if not _validate_tool_module(py_src):
            continue

        tool_dir = TOOLS_DIR / name

        if tool_dir.exists():
            # Tool already exists (e.g. another talent brought the same one).
            # Don't overwrite, but still register this employee as a user.
            logger.info(
                "Tool %s already exists in central registry — registering user only", name,
            )
        else:
            # Create central tool directory and copy the .py file
            tool_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(py_src), str(tool_dir / f"{name}.py"))

            # Generate tool.yaml
            tool_meta: dict = {
                "id": name,
                "name": name,
                "description": description,
                "type": "langchain_module",
                "added_by": f"talent:{talent_dir.name}",
                "source_talent": talent_dir.name,
            }
            if scope == "personal":
                tool_meta["allowed_users"] = [employee_id]
            # scope == "company" → omit allowed_users entirely → unrestricted

            with open_utf(tool_dir / TOOL_YAML_FILENAME, "w") as f:
                yaml.dump(tool_meta, f, default_flow_style=False, allow_unicode=True)

        # Ensure the bringing employee has access
        register_tool_user(name, employee_id)
        installed.append(name)

    return installed


# ---------------------------------------------------------------------------
# Agent config installation (agent/manifest.yaml)
# ---------------------------------------------------------------------------

def install_talent_agent_config(talent_dir: Path, emp_dir, employee_id: str) -> dict | None:
    """Install talent agent config (agent/ directory) into the employee folder.

    Copies the entire agent/ directory from the talent package to the employee
    directory, then validates runner and hooks modules if declared.

    Returns the parsed manifest dict on success, or None if no agent config exists.
    """
    agent_dir = talent_dir / AGENT_DIR_NAME
    manifest_path = agent_dir / MANIFEST_YAML_FILENAME
    if not manifest_path.exists():
        return None

    # Copy agent/ directory to employee
    dst_agent_dir = Path(emp_dir) / AGENT_DIR_NAME
    if dst_agent_dir.exists():
        shutil.rmtree(str(dst_agent_dir))
    shutil.copytree(str(agent_dir), str(dst_agent_dir))

    with open_utf(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    # Validate runner module if declared
    runner_cfg = manifest.get("runner", {})
    if runner_cfg:
        mod_name = runner_cfg.get("module", "")
        cls_name = runner_cfg.get("class", "")
        if mod_name and cls_name:
            runner_py = dst_agent_dir / f"{mod_name}.py"
            if runner_py.exists():
                try:
                    from onemancompany.agents.base import BaseAgentRunner
                    spec = importlib.util.spec_from_file_location(
                        f"_validate_runner_{employee_id}", str(runner_py)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        runner_cls = getattr(mod, cls_name, None)
                        if runner_cls is None or not (
                            isinstance(runner_cls, type) and issubclass(runner_cls, BaseAgentRunner)
                        ):
                            logger.warning(
                                "Runner class %s in %s is not a BaseAgentRunner subclass",
                                cls_name, runner_py,
                            )
                except Exception as exc:
                    logger.warning("Failed to validate runner module %s: %s", runner_py, exc)

    # Validate hooks module if declared
    hooks_cfg = manifest.get("hooks", {})
    if hooks_cfg:
        hooks_mod_name = hooks_cfg.get("module", "")
        if hooks_mod_name:
            hooks_py = dst_agent_dir / f"{hooks_mod_name}.py"
            if hooks_py.exists():
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"_validate_hooks_{employee_id}", str(hooks_py)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        for hook_key in ("pre_task", "post_task"):
                            fn_name = hooks_cfg.get(hook_key, "")
                            if fn_name:
                                fn = getattr(mod, fn_name, None)
                                if fn is None or not callable(fn):
                                    logger.warning(
                                        "Hook function %s not found or not callable in %s",
                                        fn_name, hooks_py,
                                    )
                except Exception as exc:
                    logger.warning("Failed to validate hooks module %s: %s", hooks_py, exc)

    logger.info("Installed agent config for employee %s from talent %s", employee_id, talent_dir.name)
    return manifest


def _create_agent_runner(employee_id: str, emp_dir) -> "BaseAgentRunner":
    """Create an agent runner for an employee, using custom runner if configured.

    Search order:
      1. emp_dir/vessel/vessel.yaml runner config
      2. emp_dir/agent/manifest.yaml runner config (backward compat)
      3. Default EmployeeAgent
    """
    from pathlib import Path
    from onemancompany.core.vessel_config import load_vessel_config

    emp_path = Path(emp_dir)
    config = load_vessel_config(emp_path)

    # Try vessel config first, then legacy agent/manifest.yaml
    mod_name = config.runner.module
    cls_name = config.runner.class_name

    if mod_name and cls_name:
        # Look for runner .py in vessel/ first, then agent/
        for search_dir in [emp_path / VESSEL_DIR_NAME, emp_path / AGENT_DIR_NAME]:
            runner_py = search_dir / f"{mod_name}.py"
            if runner_py.exists():
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"emp_runner_{employee_id}_{mod_name}", str(runner_py)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        runner_cls = getattr(mod, cls_name, None)
                        if runner_cls is not None:
                            logger.info(
                                "Using custom runner %s for employee %s", cls_name, employee_id,
                            )
                            return runner_cls(employee_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to load custom runner for %s: %s — falling back to default",
                        employee_id, exc,
                    )

    from onemancompany.agents.base import EmployeeAgent
    return EmployeeAgent(employee_id)


def _load_hooks_from_config(emp_dir) -> dict[str, "Callable"]:
    """Load hook functions from an employee's vessel or agent config.

    Search order:
      1. emp_dir/vessel/vessel.yaml hooks config
      2. emp_dir/agent/manifest.yaml hooks config (backward compat)

    Returns a dict with optional "pre_task" and "post_task" callable entries.
    """
    from pathlib import Path
    from onemancompany.core.vessel_config import load_vessel_config

    emp_path = Path(emp_dir)
    config = load_vessel_config(emp_path)

    hooks_mod_name = config.hooks.module
    if not hooks_mod_name:
        return {}

    # Look for hooks .py in vessel/ first, then agent/
    hooks_py = None
    for search_dir in [emp_path / VESSEL_DIR_NAME, emp_path / AGENT_DIR_NAME]:
        candidate = search_dir / f"{hooks_mod_name}.py"
        if candidate.exists():
            hooks_py = candidate
            break

    if not hooks_py:
        return {}

    result: dict[str, "Callable"] = {}
    try:
        spec = importlib.util.spec_from_file_location(
            f"emp_hooks_{emp_path.name}_{hooks_mod_name}", str(hooks_py)
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for hook_key, fn_name in [("pre_task", config.hooks.pre_task), ("post_task", config.hooks.post_task)]:
                if fn_name:
                    fn = getattr(mod, fn_name, None)
                    if fn and callable(fn):
                        result[hook_key] = fn
    except Exception as exc:
        logger.warning("Failed to load hooks from %s: %s", hooks_py, exc)

    return result


def _register_employee_hooks(employee_id: str, emp_dir) -> None:
    """Load and register hooks for an employee if agent config exists."""
    hooks = _load_hooks_from_config(emp_dir)
    if hooks:
        from onemancompany.core.agent_loop import employee_manager
        employee_manager.register_hooks(employee_id, hooks)
        logger.info("Registered hooks for employee %s: %s", employee_id, list(hooks.keys()))


# ---------------------------------------------------------------------------
# Vessel config installation
# ---------------------------------------------------------------------------

def install_talent_vessel_config(talent_dir: Path, emp_dir, employee_id: str) -> None:
    """Install vessel config (vessel.yaml) into the employee folder.

    Search order:
      1. talent_dir/vessel/vessel.yaml → direct copy
      2. Neither exists → use src/onemancompany/core/default_vessel.yaml

    Also copies vessel/ subdirectories (prompt_sections/, runner .py, hooks .py).
    """
    from onemancompany.core.vessel_config import (
        _load_default_vessel_config,
        save_vessel_config,
    )

    emp_path = Path(emp_dir)
    vessel_dir = emp_path / VESSEL_DIR_NAME

    # Already installed
    if (vessel_dir / VESSEL_YAML_FILENAME).exists():
        return

    # 1. talent has vessel/vessel.yaml
    talent_vessel = talent_dir / VESSEL_DIR_NAME
    talent_vessel_yaml = talent_vessel / VESSEL_YAML_FILENAME
    if talent_vessel_yaml.exists():
        vessel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(talent_vessel_yaml), str(vessel_dir / VESSEL_YAML_FILENAME))

        # Copy prompt_sections/
        ps_src = talent_vessel / "prompt_sections"
        if ps_src.exists() and ps_src.is_dir():
            ps_dst = vessel_dir / "prompt_sections"
            if not ps_dst.exists():
                shutil.copytree(str(ps_src), str(ps_dst))

        # Copy runner/hooks .py files
        with open_utf(talent_vessel_yaml) as f:
            raw = yaml.safe_load(f) or {}
        for key in ("runner", "hooks"):
            mod = (raw.get(key) or {}).get("module", "")
            if mod:
                py_src = talent_vessel / f"{mod}.py"
                if py_src.exists():
                    py_dst = vessel_dir / f"{mod}.py"
                    if not py_dst.exists():
                        shutil.copy2(str(py_src), str(py_dst))
        return

    # 2. Use default
    config = _load_default_vessel_config()
    save_vessel_config(emp_path, config)


# ---------------------------------------------------------------------------
# Talent asset copying
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Talent directory resolution
# ---------------------------------------------------------------------------

# Local talent store: cloned remote talents + offline library + local talent development workspace
from onemancompany.core.config import TALENTS_RUNTIME_DIR as _TALENTS_CLONE_DIR, TALENTS_DIR as _BUILTIN_TALENTS_DIR


def resolve_talent_dir(talent_id: str) -> Path | None:
    """Resolve a talent_id to a filesystem path.

    Searches talents/{id}/ first, then talents/{repo}/{id}/ for
    multi-talent repos cloned as a single directory.
    """
    if not talent_id:
        return None
    # Search runtime (cloned) first, then built-in
    for base in (_TALENTS_CLONE_DIR, _BUILTIN_TALENTS_DIR):
        candidate = base / talent_id
        if candidate.exists():
            return candidate
    return None


async def clone_talent_repo(repo_url: str, talent_id: str) -> Path:
    """Clone a talent repo and flatten sub-talent directories into talents/.

    A repo may contain multiple talents as subdirectories (each with profile.yaml).
    After cloning, those subdirectories are moved up to talents/{sub_id}/ and the
    repo wrapper is removed.

    Returns the local talent directory path for the requested talent_id.
    """
    import asyncio
    import tempfile

    _TALENTS_CLONE_DIR.mkdir(parents=True, exist_ok=True)

    # Clone into a temp dir first to inspect structure
    tmp_clone = Path(tempfile.mkdtemp(prefix="talent_clone_"))
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", repo_url, str(tmp_clone),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, f"git clone {repo_url}", stderr=stderr)

        # Check if repo itself is a single talent (has profile.yaml at root)
        if (tmp_clone / PROFILE_FILENAME).exists():
            dest = _TALENTS_CLONE_DIR / talent_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(tmp_clone), str(dest), ignore=shutil.ignore_patterns(".git"))
        else:
            # Multi-talent repo: each subdir with profile.yaml is a talent.
            # Save by profile.yaml's `id` field (falling back to dir name) so
            # resolve_talent_dir(talent_id) can find it by its canonical ID.
            # Look in repo root AND in a conventional `talent/` subdirectory
            # — many talent repos keep their bundle there to avoid cluttering
            # the project root with research code.
            import yaml as _yaml
            scan_roots = [tmp_clone]
            if (tmp_clone / "talent").is_dir():
                scan_roots.append(tmp_clone / "talent")
            seen_subs: set[Path] = set()
            for scan_root in scan_roots:
                for sub in scan_root.iterdir():
                    if sub in seen_subs:
                        continue
                    seen_subs.add(sub)
                    if sub.is_dir() and (sub / PROFILE_FILENAME).exists():
                        try:
                            profile_id = (_yaml.safe_load(read_text_utf(sub / PROFILE_FILENAME)) or {}).get("id", "")
                        except Exception:
                            profile_id = ""
                        dest_name = profile_id or sub.name
                        dest = _TALENTS_CLONE_DIR / dest_name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(str(sub), str(dest), ignore=shutil.ignore_patterns(".git"))
                        # Also save under dir name as alias if different
                        if profile_id and profile_id != sub.name:
                            alias = _TALENTS_CLONE_DIR / sub.name
                            if not alias.exists():
                                shutil.copytree(str(sub), str(alias), ignore=shutil.ignore_patterns(".git"))
    finally:
        shutil.rmtree(tmp_clone, ignore_errors=True)

    resolved = _TALENTS_CLONE_DIR / talent_id
    return resolved if resolved.exists() else _TALENTS_CLONE_DIR


def _inject_default_skills(
    skills_dir: Path,
    employee_id: str = "",
    employee_skills: list[str] | None = None,
) -> None:
    """Copy/update default skills into the employee's skills folder.

    Always overwrites SKILL.md from the source to pick up frontmatter
    changes (e.g. autoload flag). Preserves employee-specific files
    in the skill directory that don't exist in the source.

    Skill-conditional runbook injection: if the employee carries a skill
    listed in ``_SKILL_REQUIRED_RUNBOOKS``, the mapped runbook names are
    added to the injection list (e.g. ``methodology_designer`` →
    ``methodology-debate-convener``). When ``employee_skills`` is not
    provided, the function reads ``skills_dir.parent / 'profile.yaml'`` to
    discover the employee's skills; if no profile file is present, only the
    universal defaults are injected.
    """
    from onemancompany.core.config import EA_ID
    names = list(_DEFAULT_SKILL_NAMES)
    if employee_id == EA_ID:
        names.extend(_EA_SKILL_NAMES)

    # Discover the employee's skills if not passed explicitly.
    if employee_skills is None:
        profile_path = skills_dir.parent / "profile.yaml"
        if profile_path.exists():
            try:
                import yaml
                with profile_path.open(encoding="utf-8") as f:
                    profile = yaml.safe_load(f) or {}
                employee_skills = list(profile.get("skills") or [])
            except Exception as exc:
                logger.warning(
                    "[_inject_default_skills] failed to read {}: {}", profile_path, exc
                )
                employee_skills = []
        else:
            employee_skills = []

    # Add runbooks required by the employee's skills.
    for skill in employee_skills:
        for runbook in _SKILL_REQUIRED_RUNBOOKS.get(skill, []):
            if runbook not in names:
                names.append(runbook)
    for name in names:
        src = _DEFAULT_SKILLS_DIR / name
        if not src.exists():
            continue
        dst = skills_dir / name
        if not dst.exists():
            shutil.copytree(str(src), str(dst))
        else:
            # Hot-update: mirror every source file (SKILL.md, manifest.yaml,
            # static/, references/, hooks/ …) over the employee's copy so a
            # change to default_skills/ propagates on the next onboard, while
            # leaving employee-only files in place.
            _sync_skill_tree(src, dst)


def _sync_skill_tree(src: Path, dst: Path) -> None:
    """Recursively mirror ``src`` into ``dst``.

    Every file present in ``src`` is copied over its ``dst`` counterpart
    (overwrite), and directories are recursed. Files/dirs that exist only in
    ``dst`` (employee-specific additions) are left untouched — this is a
    one-way overlay, never a delete-sync. ``__pycache__`` and ``.git`` are
    skipped so we don't ship build/VCS noise into a skill folder.
    """
    _SKIP = {"__pycache__", ".git"}
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in _SKIP:
            continue
        dst_child = dst / child.name
        if child.is_dir():
            _sync_skill_tree(child, dst_child)
        elif child.is_file():
            shutil.copy2(str(child), str(dst_child))


# ---------------------------------------------------------------------------
# Cloud skill installation via fastskills MCP
# (overridable hooks so unit tests can mock the MCP transport)
# ---------------------------------------------------------------------------

# Lazy imports — fastskills + MCP libs may not be installed in every dev env,
# so we import on first use. Tests overwrite the two module-level hooks below.
_FastskillsStdioClient = None
_FastskillsClientSession = None


def _fastskills_mcp_clients():
    """Return (stdio_client, ClientSession, StdioServerParameters), importing on demand."""
    global _FastskillsStdioClient, _FastskillsClientSession
    from mcp import ClientSession as _CS, StdioServerParameters as _Params
    from mcp.client.stdio import stdio_client as _stdio
    if _FastskillsStdioClient is None:
        _FastskillsStdioClient = _stdio
    if _FastskillsClientSession is None:
        _FastskillsClientSession = _CS
    return _FastskillsStdioClient, _FastskillsClientSession, _Params


async def _search_cloud_skills_via_fastskills(
    query: str,
    api_key: str = "",
) -> str:
    """Query the SkillsMP cloud catalog via a short-lived ``fastskills`` MCP subprocess.

    company-hosted and omctalent-hosted agents do not have direct MCP access
    (only ``hosting: self`` does, via ClaudeSessionExecutor), so this helper
    bridges the gap: spin up fastskills as a stdio subprocess scoped to a tmp
    skills_dir + workdir, call ``search_cloud_skills``, return the raw output.

    Args:
        query: Search keywords passed straight through to SkillsMP.
        api_key: SkillsMP API key. Defaults to ``settings.skillsmp_api_key``.

    Returns:
        The raw textual output from the fastskills ``search_cloud_skills`` tool.

    Raises:
        RuntimeError: when ``SKILLSMP_API_KEY`` is not configured.
    """
    import os
    import tempfile

    effective_key = api_key or settings.skillsmp_api_key
    if not effective_key:
        raise RuntimeError(
            "SKILLSMP_API_KEY is not configured; cannot search cloud skills."
        )

    stdio_client, ClientSession, StdioServerParameters = _fastskills_mcp_clients()

    with tempfile.TemporaryDirectory(prefix="fastskills-search-") as tmpdir:
        params = StdioServerParameters(
            command="uvx",
            args=["fastskills", "--skills-dir", tmpdir, "--workdir", tmpdir],
            env={**os.environ, "SKILLSMP_API_KEY": effective_key},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_cloud_skills", {"query": query}
                )
                return "".join(getattr(c, "text", "") for c in result.content)


async def _install_cloud_skill_for_employee(
    employee_id: str,
    skill_github_url: str,
    api_key: str = "",
) -> str:
    """Install a SkillsMP cloud skill into an employee's own skills/ directory.

    Spawns ``fastskills`` as a short-lived MCP stdio subprocess scoped to the
    target employee's ``skills/`` and ``workspace/`` dirs, calls
    ``install_cloud_skill``, then closes.

    Args:
        employee_id: Numeric employee id (e.g. "00007").
        skill_github_url: GitHub tree URL of the skill (e.g.
            https://github.com/foo/repo/tree/main/skills/experiment-design).
            SkillsMP URLs are not accepted — fastskills requires the github URL.
        api_key: SkillsMP API key. Defaults to ``settings.skillsmp_api_key``.

    Returns:
        The install_cloud_skill tool's textual output.

    Raises:
        RuntimeError: when SKILLSMP_API_KEY is not configured.
    """
    import os

    effective_key = api_key or settings.skillsmp_api_key
    if not effective_key:
        raise RuntimeError(
            "SKILLSMP_API_KEY is not configured; cannot install cloud skills."
        )

    skills_dir = EMPLOYEES_DIR / employee_id / "skills"
    workdir = EMPLOYEES_DIR / employee_id / WORKSPACE_DIR_NAME
    skills_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    stdio_client, ClientSession, StdioServerParameters = _fastskills_mcp_clients()
    params = StdioServerParameters(
        command="uvx",
        args=["fastskills", "--skills-dir", str(skills_dir), "--workdir", str(workdir)],
        env={**os.environ, "SKILLSMP_API_KEY": effective_key},
    )

    logger.debug(
        "[_install_cloud_skill_for_employee] employee={} url={}",
        employee_id, skill_github_url,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "install_cloud_skill", {"skill_url": skill_github_url}
            )
            return "".join(getattr(c, "text", "") for c in result.content)


def _assign_default_avatar(emp_dir: Path, emp_num: str) -> None:
    """Assign a random default avatar if the employee doesn't already have one."""
    # Check if employee already has a custom avatar
    for ext in (".png", ".jpg", ".jpeg"):
        if (emp_dir / f"avatar{ext}").exists():
            logger.debug("Employee {} already has avatar, skipping default", emp_num)
            return

    from onemancompany.core.config import COMPANY_DIR
    avatars_dir = COMPANY_DIR / "human_resource" / "avatars"
    if not avatars_dir.exists():
        logger.debug("No avatars directory found at {}", avatars_dir)
        return

    avatars = sorted(
        p for p in avatars_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not avatars:
        logger.debug("No avatar files found in {}", avatars_dir)
        return

    idx = int(emp_num) % len(avatars) if emp_num.isdigit() else hash(emp_num) % len(avatars)
    pick = avatars[idx]
    dst = emp_dir / f"avatar{pick.suffix}"
    shutil.copy2(str(pick), str(dst))
    logger.info("Assigned default avatar {} to employee {}", pick.name, emp_num)


def copy_talent_assets(talent_dir: Path, emp_dir) -> None:
    """Copy all files from a talent package into an employee folder.

    Copies the entire talent directory tree first (excluding profile.yaml,
    .git, and Python tool modules), then handles special cases for tools
    (LangChain registration) and personas (prompts/talent_persona.md).
    """
    if not talent_dir.exists():
        return

    # Copy everything from the talent dir that doesn't already exist in emp_dir,
    # skipping files handled specially below and ones that shouldn't transfer.
    _SKIP_FILES = {"profile.yaml", LAUNCH_SCRIPT, HEARTBEAT_SCRIPT}  # scripts handled below with chmod
    _SKIP_DIRS = {".git", "tools", "skills"}  # tools→assets/tools/, skills handled by loop below
    for src in talent_dir.iterdir():
        if src.name in _SKIP_FILES or src.name in _SKIP_DIRS:
            continue
        dst = emp_dir / src.name
        if src.is_dir():
            if not dst.exists():
                shutil.copytree(str(src), str(dst), ignore=shutil.ignore_patterns(".git", "*.pyc", "__pycache__"))
        elif src.is_file() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    talent_skills = talent_dir / "skills"
    if talent_skills.exists():
        emp_skills = emp_dir / "skills"
        emp_skills.mkdir(exist_ok=True)
        for entry in talent_skills.iterdir():
            if entry.is_dir() and (entry / SKILL_FILENAME).exists():
                # Folder-based skill: copy entire folder
                dst_dir = emp_skills / entry.name
                if not dst_dir.exists():
                    shutil.copytree(str(entry), str(dst_dir))
            elif entry.is_file() and entry.suffix == ".md":
                # Legacy plain .md: convert to folder/SKILL.md
                dst_dir = emp_skills / entry.stem
                dst_dir.mkdir(exist_ok=True)
                dst_file = dst_dir / SKILL_FILENAME
                if not dst_file.exists():
                    shutil.copy2(str(entry), str(dst_file))

    talent_tools = talent_dir / "tools"
    if talent_tools.exists():
        employee_id = emp_dir.name
        manifest = talent_tools / MANIFEST_YAML_FILENAME
        custom_tools: list[str] = []
        if manifest.exists():
            with open_utf(manifest) as f:
                mdata = yaml.safe_load(f) or {}
            custom_tools = mdata.get("custom_tools", [])

        # Move each named tool subdir to assets/tools/ if not already there,
        # then register the employee. Do NOT keep a local tools/ folder.
        #
        # Talent-bundled tool dirs ship with ``manifest.yaml`` (type: python)
        # but no ``tool.yaml``. ``register_tool_user`` and ``load_asset_tools``
        # both key off ``tool.yaml`` (the ``source_talent`` field decides whether
        # ``_is_allowed`` enforces ``allowed_users`` or treats it as a company-
        # wide asset). Without a generated ``tool.yaml`` the tool silently
        # registers as a company asset visible to every employee, defeating the
        # whole "tool follows the talent that brought it" model. So we generate
        # one alongside the copy, mirroring what ``install_talent_functions``
        # already does for the ``functions/manifest.yaml`` flow (issue #130).
        for entry in talent_tools.iterdir():
            if entry.name == "manifest.yaml":
                continue
            if entry.is_dir():
                dst_tool = TOOLS_DIR / entry.name
                newly_installed = not dst_tool.exists()
                if newly_installed:
                    shutil.copytree(str(entry), str(dst_tool), ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
                    logger.info("Installed tool '{}' from talent {} to assets/tools/", entry.name, talent_dir.name)
                _ensure_talent_tool_yaml(dst_tool, entry.name, talent_dir.name)
                register_tool_user(entry.name, employee_id)
            elif entry.is_file() and entry.suffix not in (".py", ".pyc"):
                # Loose non-Python files (e.g. config.yaml) stay in emp/tools/
                emp_tools = emp_dir / "tools"
                emp_tools.mkdir(exist_ok=True)
                dst = emp_tools / entry.name
                if not dst.exists():
                    shutil.copy2(str(entry), str(dst))

        # Register employee for any custom_tools listed in manifest.yaml by name
        for tool_name in custom_tools:
            register_tool_user(tool_name, employee_id)

    # Copy talent persona — prefer prompts/talent_persona.md from repo,
    # fall back to system_prompt_template from profile.yaml
    prompts_dir = emp_dir / PROMPTS_DIR_NAME
    talent_persona_src = talent_dir / PROMPTS_DIR_NAME / TALENT_PERSONA_FILENAME
    if talent_persona_src.exists():
        prompts_dir.mkdir(exist_ok=True)
        dst_persona = prompts_dir / TALENT_PERSONA_FILENAME
        if not dst_persona.exists():
            shutil.copy2(str(talent_persona_src), str(dst_persona))
    else:
        talent_profile_path = talent_dir / PROFILE_FILENAME
        if talent_profile_path.exists():
            with open_utf(talent_profile_path) as f:
                talent_data = yaml.safe_load(f) or {}
            spt = talent_data.get("system_prompt_template", "")
            if spt and spt.strip():
                prompts_dir.mkdir(exist_ok=True)
                dst_persona = prompts_dir / TALENT_PERSONA_FILENAME
                if not dst_persona.exists():
                    write_text_utf(dst_persona, spt.strip() + "\n")

    # Copy CLAUDE.md for Claude CLI discovery
    dst_claude_md = emp_dir / CLAUDE_MD_FILENAME
    talent_claude_md = talent_dir / CLAUDE_MD_FILENAME
    if talent_claude_md.exists():
        if not dst_claude_md.exists():
            shutil.copy2(str(talent_claude_md), str(dst_claude_md))

    # Generate CLAUDE.md if talent didn't provide one (self-hosted employees need it)
    if not dst_claude_md.exists():
        from onemancompany.core.config import EMPLOYEES_DIR as _EMP_DIR
        emp_id = emp_dir.name
        profile_path = _EMP_DIR / emp_id / "profile.yaml"
        claude_md_content = (
            f"# Employee {emp_id}\n\n"
            f"You are an employee of One Man Company.\n"
            f"Your profile is at: {profile_path}\n"
            f"Your employee directory is: {emp_dir}\n\n"
            f"## Important\n"
            f"- All company data is stored on the filesystem. There is no database.\n"
            f"- Read your task description carefully — it contains your role, context, and acceptance criteria.\n"
            f"- Save all outputs to the project workspace path specified in your task.\n"
            f"- Do NOT loop or re-analyze. Produce output, verify once, then finish.\n"
        )
        write_text_utf(dst_claude_md, claude_md_content)

    # Copy manifest.json (frontend UI config — OAuth buttons, settings sections)
    talent_manifest_json = talent_dir / MANIFEST_FILENAME
    if talent_manifest_json.exists():
        dst_manifest_json = emp_dir / MANIFEST_FILENAME
        if not dst_manifest_json.exists():
            shutil.copy2(str(talent_manifest_json), str(dst_manifest_json))

    # Copy launch.sh / heartbeat.sh for self-hosted employees
    for script_name in (LAUNCH_SCRIPT, HEARTBEAT_SCRIPT):
        talent_script = talent_dir / script_name
        if talent_script.exists():
            dst_script = emp_dir / script_name
            if not dst_script.exists():
                shutil.copy2(str(talent_script), str(dst_script))
                dst_script.chmod(dst_script.stat().st_mode | 0o755)

    # Install agent config (agent/manifest.yaml + prompts, hooks, runner)
    install_talent_agent_config(talent_dir, emp_dir, emp_dir.name)

    # Install vessel config (vessel/vessel.yaml — uses default if talent has none)
    install_talent_vessel_config(talent_dir, emp_dir, emp_dir.name)

    # Install talent-brought functions into central registry
    employee_id = emp_dir.name
    installed = install_talent_functions(talent_dir, emp_dir, employee_id)
    if installed:
        # Append to employee's tools/manifest.yaml custom_tools
        emp_tools = emp_dir / "tools"
        emp_tools.mkdir(exist_ok=True)
        emp_manifest = emp_tools / MANIFEST_YAML_FILENAME
        if emp_manifest.exists():
            with open_utf(emp_manifest) as f:
                emp_mdata = yaml.safe_load(f) or {}
        else:
            emp_mdata = {"builtin_tools": [], "custom_tools": []}
        existing = emp_mdata.get("custom_tools", [])
        for fn in installed:
            if fn not in existing:
                existing.append(fn)
        emp_mdata["custom_tools"] = existing
        with open_utf(emp_manifest, "w") as f:
            yaml.dump(emp_mdata, f, allow_unicode=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# Core hire execution
# ---------------------------------------------------------------------------

async def execute_hire(
    name: str,
    nickname: str,
    role: str,
    skills: list[str],
    *,
    talent_id: str = "",
    talent_dir: Path | None = None,
    llm_model: str = "",
    temperature: float = 0.7,
    image_model: str = "",
    api_provider: str = "openrouter",
    hosting: str = "company",
    auth_method: str = "api_key",
    sprite: str = "employee_default",
    remote: bool = False,
    department: str = "",
    progress_callback=None,  # async callable(step, message)
) -> Employee:
    """Execute the full hire flow in code — no LLM involved.

    Assigns employee number, department, desk position, permissions,
    creates profile, copies talent assets, generates work principles,
    and registers the agent loop.

    Args:
        talent_dir: Path to the talent directory (cloned from Talent Market).
            If None, no talent assets are copied.
        department: Explicit department override (from COO request).
            If empty, auto-determined from ROLE_DEPARTMENT_MAP.

    Returns the newly created Employee.
    """
    from onemancompany.core.model_costs import compute_salary

    logger.debug("[execute_hire] Starting: name={}, nickname={}, role={}, talent_id={}, hosting={}",
                 name, nickname, role, talent_id, hosting)

    # Resolve talent_dir from talent_id if not explicitly provided
    if talent_dir is None and talent_id:
        talent_dir = resolve_talent_dir(talent_id)
        logger.debug("[execute_hire] Resolved talent_dir={}", talent_dir)

    # Use explicit department if provided (from COO), otherwise auto-assign
    if not department:
        department = ROLE_DEPARTMENT_MAP.get(role, DEFAULT_DEPARTMENT)

    # Desk position
    if remote:
        desk_pos = (-1, -1)
    else:
        desk_pos = get_next_desk_for_department(company_state, department)

    emp_num = company_state.next_employee_number()

    if progress_callback:
        await progress_callback("assigning_id", f"Assigned #{emp_num}")

    # Default permissions
    default_perms = ["company_file_access", "web_search"]
    default_tool_perms = list(DEFAULT_TOOL_PERMISSIONS.get(
        department, DEFAULT_TOOL_PERMISSIONS_FALLBACK
    ))

    # Default model from settings if not specified
    if not llm_model:
        from onemancompany.core.config import load_app_config
        _settings = load_app_config()
        llm_model = _settings.get("default_llm_model", "") if isinstance(_settings, dict) else getattr(_settings, "default_llm_model", "")

    # Salary
    salary = compute_salary(llm_model) if llm_model else 0.0

    # Auto-generate nickname if not provided
    if not nickname:
        nickname = await generate_nickname(name, role, is_founding=False)

    # Random character sprite (1-20) for pixel-art office rendering
    avatar_sprite_num = random.randint(1, 20)

    emp = Employee(
        id=emp_num,
        name=name,
        nickname=nickname,
        level=1,
        department=department,
        role=role,
        skills=skills,
        employee_number=emp_num,
        desk_position=desk_pos,
        sprite=sprite,
        remote=remote,
        permissions=default_perms,
        tool_permissions=default_tool_perms,
        salary_per_1m_tokens=salary,
        probation=True,
        onboarding_completed=False,
        avatar_sprite=avatar_sprite_num,
        talent_id=talent_id,
    )
    logger.debug("[execute_hire] Created Employee object: id={}, nickname={}, dept={}", emp_num, nickname, department)
    # Persist profile via store (single source of truth)
    await _store.save_employee(emp_num, {
        "name": name,
        "nickname": nickname,
        "level": 1,
        "department": department,
        "role": role,
        "skills": skills,
        "employee_number": emp_num,
        "desk_position": list(desk_pos),
        "sprite": sprite,
        "remote": remote,
        "llm_model": llm_model,
        "temperature": temperature,
        "image_model": image_model,
        "permissions": default_perms,
        "tool_permissions": default_tool_perms,
        "salary_per_1m_tokens": salary,
        "api_provider": api_provider,
        "hosting": hosting,
        "auth_method": auth_method,
        "probation": True,
        "onboarding_completed": False,
        "avatar_sprite": avatar_sprite_num,
        "talent_id": talent_id,
    })
    await _store.save_employee_runtime(emp_num, status=STATUS_IDLE)

    if progress_callback:
        await progress_callback("copying_skills", "Copying skill packages...")

    emp_dir = ensure_employee_dir(emp_num)
    skills_dir = emp_dir / "skills"

    # Assign a default avatar if the employee doesn't have one
    _assign_default_avatar(emp_dir, emp_num)

    # Connection config for remote and self-hosted employees
    if remote or hosting == HostingMode.SELF:
        connection = {
            "employee_id": emp_num,
            "company_url": f"http://{settings.host}:{settings.port}",
            "talent_id": talent_id,
        }
        write_text_utf(emp_dir / CONNECTION_JSON_FILENAME, _json.dumps(connection, indent=2, ensure_ascii=False))

    # Copy talent skills + tools
    if talent_dir and talent_dir.exists() and not remote:
        copy_talent_assets(talent_dir, emp_dir)

    # Copy launch.sh for self-hosted employees
    if talent_dir and hosting == HostingMode.SELF:
        talent_launch = talent_dir / LAUNCH_SCRIPT
        if talent_launch.exists():
            dst_launch = emp_dir / LAUNCH_SCRIPT
            if not dst_launch.exists():
                shutil.copy2(str(talent_launch), str(dst_launch))
                dst_launch.chmod(dst_launch.stat().st_mode | 0o111)  # ensure executable

    # Copy heartbeat.sh for employees with custom heartbeat scripts
    if talent_dir:
        talent_hb = talent_dir / HEARTBEAT_SCRIPT
        if talent_hb.exists():
            dst_hb = emp_dir / HEARTBEAT_SCRIPT
            if not dst_hb.exists():
                shutil.copy2(str(talent_hb), str(dst_hb))
                dst_hb.chmod(dst_hb.stat().st_mode | 0o111)

    # Create skill stubs (folder-based)
    for skill_name in skills:
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.exists():
            write_text_utf(skill_file,
                f"---\nname: {skill_name}\ndescription: \"{name}'s {skill_name} skill.\"\n---\n\n"
                f"# {skill_name}\n\n(Auto-created by HR during hiring.)\n")

    # Inject default skills (task_lifecycle; EA also gets project-brainstorming)
    _inject_default_skills(skills_dir, employee_id=emp_num)

    # Create initial SOUL.md in workspace
    workspace_dir = emp_dir / WORKSPACE_DIR_NAME
    workspace_dir.mkdir(exist_ok=True)
    soul_path = workspace_dir / SOUL_FILENAME
    if not soul_path.exists():
        write_text_utf(soul_path,
            f"# {name} ({nickname}) — Personal Knowledge\n\n"
            f"**Role**: {role}\n"
            f"**Department**: {department}\n\n"
            f"## Lessons Learned\n\n"
            f"(Will be updated automatically after each task.)\n")

    # Generate initial work_principles.md (unified location for all hosting modes)
    from onemancompany.core.store import WORK_PRINCIPLES_FILENAME
    wp_path = emp_dir / WORK_PRINCIPLES_FILENAME
    if not wp_path.exists():
        write_text_utf(wp_path,
            f"# {name} ({nickname}) Work Principles\n\n"
            f"**Department**: {department}\n"
            f"**Title**: {make_title(1, role)}\n"
            f"**Level**: Lv.1\n\n"
            f"## Core Principles\n"
            f"1. Complete assigned work diligently and maintain professional standards\n"
            f"2. Actively collaborate with the team and communicate progress promptly\n"
            f"3. Continuously learn and improve professional skills\n"
            f"4. Follow company rules and guidelines\n")

    # Generate standalone run.py for company-hosted employees
    if hosting == HostingMode.COMPANY:
        from onemancompany.core.standalone_runner import generate_run_py
        generate_run_py(emp_dir, name, emp_num)

    # Recompute layout
    compute_layout(company_state)

    if progress_callback:
        await progress_callback("registering_agent", "Registering agent...")

    await _store.append_activity(
        {"type": "employee_hired", "name": name, "nickname": nickname, "role": role}
    )
    hired_data = _store.load_employee(emp_num)
    await event_bus.publish(CompanyEvent(type=EventType.EMPLOYEE_HIRED, payload=hired_data, agent="HR"))

    if progress_callback:
        await progress_callback("completed", f"{name} ({nickname}) onboarded as #{emp_num}")

    logger.debug("[execute_hire] Profile saved, registering agent for {}", emp_num)
    # Register in EmployeeManager (skip remote — they use remote task queue)
    if not remote:
        from onemancompany.core.agent_loop import get_agent_loop, register_and_start_agent, register_self_hosted
        if not get_agent_loop(emp_num):
            if hosting == HostingMode.SELF:
                register_self_hosted(emp_num)
            elif (emp_dir / LAUNCH_SCRIPT).exists():
                # Company-hosted with launch.sh → SubprocessExecutor
                from onemancompany.core.subprocess_executor import SubprocessExecutor
                from onemancompany.core.vessel import employee_manager
                _executor = SubprocessExecutor(emp_num, script_path=str(emp_dir / LAUNCH_SCRIPT))
                employee_manager.register(emp_num, _executor)
            else:
                agent_runner = _create_agent_runner(emp_num, emp_dir)
                await register_and_start_agent(emp_num, agent_runner)
                _register_employee_hooks(emp_num, emp_dir)

    # Trigger onboarding routine as background task
    from onemancompany.core.routine import run_onboarding_routine
    from onemancompany.core.async_utils import spawn_background
    spawn_background(run_onboarding_routine(emp_num))

    return emp
