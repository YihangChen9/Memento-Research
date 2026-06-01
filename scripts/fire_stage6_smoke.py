#!/usr/bin/env python3
"""Fire the Stage 6 smoke fixture against a running OMC server.

Race-injects ``tests/fixtures/stage6_smoke/`` (canned Stage 1-5 outputs +
a tiny pin to ``pypa/sampleproject``) into a fresh project iteration,
then POSTs ``/api/ceo/task`` with ``start_stage=6, end_stage=6,
auto_approve=true``. Polls the pipeline state every 15 s until the
pipeline reaches ``done`` or exhausts retries at the CEO gate.

Goal: exercise the Stage 6 6a → hard-gate → 6b → critic flow end-to-end
in ~5 min without burning hours of GPU on a real benchmark harness.

Usage:
    python scripts/fire_stage6_smoke.py

Assumes:
- An OMC server running on ``http://localhost:8000``.
- A code_implementer employee on the roster (e.g. 00103 Claude Opus 4.7).
- An experiment_runner employee on the roster (e.g. 00025).
- ``OPENROUTER_DIRECT_KEY`` in ``.onemancompany/.env`` so the code-writer
  can reach Claude on real openrouter.ai.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "stage6_smoke"
PROJECTS_DIR = REPO_ROOT / ".onemancompany" / "company" / "business" / "projects"
SERVER_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 15
TIMEOUT_MINUTES = 30


def _fire() -> str:
    """POST the smoke task; return the new project_id."""
    resp = requests.post(
        f"{SERVER_URL}/api/ceo/task",
        data={
            "task": (
                "Stage 6 smoke fixture: tiny regex-vs-regex benchmark on "
                "pypa/sampleproject — exercises 6a → hard-gate → 6b → critic "
                "in under 5 minutes."
            ),
            "start_stage": "6",
            "end_stage": "6",
            "auto_approve": "true",
            "mode": "standard",
        },
        timeout=30,
    )
    resp.raise_for_status()
    pid = resp.json()["project_id"]
    print(f"[fire] new project_id = {pid}")
    return pid


def _race_inject(pid: str) -> Path:
    """Copy fixture files into the new iter_001 before Stage 6a dispatches."""
    iter_dir = PROJECTS_DIR / pid / "iterations" / "iter_001"
    iter_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(FIXTURE_DIR.iterdir()):
        if src.is_file():
            shutil.copy(src, iter_dir / src.name)
    print(f"[fire] race-injected {len(list(FIXTURE_DIR.iterdir()))} fixture files → {iter_dir}")
    return iter_dir


def _snapshot(iter_dir: Path) -> dict:
    """One-line summary of pipeline_state.yaml + receipt presence."""
    state_path = iter_dir / "pipeline_state.yaml"
    if not state_path.exists():
        return {"phase": "?", "retries": 0, "receipt": False, "upstream_commits": 0}
    try:
        state = yaml.safe_load(state_path.read_text()) or {}
    except yaml.YAMLError:
        return {"phase": "?yaml-error?", "retries": 0, "receipt": False, "upstream_commits": 0}
    receipt_path = iter_dir / "stage6_implementation_receipt.md"
    receipt_present = receipt_path.exists() and receipt_path.stat().st_size >= 200
    upstream_git = iter_dir / "upstream" / ".git"
    if upstream_git.exists():
        import subprocess
        try:
            log = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=iter_dir / "upstream",
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            commits = len(log.splitlines())
        except (subprocess.SubprocessError, OSError):
            commits = -1
    else:
        commits = 0
    return {
        "phase": state.get("phase", "?"),
        "retries": state.get("retries", 0),
        "stage": state.get("current_stage", "?"),
        "receipt": receipt_present,
        "upstream_commits": commits,
    }


def _poll_until_terminal(iter_dir: Path) -> dict:
    """Block until pipeline reaches a terminal state or times out."""
    deadline = time.time() + TIMEOUT_MINUTES * 60
    last_phase = None
    while time.time() < deadline:
        snap = _snapshot(iter_dir)
        ts = time.strftime("%H:%M:%S")
        print(
            f"  [{ts}] phase={snap['phase']} stage={snap['stage']} "
            f"retries={snap['retries']} receipt={'Y' if snap['receipt'] else 'N'} "
            f"upstream_commits={snap['upstream_commits']}"
        )
        # Terminal states
        if snap["phase"] in ("done", "failed"):
            return snap
        # Treat "gate" as terminal only after retries are exhausted (auto_approve
        # should keep the pipeline moving past PASS gates automatically; if we
        # land at gate it means a hard failure was held for CEO).
        if snap["phase"] == "gate" and last_phase == "gate" and snap["retries"] >= 1:
            print("  → held at gate (likely hard-gate or critic REJECT exhausted)")
            return snap
        last_phase = snap["phase"]
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"  → TIMEOUT after {TIMEOUT_MINUTES} min")
    return _snapshot(iter_dir)


def _summarise(iter_dir: Path, final: dict) -> int:
    """Print final acceptance-criteria summary; return shell exit code."""
    print()
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"  phase             = {final['phase']}")
    print(f"  retries           = {final['retries']}")
    print(f"  receipt present   = {final['receipt']}")
    print(f"  upstream commits  = {final['upstream_commits']} (expect ≥ 2: pinned base + Stage 6 adaptation)")

    expected_files = [
        "stage6_implementation_receipt.md",
        "stage6_experimentalist.md",
        "stage6_gate_review.md",
    ]
    print("\n  expected artifacts:")
    missing = 0
    for name in expected_files:
        p = iter_dir / name
        status = "✓" if p.exists() else "✗"
        size = p.stat().st_size if p.exists() else 0
        print(f"    {status} {name}  ({size} B)")
        if not p.exists():
            missing += 1

    success = (final["phase"] == "done") and (missing == 0) and final["receipt"]
    print()
    print("VERDICT:", "✅ SMOKE PASS" if success else "❌ SMOKE FAIL")
    return 0 if success else 1


def main() -> int:
    if not FIXTURE_DIR.exists():
        print(f"FATAL: fixture dir not found at {FIXTURE_DIR}", file=sys.stderr)
        return 2

    try:
        ping = requests.get(f"{SERVER_URL}/api/employees", timeout=5)
        ping.raise_for_status()
    except requests.RequestException as exc:
        print(f"FATAL: OMC server not reachable at {SERVER_URL}: {exc}", file=sys.stderr)
        return 2

    pid = _fire()
    iter_dir = _race_inject(pid)
    final = _poll_until_terminal(iter_dir)
    return _summarise(iter_dir, final)


if __name__ == "__main__":
    sys.exit(main())
