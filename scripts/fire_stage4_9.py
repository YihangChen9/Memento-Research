#!/usr/bin/env python3
"""Fire a Stage 4→9 end-to-end run against a running OMC server.

Exercises the full back half of the pipeline — methodology design,
experiment design, auto-experiment (real Qwen2.5-7B on H100), result
analysis, paper writing, self-review — with auto_approve so it runs
unattended. The topic is deliberately scoped TINY (1 model, 10 GSM
problems, greedy) so Stage 6 finishes in minutes, not hours.

Primary purpose: validate the #27 hard data gate + #107 producer_b_waiting
+ #106 exhausted-gate fixes don't break a healthy end-to-end run, and
that a real-data run flows through all four gates (6/7/8/9) cleanly.

Usage:
    python scripts/fire_stage4_9.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / ".onemancompany" / "company" / "business" / "projects"
SERVER_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 30
TIMEOUT_MINUTES = 90

TOPIC = (
    "Quick validation study: does chain-of-thought prompting beat direct-answer "
    "prompting on Qwen2.5-7B-Instruct for grade-school arithmetic? "
    "SCOPE THIS TINY AND FAST — this is a pipeline-validation run, not a "
    "publication: exactly ONE model (Qwen2.5-7B-Instruct at "
    "/mnt/data0/hf_models/Qwen/Qwen2.5-7B-Instruct), exactly TEN hand-curated "
    "GSM-style word problems embedded as a constant (do NOT download a full "
    "dataset), greedy decoding (do_sample=False), one H100. Two conditions: "
    "(A) direct — system prompt asks for only the final integer, max_new_tokens=16; "
    "(B) cot — system prompt asks to think step by step then answer, "
    "max_new_tokens=256. Use tokenizer.apply_chat_template. Extract the integer "
    "with a regex (prefer 'answer is N', fall back to last integer). Metric: "
    "paired accuracy; H1 = accuracy_cot - accuracy_direct >= 0.20. Smoke = 3 "
    "problems (<=5 min), full = 10 problems (<=3 min). Pin pypa/sampleproject as "
    "the host repo and add src/sample/benchmark.py. Keep the whole Stage 6 run "
    "under ~5 minutes of GPU."
)


def _fire() -> str:
    resp = requests.post(
        f"{SERVER_URL}/api/ceo/task",
        data={
            "task": TOPIC,
            "start_stage": "4",
            "end_stage": "9",
            "auto_approve": "true",
            "mode": "standard",
            "paper_format": "markdown",
        },
        timeout=30,
    )
    resp.raise_for_status()
    pid = resp.json()["project_id"]
    print(f"[fire] new project_id = {pid}  (Stage 4 -> 9, auto_approve)")
    return pid


def _iter_dir(pid: str) -> Path:
    return PROJECTS_DIR / pid / "iterations" / "iter_001"


def _snapshot(iter_dir: Path) -> dict:
    sp = iter_dir / "pipeline_state.yaml"
    if not sp.exists():
        return {"phase": "?", "stage": "?", "retries": 0}
    try:
        st = yaml.safe_load(sp.read_text()) or {}
    except yaml.YAMLError:
        return {"phase": "?yaml?", "stage": "?", "retries": 0}
    return {
        "phase": st.get("phase", "?"),
        "stage": st.get("current_stage", "?"),
        "retries": st.get("retries", 0),
        "failure_reason": st.get("failure_reason", ""),
        "stage_results": sorted((st.get("stage_results") or {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 99),
    }


def _poll(iter_dir: Path) -> dict:
    deadline = time.time() + TIMEOUT_MINUTES * 60
    last = None
    while time.time() < deadline:
        snap = _snapshot(iter_dir)
        ts = time.strftime("%H:%M:%S")
        line = (f"  [{ts}] stage={snap['stage']} phase={snap['phase']} "
                f"retries={snap['retries']} done={snap['stage_results']}")
        if line != last:
            print(line, flush=True)
            last = line
        if snap["phase"] in ("done", "failed"):
            return snap
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"  -> TIMEOUT after {TIMEOUT_MINUTES} min")
    return _snapshot(iter_dir)


def main() -> int:
    # The first request after idle can spike past a few seconds (GC / event
    # processing right after a prior run), so retry with a generous timeout
    # rather than bailing on a single tight ping.
    last_exc = None
    for attempt in range(5):
        try:
            requests.get(f"{SERVER_URL}/api/employees", timeout=30).raise_for_status()
            last_exc = None
            break
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(3)
    if last_exc is not None:
        print(f"FATAL: server not reachable: {last_exc}", file=sys.stderr)
        return 2
    pid = _fire()
    iter_dir = _iter_dir(pid)
    final = _poll(iter_dir)
    print()
    print("=" * 60)
    print(f"FINAL: stage={final['stage']} phase={final['phase']} "
          f"retries={final['retries']}")
    if final.get("failure_reason"):
        print(f"failure_reason = {final['failure_reason']}")
    print(f"stages with results: {final['stage_results']}")
    print(f"project_id = {pid}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
