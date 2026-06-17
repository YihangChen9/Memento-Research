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
    "Quick validation study: does feature standardization (z-score scaling) "
    "improve classifier accuracy on a small tabular dataset? "
    "THIS IS A CPU-ONLY PIPELINE-VALIDATION RUN, NOT A PUBLICATION — scope it "
    "TINY AND FAST. **CPU ONLY: do NOT use GPU/CUDA, do NOT request a GPU, do "
    "NOT load any LLM or download any model/dataset.** Use scikit-learn's "
    "BUILT-IN Breast Cancer Wisconsin dataset (sklearn.datasets."
    "load_breast_cancer — ships with sklearn, 569 samples, no download). "
    "Compare two conditions on LogisticRegression(max_iter=5000): "
    "(A) raw features; (B) StandardScaler-scaled features (fit the scaler "
    "INSIDE each CV fold to avoid leakage). Evaluation: stratified 5-fold "
    "cross-validation repeated over seeds [0, 1, 2] (n_seeds=3); report the "
    "MEAN and STD of accuracy per condition AND the paired mean difference. "
    "H1 = mean(acc_scaled) - mean(acc_raw) >= 0.01. Emit a RESULT_JSON envelope "
    "with fields: acc_raw_mean, acc_raw_std, acc_scaled_mean, acc_scaled_std, "
    "delta_mean, n_seeds, n_folds. Smoke = seed [0] only (<1 min); full = seeds "
    "[0,1,2] (<2 min) — both run in SECONDS on CPU. Pin pypa/sampleproject as "
    "the host repo and add src/sample/benchmark.py with a --smoke / --seed CLI. "
    "Dependencies: scikit-learn + numpy only (already standard)."
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
