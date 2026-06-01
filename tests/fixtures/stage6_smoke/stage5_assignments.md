# Stage 5 — Coordination Assignments

| # | Task | Assignee | Skill | Due | Acceptance criterion |
|---|------|----------|-------|-----|----------------------|
| T0 | Clone upstream `pypa/sampleproject@621e4974ca`, run trivial `pytest -q`, add `src/sample/benchmark.py` with the 10-problem GSM-style DATASET + two strategies + main(--smoke), edit `setup.py` per `stage5_codebase_pin.md` adaptation surface, commit on top of pinned SHA. | code_implementer | code_implementer | day 1 | Upstream cloned cleanly + one `Stage 6 adaptation:` commit on top of pinned SHA + `git status --short` empty + `stage6_implementation_receipt.md` written with `path_taken: pin` and the runnable entrypoint command (`python -m sample.benchmark --smoke`). |
| T1 | Run `python -m sample.benchmark --smoke` once, capture its `=== RESULT_JSON ===` envelope, write `stage6_experimentalist.md` populated with the run command, stdout tail, parsed `accuracy_direct` (Strategy A) / `accuracy_cot` (Strategy B) / `n_problems`, and the H1 verdict (`paired_diff ≥ 0.30`). | experiment_runner | experiment_runner | day 1 | A valid `RESULT_JSON` block appears in the runner report; `accuracy_cot - accuracy_direct ≥ 0.30` (H1 trivially supported by the hand-curated dataset, ~0.6 pp gap expected). |

## Dependencies

T1 depends on T0 (smoke run cannot start until `benchmark.py` exists,
is committed, and is pushed to the remote working dir).

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Session budget exhausted on remote infra | medium | If `fast_query_budget.sh` returns `$0`, the runner reports `BLOCKED: budget` honestly — the engine's hard-gate work in PR #66 ensures this surfaces correctly rather than auto-passing an empty stage. |
| Upstream SHA `621e4974ca…` is unreachable | low | The runbook tells 6a to fall back to the latest reachable commit on `main` (and document the deviation in the receipt). |
| Claude Opus skips Phase 5 (writes patches, forgets receipt) | medium (observed in earlier runs) | Engine hard-gate from PR #66 catches this and retries 6a with an explicit "you skipped the receipt" feedback. |
| `eval()` fails on a malformed `Compute:` expression | low | The fixture's dataset is hand-curated to never produce malformed expressions; if a future expansion introduces them, Strategy B catches `SyntaxError` and counts the row as incorrect (still passes H1 because A doesn't recover either). |
