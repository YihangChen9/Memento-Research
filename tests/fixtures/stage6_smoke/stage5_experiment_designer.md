# Stage 5 — Experiment Plan

## 1. Setup

- **Codebase**: pinned in `stage5_codebase_pin.md` (pypa/sampleproject, MIT).
- **New module**: `src/sample/benchmark.py` (~80 LOC) added by Stage 6a.
- **Dataset**: 10-row constant inside `benchmark.py`. Each row is a
  Python dict `{"id": int, "text": "Q...Compute: <expr> = ?", "gt": int}`.
- **Strategies**:
  - **A (last_number)**: `re.search(r"(-?\d+)(?!.*\d)", text)` — capture the LAST integer in the problem text. Acts as a naive baseline.
  - **B (eval_expression)**: `re.search(r"Compute:\s*([0-9+\-*/()\s]+?)\s*=\s*\?", text)`, then `int(eval(expr_substring))`. Acts as the principled approach.
- **Scorer**: `correct = (predicted == gt)`; aggregate to per-strategy
  accuracy over the 10 problems.

## 2. Procedure

```python
def evaluate(strategy_fn):
    correct = 0
    for problem in DATASET:
        try:
            predicted = strategy_fn(problem["text"])
            if predicted == problem["gt"]:
                correct += 1
        except Exception:
            pass  # extraction failure counts as incorrect
    return correct / len(DATASET)

acc_a = evaluate(strategy_a)   # last_number heuristic
acc_b = evaluate(strategy_b)   # eval_expression
diff  = acc_b - acc_a
```

## 3. Smoke vs full

The full experiment runs in <1 s on the 10-row dataset. The `--smoke`
flag exists to satisfy the experiment-execution-runbook contract;
functionally, **smoke == full** here. (If a future version expands the
dataset to 100+ problems, `--smoke` would shrink to a 5-row subset;
for now both run the same 10.)

## 4. Output contract

`benchmark.py` MUST print exactly one final line to stdout in this
format (the runbook's `Step 1b'` smoke-quality gate parses this
envelope):

```
=== RESULT_JSON: {"accuracy_direct": <acc_a>, "accuracy_cot": <acc_b>, "direct_truncated": 0, "cot_truncated": 0, "n_problems": 10} ===
```

Field-name mapping (intentional reuse of existing field names so the
engine's quality gate works unchanged):

| Field | This experiment |
|-------|-----------------|
| `accuracy_direct` | Strategy A (last_number) |
| `accuracy_cot`    | Strategy B (eval_expression) |
| `direct_truncated` | 0 (no LLM, no truncation possible) |
| `cot_truncated`   | 0 (same reason) |
| `n_problems`      | 10 |

## 5. Hypothesis re-statement

H1: `accuracy_cot - accuracy_direct ≥ 0.30` on this fixed 10-row dataset.

## 6. Decision rule

H1 holds iff the runner's emitted `accuracy_cot - accuracy_direct ≥ 0.30`.
For the hand-curated dataset shipped with the fixture, this is
deterministic (expected `paired_diff ≈ 0.6`), so the decision is
trivially "supported".

## 7. Sample size / power

n=10 paired is under-powered for a publication claim, and is
acknowledged in the threats-to-validity section of Stage 4. The
fixture is for pipeline validation, not for a research finding —
the large expected effect (≥0.3 pp gap) is what makes the test
robust despite the tiny n.

## 8. Threats / exclusions (recap from Stage 4 §7)

This is a fixture for pipeline validation. No external generalisability
is claimed. The `eval()` call is restricted to a regex-whitelisted
arithmetic substring.
