# Stage 4 — Methodology

## 1. Variables

| Symbol | Role | Type | Definition |
|--------|------|------|------------|
| `strategy` | IV (independent) | categorical(2) | `A` (last_number heuristic) or `B` (eval_expression) |
| `problem_id` | IV (control, paired) | int 0..9 | dataset index, identical for both strategies |
| `predicted` | mediator | int | numeric value returned by the strategy |
| `correct` | DV (primary, per-problem) | bool | `predicted == ground_truth` |
| `accuracy_A` | DV (aggregate) | float ∈ [0,1] | mean `correct` over 10 problems under Strategy A |
| `accuracy_B` | DV (aggregate) | float ∈ [0,1] | mean `correct` over 10 problems under Strategy B |
| `diff` | DV (derived) | float ∈ [-1,1] | `accuracy_B − accuracy_A` |

## 2. Dataset construction

Ten GSM8K-shaped problems, hand-curated. Each problem has the form:

```
Q<i>: <natural-language word problem>. Compute: <arithmetic expression> = ?
Ground truth: <integer>
```

The arithmetic expression is **explicit** in the problem text so that
Strategy B has a deterministic substring to parse. This is the key
fixture invariant — it lets the experiment run without an LLM while
still mirroring the GSM8K I/O shape.

Distribution: ~4 single-step problems where Strategy A's
"last-number" heuristic happens to coincide with the ground truth
(so A scores some non-zero accuracy), and ~6 multi-step problems
where the last literal is part of an intermediate computation (so A
fails systematically).

## 3. Algorithm

```python
LAST_NUMBER_PATTERN  = r"(-?\d+)(?!.*\d)"          # last integer in text
EXPRESSION_PATTERN   = r"Compute:\s*([0-9+\-*/()\s]+?)\s*=\s*\?"

def strategy_A(problem_text):
    m = re.search(LAST_NUMBER_PATTERN, problem_text)
    return int(m.group(1)) if m else None

def strategy_B(problem_text):
    m = re.search(EXPRESSION_PATTERN, problem_text)
    if not m:
        return None
    return int(eval(m.group(1)))   # ast.literal_eval is too strict for arithmetic ops

correct_A = sum(strategy_A(p.text) == p.gt for p in DATASET)
correct_B = sum(strategy_B(p.text) == p.gt for p in DATASET)
accuracy_A = correct_A / 10
accuracy_B = correct_B / 10
diff = accuracy_B - accuracy_A
```

**Safety note**: `eval` is invoked here only on strings that pass the
`[0-9+\-*/()\s]+` whitelist regex, so the call site is effectively
restricted to pure arithmetic. Stage 6a may also use `ast.parse` +
manual tree walk if it prefers, but `eval` on a regex-validated
substring is acceptable for a fixture this small.

## 4. Statistical test

Exact paired binomial test on McNemar's discordant cells, one-sided,
α = 0.05. With n=10 the test is borderline-powered, but the expected
gap (≥30 pp) is large enough that we expect discordance to fall
heavily in one direction. The test is included so the deliverable
exercises the same `paired_diff` field shape the critic looks for.

## 5. Output contract (CRITICAL)

`benchmark.py` MUST print exactly one final stdout line in this
format (the runbook's `Step 1b'` smoke-quality gate parses this
envelope):

```
=== RESULT_JSON: {"accuracy_direct": <accuracy_A>, "accuracy_cot": <accuracy_B>, "direct_truncated": 0, "cot_truncated": 0, "n_problems": 10} ===
```

We deliberately reuse the existing `accuracy_direct` / `accuracy_cot`
field names from `experiment-execution-runbook` Step 1b'. Semantically
**`direct` = Strategy A**, **`cot` = Strategy B**. This keeps the
existing engine-side quality gate functional for our fixture without
any engine changes.

## 6. Reproducibility

- Deterministic: regex + `eval` are pure functions of input.
- No external dependencies beyond the Python stdlib (`re`, `json`).
- Pinned upstream commit (see `stage5_codebase_pin.md`).

## 7. Threats to validity

1. **Dataset is hand-curated and tiny (n=10)** — not generalisable to
   real GSM8K; this is a fixture for pipeline smoke-testing.
2. **`eval` is restricted to a regex-whitelisted substring** — safe
   in this fixture's context but would need `ast.parse` + visitor for
   a production deployment.
3. **No real LLM is used** — the "model" is the deterministic
   strategy itself. This is intentional: the fixture's job is to
   validate the *pipeline*, not the *reasoning*.

## 8. Why no framework figure was rendered

This fixture intentionally skips the Stage 4 nano-banana figure
render: a 2-strategy comparison on 10 problems is not visually
interesting. A 1×1 PNG placeholder is committed so downstream code
that expects the file does not crash.
