# Stage 1 — Topic Refinement

**Refined topic**: A 10-problem GSM8K-style arithmetic-reasoning smoke
benchmark comparing two answer-extraction strategies — a naive
"last-number" heuristic vs an expression-evaluation strategy that
parses and evaluates the arithmetic embedded in the problem text.

## Research question

When evaluating a simple arithmetic-word-problem benchmark
(GSM8K-style), how much accuracy do we gain by *actually evaluating
the embedded arithmetic expression* (Strategy B) versus the naive
baseline of *taking the last number in the problem text as the answer*
(Strategy A)?

## Scope

- **In scope**: paired comparison of two pure-Python extraction
  strategies on 10 hand-curated GSM8K-shaped problems. Each problem
  has the ground-truth answer encoded into its arithmetic expression
  (e.g. `Q: A farmer has 14 apples and gives 3 to each of 4
  children. How many remain? Compute: 14 - 3 * 4 = ?`). Both
  strategies score on the same problems. All compute is CPU-only,
  runs in milliseconds.
- **Out of scope**: real LLM inference, multi-step reasoning beyond
  a single arithmetic expression, distractor problems requiring
  natural-language understanding, GSM8K's actual 1,319-problem suite.

## Why this is useful as a smoke test

This topic is deliberately tiny: the full evaluation runs in under
one second on a laptop, the implementation is ~60 LOC of Python, and
no external API or GPU is needed. It exists to exercise the Stage 6
pipeline (6a code-writer → hard-gate → 6b runner → critic) end-to-end
without burning hours of GPU on a real GSM8K eval. The two strategies
mirror the kind of "naive baseline vs principled approach" comparison
that real GSM8K papers run, so the pipeline exercises the same shape
of output (RESULT_JSON with `accuracy_direct` / `accuracy_cot` /
`n_problems`) the runbook's smoke-quality gate already understands.
