# Stage 3 — Idea & Hypothesis

## Single hypothesis (H1, primary)

**H1**: On a 10-problem hand-curated GSM8K-style benchmark whose
problems each embed an explicit arithmetic expression as a
"Compute: <expr> = ?" sentence, an extraction strategy that parses
and evaluates the expression (**Strategy B**) achieves at least
**30 percentage points** higher accuracy than the naive baseline of
returning the *last numeric token* in the problem text
(**Strategy A**).

## Rationale

GSM8K problems typically state intermediate quantities before the
final answer (e.g. "A has 14 apples, gives 3 to each of 4 children,
how many remain?"). The "last number" heuristic latches onto whichever
literal happens to appear last (often `4` in this example, not the
correct `2`), producing systematically low accuracy on problems with
multi-step arithmetic. Evaluating the parenthesised expression
(`14 - 3 * 4`) yields the correct answer in O(1) per problem with no
NLP.

We expect Strategy B to score ~90% (perfect modulo eval failures
on malformed expressions; the fixture deliberately includes no
malformed problems) and Strategy A to score ~30–50% (matching the
last-literal value happens to be correct for some single-step
problems but fails for the multi-step ones). A 30 pp gap is the
minimum claim that's robustly true for our hand-curated fixture.

## Why a single hypothesis

This fixture's role is to validate the Stage 6 pipeline (6a → hard-gate
→ 6b → critic), not to publish a paper. A single, easily-evaluable,
deterministic claim minimises the implementation surface (~60 LOC for
6a) while still exercising the same data-flow shape (paired accuracy
per strategy → `RESULT_JSON` envelope) the runbook's smoke-quality
gate already validates against. H2/H3 ("does the gap shrink on
distractor-rich problems?", "is the gap larger for multi-step
problems?") would be appropriate for a real study but are deliberately
out of scope to keep total wall-clock under one second.
