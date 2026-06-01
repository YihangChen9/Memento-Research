# Stage 2 — Literature Survey

## Summary

This fixture references the foundational GSM-arithmetic-benchmark
literature only — enough context for the critic to verify the
research framing is sound, kept short to avoid hallucinated citations.

## Key references

1. **Cobbe, K. et al. *Training Verifiers to Solve Math Word
   Problems.* arXiv:2110.14168 (2021).**
   Introduces GSM8K, the 8,500-problem grade-school math benchmark
   used as the standard reasoning eval for LLMs. Our 10-problem
   smoke fixture is shaped after GSM8K's problem format
   (natural-language problem statement + numeric ground truth).

2. **Wei, J. et al. *Chain-of-Thought Prompting Elicits Reasoning
   in Large Language Models.* NeurIPS (2022).**
   Original CoT-vs-direct-answer comparison on GSM8K. Establishes
   the experimental shape (per-problem accuracy, paired comparison
   of two prompting/reasoning strategies) that our fixture mirrors
   without invoking an actual LLM.

3. **Cobbe, K. *GSM8K canonical answer-extraction regex* (released
   alongside the dataset).**
   The published `####\s*(-?\d+)` extraction pattern serves as the
   reference for "principled extraction" in the GSM literature; our
   Strategy B (expression-evaluation) follows the same shape:
   *parse the structured answer field, do not heuristic the prose*.

## Gap addressed

Comparing answer-extraction strategies *without* the confound of
varying model quality is rare in the literature — most GSM ablations
mix prompting and extraction. This fixture isolates extraction by
holding "model output" constant (the problem text itself) and only
varies the extraction logic. The point is not to make a claim about
GSM, but to provide a deterministic benchmark whose accuracy gap
is large and stable, suitable as a pipeline smoke test.

## Notes

This is a fixture for Stage 6 pipeline validation; literature
coverage is intentionally minimal.
