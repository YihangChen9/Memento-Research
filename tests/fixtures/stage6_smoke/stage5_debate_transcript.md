# Stage 5 — Debate Transcript (abbreviated, fixture-only)

This is a fixture record of the Stage 5 debate. The original convener
flow was bypassed for this smoke run (the entire stage 5 directory is
race-injected as a unit), but the critic in Stage 6 expects a debate
transcript file to exist, so we ship one with the minimum sufficient
content.

---

**Convener**: We have a 10-problem GSM-style benchmark and two
extraction strategies. The primary risk is over-claiming
generalisability from such a small n. Anything I'm missing?

**Methodologist (00019)**: Power for n=10 paired is low — McNemar's
exact test wants ≥5 discordant pairs. Given the hand-curated design
we expect ~6 multi-step problems where A fails and B succeeds, so
discordance should be ~6 cells in the (A=wrong, B=right) corner.
That's plenty for the test to behave; the small-n caveat needs to be
explicit in the threats section of Stage 4. Approved.

**Experimentalist (00025)**: No issue on the runner side. The
`--smoke` command is the same as the full command for this
experiment, so I won't have to worry about smoke-vs-full divergence
or runtime budget. The eval finishes in well under a second on CPU.

**Code Writer (00103)**: The adaptation surface is tight (~82 LOC,
2 files). The dataset is the trickiest part — I need to hand-curate
10 problems where (a) at least 4 have last-literal == ground-truth
(so A scores non-zero) and (b) at least 5 are multi-step (so A fails
and B succeeds). The pin file's "Implementation hints" section lists
both invariants — I'll mirror them in the DATASET docstring so it's
audit-able. The main risk for me is forgetting to commit before
writing the receipt — the engine hard-gate from PR #66 catches that,
so I'll be careful.

**Critic (00017)**: The output contract reuses the runbook's
`accuracy_direct` / `accuracy_cot` field names from
experiment-execution-runbook Step 1b'. That's intentional — the
existing smoke quality gate can validate this fixture without engine
changes. The `eval()` call is regex-whitelisted, which I'd normally
flag, but the regex `^[0-9+\-*/()\s]+$` is genuinely restrictive
enough that this is fine for a fixture. Approved.

**Resolution**: Proceed. Document the small-n caveat and the
`eval()`-restriction in Stage 4 § 7 (already done).

---

This transcript is fictional but consistent with the real convener
flow's emitted format. Stage 6's critic only checks for the file's
existence, not its veracity.
