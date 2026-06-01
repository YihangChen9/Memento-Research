# Stage 5 — Upstream Codebase Pin

## Primary upstream

- **Repository**: `https://github.com/pypa/sampleproject`
- **Commit**: `621e4974ca0708e6a4ddec3e2f5da50d56e6a1bb` (latest stable on `main` at fixture authoring time; Stage 6 may resolve to whichever SHA `git fetch` returns if the exact commit is unreachable — document the deviation in the receipt).
- **License**: MIT
- **Test command**: `python -m pytest -q` (the upstream ships an empty test scaffold; this should exit 0 trivially after install)

## Why this repo

`pypa/sampleproject` is the canonical Python packaging template
(~50 LOC of real code in `src/sample/__init__.py`). It is small,
public, MIT-licensed, dependency-free, and *deliberately empty* —
the perfect host for a tiny new benchmark module added by Stage 6a.

This fixture exists to exercise the Stage 6 pipeline end-to-end in
under five minutes, not to make a research claim. The pin choice
reflects that.

## Adaptation surface (what we change, with exact paths)

| File | Change | Reason | Estimated LOC |
|------|--------|--------|---------------|
| `src/sample/benchmark.py` *(new)* | Implement the GSM-style 10-problem benchmark: `DATASET` constant (10 hand-curated problems each containing a natural-language statement + an explicit `Compute: <expr> = ?` clause + ground-truth integer), `strategy_a` (last-number regex), `strategy_b` (eval the `Compute:` expression), `main(--smoke)` that scores both, computes `paired_diff`, and prints the canonical `=== RESULT_JSON: ... ===` envelope. | The entire experiment. | ~80 |
| `setup.py` | Add `regex-benchmark = sample.benchmark:main` console-script entry under the existing `entry_points={"console_scripts": ...}` block so the smoke runner can invoke `regex-benchmark --smoke` from PATH. | Make the entrypoint discoverable; this matches the `Runnable entrypoint` field of the receipt template. | 2 |

**Total adaptation surface: ~82 LOC across 2 files.**

Files **NOT** to touch (use as-is):
- `src/sample/__init__.py` — keep the package skeleton untouched
- `tests/` — the upstream's empty test scaffold
- `README.md`, `LICENSE`, `pyproject.toml` — upstream metadata
- Any other file in the upstream tree

## Implementation hints for Stage 6a (Claude Opus)

The dataset MUST satisfy three invariants so the experiment is
deterministic and the H1 (≥30 pp gap) holds:

1. **At least 4 of the 10 problems** must have the ground-truth
   answer equal to a literal that appears anywhere in the problem
   text — so Strategy A (last-number) scores non-zero. Example:
   `Q: Sam has 7 apples. How many apples? Compute: 7 = ?` →
   last literal `7` matches ground truth `7`.

2. **At least 5 of the 10 problems** must have a final arithmetic
   expression whose result differs from any single literal in the
   text — so Strategy A fails. Example: `Q: A has 14 apples, gives
   3 to each of 4 kids. How many remain? Compute: 14 - 3*4 = ?` →
   ground truth `2`, last literal `4`, mismatched.

3. **All `Compute:` expressions must be parseable by `eval()` after
   passing a `^[0-9+\-*/()\s]+$` whitelist regex** — so Strategy B
   scores 10/10 (modulo no malformed expressions in the fixture).

Expected outcome: `accuracy_A ≈ 0.4`, `accuracy_B = 1.0`,
`paired_diff = 0.6`. H1 (`diff ≥ 0.3`) trivially holds.

## Smoke command

`python -m sample.benchmark --smoke`

For this experiment the smoke run **is** the full run: it evaluates
both strategies on all 10 problems in well under one second on CPU,
prints the `RESULT_JSON` envelope, and exits 0.

## Vendoring policy

Stage 6 will:

1. `git clone --depth 1 https://github.com/pypa/sampleproject upstream` into the project workspace.
2. `cd upstream && git fetch --depth 1 origin <commit_sha> && git checkout <commit_sha>` to lock the SHA.
3. Add `src/sample/benchmark.py` and edit `setup.py` per the adaptation surface table.
4. `git add -A && git commit -m "Stage 6 adaptation: GSM-style smoke benchmark"` on top of the pinned SHA.
5. Write `stage6_implementation_receipt.md` with `path_taken: pin`, the entrypoint `python -m sample.benchmark --smoke`, and the `git log <pin_sha>..HEAD --oneline` summary.

## Failure mode handling

- Upstream tests are trivial / non-existent — `pretest_status` should
  be `PASSED` on Linux and `SKIPPED_ENV` on macOS (no functional
  difference).
- If `git clone` fails for any reason, the runner should report
  `BLOCKED: upstream_unreachable` and stop — there is no fallback
  pin for this fixture (it's deliberately tied to one tiny public
  scaffold).
