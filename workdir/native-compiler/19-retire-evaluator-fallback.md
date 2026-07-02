# Stage 19 — retire the `Nx.Defn.Evaluator` fallback lane

Status: done. Depends on Stages 16–18.

## Why this stage exists

Despite `README.md`'s "single-mode, no fallback" claim (Resolved decision
#1), `emlx.ex`'s `__compile__/4` → `try_native_compile/3`
(`emlx.ex:1326-1405`) still rescues any `ArgumentError` whose message starts
with `"does not yet lower op"` and silently delegates the **whole** `defn` to
`Nx.Defn.Evaluator`. The plan's stated architecture and the actual code have
been out of sync since this lane was added as an incremental-development
safety net. Once Stages 16–18 close every reachable raise path — `fun`/
`optional`/`from_binary` were already non-issues (Stage 16), `block`/linalg
while-in-`default_expr` closed (Stage 17), `token`/`attach_token` resolved
one way or another (Stage 18) — this lane is provably dead code and should be
**deleted**, not hardened or config-gated, so the single-mode claim is
literally true in the code, not just the docs.

(Contrast with Emily's own design: Emily deliberately *keeps* an
`Nx.Defn.Evaluator` fallback as a permanent safety net, with a
`native_fallback: :eval | :raise` option and `[:emily, :compiler, :fallback]`
telemetry — `:raise` is opt-in, used only by their parity gates. EMLX's
resolved decision #1 is stricter than that by design: no fallback lane at
all, ever. Do not import Emily's fallback-with-telemetry model here; that
question was already settled and this stage enforces it.)

## Procedure

1. **Prerequisite check**: Stages 16–18 landed. If Stage 18 ended in a no-go
   that accepted option (a) (token/attach_token stays a permanent hard-raise)
   or (b) (a narrowly-scoped hook-only fallback), adjust this stage's scope
   accordingly — (a) needs no change here since a hard-raise was already the
   design; (b) means "delete the whole-defn fallback lane, replace it with
   the narrower hook-scoped one from Stage 18," not a full deletion.
2. Delete the `:not_supported` branch in `__compile__/4` and the
   `rescue`/`try_native_compile` split; let `try_native_compile`'s
   `ArgumentError` propagate directly (fold `try_native_compile` back into
   `__compile__/4` if the rescue-isolation reason for splitting it out no
   longer applies).
3. Remove the now-unused `Nx.Defn.Evaluator` reference from `emlx.ex` if
   nothing else in the file needs it.
4. Add a regression test asserting that calling `__compile__`/`__jit__` on a
   genuinely unsupported construct raises `ArgumentError` (not a silent
   evaluator run) — guards against reintroduction of a fallback lane.
5. Run the full op-coverage probe (`scripts/expr_op_coverage.exs`) and the
   full Bumblebee parity suite once more post-deletion to confirm
   nothing was silently relying on the deleted path.
6. Update `README.md`: Resolved decision #1 currently states only the
   aspiration ("There is no `:native` flag and no eager-Evaluator fallback
   lane") — add a line confirming this is now enforced in code, pointing at
   this stage. Sweep Stage 09/10/15/`EXPR_NODES.md` for any remaining "→
   Evaluator fallback" language and remove it (superseded by Stages 17–19).

## Acceptance

- `try_native_compile`'s Evaluator-delegation branch is deleted from
  `emlx.ex`.
- `mix test` (full suite, including Bumblebee parity) is green.
- A regression test proves an unsupported construct raises rather than
  silently falling back.
- `README.md`'s single-mode claim has zero remaining discrepancy between docs
  and code.

## Results

**Advisor sign-off (before starting).** Confirmed proceeding was safe despite
`triangular_solve`'s non-default variants still raising `does not yet lower
op` (a real, Stage-17-descoped gap, not closed by Stages 16–18): the deletion
itself, gated by a green full-suite run before and after, is the correct
oracle for whether anything still silently depended on the fallback lane —
no a-priori decision was needed. Advisor also flagged a stale docstring
(`expr.ex:187-189`, still describing the fallback) that the original
Procedure's sweep list (step 6, scoped to Stage 09/10/15/`EXPR_NODES.md`)
would have missed, and recommended `triangular_solve left_side: false` as
the regression-test sentinel (a genuine permanent gap, not a maintenance
trap) rather than an arbitrary "not implemented" stand-in.

**Baseline (before deletion).** `mix test` in `emlx/`: 2564 passed (825
doctests, 1739 tests), 0 failures, 1 excluded (`:large_memory`). `mix test`
in `emlx_axon/`: 7 passed, 2 excluded (`:bumblebee`, `:metal`) — this
includes `test/emlx/qwen3_quantized_test.exs`'s 2 tests (a local-checkpoint,
`compiler: EMLX`-driven Qwen3-0.6B-MLX-4bit greedy-decode parity test,
tagged `:quantized_inference`, not excluded by `test_helper.exs`; it runs as
part of the standard `mix test` in this environment since the checkpoint is
present at `~/models/Qwen3-0.6B-MLX-4bit` — reviewer-caught correction: an
earlier draft of this doc incorrectly described it as separately-targeted
and excluded by default).

**Change.** Deleted `try_native_compile/3`'s `rescue`/`:not_supported`
branch and folded its body into `__compile__/4` (renamed `native_compile/3`
— the rescue-isolation reason for the split no longer applies, since there's
nothing left to isolate: an `ArgumentError` now simply propagates). Removed
the now-dead `split_compiler_opts/1` helper and its `rest_opts` half — with
`Keyword.validate!/2` already restricting `opts` to `@valid_compiler_keys`
before the split ran, `rest_opts` (opts *outside* that list, forwarded to
`Nx.Defn.Evaluator.__compile__/4`) was always `[]` in practice once the
Evaluator branch was gone, so keeping the split would have left dead code
behind it. `__compile__/4`'s unused `key` param is now `_key` (was only
threaded through to the deleted `Nx.Defn.Evaluator.__compile__/4` call).
Fixed the stale docstring at `expr.ex:187-189` (flagged by the advisor) that
described `EMLX.__compile__/4` as catching the `"does not yet lower op"`
message.

**Regression test (Acceptance bullet 3).** Replaced
`test/emlx/native/expr_test.exs`'s `"unsupported op falls back to Evaluator
transparently"` test — which asserted the now-deleted behavior, and was
already misleadingly named even before this stage since `Nx.sum/1` (its
example "unsupported" op) has lowered natively since Stage 04 — with `@tag
:stage19 test "unsupported op raises through the compiler seam (no
Evaluator fallback)"`, which drives `triangular_solve left_side: false`
through `Nx.Defn.jit(f, compiler: EMLX)` (the actual `__compile__/4` seam,
not the lower-level `Expr.lower/2` call the existing Stage 17 test at
`expr_test.exs:3546` already covers) and asserts it raises `ArgumentError`
matching `"does not yet lower op :triangular_solve"`.

**Post-deletion verification.** `mix test` in `emlx/`: 2564 passed (825
doctests, 1739 tests), 0 failures, 1 excluded — identical count to baseline
(net zero: one stale test replaced by one new test). `mix test` in
`emlx_axon/`: 7 passed, 2 excluded — identical to baseline, including the
Qwen3 e2e parity tests above (the closest thing this repo has to a
"Bumblebee parity suite" — reviewer-confirmed there is no literal
`:bumblebee`-tagged test anywhere in the repo, i.e. `test_helper.exs`'s
`exclude: [:bumblebee]` is currently a no-op; and no
`scripts/expr_op_coverage.exs` op-coverage probe exists yet either, despite
both being referenced aspirationally in `README.md`/`EXPR_NODES.md`; neither
gap is closed by this stage, they're pre-existing "to be written" items, not
something Stage 19 regressed) still passing end-to-end with the fallback
lane gone. No hidden dependency on the deleted lane surfaced.

**`triangular_solve` accepted as a permanent hard-raise (Acceptance bullet
1, decided per advisor sign-off).** It joins the cond-branch-local hook
(Stage 18) as the second and only other permanent, by-design "does not yet
lower op"-style raise. Documented explicitly in `README.md` (Resolved
decision #1) and `EXPR_NODES.md` (section K) so it reads as an accepted,
named gap rather than an implicit "still raises."

**Doc sweep (Procedure step 6).** `README.md`: Resolved decision #1 now
states the enforcement is real in code, naming both permanent hard-raises;
the "Known discrepancy" callout above it is updated to past tense
("closed"); Stage 19's checklist box flips to `[x]`; Stage 10's summary line
("prefill RoPE raises → Evaluator fallback") is corrected — that was already
stale by Stage 15's fix, not by this stage, and predates the fallback
deletion. `EXPR_NODES.md` section K's `triangular_solve` line and
`09-blocks-linalg.md`'s results table drop the "→ Evaluator" phrasing.
`10-fast-kernels.md`'s results-table row for RoPE now reflects Stage 15's
fix instead of the pre-Stage-15 state (a doc bug unrelated to this stage,
caught during the sweep). Historical narrative prose in `09-blocks-linalg.md`
/`10-fast-kernels.md`/`14-while-childprogram.md`/`11-bench-regression.md`
describing what the Evaluator fallback *was* at the time those stages ran is
left alone — it's accurate history, not a live claim.

**Full suite, final.** `mix test` in `emlx/`: 2564 passed (825 doctests,
1739 tests), 0 failures. `mix test` in `emlx_axon/`: 7 passed, 2 excluded.
All acceptance bullets met.

**Reviewer sign-off (clean, no blockers).** A fresh reviewer subagent (fed
only the Acceptance criteria + diffs + test output, no reasoning)
independently re-verified against the live repo (re-ran `mix test` in both
`emlx/` and `emlx_axon/`, `mix test --only stage19`, `mix compile
--warnings-as-errors`, `mix format --check-formatted`) and confirmed: the
success path is behaviorally unchanged; `split_compiler_opts`/`rest_opts`
was genuinely dead code once `Keyword.validate!/2` ran first; the new
regression test is a real guard (cross-referenced `EMLX.Backend`'s eager
`triangular_solve` to confirm a reintroduced fallback would silently compute
a wrong-looking-right answer instead of raising, so `assert_raise` would
catch it); and there is no live "→ Evaluator fallback" claim left in the
docs. Flagged two non-blocking doc issues (a "see ... above" pointer that
should have read "below", and this doc's Qwen3-test framing) — both fixed
above.
