# Stage 19 — retire the `Nx.Defn.Evaluator` fallback lane

Status: not started. Depends on Stages 16–18.

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
telemetry — `:raise` is opt-in, used only by their conformance gates. EMLX's
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
   full Bumblebee conformance suite once more post-deletion to confirm
   nothing was silently relying on the deleted path.
6. Update `README.md`: Resolved decision #1 currently states only the
   aspiration ("There is no `:native` flag and no eager-Evaluator fallback
   lane") — add a line confirming this is now enforced in code, pointing at
   this stage. Sweep Stage 09/10/15/`EXPR_NODES.md` for any remaining "→
   Evaluator fallback" language and remove it (superseded by Stages 17–19).

## Acceptance

- `try_native_compile`'s Evaluator-delegation branch is deleted from
  `emlx.ex`.
- `mix test` (full suite, including Bumblebee conformance) is green.
- A regression test proves an unsupported construct raises rather than
  silently falling back.
- `README.md`'s single-mode claim has zero remaining discrepancy between docs
  and code.
