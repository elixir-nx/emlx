# Stage 32 — `runtime_call` dispatch cache (EXLA-style custom-call reuse)

Status: not started. Named by
[`31-runtime-call-split-points`](31-runtime-call-split-points.md)'s
Results, per user directive.

## Why this stage exists

Stage 31 made an *unrecognized* `runtime_call` (any callback that isn't
one of Stage 10's `EMLX.Fast.*` fused kernels — e.g.
`EMLXAxon.native_kv_attn_callback/2`) **correct**: it's handled as a
graph-split point exactly like `while`, closing the `does not yet lower
op :runtime_call` hard-raise that made `bb+rewrite` (`EMLXAxon.rewrite/2`
+ quantized weights) unusable.

It did not make it **fast**. Real-model validation
(`emlx_axon/bench/validate_qwen3.exs`, `bb+rewrite` path) showed a single
decode pass through Qwen3's 28 attention layers (each containing one
`native_kv_attn_callback` split point) taking upwards of 20+ minutes,
still climbing in memory when killed. Root cause (see Stage 31 Results
item 5): `Nx.Defn.Graph.split` + `Nx.Defn.Graph.run` re-splits the whole
surrounding graph and re-compiles every stage from scratch **on every
call** — there is zero reuse of a compiled stage across the 28
structurally-identical layers within one call, or across successive
decode steps. `EMLX.get_or_compile_program/6`'s ETS cache (Stage 25) is
scoped per-stage-per-call (a fresh table each time `build_native_eval_fn`
runs), so it can't help here even in principle.

EXLA solves the analogous problem (calling out to host/custom code from a
compiled XLA computation) with a **custom-call dispatch table**: a custom
call is registered once, keyed by a stable identity (not by call), and
the compiled executable simply invokes the registered handler by that key
on every subsequent run — no re-tracing, no re-compiling the surrounding
graph, just a dispatch. This stage's charter is to build the EMLX
equivalent for `runtime_call` split points.

## Procedure (sketch — refine at stage start)

1. **Stable stage identity.** Today, `Nx.Defn.Graph.split/2` assigns each
   stage a fresh `make_ref()` every call, so there is no way to recognize
   "this is the same shape of split-point stage I already compiled." Need
   a content-addressable key for a stage (e.g. hash of its `Expr` shape +
   the `runtime_call` callback's `{module, function, arity}` + operand
   shapes/types) that's stable across calls and across structurally
   identical layers within one call.
2. **Persistent (cross-call) program cache.** Replace or extend
   `EMLX.get_or_compile_program/6`'s per-call ETS table with a
   process-lifetime (or `:persistent_term`-backed) cache keyed by the
   stable identity from (1), so a stage compiled once for layer 1 is
   reused verbatim for layers 2–28 and for every subsequent decode step,
   as long as shapes match.
3. **Avoid re-tracing/re-splitting entirely on a cache hit.** Ideally the
   *tracing* (`fun.(vars)` → `Nx.Defn.Graph.split`) is also skipped on a
   hit, not just the native compile — tracing cost for a 28-layer model is
   itself non-trivial. This likely needs caching at the `EMLX.__compile__/
   4` / `native_compile/3` level, keyed on something stable across the
   outer `defn`'s repeated invocations (today that's `Nx.Defn.Compiler`'s
   own `key`, but a single top-level `defn` call producing 28 sub-stage
   compiles is one `key` for all 28 — need a finer-grained key per stage).
4. **Validate against `validate_qwen3.exs`.** Concrete acceptance target:
   `bb+rewrite` runs at a tok/s figure in the same ballpark as `bb base`
   (Stage 25, ~26 tok/s) or the hand-written `native` path (~70+ tok/s),
   not tens-of-minutes-per-token.
5. **Regression tests.** Extend Stage 31's `:stage31` tests (or a new
   `:stage32` block) to assert cache-hit behavior explicitly: calling a
   `defn` with a `runtime_call` split point twice (same shapes) compiles
   the split-point stage exactly once.

## Open questions (resolve before/while implementing)

- Does a cache hit require *bit-identical* stage `Expr` structure, or is
  there a coarser notion of "the same kernel call site" (e.g. same
  callback + same shapes, regardless of surrounding graph) that's safe to
  key on?
- How does this interact with `EMLX.CommandQueue`/worker dispatch — is a
  cached compiled program tied to the worker/device it was compiled on,
  same as today's `quant_signature` cache?
- Does this subsume or coexist with Stage 25's `quant_signature`-keyed
  cache (both are "compile once per call-time-derived signature, reuse
  across calls" — could plausibly unify into one caching layer)?

## Acceptance

- `bb+rewrite` in `validate_qwen3.exs` runs end-to-end at a practical
  tok/s (same order of magnitude as `bb base`/`native`), not
  tens-of-minutes-per-token.
- A `runtime_call` split-point stage is provably compiled once and reused
  across structurally-identical call sites (regression test).
- Full `emlx`/`emlx_axon`/`Nx` suites remain green; no regression to
  Stage 31's correctness (`bb+rewrite` still produces coherent,
  Evaluator-equivalent output, just faster).

## Results

(not started)
