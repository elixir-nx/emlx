# Stage 32 — `runtime_call` dispatch cache (EXLA-style custom-call reuse)

Status: superseded (partial). Named by
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

**Status: superseded (partial) — user directive, 2026-07-02.** The dispatch
cache mechanism itself was implemented, correctness-tested, and works; but
"a couple of seconds, not tens of minutes" turned out to be the wrong bar to
clear with this architecture. See
[`32a-inline-runtime-call`](32a-inline-runtime-call.md), which replaces
split-and-cache with not-splitting-at-all.

1. **Implemented per the Procedure/advisor sign-off**: `EMLX.dispatch_key/3`
   builds a structural (id-independent) signature of a stage `Expr` —
   `EMLX.Defn.Tree.post_order/1`'s node list with tensor operands replaced by
   their post-order position (not their trace-time `id`), functions reduced
   to `{module, name, arity}`, opaque sub-scopes (`while`/`block`/`fun`
   bodies, not visited by the parent `post_order/1`) recursed into via their
   own self-contained signature. `EMLX.get_or_compile_program/6` now looks
   this key up in a process-lifetime, named public ETS table
   (`:emlx_native_dispatch_cache`, lazily created, idempotent under races)
   instead of Stage 25's original per-`build_native_eval_fn`-closure table —
   unifying with Stage 25's `quant_signature` cache per the stage doc's Open
   Question 3 (cache key is now `{dispatch_key, quant_signature}`).
2. **Found and fixed a real bug in this stage's own new code before it ever
   reached a real model**: `sanitize_key_term/2`'s opaque-scope fallback
   recomputed a shared sub-expression's structural signature from scratch on
   *every* reference to it, with no memoization — the same
   unmemoized-shared-subexpression blowup pattern
   `nx-graph-split-bugreport.md`'s Bug 1 hit in `Nx.Defn.Graph`'s
   `rewrite_subtree`. Fixed with a process-dictionary-scoped memo
   (`id => signature`, live only for one `dispatch_key/3` call). Caught by
   the real-model validation below, not by the unit suite (the unit tests'
   expressions are too small to exhibit the blowup).
3. **Regression tests**: `emlx/test/emlx/native/expr_test.exs`'s new
   `:stage32` describe block (2 tests) — calling the same runtime_call-split
   defn twice with different quantized weights (same shapes) shares one
   cache entry across both calls, and two separately-defined-but-op-for-op-
   identical defns (standing in for "two of Qwen3's 28 attention layers")
   share one cache entry despite tracing to distinct `Expr` ids. Both
   equivalence-tested against `Nx.Defn.Evaluator`. Full `emlx` suite:
   2671 passed (827 doctests, 1844 tests), 0 failed — no regression.
4. **Real-model validation against `validate_qwen3.exs` did not clear the
   acceptance bar, and revealed the bar itself was set wrong.** Even after
   fix #2, a `bb+rewrite` run (`EMLX_QWEN3_MAX_NEW=3`,
   `EMLX_QWEN3_WARMUP_RUNS=1`, local `Qwen3-0.6B-MLX-4bit`) did not
   complete within a 10-minute bound and was killed. This stage's original
   Acceptance criterion ("same order of magnitude as `bb base`/`native`, not
   tens-of-minutes-per-token") was too permissive — **user directive:
   anything larger than a couple of seconds is unacceptable.** Root cause is
   architectural, not a caching-completeness gap: `Nx.Defn.Graph.split`
   fragments the model into ~2 flat stages per attention layer (Stage 31
   Results item 5) *plus* the split/retrace bookkeeping itself scales with
   real-model size (unlike the small synthetic repros both this stage and
   Stage 31 validated against); caching the *compiled artifact* per
   structural key (this stage's charter) does not remove that fragmentation
   or retrace cost, it only avoids re-paying the NIF-compile portion of it
   on a hit. A cold cache still pays real compile cost once per distinct
   structural site, and a real 28-layer model has enough genuine structural
   variation (or enough split-machinery overhead near that scale) to land
   nowhere close to "a couple of seconds."
5. **Deferred: Stage 32a.** The dispatch cache built here is retained (it is
   correct, tested, and strictly beneficial for any stage that *does* get
   split, e.g. a bare `while`'s surrounding flat stages), but it does not by
   itself meet the real bar. Stage 32a takes a different architectural
   approach — make an unrecognized `runtime_call` an **in-graph** compiled
   instruction (no `Nx.Defn.Graph.split`, no host round-trip stage boundary
   at all), mirroring how Stage 10's `EMLX.Fast.*` kernels already fuse into
   the single compiled program. See that stage doc for the design.
