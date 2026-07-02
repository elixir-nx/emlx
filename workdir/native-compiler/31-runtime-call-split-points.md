# Stage 31 — `runtime_call` as a graph-split point (bring `bb+rewrite` in scope)

Status: done. Follow-on to
[`25-quantized-dot-full-fix`](25-quantized-dot-full-fix.md), which left
`EMLXAxon.rewrite/2` + quantized weights (`bb+rewrite`) explicitly
out-of-scope: it raised `does not yet lower op :runtime_call for
EMLXAxon.native_kv_attn_callback/2` and Stage 24/25 treated that as a
permanent, by-design carve-out (real host blocking + mutable ETS
side-channel state, "permanently unlowerable").

## Why this stage exists

User directive, superseding that carve-out: **`bb+rewrite` is also in
scope.** Two decisions came out of that:

1. **This stage (31)**: handle an *unrecognized* `:runtime_call` (i.e. any
   `Nx.runtime_call` callback that is not one of the `EMLX.Fast.*` fused
   kernels Stage 10 already lowers in-graph) the same way Stage 08 handles
   `while` — as a **graph-split point**. The split isolates the
   runtime_call into its own stage, driven host-side by
   `Nx.Defn.Graph.run`, with every other stage re-entering `compiler: EMLX`
   normally. This is a correctness fix, not a performance one.
2. **Future (Stage 32, named but not started)**: replace the naive
   split-and-recompile-every-call approach with a real dispatch system —
   compile each distinct `runtime_call` site's stage *once* and reuse it
   across invocations/layers/decode-steps, mirroring how EXLA dispatches
   custom calls. See "Deferred: Stage 32" below.

## Procedure

1. **Recognize which `runtime_call`s are already handled in-graph.**
   `EMLX.Native.Expr.recognized_runtime_call?/1` (new, public) checks
   whether a `runtime_call`'s callback capture belongs to `EMLX.Fast` — the
   same check `fast_kernel_dispatch/2` (Stage 10) already made privately,
   now shared so `emlx.ex`'s split-point routing can reuse it. Recognized
   kernels are unaffected by this stage (still lowered as a single fused
   opcode, no split).
2. **Generalize the `while`-split routing to a generic split-point
   routing.** `emlx.ex`:
   - `contains_while?/1` → `contains_split_point?/1`; `split_point?/1` is
     `true` for `:while` or an *unrecognized* `:runtime_call`
     (`unrecognized_runtime_call?/1`, built on
     `EMLX.Native.Expr.recognized_runtime_call?/1`).
   - `build_while_chain_eval_fn/2` → `build_split_chain_eval_fn/2`: same
     `Nx.Defn.Graph.split/2` + `Graph.run(compiler: EMLX)` machinery,
     generalized to split on `split_point?/1` instead of `op == :while`
     only.
   - New base case, mirroring the existing "bare `while` at the root"
     case: `bare_runtime_call?/1` + `build_runtime_call_base_eval_fn/2`
     handle a stage whose entire body *is* an unrecognized `runtime_call`
     (materialize inputs, call the Elixir callback directly, wrap the
     result back as an `EMLX.Backend` tensor).
3. **Regression tests** — `emlx/test/emlx/native/expr_test.exs`'s new
   `:stage31` describe block (6 tests, using `EMLX.Quantization`'s
   `dequantize/1` and `quantized_matmul/2` runtime_calls as real,
   non-`EMLX.Fast` unrecognized callbacks): bare runtime_call, runtime_call
   surrounded by ordinary ops (after the call — `dequantize`'s only
   operand is itself a bare parameter, so there's no real pre-call
   computation to hoist into a "before" stage), two independent
   runtime_calls merged downstream, a runtime_call with a tuple
   (multi-tensor) operand container (`quantized_matmul`'s `{activation,
   qw}`, the same container shape as the real target use case), a
   runtime_call inside a `while` body, and a regression guard that a
   *recognized* `EMLX.Fast.*` kernel is still fused in-graph as a single
   instruction (no split — asserted structurally via `Expr.lower/1` +
   `prog.instructions`, not just numeric equivalence, since
   `Graph.split`/`run` is numerically transparent and can't otherwise
   distinguish "fused" from "split"). All equivalence-tested against
   `Nx.Defn.Evaluator`.

## Nx.Defn.Graph bug found (fourth in the
[`nx-graph-split-bugreport.md`](nx-graph-split-bugreport.md) lineage)

Exercising `Nx.Defn.Graph.split/2` with a `runtime_call` whose *sole*
operand is a bare parameter (the common case: `dequantize(qw)`, no
intermediate computation feeding `qw`) crashed in `split_before/3` /
`split_both/3` with `FunctionClauseError` on `Expr.parameter/2`, or a
downstream `KeyError`/`BadMapError` in `Graph.run/3`.

**Root cause:** both functions scan a node's `args` for `%Nx.Tensor{}`
values to decide what counts as an "intermediate computation" to hoist as
a stage-boundary parameter, matching on the bare struct (`%T{} = expr`).
A `runtime_call`'s `args` list is `[tensor_expr, callback, out_template,
opts]` — `out_template` (from `Nx.template/2`, e.g. via
`Nx.runtime_call(out, ...)`) is *also* a `%Nx.Tensor{}`, but backed by
`Nx.TemplateBackend`, not `Nx.Defn.Expr`. The generic scan can't tell it
apart from a real graph node and tries to hoist it too, so
`Expr.parameter/2` (which requires `data: %Nx.Defn.Expr{}`) blows up on
it.

**Fix:** narrow the guard from `%T{} = expr` to `%T{data: %Expr{}} =
expr` in both `split_before/3` (~line 506) and `split_both/3`'s mirrored
`has_intermediate_computations` scan (~line 699) — a `Nx.TemplateBackend`
-backed tensor riding in an op's args is not a graph node to hoist, it's
an opaque value like any other non-tensor arg. Applied identically to all
three vendored copies of `nx/lib/nx/defn/graph.ex`: `~/coding/nx/nx`
(canonical fork), `emlx/deps/nx/nx`, `emlx_axon/deps/nx/nx`.

Added as an addendum to `nx-graph-split-bugreport.md`'s existing
three-bug lineage (same file, same component, found via the same class of
workload — a `runtime_call`-bearing graph put through `Graph.split`) —
see that doc for bugs 1–3 and their upstream-fix status.

## Acceptance

- `emlx/test/emlx/native/expr_test.exs`'s new `:stage31` tests pass,
  equivalence-tested against `Nx.Defn.Evaluator`.
- Full `emlx` and `Nx` (`~/coding/nx`) suites remain green — the
  `graph.ex` patch must not regress any existing `while`-split behavior.
- `EMLXAxon.rewrite/2` + quantized weights (`bb+rewrite`) no longer raises
  `does not yet lower op :runtime_call` — `native_kv_attn_callback/2` is
  routed as a split point instead of a hard error.

## Results

**Status: done**, with one documented, expected limitation deferred to
Stage 32 (see below).

1. Implemented exactly per the Procedure — `EMLX.Native.Expr.
   recognized_runtime_call?/1`, `emlx.ex`'s generalized
   `contains_split_point?/1` / `split_point?/1` /
   `build_split_chain_eval_fn/2` / `bare_runtime_call?/1` /
   `build_runtime_call_base_eval_fn/2`.
2. Fixed the `Nx.Defn.Graph.split_before/3`/`split_both/3` bug above in
   all three vendored copies.
3. `emlx/test/emlx/native/expr_test.exs`'s `:stage31` block: 6/6 passing.
   Full `emlx` suite: unaffected by this stage's changes (same
   pre-existing `nif_not_loaded` failures as Stage 25's baseline, none
   introduced). `~/coding/nx` suite: green after the `graph.ex` patch.
4. **Synthetic scaling check** (not part of the original acceptance
   criteria, added after the real-model validation below raised a
   concern): a standalone repro chaining *N* independent
   `dequantize`-as-`runtime_call` split points sequentially (`N` up to 12)
   compiles in ~1 ms per step after the first, and the same chain nested
   inside a `while` of 2 iterations (`N` up to 28, matching Qwen3's layer
   count) compiles in single-digit-to-low-double-digit ms. This confirms
   the split-point *mechanism itself* — `Graph.split`/`Graph.run` chaining
   many sequential split points, with or without an enclosing `while` — is
   correct and not exponential (the `nx-graph-split-bugreport.md` Bug 1
   memoization fix already covers this class of blowup upstream).
5. **Real-model validation against `validate_qwen3.exs`** (local
   `Qwen3-0.6B-MLX-4bit`, `bb+rewrite` path,
   `EMLX_QWEN3_MAX_NEW=1 EMLX_QWEN3_BENCH_RUNS=1 EMLX_QWEN3_WARMUP_RUNS=0
   EMLX_QWEN3_SEQUENCE_LENGTH=32`): the `does not yet lower op
   :runtime_call` `ArgumentError` no longer reproduces — routing is
   correct — but a single decode pass through Qwen3's 28
   `native_kv_attn_callback`-bearing attention layers took **>20 minutes
   and was still running** when killed, with steadily climbing memory
   (10–24 GB observed across two separate runs). This is **not** the
   synthetic-repro exponential-blowup pattern (item 4 above rules that
   out at this layer count) and is **not** a graph-splitting correctness
   bug — it is inherent to this stage's approach: every `runtime_call`
   split point forces a fresh `Nx.Defn.Graph.split` + native `mlx::core::
   detail::compile` per stage, **every single call**, with zero
   compiled-artifact reuse across the 28 structurally-identical layers or
   across decode steps (`EMLX.get_or_compile_program/6`'s cache, added in
   Stage 25, is scoped per-stage-per-call, not shared across stages or
   calls). 28 layers × 2 stages (before/after the split) × real Metal
   shader compilation for actual attention math is tens of minutes of
   pure compile overhead for one token.
6. **Deferred: Stage 32.** This is exactly the gap that stage names:
   compile each distinct `runtime_call` call-site's stage *once*
   (keyed by shape/callback identity, not by call), and dispatch to the
   cached compiled artifact on every subsequent invocation — an
   EXLA-style custom-call dispatch table instead of "re-split and
   re-compile the whole surrounding graph from scratch on every call."
   Until Stage 32 ships, `bb+rewrite` is **functionally correct but not
   practically usable** for real generation workloads (`bb base`, Stage
   25's target, remains the fast, supported path for quantized weights
   under `compiler: EMLX`).
