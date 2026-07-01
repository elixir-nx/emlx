# Stage 24 — investigation: quantized `Nx.dot` is invisible to the native compiler

Status: done (interim). Root-caused and given a clear pre-flight raise +
regression test; the full fix (call-time program specialization) is scoped
below but **not implemented** — needs a design sign-off first (see
"Deferred: the full fix").

## Symptom

`emlx_axon/bench/validate_qwen3.exs`'s `bb base` path (stock
Bumblebee-generated Axon graph, `defn_options: [compiler: EMLX]`, no
`EMLXAxon.rewrite/2`) crashed on Qwen3-0.6B-MLX-4bit with:

```
** (EMLX.NIFError) [tensordot] a and b must have the same shape on the contracted axes. in NIF.eval_program/2
    (emlx 0.3.1) lib/emlx.ex:1272: EMLX.await_worker/1
    (emlx 0.3.1) lib/emlx.ex:1491: anonymous fn/6 in EMLX.build_native_eval_fn/4
    ...
    (emlx 0.3.1) lib/emlx.ex:1538: anonymous fn/3 in EMLX.build_while_chain_eval_fn/2
```

Found immediately after Stage 19 closed (via a user-attached terminal log),
initially suspected to be a Stage-11-style `Nx.Defn.Graph.split` regression,
or a Stage-19 regression. It's neither.

## Root cause

`EMLXAxon.QuantizeParams.quantize/2` re-quantizes eligible Bumblebee weight
tensors to MLX 4-bit packed format specifically "so that `Nx.dot` dispatch
routes to `EMLX.quantized_matmul` at serving time" (its own doc comment).
That dispatch lives entirely in the **eager per-op backend callback**
`EMLX.Backend.dot/7`: it inspects the actual runtime tensor's
`quantization_config` metadata and, if present, reroutes to
`EMLX.quantized_matmul/7` — passing two *extra* hidden tensor operands
(`scales`, `biases`) that never appear anywhere in the traced `Nx.Defn.Expr`
graph.

A quantized tensor's Nx-visible `.shape`/`.type` deliberately mirror its
original logical dense shape (e.g. `{1024, 3072}`, `{:bf, 16}`) — a
deliberate illusion so ordinary Nx/Axon code that only ever calls plain
`Nx.dot` "just works" via this eager runtime dispatch. The real
`%EMLX.Backend{ref: ...}` NIF resource holds a physically different packed
representation.

`EMLX.Native.Expr` traces and compiles **once**, before any real tensor is
bound. At that point a weight is just a `:parameter` template — confirmed by
instrumenting the `:dot` lowering clause (`emlx/lib/emlx/native/expr.ex`
~line 728): the `right` operand showed up as `{:parameter, {1024, 2048},
{:bf, 16}}`, indistinguishable from an ordinary dense parameter. Separately
instrumenting `materialise_input_refs` (`emlx/lib/emlx.ex`) confirmed several
call-time-bound parameter positions do carry a non-nil `quantization_config`
with real scale/bias tensors — invisible at trace time, visible only once a
real tensor is bound at call time.

Consequence: the `:dot` lowering always emits a plain `:dot` IR opcode, with
no way to know ahead of time that the eventual runtime operand will need
`quantized_matmul` instead. At NIF replay time that opcode runs a real MLX
tensordot against the *packed* physical array — whose true last-dim size is
smaller (grouped by `bits`/`group_size`) than the logical shape the compiled
program was built assuming — hence the "contracted axes" shape mismatch.

This is structurally different from Stage 10's `EMLX.Fast.*` fused-kernel
design: those work because the call is *explicit* in the traced defn body
(`EMLX.Fast.rms_norm(x, w)` compiles to a `:runtime_call` node carrying every
real operand, including any "extra" ones, since they're literal arguments to
the traced call). Quantized-dot dispatch is *implicit* polymorphism resolved
only by inspecting a runtime tensor's backend-specific metadata inside an
eager callback — invisible to a compile-once/replay-many compiler by
construction, not by a missing feature.

### Confirmed not a model/graph bug, not a Stage 19 regression

- `Nx.Defn.jit(&Nx.dot/2, compiler: Nx.Defn.Evaluator)` on the same
  quantized input runs correctly (produces the right output shape) — the
  model/params/graph are fine; the gap is specific to `compiler: EMLX`.
- Bisected by stashing the Stage 19 `emlx.ex`/`expr.ex`/test edits and
  re-running the same repro: identical crash, same message, only the
  `emlx.ex` line numbers in the stack trace shift (matching the lines Stage
  19 deleted). The old `try_native_compile` rescue only ever caught
  `ArgumentError`s starting with `"does not yet lower op"`; `EMLX.NIFError`
  was never one of them, so this crashed exactly the same way before Stage
  19 too.

### Related but distinct finding: `bb+rewrite` was never exercising the native compiler either

`EMLXAxon.rewrite/2`'s native-attention rewrite introduces
`EMLXAxon.native_kv_attn_callback/2`, which calls `Nx.to_number/1` (a
blocking host sync) and manages KV-cache offset state via ETS across calls —
genuinely, permanently unlowerable (real host blocking + mutable
side-channel state, not just a missing-coverage gap). It has always raised
`"does not yet lower op :runtime_call"` from the native lowerer. Before
Stage 19, that unconditionally routed the *entire* `bb+rewrite` defn through
`Nx.Defn.Evaluator` — meaning Stage 11's recorded 23.4–34.5 tok/s for
`bb+rewrite` was never actually exercising EMLX's native single-NIF-replay
compiler at all; it was `Nx.Defn.Evaluator` plus Axon-graph-level
optimizations. Now, with the fallback deleted, it hard-crashes instead of
silently (and misleadingly) "working." Per discussion with the user: **left
as-is, not fixed** — it correctly exposes that this combination was never a
real native-compiler path, and `EMLXAxon.rewrite/2` is documented as
incompatible with `load_quantized` regardless (`emlx_axon.ex` ~line 198-201:
"Do not apply `EMLXAxon.rewrite/2` after `load_quantized`").

The `native` bench path (`EMLXAxon.TextGeneration`/`Qwen3.Generate`) is
unaffected by either finding: it's a hand-written eager pipeline (`EMLX.eval`
once per token) that never touches `Nx.Defn.jit`/`compiler: EMLX`, so
quantized-dot dispatch goes through `EMLX.Backend.dot/7`'s eager path
directly and "just works" by construction. This is why
`emlx_axon/test/emlx/qwen3_quantized_test.exs` (green, part of Stage 19's
verification) never caught this: it exercises `Qwen3.Generate`, not
`compiler: EMLX`. No test anywhere in `emlx`/`emlx_axon` exercised
`compiler: EMLX` against real quantized weights before this stage.

## What shipped (interim)

- `EMLX.materialise_input_refs/2` (`emlx/lib/emlx.ex`) now calls
  `reject_quantized_native_input!/1` on every bound input, which raises a
  clear, actionable `ArgumentError` (pointing at `Nx.Defn.Evaluator`,
  `EMLX.dequantize/1`, or a hand-written eager pipeline as alternatives)
  instead of letting a quantized tensor reach the NIF and crash there with
  an opaque MLX-level shape-mismatch message.
- Regression test added: `emlx/test/emlx/native/expr_test.exs`, `describe
  "Stage 24 — quantized Nx.dot input is a documented, permanent hard-raise
  (no fix yet)"` (tag `:stage24`) — pins the clear raise, plus a sanity test
  confirming the same defn runs correctly under `Nx.Defn.Evaluator`.
- Full `emlx` suite green: 2566 passed (825 doctests, 1741 tests), 1
  excluded — up from 2564 (the two new tests).

## Deferred: the full fix

Advisor-recommended direction, not implemented pending a design decision:
**call-time program specialization.** `build_native_eval_fn`'s closure
already sees real bound tensors (via `materialise_input_refs`) before the
NIF call — at that point, each parameter's `quantization_config` is visible.
The fix would:

1. On first call, inspect which parameter positions are quantized (a
   "quantization signature" — e.g. a bitset of positions).
2. If any are, compile a *second*, specialized program (cached alongside the
   original, keyed by `{original compile key, quantization signature}`)
   whose `:dot` lowering for those positions emits a new `quantized_matmul`
   IR opcode instead of plain `:dot`, mirroring `EMLX.Backend.quantized_dot/4`
   (`transpose`/`group_size`/`bits` as iattrs; `scales`/`biases` threaded
   through as additional hidden inputs appended at call-construction time,
   not part of the originally-traced Expr).
3. New C++ opcode in `emlx_compiler.cpp` wrapping
   `mlx::core::quantized_matmul` (already NIF-exposed via
   `EMLX.quantized_matmul/7`, used eagerly by `quantized_dot/4` today).

This only helps `bb base`-shaped usage (stock Axon graph, no
`EMLXAxon.rewrite/2`) — `bb+rewrite` remains out of scope regardless (see
above). Before building this, decide: is "stock Bumblebee graph + quantized
weights + `compiler: EMLX`" (`bb base`) a configuration this project
actually needs to support, given `native`
(`EMLXAxon.TextGeneration`/`Qwen3.Generate`) already covers real deployment
correctly and performs far better (62.6–71.4 tok/s vs `bb base`'s 7.3–9.1
tok/s even when it worked)? If yes, size this as its own stage. If no,
Stage 24's interim raise is the permanent answer and this section can be
closed as "descoped."

## Acceptance (for this investigation)

- [x] Root cause documented (which layer, why, confirmed via instrumentation
  — now reverted).
- [x] Confirmed independent of Stage 19 (bisected).
- [x] Opaque NIF crash replaced with a clear, actionable `ArgumentError`.
- [x] Regression test added; full `emlx` suite green.
- [ ] Full fix (call-time specialization) — deferred pending a scoping
  decision; not blocking Stage 19 or any other closed stage.
