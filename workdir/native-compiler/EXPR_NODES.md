# Nx.Defn.Expr node taxonomy & coverage checklist

The complete set of node types the native lowerer (`EMLX.Native.Expr`) must
eventually handle, split into **syntax/control-flow nodes** (compiler-level —
the lowerer must handle these itself; `Nx.Defn.Evaluator` does today) and
**backend-callback ops** (EMLX.Backend already implements eager semantics —
lowering reuses them).

The compiler is **single-mode** (no fallback lane): a not-yet-lowered node
**raises**. Product-level lowering control is structural, via `Nx.Defn.Block`
(see PLAN.md §6) — a `block` node is lowered either by recognizing its
`Nx.Block.*` struct (native/fused path) or by descending into its
`default_expr` (primitive decomposition, the per-block default).

Status legend: `[ ]` not lowered → raises · `[~]` partial · `[x]` lowered +
equivalence-tested. Update as milestones land.

Source of truth:
- syntax nodes: `emlx/deps/nx/lib/nx/defn/expr.ex` (moduledoc "Syntax nodes")
- callbacks: `emlx/deps/nx/lib/nx/backend.ex`
- unary math fun list: `emlx/deps/nx/lib/nx/shared.ex` `unary_math_funs/0`

---

## A. Syntax / control-flow nodes (compiler-level)

| Node | Args | Lowering approach | Status |
|------|------|-------------------|--------|
| `parameter` | `(index)` | `{:input, i}` ref | [x] |
| `constant` | `(number)` | materialize → `{:const, i}` | [x] |
| `tensor` | `(tensor)` | bake weight → `{:capture, i}` | [x] |
| `metadata` | `(expr, meta)` | passthrough to inner expr | [x] |
| `elem` | `(tuple, pos)` | index into list of refs stored for tuple-output op | [x] |
| `fun` | `(params, expr, mfa)` | inline at call site / child program | [x] (unreachable / already-subsumed — Stage 16) |
| `cond` | `(clauses, last)` | right-folded nested `:select` ops (all branches in parent scope) | [x] |
| `while` | `(initial, arg, pred, body)` | parent scope: `Nx.Defn.Graph.split` + host loop. Nested in a block's `default_expr`: statically-counted loops (fixed trip count at trace time) unroll in place in `expand_node` (Stage 17); a genuinely data-dependent nested `while` still raises | [x] |
| `block` | `(struct, args, default, fun)` | dispatch on `Nx.Block.*` (see F) | [x] |
| `optional` | `(name, args, default)` | lower default expr, or route to native | [x] (already-subsumed — dead node type in current Nx fork, Stage 16; re-audit on Nx version bump) |
| `attach_token` / `token` | hooks | fire-and-forget passthrough; hooked value(s) ride as extra program outputs, host fires the callback after the one NIF call returns (see K note) | [x] (Stage 18; cond-branch-local hooks raise deliberately) |
| `runtime_call` | `(expr, cb, out, opts)` | recognize `EMLX.Fast.*` callback → fused opcode (see L); else raises | [x] |

Notes:
- `cond`: all predicate and body tensors are in the **parent scope** (`apply_args`
  for `:cond` traverses everything without scope distinction). By the time the
  `:cond` node is expanded, every ref is already in `node_to_ref`. Lowered as
  right-folded `:select` ops — no child programs, no C++ changes.
- `while`: handled structurally, not as an IR instruction. `build_eval_fn/3`
  splits the expression on its `:while` nodes (`Nx.Defn.Graph.split`, `:both`)
  and replays the chain with `Nx.Defn.Graph.run(…, compiler: EMLX)`, so each
  stage re-enters this compiler. An isolated `while` stage (the base case: its
  `initial` carry is exactly the stage parameters) runs a host loop whose
  condition/body are recompiled via `Nx.Defn.jit(compiler: EMLX)` — recursing
  for nested whiles. Stage inputs are reordered into carry-parameter order
  before binding. Non-tail and nested `while`s (and `while`-as-input) compile
  natively without an Evaluator fallback.
- `block` is the lowering-control lever (PLAN.md §6): recognize the
  `Nx.Block.*` struct for a native/fused path, else lower `default_expr`.
- **Stage 17 (while-in-`default_expr` descent):** a top-level parent-scope
  `while` is handled at `build_eval_fn` level (splits the expression on
  `:while` nodes, Stage 08) before `default_expr` descent ever runs — that
  split never sees a `while` nested inside a block's `default_expr`. Instead
  of extending the split to recurse into blocks, `expand_node` now has a
  `:while` clause that fires only in that nested position: it recognizes the
  exact shape `Nx.Defn.Expr.while_range/7` emits for a range-generator loop
  with `unroll: false` (the default) — start index, bound, and step are all
  compile-time constants — and statically unrolls the body that many times,
  chaining each iteration's carried refs into the next (same idiom as the
  Stage 12/13 custom-fun static unroll, generalized from a single accumulator
  to a loop-carried tuple). A nested `while` that doesn't match this shape
  (genuinely data-dependent trip count) still raises `does not yet lower op
  :while`, deliberately — no silent wrong answer. This closes QR `mode:
  :complete` (Householder loop) and, transitively, three unrelated
  pre-existing gaps that blocked reaching the `while` node at all: `:eye` and
  `:constant` both assumed no leading batch/vectorized dim (broke under
  `Nx.revectorize`'s `collapsed_axes` wrapping, used unconditionally by both
  `Nx.LinAlg.qr` and `Nx.LinAlg.svd`), and `:metadata` assumed a single-tensor
  inner expr (broke when wrapping a tuple, as `Nx.LinAlg.svd`'s Gram-matrix
  path does around its `cond`). SVD `full_matrices?: false` needed only the
  latter two fixes — its default_expr no longer contains a `while` at all in
  the current Nx fork (rewritten to a non-iterative Gram-matrix/`eigh`
  decomposition instead of the old QDWH iteration). `triangular_solve` with
  `left_side: false` or `transform_a != :none` is a *different* gap not
  addressed here: those opts land directly on a `:triangular_solve` op node
  (not a block whose `default_expr` contains a `while`), and the raise is a
  deliberate "not yet implemented" for that native op's unsupported operand
  layout — orthogonal to this stage's structural-boundary charter, left
  raising.
- **Stage 18 (hooks):** `token`/`attach_token` (`Nx.Defn.Kernel.hook/2,3`) turned
  out **not** to be control flow — `attach_token`'s runtime value is its
  wrapped expr unchanged, and a hook's callback return value is never read
  back into the graph (verified against `Nx.Defn.Evaluator.eval_apply/5`).
  So the `while`-style `Nx.Defn.Graph.split` host-round-trip the stage doc
  proposed buys nothing: `:attach_token` lowers as a zero-instruction
  passthrough, and `:token` records each hook's already-lowered ref(s) +
  callback as extra program outputs (`EMLX.Native.Expr.t/0`'s new `hooks`
  field); `EMLX.__compile__` fires each callback host-side once, right after
  the single `eval_program` NIF call returns — still one NIF call per `defn`
  invocation, strictly better than `while`'s N-calls-per-loop. A hook
  reachable only from inside a `cond` branch (per `Nx.Defn.Tree.scope_ids/1`)
  raises deliberately: EMLX's `cond` evaluates every branch unconditionally
  (`:select`), which would fire such a hook on every call regardless of which
  branch is taken — a genuine correctness divergence from `Evaluator`, not a
  coverage gap to silently paper over. A hook inside a `while` body,
  reduce/window_reduce custom-fun body, or a statically-unrolled nested
  `while` needs no such guard — all three always execute in full (never
  conditionally, unlike `cond`), so each fires once per iteration exactly
  like `Evaluator` (equivalence-tested). This required care: a reviewer
  subagent caught a false positive where a hook in a plain reduce body (no
  `cond` involved) wrongly raised the cond-branch message, because
  `Nx.Defn.Tree.scope_ids/1`'s `:scope`-mode walk never sees inside a
  `:fun`/`:while` body by design — fixed by extending the top-scope id set
  with a fresh `scope_ids` pass over each such body right before lowering it
  inline, which still correctly excludes a `cond` nested deeper inside.
  Chasing that also surfaced an independent, silent-wrong-answer bug (found
  by executing, not reading): the reducer/unroll body lowering helpers
  reconstructed their returned state from an explicit field list that
  predated hooks and never included the new `hooks` field, so any hook fired
  from inside such a body was dropped with no error. Descoped: a
  runtime `hooks:` jit-option override (only trace-time callbacks are
  supported); `EMLX.__compile__` doesn't thread `opts` into the native path
  for this at all today. **Found and fixed a real `Nx.Defn.Graph` bug along
  the way** (see the stage doc's Results): `do_rewrite_subtree/3` had no
  `:token` clause, so a hook's payload was silently skipped during
  `Graph.split`'s per-stage parameter renumbering whenever a `while` had
  surrounding work on both sides — same "found via testing, not static
  reading" pattern as Stages 11/17.
  `runtime_call` is fully lowered within its designed scope: `EMLX.Fast.*`
  callbacks (incl. per-token prefill RoPE, Stage 15 Part B) recognize to a
  fused opcode; any other callback is a genuine host side effect and raises
  deliberately (not a gap).
- **Stage 16 doc audit (confirmed, not real gaps):** `fun` is only ever
  produced as a `reduce`/`window_reduce` operand (`nx/lib/nx/defn/expr.ex:
  996,1017`, single producer verified fork-wide via `apply_fun/4` at
  `expr.ex:1376`), which already extracts and re-lowers its body directly
  (Stages 12–13) — `expand_node`'s `op: :fun` clause (`expr.ex:1768`) is a
  documented no-op because a standalone `:fun` node is unreachable; pinned by
  a regression test (`expr_test.exs`, "Stage 16 — :fun node unreachability")
  asserting no `:fun` instruction is ever emitted. `optional` has no op tag
  anywhere in the vendored Nx fork at all (dead node type from an older Nx).
  Section I's "`from_binary` / constant materialization" line is similarly
  moot: `Nx.Defn.Expr.from_binary/3` resolves to a `constant`/`tensor` node
  during tracing, not a distinct op. `optional`/`from_binary` are properties
  of the vendored Nx fork's node set, not of anything EMLX owns — **re-audit
  on Nx version bump.**

## B. Unary elementwise (Nx.Backend unary_ops)

Math funs (`unary_math_funs/0`): exp, expm1, log, log1p, sigmoid, cos, sin,
tan, cosh, sinh, tanh, acos, asin, atan, acosh, asinh, atanh, sqrt, rsqrt,
cbrt, erf, erfc, erf_inv.

Plus: abs, bitwise_not, ceil, conjugate, floor, negate, round, sign,
count_leading_zeros, population_count, real, imag, is_nan, is_infinity.

- [x] math funs (23)
- [x] sign/abs/negate/ceil/floor/round
- [x] bitwise_not
- [x] count_leading_zeros, population_count (raise — not supported by EMLX)
- [x] is_nan, is_infinity
- [x] complex: conjugate, real, imag

## C. Binary elementwise (Nx.Backend binary_ops)

Arithmetic/bitwise: add, subtract, multiply, pow, remainder, divide, atan2,
min, max, quotient, bitwise_and, bitwise_or, bitwise_xor, left_shift,
right_shift.

Compare/logical: equal, not_equal, greater, less, greater_equal, less_equal,
logical_and, logical_or, logical_xor.

- [x] arithmetic (add/subtract/multiply/divide/pow/remainder/atan2/min/max/quotient)
- [x] bitwise + shifts
- [x] compare (6)
- [x] logical (and/or/xor) + logical_not (unary composite via Nx.Block.LogicalNot)

## D. Shape / movement

- [x] reshape
- [x] squeeze
- [x] transpose
- [x] broadcast
- [x] as_type
- [x] bitcast
- [x] pad (fully closed, Stage 33: interior padding and negative lo/hi decompose into reshape/pad/slice/squeeze in `EMLX.Native.Expr.expand_pad_general/5`, no C++ change)
- [x] reverse
- [x] concatenate (variadic — args is `[list | ...]`)
- [x] stack (variadic)

## E. Reductions / contraction / conv

- [x] sum, product, all, any
- [x] reduce_max, reduce_min
- [x] argmax, argmin
- [x] reduce (custom fun — static trace-time unroll: reducer body re-lowered inline once per reduce-extent element, Stages 12–13)
- [x] dot
- [x] conv

## F. Indexing / selection

- [x] select
- [x] clip
- [x] slice (static indices + dynamic tensor start indices via take-based fallback)
- [x] put_slice (static + dynamic via MLX dynamic `slice_update` overload)
- [x] gather
- [x] take
- [x] take_along_axis
- [x] indexed_add
- [x] indexed_put

## G. Sort / window / cumulative

- [x] sort, argsort
- [x] window_sum/max/min/product
- [x] window_scatter_max/min
- [x] cumulative_sum/product/max/min
- [x] window_reduce (custom fun — static unroll: pad with acc, then fold the reducer body inline over the prod(window_dims) within-window offsets via strided per-offset slices, Stage 13)

## H. FFT

- [x] fft, ifft (1-D)
- [x] fft2/ifft2
- [x] rfft/irfft (via default_expr descent — Nx.Block.RFFT/IRFFT → fft+slice decomp)

## I. Creation

- [x] iota (flat + axis-specific; all dtypes)
- [x] eye (rectangular; all dtypes)
- [x] from_binary / constant materialization (boundary handling) — unreachable / already-subsumed (Stage 16): `from_binary` resolves to `constant`/`tensor` during tracing, no distinct op reaches the lowerer; re-audit on Nx version bump

## J. RNG (Nx.Random primitives)

- [x] random_uniform / random_normal — decompose via threefry2x32 (bitwise + iota); key threading is ordinary tensor operand threading; deterministic vs eager EMLX.Backend verified

## K. Nx.Block.* (block node, dispatch on struct)

- [x] LinAlg: cholesky, triangular_solve, solve, qr, eigh, lu, svd (native CPU-pinned MLX ops); determinant (default_expr descent — 2×2/3×3 pure primitives, N>3 descends through the recognized native LU block)
  - Native multi-output ops (qr/eigh/svd/lu) use the new multi-output IR (instruction result is a list of refs; `to_wire/1` flat-indexes outputs; C++ `multi_op_registry`).
  - Linalg outputs are `mlx::core::contiguous`-wrapped: MLX can otherwise emit a strided fused CPU `Compiled` kernel for the factorization tails (e.g. solve permutation, LU L/U masks) that fails to JIT (`pclose()`).
  - QR `:complete` and SVD `full_matrices?: false` descend into `default_expr`; both lower natively (Stage 17 — see the `while`-in-`default_expr` note above). `triangular_solve` with `left_side: false` or `transform_a != :none` is a direct op-node gap (not a `default_expr` descent) and still raises `does not yet lower op :triangular_solve`, permanently (out of Stage 17's scope; accepted as a permanent hard-raise by Stage 19, which removed the whole-defn Evaluator fallback lane this used to route through).
  - Batched (rank>2) and chained linalg→linalg are **correct** (verified: batched `cholesky` on CPU; batched `lu` `P·L·U` reconstruction + chained `cholesky→solve` on GPU default — the LU pivot→`P` rebuild via `:eye`/`:take` broadcasts over batch dims). Known env limitation: batched `lu`/`solve` can still hit the CPU `pclose()` JIT failure for the rank-3 strided permutation/mask kernels even with the `contiguous`-wrap, so those batched variants are not exercised in the CPU CI suite.
- [x] all_close, phase, top_k (tuple-output `default_expr`, via `flat_refs`), and unrecognized-struct `default_expr` descent — equivalence-tested (Stage 15 Part A)

## L. EMLX.Fast fused kernels (optimization, not correctness)

`EMLX.Fast.*` functions surface as `:runtime_call` nodes (not blocks/primitive
subgraphs — see Stage 10). The lowerer recognizes the callback (by
module+name+arity via `fast_kernel_dispatch/2`) and emits a single fused opcode
calling `mlx::core::fast::*` inside the compiled graph (one NIF replay, no host
hop). Float opts (eps/scale/base) ride the int64 attr channel as IEEE-754 bits.
Metal-only kernels → E2E tests run on a GPU worker (`device: :gpu`, `:metal`).

- [x] rms_norm
- [x] layer_norm (+ no-bias variant)
- [x] rope / rope_with_positions / rope_with_freqs (decode/T=1 fast callbacks call `mlx::core::fast::*`; per-token prefill T>1 callbacks lower to an in-graph cos/sin/rotate primitive composition, no new C++ kernel — Stage 15 Part B). **Known issue (out of scope, filed):** `mlx::core::fast::rope` itself miscomputes non-head-0 rotations for multi-head (H>1) input in EMLX's non-transposed layout — see `mlx-fast-rope-multihead-bugreport.md`. Affects the decode/T=1 fast callbacks (both eager and compiled call the same buggy primitive, so they agree with each other while both disagree with the textbook formula); does not affect the new prefill composition, which never calls `fast::rope`.
- [x] scaled_dot_product_attention (+ causal / additive-mask / causal-key-masked variants), incl. attention **sinks** (`mlx::core::fast::sdpa`'s `sinks` param) — `_sinks`-suffixed opcode variants (`fast_sdpa_sinks`, `fast_sdpa_masked_sinks`, `fast_sdpa_causal_sinks`, `fast_sdpa_causal_key_masked_sinks`), eager + compiled parity (Stage 22, Emily M26)
- [x] swiglu

---

### Coverage burndown

Run the ported probe to regenerate MISS lists (single mode — a not-yet-lowered
op raises, which the probe records as MISS):

```
mix run scripts/expr_op_coverage.exs   # compiler: EMLX
```

Each milestone (M2–M10 in PLAN.md) clears one or more sections above, in both
the Elixir lowerer and the C++ program.
