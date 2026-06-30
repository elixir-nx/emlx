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
| `fun` | `(params, expr, mfa)` | inline at call site / child program | [ ] |
| `cond` | `(clauses, last)` | right-folded nested `:select` ops (all branches in parent scope) | [x] |
| `while` | `(initial, arg, pred, body)` | `Nx.Defn.Graph.split` on `:while`; cond/body recompiled via `compiler: EMLX`; Elixir host loop | [x] |
| `block` | `(struct, args, default, fun)` | dispatch on `Nx.Block.*` (see F) | [~] |
| `optional` | `(name, args, default)` | lower default expr, or route to native | [ ] |
| `attach_token` / `token` | hooks | unsupported → raises (side effects) | [ ] |
| `runtime_call` | `(expr, cb, out, opts)` | recognize `EMLX.Fast.*` callback → fused opcode (see L); else raises | [~] |

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
- `token`/`runtime_call`/hooks imply host side effects → not lowerable to a
  pure replay; they raise (no silent fallback in single mode).

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
- [x] pad (simple: non-negative lo/hi, interior=0; interior/negative raises — not yet lowered)
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
- [ ] from_binary / constant materialization (boundary handling)

## J. RNG (Nx.Random primitives)

- [x] random_uniform / random_normal — decompose via threefry2x32 (bitwise + iota); key threading is ordinary tensor operand threading; deterministic vs eager EMLX.Backend verified

## K. Nx.Block.* (block node, dispatch on struct)

- [x] LinAlg: cholesky, triangular_solve, solve, qr, eigh, lu, svd (native CPU-pinned MLX ops); determinant (default_expr descent — 2×2/3×3 pure primitives, N>3 descends through the recognized native LU block)
  - Native multi-output ops (qr/eigh/svd/lu) use the new multi-output IR (instruction result is a list of refs; `to_wire/1` flat-indexes outputs; C++ `multi_op_registry`).
  - Linalg outputs are `mlx::core::contiguous`-wrapped: MLX can otherwise emit a strided fused CPU `Compiled` kernel for the factorization tails (e.g. solve permutation, LU L/U masks) that fails to JIT (`pclose()`).
  - Unsupported variants (QR `:complete`, SVD `full_matrices?: false`, `triangular_solve` with `left_side: false` or `transform_a != :none`) descend into `default_expr`; while-containing decompositions raise `does not yet lower op` → Evaluator fallback.
  - Batched (rank>2) and chained linalg→linalg are **correct** (verified: batched `cholesky` on CPU; batched `lu` `P·L·U` reconstruction + chained `cholesky→solve` on GPU default — the LU pivot→`P` rebuild via `:eye`/`:take` broadcasts over batch dims). Known env limitation: batched `lu`/`solve` can still hit the CPU `pclose()` JIT failure for the rank-3 strided permutation/mask kernels even with the `contiguous`-wrap, so those batched variants are not exercised in the CPU CI suite.
- [ ] all_close, phase, and other Nx.Block.* helpers

## L. EMLX.Fast fused kernels (optimization, not correctness)

`EMLX.Fast.*` functions surface as `:runtime_call` nodes (not blocks/primitive
subgraphs — see Stage 10). The lowerer recognizes the callback (by
module+name+arity via `fast_kernel_dispatch/2`) and emits a single fused opcode
calling `mlx::core::fast::*` inside the compiled graph (one NIF replay, no host
hop). Float opts (eps/scale/base) ride the int64 attr channel as IEEE-754 bits.
Metal-only kernels → E2E tests run on a GPU worker (`device: :gpu`, `:metal`).

- [x] rms_norm
- [x] layer_norm (+ no-bias variant)
- [x] rope / rope_with_positions / rope_with_freqs (decode/T=1 fast callbacks; per-token prefill paths raise → Evaluator fallback)
- [x] scaled_dot_product_attention (+ causal / additive-mask / causal-key-masked variants)
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
