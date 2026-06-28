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
| `parameter` | `(index)` | `{:input, i}` ref | [ ] |
| `constant` | `(number)` | materialize → `{:const, i}` | [ ] |
| `tensor` | `(tensor)` | bake weight → `{:capture, i}` | [ ] |
| `metadata` | `(expr, meta)` | passthrough to inner expr | [ ] |
| `elem` | `(tuple, pos)` | select sub-result of a multi-output instr | [ ] |
| `fun` | `(params, expr, mfa)` | inline at call site / child program | [ ] |
| `cond` | `(clauses, last)` | child programs + select, or native cond | [ ] |
| `while` | `(initial, arg, pred, body)` | child programs (cond+body); host loop | [ ] |
| `block` | `(struct, args, default, fun)` | dispatch on `Nx.Block.*` (see F) | [ ] |
| `optional` | `(name, args, default)` | lower default expr, or route to native | [ ] |
| `attach_token` / `token` | hooks | unsupported → raises (side effects) | [ ] |
| `runtime_call` | `(expr, cb, out, opts)` | unsupported → raises | [ ] |

Notes:
- `cond`/`while` introduce sub-scopes — Layer A (`EMLX.Defn.Tree.post_order/1`)
  treats them as opaque single nodes; the lowerer recurses into `args` and
  topo-sorts each inner scope as a child program.
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

- [ ] sum, product, all, any
- [ ] reduce_max, reduce_min
- [ ] argmax, argmin
- [ ] reduce (custom fun — may need fallback / fun lowering)
- [ ] dot
- [ ] conv

## F. Indexing / selection

- [ ] select
- [ ] clip
- [ ] slice (start_indices may be tensors — see `apply_args` special case)
- [ ] put_slice
- [ ] gather
- [ ] take
- [ ] take_along_axis
- [ ] indexed_add
- [ ] indexed_put

## G. Sort / window / cumulative

- [ ] sort, argsort
- [ ] window_sum/max/min/product (+ window_reduce custom fun → maybe fallback)
- [ ] window_scatter_max/min
- [ ] cumulative_sum/product/max/min (last-axis fast path + interior-axis)

## H. FFT

- [ ] fft, ifft (1-D)
- [ ] fft2/ifft2, rfft/irfft (route via n-D transform)

## I. Creation

- [ ] iota
- [ ] eye
- [ ] from_binary / constant materialization (boundary handling)

## J. RNG (Nx.Random primitives)

- [ ] random_uniform / random_normal primitives + key threading

## K. Nx.Block.* (block node, dispatch on struct)

- [ ] LinAlg: cholesky, triangular_solve, solve, qr, eigh, lu, svd, determinant
- [ ] all_close, phase, and other Nx.Block.* helpers

## L. EMLX.Fast fused kernels (optimization, not correctness)

Recognize lowered patterns and route to `EMLX.Fast` instead of the primitive
expansion:

- [ ] rms_norm
- [ ] layer_norm
- [ ] rope / rope_with_positions / rope_with_freqs
- [ ] scaled_dot_product_attention (+ causal / key-masked variants)
- [ ] swiglu

---

### Coverage burndown

Run the ported probe to regenerate MISS lists (single mode — a not-yet-lowered
op raises, which the probe records as MISS):

```
mix run scripts/expr_op_coverage.exs   # compiler: EMLX
```

Each milestone (M2–M10 in PLAN.md) clears one or more sections above, in both
the Elixir lowerer and the C++ program.
