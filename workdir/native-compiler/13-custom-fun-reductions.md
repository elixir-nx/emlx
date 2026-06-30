# Stage 13 — Custom-fun reductions (`reduce`, `window_reduce`)

Status: open (planned). **Depends on Stage 12** (mechanism decided by its gate).

## Why this stage exists

`reduce` (`EXPR_NODES.md` §E line 109) and `window_reduce` (§G line 131) are the
only `[~]` ops blocked purely on lowering a user-supplied scalar reducer `fun`.
Today both raise `does not yet lower op` → the whole containing defn falls back
to the Evaluator (`expr.ex:600` for `reduce`; `window_reduce` hits the catch-all
at `expr.ex:1646`).

## What's missing

The Nx nodes carry a `:fun` `[params, expr, mfa]` over **two scalar parameters**
(element, acc) returning a scalar (`deps/nx/lib/nx/defn/expr.ex:992`, `:1006`).
MLX has no arbitrary-fun reduce primitive, and `EMLX.Backend` has no eager
`reduce` (only `reduce_max`/`reduce_min`), so the equivalence oracle is
`Nx.Defn.Evaluator` (+ BinaryBackend), not eager EMLX.

## Procedure

1. Lower the reducer `fun`'s inner expr into a sub-IR (or inline subgraph) via
   the helper from Stage 12 (generalized `expand_block_via_default/4`).
2. `reduce`: fold the reducer over the reduce axes, seeded with `acc`,
   vectorized across kept dims; honor `keep_axes`/dtype.
3. `window_reduce`: reuse the existing strided window view
   (`compiler_sliding_window_view`) then fold over the flattened window dims.
4. Use the Stage-12-blessed mechanism: C++ `:fold` opcode, or pure-Elixir
   inline-unroll if the gate preferred it.
5. Equivalence vs `Nx.Defn.Evaluator`; flip `EXPR_NODES.md` lines 109, 131.

## Acceptance

- `reduce` / `window_reduce` with non-trivial reducers match the Evaluator
  within tolerance (multi-axis, windowed, dtype-changing cases covered).
- Lines 109 and 131 flipped to `[x]`; suites green.
