# Stage 12 — Spike: C++ child-program substrate

Status: open (spike). Depends on Stage 11 (a working benchmark to measure on).

## Why this stage exists

Custom-fun `reduce` / `window_reduce` (`EXPR_NODES.md` §E/§G, `[~]`) need a way
to lower the user's scalar reducer `fun` and apply it over a reduction extent.
Rather than inline-unrolling in Elixir, we spike a reusable **child-program**
(sub-IR) channel in the C++ program, with an eye to also re-expressing `while`
through it later (Stage 14).

## The constraint that shapes everything

EMLX builds against **MLX 0.31.2**, whose public C++ core
(`mlx::core::detail::compile`) is **trace-once / replay**. MLX has **no lazy,
data-dependent control-flow primitive** (no `while_loop` / `fori_loop` / `scan`
/ `cond` that lives inside a traced graph) — which is exactly why Stage 08 put
`while` host-side via `Nx.Defn.Graph.split`. So a child program splits cleanly:

- **Static fold** (`reduce`, `window_reduce`): extent is known at trace time
  (shapes are static), so a child applied N times *inside* the parent trace is
  legitimate → one cached graph. Graph-equivalent to Elixir inline-unroll; the
  C++ version's only edge is a smaller wire payload + a reusable abstraction.
- **Dynamic loop** (`while`): trip count is data-dependent → cannot be traced
  into one graph. A C++ `while` must `mlx::core::eval` the predicate scalar each
  iteration and replay a held body subprogram (eval-per-iteration on the worker,
  off the BEAM). Real but unproven win vs. the working `Graph.split` host loop.

## Procedure (minimal spike)

1. **Verify the API** against the real 0.31.2 headers (`mlx/compile_impl.h`,
   `mlx/ops.h`, `mlx/transforms.h`): confirm there is no lazy `while`/`scan`,
   and confirm mid-trace `eval` of a scalar behaves for the dynamic case.
   (Headers are fetched at build time; the local cache may be empty.)
2. **Refactor the interpreter:** extract the lambda body in `compile_program`
   (`emlx_compiler.cpp` ~1619–1662 — the `resolve` closure + instruction walk +
   output collection) into a reusable `run_program(ir, inputs)` so the top-level
   program and any subprogram share one code path.
3. **Extend the wire format:** add a `subprograms` argument to `compile_program`
   (a list of sub-IRs, each its own `{op_names, operands, attrs, output_refs,
   num_inputs}`). An instruction references a subprogram by index via the
   existing int64 attr channel — no new resource type.
4. **Add one opcode — `:fold`:** operands `[init_acc, tensor]`, attrs
   `[subprogram_idx, axis, extent]`. C++ loops `extent` times applying the child
   interpreter (static unroll within the trace). This single opcode serves both
   `reduce` (fold over a reduce axis) and `window_reduce` (fold over the
   flattened window dims, reusing `compiler_sliding_window_view`, ~lines 76–104).
5. **Elixir lowering (narrow):** generalize `expand_block_via_default/4`'s
   param-remapping (`expr.ex` ~1706–1746) into a "lower a `:fun` sub-expr into a
   sub-IR" helper; emit a `:fold` for `Nx.reduce(t, 0, fn x, acc -> x + acc end)`.
6. **Validate** vs `Nx.Defn.Evaluator` — there is no eager EMLX `reduce` oracle
   (`emlx/lib/emlx/backend.ex` only has `reduce_max`/`reduce_min`).

## Go/no-go gate

- Static-fold `:fold` reduce works end-to-end and matches the Evaluator → green
  for Stage 13.
- Benchmark `:fold` reduce vs (a) the Evaluator fallback and (b) a pure-Elixir
  inline-unroll. If the C++ child program is not meaningfully better than the
  Elixir unroll for the static case, **drop the C++ path for reductions** and do
  Stage 13 as an Elixir unroll.
- From task 1's findings, decide whether a C++ eval-per-iteration `while`
  (Stage 14) is worth pursuing over the working `Graph.split` host loop. This is
  the riskier half — treat it as a stretch goal, not a commitment.

## Acceptance

- Sub-IR plumbing lands behind one `:fold` opcode, validated on `Nx.reduce`.
- Gate decisions recorded here (fold mechanism for Stage 13; go/no-go for the
  Stage 14 `while` refactor).
