# Stage 12 — Spike: C++ child-program substrate

Status: done. **Gate outcome: no-go on the C++ child-program path.** The spike
pivoted (advisor-blessed) to the cheaper Elixir inline-unroll baseline first;
that baseline turned out to be graph-equivalent to the proposed C++ `:fold`, so
the C++ subprogram channel buys nothing measurable and was not built. Stage 13
proceeds as an Elixir unroll; the speculative Stage 14 C++ `while` is dropped.

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

## Results

### What was built (deviation from procedure, advisor-blessed)

The advisor flagged that for a **static** fold, a C++ `:fold` child program and
a pure-Elixir inline-unroll compile to the **identical** cached MLX graph — so
they are graph-equivalent and replay identically. The only axis on which they
can differ is trace-time cost and wire-payload at large extent. The leanest path
to a defensible gate is therefore: build the Elixir unroll first (cheap), then
measure whether its trace cost/payload blows up enough to justify the C++
plumbing. It did not — so the C++ subprogram channel + `run_program` refactor
(procedure tasks 2–4) were **not built**.

What landed instead (`emlx/lib/emlx/native/expr.ex`):

- `:reduce` lowered by **static trace-time unroll**: transpose reduce axes last,
  collapse them into one trailing axis of size `extent`, slice that axis into
  `extent` kept-shape elements, and fold the reducer over them — vectorised
  across the kept axes. Each fold step re-lowers the reducer body inline with
  `acc` bound to the previous step's result. Reuses existing opcodes
  (`slice`/`squeeze`/`broadcast`/`transpose`/`reshape` + the reducer's own ops):
  **zero C++ change**.
- `lower_fun_body/3` — generalises `expand_block_via_default/4`'s param-remapping
  into a "lower a `:fun` sub-expr into emitted ops" helper, with a body-local
  `node_to_ref` that does not leak across fold iterations (constant body node ids
  would otherwise alias iteration 0's results).
- A no-op `:fun` `expand_node` clause (the `:fun` leaf is opaque in the parent
  ordering; the owning `:reduce` reaches into `fun.data.args` itself).

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| Task 1 — verify MLX 0.31.2 API | ✅ | `compile.h`/`transforms.h`/`ops.h`/`compile_impl.h` confirm **no** `while_loop`/`fori_loop`/`scan`/`cond` in the public core. `eval(std::vector<array>)` + `array::item<T>()` exist ⇒ a dynamic loop can only be eval-per-iteration (trace broken each iter). Confirms the static/dynamic split. |
| `:reduce` static unroll | ✅ | 12/12 ad-hoc cases + 8/8 committed tests (`describe "Stage 12 …"`) match the Evaluator/BinaryBackend oracle, incl. multi-axis, `keep_axes`, a **non-commutative** affine reducer (validates fold order), int, runtime acc. |
| Validation oracle | ✅ | Eager EMLX has no `reduce`; oracle is `Nx.Defn.Evaluator` on `BinaryBackend` (`check_reduce_equiv/3`). |
| Suite | ✅ | `mix test test/emlx/native/expr_test.exs` → 227 passed. Old reduce fallback-sentinel test repointed to `window_reduce` (still unlowered). |

### Benchmark — the decisive axis (trace/payload vs extent, not replay)

1-D `Nx.reduce(x, 0.0, &Nx.add/2)`, Apple M-series CPU. Steady-state `replay` is
shown only to demonstrate graph-equivalence — it is **not** used to choose the
mechanism (both flavours produce the same O(extent)-op graph).

| extent | instrs | payload (int64) | lower µs | replay µs | Evaluator µs |
|-------:|-------:|----------------:|---------:|----------:|-------------:|
| 10     | 32     | 114             | ~1.4k*   | 334       | 11           |
| 100    | 302    | 1,104           | 123      | 1,140     | 51           |
| 500    | 1,502  | 5,504           | 549      | 9,045     | 221          |
| 1000   | 3,002  | 11,004          | 1,020    | 30,918    | 523          |
| 2000   | 6,002  | 22,004          | 2,036    | 143,998   | 852          |

(* first-iteration warmup.) `instrs ≈ 3·extent`, `payload ≈ 11·extent` — both
linear and tiny in absolute terms (≤176 KB, sent once per compile, then cached).
Elixir lowering stays sub-2 ms. Replay is O(extent) and dominates.

## Go/no-go gate — decisions

1. **Static-fold reduce works + matches the Evaluator → GREEN for Stage 13.** ✅
2. **C++ `:fold` vs Elixir unroll → drop the C++ path; Stage 13 = Elixir unroll.**
   They are graph-equivalent (identical cached graph, identical O(extent) replay).
   The only axis where they differ — wire-payload and Elixir build-time — is
   negligible (≤176 KB once; ≤2 ms). The C++ child-program substrate +
   `run_program` refactor would be pure cost for zero measurable benefit. The
   reduce unroll already landed here is the Stage 13 mechanism.
3. **Stage 14 C++ `while` → no-go (dropped).** Task 1 confirms MLX 0.31.2 has no
   in-trace control flow, so a C++ `while` is eval-per-iteration: the trace
   breaks every iteration (no cross-iteration fusion) — the **same** fusion
   profile as the proven Stage-08 `Graph.split` host loop. Its only theoretical
   edge is avoiding a per-iteration BEAM↔NIF round-trip, and the spike shows the
   child-program abstraction's costs are not justified by the measurable case.
   Revisit only if a concrete decode-loop benchmark shows per-iteration BEAM
   round-trips dominate.

### Hand-off note for Stage 13 (not a spike blocker)

The unroll produces an O(extent)-op graph that, at large extents, is far slower
to replay than the eager Evaluator loop (~170× at extent 2000). Stage 13 should
gate on extent: small extents unroll natively (keeps the defn single-NIF); large
extents should stay Evaluator-fallback, or — when the reducer matches a known
associative op (add/mul/max/min) — route to the native primitive
(`sum`/`product`/`reduce_max`/`reduce_min`). Also add `window_reduce` (reuse
`compiler_sliding_window_view` + the same `lower_fun_body/3` fold) before
flipping `EXPR_NODES.md` lines 109 / 131.
