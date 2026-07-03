# Stage 13 — Custom-fun reductions (`reduce`, `window_reduce`)

Status: done. **Mechanism = Elixir static unroll** (Stage 12 gate). `reduce`
already landed in Stage 12; this stage added `window_reduce` on the same
mechanism and flipped `EXPR_NODES.md` lines 109/131 to `[x]`.

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
`reduce` (only `reduce_max`/`reduce_min`), so the equivalence reference is
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

## Results

`reduce` already lowered in Stage 12 (`expand_reduce_unroll/8` +
`lower_fun_body/3`). This stage added **`window_reduce`** on the same
Elixir-unroll mechanism — zero C++ change — in `emlx/lib/emlx/native/expr.ex`.

### What landed

- `:window_reduce` `expand_node` clause (after the `window_scatter` loop):
  1. cast tensor + acc to `out_type` (the node's declared type, which for
     `window_reduce` is the **input tensor type**, not the acc type) before any
     fold;
  2. `:pad` the input with `acc` as the pad-value operand per the resolved
     padding config (interior 0; negative lo/hi raises);
  3. seed `acc` broadcast to the output shape;
  4. for each of `W = prod(window_dims)` within-window offsets, in **row-major
     order** (last window dim fastest, via `window_offsets/2`), emit a strided
     `:slice` of the padded input and fold the reducer body over it with
     `lower_fun_body/3`, mirroring `Nx.BinaryBackend.window_reduce`
     (`fun(element, acc)`, row-major window traversal).
- Helpers: `emit_pad_with/5` (pad with a scalar operand), `emit_static_slice/6`
  (static `:slice` wire format), `window_offsets/2` (mixed-radix decomposition).

### Key correctness detail (slice span vs stride)

`:slice` treats `lengths` as a **span** (`stop = start + length`); with a
`stride > 1` it yields `ceil(length/stride)` elements. To get `out_dim` outputs
stepping by `stride`, the span is `(out_dim - 1)*stride + 1` (caught by the
`strides: [2, 2]` test — initial naive `length = out_dim` collapsed each window
slice to a single element).

### Tests (`describe "Stage 13 …"`, tag `:stage13`)

Reference = `Nx.Defn.Evaluator` on `BinaryBackend` (eager EMLX has no custom-fun
`reduce`/`window_reduce`). Cases: dtype-changing `reduce` (s32→f32), 1-D window
sum (valid), 1-D max with `:same` padding, 2-D sum with strides, dilations,
**non-commutative affine reducer** (validates fold order), asymmetric explicit
padding, integer (s32) window, runtime acc. The old `window_reduce`
fallback-sentinel test was repointed to interior `:pad` (still unlowered).

| Item | Outcome |
|------|---------|
| `window_reduce` static unroll | ✅ matches Evaluator across all 8 cases |
| `reduce` dtype-changing coverage | ✅ s32 input, f32 acc → f32 |
| `EXPR_NODES.md` 109 / 131 | ✅ flipped to `[x]` |
| `mix test test/emlx/native/expr_test.exs` | ✅ 236 passed (was 227) |

### Deferred (named follow-up, not load-bearing for this stage)

The unroll is O(W)-op like `reduce` is O(extent)-op; large windows replay
slowly. The advisor-blessed perf follow-up — **route associative reducers
(add/mul/max/min) to the existing native `window_*` opcodes** (`expr.ex` ~1075),
which is single-mode-safe — is deferred to a later perf stage. The Stage-12
hand-off's "large-extent → Evaluator fallback" idea was **discarded**: it
violates the single-mode (no-fallback) design.
