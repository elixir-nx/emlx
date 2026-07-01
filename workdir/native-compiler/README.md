# EMLX Native Expr Compiler — planning

Overall task overview and stage index. Each stage is a separate doc in this
directory, designed for `/tackle-step <planning_dir> <stage_name>`.
`EXPR_NODES.md` is the companion node taxonomy / coverage checklist.

## Goal

Give EMLX a single-NIF graph-replay compiler for `defn`. The win: **collapse
the per-op BEAM↔NIF round-trips a `defn` pays today into one NIF call per
invocation**, and cross weights over the NIF boundary once instead of per op.

Three decoupled layers, with op coverage grown **iteratively** per op class:

1. **Layer A** — an isolated, Nx-upstreamable topological sort
   (`EMLX.Defn.Tree.post_order/1`): an `Nx.Defn.Expr` DAG → a scope-local,
   dependency-ordered node list.
2. **Layer B** — `EMLX.Native.Expr` (the IR): expand each topo-ordered node into
   ≥1 instruction(s) with tagged operand refs (kind + index packed into int64),
   op-name atoms, and an integer attribute channel; control flow and blocks become
   nested child programs.
3. **Layer C** — a C++ program backed by an op-name→function registry
   (`emlx_compiler.cpp`); `compile_program` bakes the interpreter into a lambda
   wrapped with `mlx::core::detail::compile` (unique ID per Expr), so MLX traces
   and caches the graph on first call and replays it on subsequent calls. One NIF
   call per `defn` invocation.

The `EMLX` compiler is **single-mode**: it always lowers via this structure.
There is no `:native` flag and no eager-Evaluator fallback lane; lowering
control is structural, via `Nx.Defn.Block` (see "Lowering control" below).

> **Known discrepancy (as of Stage 15), closed by Stages 16–19:** the code
> today (`emlx.ex`'s `try_native_compile/3`) still catches unsupported-op
> errors and silently delegates the whole `defn` to `Nx.Defn.Evaluator` — a
> leftover incremental-development safety net that contradicts this
> paragraph. Stages 16–19 close every reachable raise path and delete that
> lane so the claim above is true in the code, not just here.

Stages 20–23 extend this plan's charter beyond the compiler itself: EMLX's
sibling/successor project `~/coding/emily` has shipped a materially larger
feature set (native-lane observability, SDPA attention sinks, microscaled
quantization, mixed-precision training, …). Those stages audit and close the
gap where it's real (several items already exist in EMLX under a different
name — see Stage 20).

## Resolved decisions (drive everything)

1. **Single-mode compiler.** One execution path: the new lowering structure.
   Lowering is total over the primitive op set; control over native-vs-default
   lowering is expressed through `Nx.Defn.Block`, not compiler options.
2. **Topo-sort vendored as `EMLX.Defn.Tree.post_order/1`** —
   `emlx/lib/emlx/defn/tree.ex`, namespaced to mirror `Nx.Defn.Tree` so the
   eventual upstream move is a rename.
3. **C++ compile/eval lands early** (Stage 01) so perf is validated from the
   start; each op class is then implemented in steps.
4. **Module home**: `emlx/lib/emlx/defn/`.
5. **`post_order/1` emits the same `%Nx.Tensor{}` structs it received**,
   reordered into dependency-first sequence (pure reordering of one scope).
6. **Control-flow sub-scopes are resolved in Elixir, not the IR** (revised in
   Stage 08): `cond` lowers inline as `:select` ops, and `while` is split out
   with `Nx.Defn.Graph` and replayed by recursively re-entering this compiler
   (`Graph.run(compiler: EMLX)`), with the loop driven host-side. `fun` is
   still TBD.

## Why feasible for EMLX specifically

The compiler is a graph-compiler front end layered on EMLX's existing backend
and queue dispatch. EMLX already owns most of the substrate:

- A complete `Nx.Backend` — **every backend-callback op already has C++/MLX
  semantics** (`emlx/c_src/emlx_nif.cpp`). Lowering reuses those; we build each
  op into a graph instead of eval'ing it eagerly. No new kernels.
- `EMLX.CommandQueue` worker/queue dispatch — the substrate the replay NIF runs on.
- `EMLX.Fast` fused kernels (RMSNorm / RoPE / SDPA / LayerNorm).
- A `Nx.Defn.Compiler` already wired up in `emlx/lib/emlx.ex`
  (`__jit__/__compile__/__partitions_options__/__to_backend__`) that today
  **delegates to `Nx.Defn.Evaluator`** — the seam we replace.

MLX is lazy, so EMLX already gets intra-defn graph fusion before the final read.
What it lacks is dispatch-cost amortization — every Expr node is its own NIF
call today. That is the cost this compiler removes.

## Architecture (single lane)

```
EMLX (Nx.Defn.Compiler)  — one path: trace -> topo-sort -> lower -> compile -> replay
  │
  ├─ Layer A: EMLX.Defn.Tree.post_order/1   (PURE, no EMLX deps — upstream candidate)
  ├─ Layer B: EMLX.Native.Expr              (the IR; tagged refs, op-name atoms, iattrs, subprograms)
  └─ Layer C: C++ program                   (op-name registry; mlx::core::detail::compile per Expr)
```

Per-layer oracle (a bug can only live in the layer whose test fails): Layer A
vs hand-checked orderings; Layer B vs eager `EMLX.Backend` (via an Elixir IR
interpreter); Layer C vs Layer B.

## Lowering control via `Nx.Defn.Block`

Single-mode ⇒ no "route the whole defn through the Evaluator" escape hatch.
Control over native-vs-primitive lowering is structural:

- A `block` Expr node carries `[struct, block_args, default_expr, fun]`. The
  `struct` is an `Nx.Block.*` value (e.g. `Nx.Block.LinAlg.QR`,
  `Nx.Block.CumulativeSum`, `Nx.Block.TopK`, `Nx.Block.FFT2`,
  `Nx.Block.AllClose`, `Nx.Block.Phase`, …); `default_expr` is the traced
  primitive decomposition.
- The lowerer handles a `block` node either by **recognizing the struct**
  (emit a native / fused instruction — the LinAlg / `EMLX.Fast` path), or by
  **descending into `default_expr`** (lower the primitive expansion — the
  built-in, always-available per-block default).
- Genuinely unlowerable nodes (`token`/`attach_token` hooks, `runtime_call`,
  any host side-effecting construct) **raise** — no silent fallback. During
  incremental development, a not-yet-implemented op class also raises; that is
  expected and bounded by the burndown.

## Stages

Tackle in order. Stages 00–01 are foundational; 02–10 grow op coverage and are
each independently shippable. Run with
`/tackle-step workdir/native-compiler <stage_name>`.

- [x] [`00-topo-sort`](00-topo-sort.md) — `EMLX.Defn.Tree.post_order/1` (Layer A), pure, no C++.
- [x] [`01-ir-cpp-substrate`](01-ir-cpp-substrate.md) — `EMLX.Native.Expr` IR + C++ `compile_program`/`eval_program` + compiler seam + `add` end-to-end + perf baseline. Post-stage: `mlx::core::detail::compile` with unique IDs; op-name string registry replaces enum + wire integers. **Perf gate soft-pass — see stage doc § Perf findings.**
- [x] [`02-elementwise`](02-elementwise.md) — unary + binary + compare/logical.
- [x] [`03-shape-movement`](03-shape-movement.md) — reshape, transpose, squeeze, broadcast, pad, reverse, as_type, bitcast, concatenate, stack.
- [x] [`04-reductions-dot-conv`](04-reductions-dot-conv.md) — reductions + argmax/argmin + dot + conv.
- [x] [`05-indexing-selection`](05-indexing-selection.md) — select, clip, slice, put_slice, gather, take, take_along_axis, indexed_add/put.
- [x] [`06-sort-window-cumulative-fft`](06-sort-window-cumulative-fft.md) — sort/argsort, window reductions, cumulative, fft family. **`expand_block_via_default` fallback enables rfft/irfft and future unrecognized blocks.**
- [x] [`07-creation-rng`](07-creation-rng.md) — iota, eye, `Nx.Random` primitives (via threefry2x32 decomposition).
- [x] [`08-control-flow`](08-control-flow.md) — `cond`, `while`. **`cond` = inline `:select` ops; `while` = `Nx.Defn.Graph.split` + recursive `Graph.run(compiler: EMLX)`, Elixir host loop for each isolated while. Non-tail/nested/while-as-input compile natively.**
- [x] [`09-blocks-linalg`](09-blocks-linalg.md) — `Nx.Block.LinAlg.*` recognize-struct path + `default_expr` descent. **Native CPU-pinned `mlx::linalg` opcodes (cholesky/solve/triangular_solve + multi-output qr/eigh/svd/lu via new multi-output IR); determinant via `default_expr` descent (N>3 through recognized native LU). cpu-pin composes in compiled graph on both `:cpu`/`:gpu`; linalg outputs `contiguous`-wrapped to avoid a strided CPU `Compiled`-kernel JIT failure.**
- [x] [`10-fast-kernels`](10-fast-kernels.md) — pattern-route to `EMLX.Fast`. **`EMLX.Fast.*` surface as `:runtime_call` nodes (not blocks); recognize the callback (module+name+arity) → single fused `mlx::core::fast::*` opcode in the compiled graph. Float opts ride the int64 attr channel as IEEE-754 bits. Decode/T=1 callbacks fused; prefill RoPE raises → Evaluator fallback. ~1.3–1.4× over primitive replay on a decode block.**
- [x] [`11-bench-regression`](11-bench-regression.md) — **investigation, resolved.** `validate_qwen3.exs` regression root-caused to three `Nx.Defn.Graph.split` bugs (not `emlx.ex`): exponential `rewrite_subtree` (hang), `runtime_call` operand under-collection (param-index crash), and non-tuple final-stage output in `run/3`. Fixed in the nx fork; bench runs end-to-end (`bb base` 7.3 / `bb+rewrite` 23.4 / `native` 71.4 tok/s); regression tests added; suites green.
- [x] [`12-childprogram-spike`](12-childprogram-spike.md) — spike resolved. **No-go on the C++ child-program path.** Static fold is graph-equivalent to a pure-Elixir inline-unroll, so the C++ `:fold` buys nothing measurable (payload/build savings negligible; replay identical). `:reduce` now lowers via static trace-time unroll (Elixir, reuses existing opcodes, zero C++ change), validated vs the Evaluator. Stage 13 = Elixir unroll; Stage 14 C++ `while` dropped (MLX has no in-trace control flow → eval-per-iteration matches the proven Stage-08 host loop).
- [x] [`13-custom-fun-reductions`](13-custom-fun-reductions.md) — full `reduce` / `window_reduce` custom-fun lowering via Elixir static unroll (Stage-12-blessed). `window_reduce` = pad-with-acc + fold reducer body over the prod(window_dims) within-window offsets via strided per-offset slices. Flipped `EXPR_NODES.md` 109/131; suite 236 passed. Associative-reducer→native-`window_*` perf routing deferred.
- [~] [`14-while-childprogram`](14-while-childprogram.md) — **dropped; no-go re-affirmed by measurement (2026-06-30 revisit).** MLX 0.31.2 has no lazy control flow, so a C++ `while` still hits an `eval` barrier per iteration (no cross-iteration fusion). Benchmark (`emlx/bench/while_dispatch_bench.exs`): C++ saves ≤30 % (GPU) per iter for **convergent** loops, shrinking with body weight; for **counted** loops the host loop already fuses the body lazily, so a C++ eval-per-iteration `while` is a **regression**. `Graph.split` fragmentation is a fixed per-invocation cost, amortized to noise. Host loop retained. **Side finding, fixed:** counter-only bare-while (cond doesn't read the full carry) had a correctness bug — `EMLX.Native.Expr.lower/2` now densifies its wire input list by arity hint instead of compacting to referenced positions only; regression tests added.
- [x] [`15-block-completeness-rope-prefill`](15-block-completeness-rope-prefill.md) — (a) AllClose/Phase/TopK block descent equivalence-tested (TopK needed a `default_expr`-is-a-tuple fix in `expand_block_via_default`, via `flat_refs`) → `EXPR_NODES.md` line 156 flipped. (b) Prefill RoPE (`rope_with_positions_callback`/`rope_with_freqs_callback`, T>1) lowers to an in-graph cos/sin/rotate primitive composition (no new C++ kernel) → `runtime_call` flipped to `[x]`. **Side finding, filed not fixed:** `mlx::core::fast::rope` itself miscomputes non-head-0 rotations for multi-head (H>1) input in EMLX's non-transposed layout, affecting the existing decode/T=1 fast callbacks (out of scope here) — see `mlx-fast-rope-multihead-bugreport.md`.

### Zero evaluator-fallback (closes the single-mode gap left open since Stage 01)

- [x] [`16-expr-nodes-doc-audit`](16-expr-nodes-doc-audit.md) — audit stale `EXPR_NODES.md` `[ ]` boxes (`fun`/`optional`/`from_binary`); confirmed all three are unreachable/subsumed (not real gaps) via re-grep against the vendored Nx fork; flipped the doc; two regression tests pin the `:fun` no-op invariant. No `expr.ex` code changes needed.
- [x] [`17-block-while-descent`](17-block-while-descent.md) — close the `while`-nested-inside-a-block's-`default_expr` structural boundary. Statically unrolls counted `while` loops reached via block descent (fixes `Nx.Block.LinAlg.QR :complete`); `SVD full_matrices?: false` turned out to have no `while` at all in the current Nx fork (rewritten to a Gram-matrix decomposition) — needed only prerequisite `:eye`/`:constant`/`:metadata` fixes for non-scalar/vectorized shapes hit via the same descent path. `triangular_solve`'s non-default variants are a separate, unrelated gap (direct op-node, not a `default_expr` `while`) — descoped, still raises.
- [x] [`18-hooks-token-splitting`](18-hooks-token-splitting.md) — answered "no" to the `while`-style split question: hooks are fire-and-forget, not control flow, so they lower in the *same* single NIF-call program via an extra-output design (no `Graph.split`, no host round-trip) — `:attach_token` is a zero-instruction passthrough, `:token` rides its hook(s) as extra program outputs fired host-side after the one `eval_program` call returns. **Cond-branch-local hooks hard-raise** (EMLX's `cond` evaluates every branch unconditionally, which would double-fire such a hook — a correctness carve-out, not a coverage gap); while-body hooks need no such guard (equivalence-tested vs Evaluator). **Found and fixed a real `Nx.Defn.Graph.split` bug** (`do_rewrite_subtree/3` had no `:token` clause, silently dropping hook-payload parameter remapping across a `while`'s stage boundary) — same "found via testing" pattern as Stages 11/17.
- [ ] [`19-retire-evaluator-fallback`](19-retire-evaluator-fallback.md) — once 16–18 land, delete `try_native_compile`'s `Nx.Defn.Evaluator` delegation branch from `emlx.ex` entirely; unsupported ops hard-raise, no silent whole-defn fallback, matching this README's single-mode claim in code.

### Emily backend-parity (expanded charter — see the note above "Resolved decisions")

- [ ] [`20-emily-parity-audit`](20-emily-parity-audit.md) — docs-only gap audit against `~/coding/emily`'s PLAN.md/ROADMAP.md (M0–M27); marks already-at-parity items closed and scopes Stages 21–23.
- [ ] [`21-observability`](21-observability.md) — `:telemetry` spans (eval/to_binary/memory-stats) + compile-time `:debug_bounds_check`/`:debug_detect_nan_inf` flags (Emily M18/M22 parity).
- [ ] [`22-fast-kernel-quant-parity`](22-fast-kernel-quant-parity.md) — SDPA attention sinks, microscaled quantization modes (mxfp4/mxfp8/nvfp4), public `einsum` helper (Emily M25/M26/M27 parity).
- [ ] [`23-gradient-training-conformance`](23-gradient-training-conformance.md) — scoping-only epic: triage grad/training behavior under `compiler: EMLX` (currently untested), name follow-on stages for real gaps (Emily M9/M13/M16/M17 parity).

## Decision gates

- **After 00**: confirm the `post_order/1` shape — minimal (lowerer recurses
  into child scopes) vs richer (`{ordered_nodes, child_scopes}`). **Decision:
  minimal.** Richer shape couples Stage 00 to IR concerns and hurts
  Nx-upstreamability; Stage 08 will own child-scope recursion.
- **After 01**: perf gate — the single-NIF replay must beat the current
  op-by-op Evaluator path on a multi-op `defn` (dispatch-collapse thesis). If
  it does not, stop and rethink before growing coverage.  
  **Status:** Hard-pass as of Stage 02. The Stage 01 benchmark used `Nx.add(x, 1)` chained 10×; Nx.Defn constant-folds repeated scalar additions into a single op, so the "10-add chain" was a 1-op graph. Stage 02 switched to `Nx.add(x, y)` with a runtime `y` — a genuine 10-instruction program. Native path is dramatically faster. `eval_program` no longer calls `mlx::core::eval` eagerly (lazy outputs since Stage 02).
- **Ongoing**: every op added must pass an equivalence test vs eager
  `EMLX.Backend` (within tolerance) before its `EXPR_NODES.md` box flips.
- **After 18**: decided — hooks lower natively via an extra-output design
  (no structural split needed; they aren't control flow), except a
  cond-branch-local hook, which raises permanently and deliberately (a
  correctness carve-out `Nx.Defn.Evaluator` doesn't have to make, since it
  evaluates only the taken branch). Stage 19 should name this one construct
  explicitly as the sole intentional hard-raise, distinct from a coverage gap.

## Testing philosophy (per-layer oracle)

| Layer | Oracle |
|-------|--------|
| A (topo-sort) | Hand-checked orderings; property: every node after its operands |
| B (lowering)  | Eager `EMLX.Backend` via the IR interpreter, same inputs |
| C (replay)    | Layer B interpreter output |
| E2E           | Existing EMLX conformance / Bumblebee suites |

## Key file references

- EMLX compiler seam: `emlx/lib/emlx.ex` (`__compile__/4` ~line 1320).
- Nx traversal: `emlx/deps/nx/lib/nx/defn/tree.ex` (`apply_args/4` `:scope`,
  `scope_ids/1`), `emlx/deps/nx/lib/nx/defn/composite.ex`.
- Node taxonomy: `emlx/deps/nx/lib/nx/backend.ex` (callbacks) +
  `emlx/deps/nx/lib/nx/defn/expr.ex` (syntax nodes) +
  `emlx/deps/nx/lib/nx/shared.ex` (`unary_math_funs/0`) +
  `emlx/deps/nx/lib/nx/block.ex` (`Nx.Block.*` structs).
- Coverage probe: an op-coverage script (to be written) that probes every Nx
  op through `compiler: EMLX` and reports OK/MISS for the burndown.
- C++ to reuse: `emlx/c_src/emlx_nif.cpp`, `emlx/c_src/emlx_fast.cpp`,
  worker/queue in `emlx/c_src/emlx_worker.hpp`.
