# Stage 17 — close the while-in-`default_expr` structural boundary

Status: done.

## Why this stage exists

`build_eval_fn` finds `:while` nodes by scanning the *top-level* traced
expression before handing off to `Nx.Defn.Graph.split` (Stage 08). It never
looks inside a `Nx.Block.*` node's `default_expr`, so a `while` nested in a
block's decomposition still raises `"does not yet lower op"` and fires the
Evaluator fallback. Per Stage 09/15's notes, this affects at least:

- `Nx.Block.LinAlg.QR` with `mode: :complete`
- `Nx.Block.LinAlg.SVD` with `full_matrices?: false`
- `Nx.Block.LinAlg.TriangularSolve` with `left_side: false` or
  `transform_a != :none`

Combined with Stage 16 (which closes out `fun`/`optional`/`from_binary` as
non-issues), this is the last *reachable* non-hook gap standing between EMLX
and true zero-fallback.

## Procedure

1. Re-enumerate exactly which `default_expr` decompositions (in
   `nx/lib/nx/block.ex` and any EMLX-side default_exprs) currently contain a
   `while` — confirm against the present Nx fork; the set may have drifted
   since Stage 09/15 were written.
2. Extend the pre-split `:while`-discovery pass so it recurses into `block`
   nodes' `default_expr` (and any other node whose args carry a full
   sub-expression) before `Nx.Defn.Graph.split` runs, so nested whiles are
   found and split just like top-level ones.
3. Verify the recompile side composes: when a split stage's boundary falls
   *inside* a block's `default_expr`, `expand_block_via_default`'s
   "recognize struct vs descend into default_expr" dispatch happens per-node
   during lowering, not once per top-level expression, so it should already
   work when re-entered per split stage — but this must be tested explicitly,
   not assumed.
4. Equivalence tests (vs eager `EMLX.Backend`) for each of the three named
   LinAlg variants above, plus any others step 1 turns up.
5. Flip `EXPR_NODES.md`'s `block` line (section A) and the K-section caveat
   to `[x]`; delete the "→ Evaluator fallback" language left in Stage 09/15's
   docs (superseded by this stage).

## Acceptance

- QR `:complete` and SVD `full_matrices?: false` lower natively (no raise)
  with equivalence tests.
  ~~and non-default `triangular_solve`~~ — **descoped, not met.** Investigation
  (see Results) found `triangular_solve`'s `left_side: false`/
  `transform_a != :none` raise comes from a direct `:triangular_solve`
  op-node clause, not a block whose `default_expr` contains a `while` — it
  was miscategorized as a while-in-block case in this doc and in Stage
  09/15's notes. Not a structural-boundary issue this stage's charter covers;
  advisor-approved descope. It still raises, unchanged, after this stage.
- `block`'s structural-boundary caveat is removed from `EXPR_NODES.md`; no
  `Nx.Block.*` variant remains that raises solely because of a nested `while`.
  **Met** — `triangular_solve`'s remaining raise is a direct op-node gap, not
  a nested-`while` case, so it doesn't violate this criterion.

## Results

**Re-enumeration (step 1) changed the target set.** Against the actual `nx`
fork this project builds against (a `path:` dependency at
`/Users/valente/coding/nx/nx` — *not* the stale, gitignored, unused mirror at
`emlx/deps/nx`, which still has an older copy and caused a lot of wasted
debugging before this was caught), `Nx.LinAlg.SVD`'s `full_matrices?: false`
path was rewritten to a non-iterative Gram-matrix (`AᵀA` → `eigh`)
decomposition — its `default_expr` contains **no `while` node at all**
anymore. `Nx.LinAlg.QR`'s `mode: :complete` Householder decomposition still
has one (a statically-counted range loop). `triangular_solve`'s
`left_side: false`/`transform_a != :none` raise comes from a direct
`:triangular_solve` op-node clause, not a block `default_expr` — it was never
a `while`-in-`default_expr` case, just miscategorized in Stage 09/15's notes.
Per advisor sign-off, descoped from this stage (see below); it still raises,
unchanged.

**Approach taken (deviates from the doc's original step 2–3 plan).** Rather
than teaching the pre-split `:while`-discovery pass to recurse into blocks
(extending `Nx.Defn.Graph.split`'s reach, which has its own correctness gaps
for remapping params inside a `default_expr`), `expand_node` gained a direct
`:while` clause that fires only when reached via block descent (a top-level
`while` is still intercepted earlier by `build_eval_fn`, never reaching this
clause). It detects the exact shape `Nx.Defn.Expr.while_range/7` emits for a
range-generator loop with `unroll: false` (the default for
`while acc, i <- lo..hi//step do ... end`): start index, bound, and step are
all trace-time constants, so the trip count is knowable without inspecting
runtime values. When detected, the loop body (a tuple of expr roots) is
re-lowered `count` times, chaining each iteration's output refs into the
next's parameter bindings — the same "re-lower body once per iteration,
overwrite `node_to_ref` for the shared param ids" idiom Stage 12/13 already
used for custom-fun `reduce`/`window_reduce`, generalized from one
accumulator to a loop-carried tuple. A nested `while` that doesn't match this
shape still raises `does not yet lower op :while`.

**Three unrelated pre-existing bugs blocked reaching the `while` node (and
SVD's non-`while` path) at all**, discovered by testing rather than static
reading — fixed as prerequisites:
1. `:eye`'s handler assumed exactly rank-2 output (`[m, n] = Tuple.to_list(node.shape)`); both `Nx.LinAlg.qr` and `Nx.LinAlg.svd` unconditionally wrap their input in `Nx.revectorize([collapsed_axes: :auto], ...)`, which prepends a batch dim (size 1 for non-batched input too) to every op inside — including the internal `eye` calls used to seed the algorithms. Fixed by emitting the rank-2 `:eye` then broadcasting to the full shape (MLX's `eye` primitive itself has no batch support).
2. `:constant`'s handler stored only a scalar value + type, silently ignoring `node.shape`. Nx's tracer folds `Nx.broadcast(scalar, shape)` into a single wider `:constant` node in some paths (hit by SVD's all-zeros-branch fallback, traced unconditionally alongside the non-zero branch even for non-zero test inputs); the wire format only carries scalars, so a non-scalar constant silently became a 1-element array, and a later reshape to the real shape failed with an MLX runtime error (not even a clean Elixir raise). Fixed the same way as `:eye` — emit scalar, broadcast to `node.shape`.
3. `:metadata`'s handler assumed a single-tensor inner expr (`inner.data.id`); `Nx.Defn.Expr.metadata/2` on a *container* (tuple) produces one metadata node wrapping the raw tuple directly (used by SVD's Gram-matrix path around a `cond`'s tuple result). Fixed by storing a list of refs when `inner` is a tuple, mirroring the existing multi-output convention `:elem` already reads from.

**Verification.** Direct `EMLX.Native.Expr.lower/2` calls (bypassing the
Evaluator-fallback rescue in `try_native_compile`, which otherwise silently
masks a `does not yet lower op` raise as a successful-looking eager result)
confirm both QR `:complete` and SVD `full_matrices?: false` lower with zero
raises and zero `:while` instructions in the compiled program. Added 8
equivalence tests (`expr_test.exs`, `@tag :stage17`): QR reconstruction +
orthonormality (square, tall) and vs-Evaluator comparison; SVD reconstruction
(tall, wide, square) and vs-Evaluator singular-value comparison; a
regression-guard asserting no `:while` instruction survives QR's lowering;
and a guard confirming `triangular_solve left_side: false` is unaffected
(still raises, unrelated code path). Full suite: 2555 passed (825 doctests,
1730 tests), 0 failures, 0 regressions.

**Deviation from acceptance criteria as originally written:**
`triangular_solve`'s non-default variants do **not** lower natively — this
was descoped per advisor recommendation once investigation showed it isn't a
`while`-in-`default_expr` case (see re-enumeration above). The stage's
*charter* (close the while-in-block structural boundary) is fully met; the
doc's acceptance bullet literally naming `triangular_solve` is not.
