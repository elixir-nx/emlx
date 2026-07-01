# Stage 17 — close the while-in-`default_expr` structural boundary

Status: not started.

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

- QR `:complete`, SVD `full_matrices?: false`, and non-default
  `triangular_solve` all lower natively (no raise) with equivalence tests.
- `block`'s structural-boundary caveat is removed from `EXPR_NODES.md`; no
  `Nx.Block.*` variant remains that raises solely because of a nested `while`.
