# Stage 09 — Blocks / LinAlg

Status: not started

## Why this stage exists

The `Nx.Block.LinAlg.*` family (Cholesky, Solve, QR, Eigh, SVD, LU,
Determinant, TriangularSolve) is the most complex use of the block-recognition
lowering path and the main remaining gap for scientific / classical-ML defns.
This stage proves the README "Lowering control" design at full strength:
recognize the block struct for a native path, else lower its `default_expr`.

## Procedure

1. For each `Nx.Block.LinAlg.*` struct (and other blocks: AllClose, Phase,
   LogicalNot, etc. not already handled): implement the **recognize-struct**
   lowering, routing to a native Metal/LinAlg path where one exists.
2. Where no native path exists yet, **descend into `default_expr`** (the traced
   primitive decomposition) — confirm it lowers fully on the primitives shipped
   in Stages 02–07.
3. Add any new opcodes + C++ replay; parity test.
4. Equivalence tests vs eager `EMLX.Backend` and, where tolerance is delicate
   (svd/eigh), against an EXLA/`Nx.BinaryBackend` golden with documented
   tolerances; flip `EXPR_NODES.md` §K boxes.

## Acceptance

- Every `Nx.Block.LinAlg.*` op either lowers via a native path or via
  `default_expr` descent, with results within documented tolerance vs the
  reference oracle.
- `default_expr` descent demonstrated for at least one block whose primitives
  all come from earlier stages.
- `EXPR_NODES.md` §K boxes flipped; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| LinAlg blocks (native vs default) | | |
| Other Nx.Block.* | | |
| tolerance-sensitive goldens | | |
