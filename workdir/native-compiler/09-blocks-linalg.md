# Stage 09 — Blocks / LinAlg

Status: done

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
   (svd/eigh), against an EXLA/`Nx.BinaryBackend` reference with documented
   tolerances; flip `EXPR_NODES.md` §K boxes.

## Acceptance

- Every `Nx.Block.LinAlg.*` op either lowers via a native path or via
  `default_expr` descent, with results within documented tolerance vs the
  reference reference.
- `default_expr` descent demonstrated for at least one block whose primitives
  all come from earlier stages.
- `EXPR_NODES.md` §K boxes flipped; CI green.

## Results

### Design decision (advisor-reviewed)

Realized LinAlg as **native C++ opcodes inside the compiled program** (not
host-split points), per the advisor's recommendation, gated on a spike. The
spike proved that pinning `mlx::core::linalg::*` to the **CPU device**
(`k_linalg_cpu`) lets the primitive compose inside a `detail::compile`d graph
**regardless of the graph's default device** — validated on both `:cpu` and
`:gpu` defaults. This removed the need for the device-gated / `while`-splice
fallback that was originally feared (user point 3): the same cpu-pinned opcode
works on GPU graphs too, so no descent into the `while`-containing default
decomposition is required for the supported variants.

### Multi-output IR

`qr`/`eigh`/`svd`/`lu` are multi-output. Extended the IR minimally: an
instruction's result field may be a **list of refs**; `to_wire/1` assigns each
output a consecutive **flat** result index (single-output programs unchanged);
C++ gained a `multi_op_registry` whose outputs are appended to the flat results
accumulator; the interpreter binds each output ref in order. Hand-rolled
recognize clauses (no block protocol yet — deferred, per user point 2).

### CPU strided-kernel pitfall (resolved)

Under `detail::compile`, MLX fuses a factorization's elementwise tail (solve's
permutation; LU's triangular L/U masks) into a **strided** `Compiled` CPU
kernel that fails to JIT in some environments (`[Compile::eval_cpu] … pclose()
failed`). Fix: wrap every linalg output in `mlx::core::contiguous` (a plain Copy
primitive). cholesky/qr/eigh/svd were unaffected; solve/lu needed it.

For **batched** (rank-3) `lu`/`solve` the strided kernel becomes rank-3 and can
still trip `pclose()` on CPU even with the `contiguous`-wrap — an MLX/env
limitation, not a correctness bug (see below). Those batched variants are kept
out of the CPU CI suite; the 2-D paths and batched `cholesky` are exercised.

### Determinant

No MLX determinant primitive: lowers via **`default_expr` descent**. 2×2/3×3 are
pure primitives (no `while`); N>3 descends through the **recognized native LU
block** (so no `while` is ever materialized). Note: EMLX.Backend's *eager* N>3
determinant has a pre-existing `{:u,32}`/`{:s,64}` type bug, so the 4×4 test uses
a `Nx.BinaryBackend` reference reference.

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| cholesky / solve / triangular_solve | native (CPU-pinned) | `expr.ex` recognize clauses + C++ `op_registry`; element-wise vs eager `EMLX.Backend` |
| qr / eigh / svd | native (CPU-pinned, multi-output) | reconstruction tests (Q·R, V·diag(W)·Vᵀ, U·diag(S)·Vᵀ) — robust to sign/order ambiguity |
| lu | native (multi-output) + in-graph eye/take for P | factors vs eager + P·L·U reconstruction |
| determinant | `default_expr` descent | 2×2/3×3 pure primitives; 4×4 via recognized native LU; vs `Nx.BinaryBackend` |
| triangular_solve variants | `left_side` + `transform_a: :none` native; others raise | permanent hard-raise `does not yet lower op` (accepted by Stage 19, no Evaluator fallback exists) |
| batched / chained | correct (verified) | batched `cholesky` (CPU) + batched `lu` `P·L·U` & chained `cholesky→solve` (GPU); LU pivot→`P` rebuild broadcasts over batch dims. Batched `lu`/`solve` may still hit CPU `pclose()` (env limit) |
| tests | 15 Stage 09 tests green on `:cpu` and `:gpu` defaults (added chained, batched cholesky, det-sign); full suite passing, no regressions | `test/emlx/native/expr_test.exs` |
