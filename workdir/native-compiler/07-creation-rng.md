# Stage 07 — Creation + RNG

Status: done

## Why this stage exists

Creation ops (`iota`, `eye`) and `Nx.Random` primitives have no tensor input
operands — they are pure producers parameterized by shape/dtype (and, for RNG,
a key/seed that must thread through the graph deterministically). They test the
lowering of nodes with only `iattrs` (and key) operands, and are needed for
sampling / dropout / weight-init paths.

## Procedure

1. Lower (`EXPR_NODES.md` §I, §J): iota (with optional axis), eye, and the
   `Nx.Random` primitives (random_uniform / random_normal and key splitting).
2. Decide RNG key handling: keys are tensors threaded as ordinary operand refs;
   confirm the lowering preserves `Nx.Random`'s split/derivation so results are
   bit-reproducible vs eager `EMLX.Backend` with the same key.
3. Encode shape/dtype/axis into `iattrs`; add opcodes + C++ replay; parity test.
4. Equivalence tests vs eager `EMLX.Backend` (fixed key for RNG); flip boxes.

## Acceptance

- iota / eye lower and replay correctly within tolerance vs eager
  `EMLX.Backend`.
- `Nx.Random` primitives produce results matching eager `EMLX.Backend` for a
  fixed key (deterministic key threading verified).
- `EXPR_NODES.md` §I/§J boxes flipped; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| iota (flat + axis-specific) | ✅ | `iattrs = [dtype_int, n_dims, axis_int, d0..dn-1]`; axis_int = −1 encodes nil (flat); C++ uses `arange` + `reshape` (flat) or `arange` + `reshape` + `broadcast_to` (axis); all dtypes tested |
| eye (rectangular) | ✅ | `iattrs = [dtype_int, m, n]`; C++ calls `mlx::core::eye(m, n, 0, dtype)`; 3×3 and 2×4 tested |
| RNG primitives | ✅ | `Nx.Random.uniform` and `Nx.Random.normal` work via threefry2x32 decomposition — no special lowering needed; all internal ops (bitwise, add, iota, reshape, slice) already lowered |
| key threading deterministic | ✅ | Same key → same samples in consecutive JIT calls; native matches `Nx.Defn.Evaluator` to 1e-5 |
| "does not yet lower" sentinel | updated | Test sentinel changed from `:iota` (now lowered) to custom-fun `:reduce` (still deferred, Stage 08) |
| Tests | 16/16 ✅ | 16 Stage 07 tests: iota IR shape + interpreter + C++ parity + 3 E2E; eye same; RNG uniform determinism + normal |
| compile/format clean | ✅ | `mix compile --warnings-as-errors` + `mix format --check-formatted` both clean |
