# Stage 07 — Creation + RNG

Status: not started

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
| iota / eye | | |
| RNG primitives | | |
| key threading deterministic | | |
