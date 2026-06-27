# Stage 02 — Elementwise (unary + binary + compare/logical)

Status: not started

## Why this stage exists

Elementwise ops are the bulk of any model graph and the simplest to lower
(no shape/axis bookkeeping), so they validate the per-op expansion pattern at
volume before tackling shape and indexing ops. They also exercise
`EMLX.Backend`'s dtype-coercion rules, which the lowering must port verbatim so
the native lane matches the eager backend numerically.

## Procedure

1. Lower **unary** ops (`EXPR_NODES.md` §B): the 23 `unary_math_funs` (exp,
   log, sin, cos, tanh, sigmoid, sqrt, rsqrt, erf, … ) plus abs, negate, sign,
   ceil, floor, round, bitwise_not, count_leading_zeros, population_count,
   is_nan, is_infinity, and complex real/imag/conjugate.
2. Lower **binary** arithmetic/bitwise (`§C`): add (done), subtract, multiply,
   divide, pow, remainder, atan2, min, max, quotient, bitwise_and/or/xor,
   left_shift, right_shift — porting `EMLX.Backend`'s out-type coercion casts.
3. Lower **compare/logical** (`§C`): equal, not_equal, less, less_equal,
   greater, greater_equal, logical_and/or/xor, and unary logical_not — including
   the merge-type cast and bool→out-type coercion.
4. Add the matching opcodes to the Elixir table and C++ enum (parity test),
   reusing the existing `emlx_nif.cpp` implementations in `eval_program`.
5. Equivalence tests vs eager `EMLX.Backend` across representative dtypes
   (f32/bf16/s32/u8) and the IR-interpreter check; flip `EXPR_NODES.md` boxes.

## Acceptance

- Every op in `EXPR_NODES.md` §B and §C lowers and replays correctly, matching
  eager `EMLX.Backend` within tolerance across the tested dtypes.
- Dtype-coercion behavior matches `EMLX.Backend` (e.g. integer/float mixing,
  compare→u8) — covered by tests.
- Opcode-parity test passes with the new opcodes.
- `EXPR_NODES.md` §B and §C boxes flipped; compile/format clean; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| Unary lowered (count) | | |
| Binary lowered (count) | | |
| Compare/logical lowered | | |
| Equivalence tests | | |
