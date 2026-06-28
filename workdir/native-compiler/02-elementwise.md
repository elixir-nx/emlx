# Stage 02 — Elementwise (unary + binary + compare/logical)

Status: done

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
| Unary lowered (count) | ✅ 36 | 23 math funs + abs/negate/sign/ceil/floor/round/bitwise_not/is_nan/is_infinity/conjugate/real/imag/logical_not/cbrt/erfc; count_leading_zeros/population_count raise (EMLX unsupported) |
| Binary lowered (count) | ✅ 15 | add/subtract/multiply/divide/pow/remainder/atan2/min/max/quotient + bitwise_and/or/xor/left_shift/right_shift |
| Compare/logical lowered | ✅ 9 | equal/not_equal/greater/less/greater_equal/less_equal + logical_and/or/xor (+ logical_not above) |
| dtype coercion | ✅ | Explicit `astype` IR instructions around binary ops; `@mlx_type_to_int` / `int_to_dtype` maintain Elixir↔C++ parity |
| Equivalence tests | ✅ 23 tests | f32/bf16/s32/u8 across all op groups; mixed-dtype upcast; compare→u8; interpreter↔C++ parity |
| compile/format clean | ✅ | `mix compile --warnings-as-errors` + `mix format --check-formatted` both clean |
| astype opcode | ✅ | First-class IR opcode with dtype int in `attrs[0]`; synced Elixir `@mlx_type_to_int` ↔ C++ `int_to_dtype()` table |
| instruction format | ✅ | 4-tuple `{ref, op, operands, attrs}`; all Stage 01 + tree tests remain green |

All 53 tests (24 Stage 02 + 23 Stage 01/seam + 6 tree) pass. 1 perf gate excluded.

**Parity note:** The original integer-opcode parity table was removed in Stage 01's post-stage refactor (string registry replaced it). The C++ `compile_program` NIF validates every op name against the registry at compile time and returns `"emlx::native: unknown op \"foo\""` for any unknown key. The 24 Stage 02 tests therefore implicitly verify Elixir↔C++ op-name parity across all registered ops.
