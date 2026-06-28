# Stage 03 — Shape / movement ops

Status: done

## Why this stage exists

Shape/movement ops introduce the **integer attribute channel** (`iattrs`) in
earnest — axes, target shapes, padding configs encoded as integers across the
NIF boundary — and the variadic-operand pattern (`concatenate`/`stack` take a
list). Getting these right unblocks reductions and indexing, which lean on the
same axis-encoding machinery.

## Procedure

1. Lower (`EXPR_NODES.md` §D): reshape, squeeze, transpose, broadcast, as_type,
   bitcast, pad, reverse, concatenate, stack.
2. Establish `iattrs` conventions: axis lists, dtype codes (for as_type/bitcast),
   pad config triples `{lo, hi, interior}` — document the encoding next to the
   opcode table and keep it in lockstep with the C++ decode.
3. Handle variadic operands (`concatenate`/`stack` — `args` is `[list | rest]`);
   ensure `post_order` ordering already linearizes the list elements.
4. Add opcodes + C++ decode/replay (reuse `emlx_nif.cpp`); parity test.
5. Equivalence tests vs eager `EMLX.Backend` (varied ranks, axes, neg axes,
   broadcasting edge cases); flip `EXPR_NODES.md` §D boxes.

## Acceptance

- All §D ops lower and replay correctly, matching eager `EMLX.Backend` within
  tolerance, including negative axes and rank-changing cases.
- `iattrs` encoding for axes / shapes / pad configs documented and parity-tested
  against the C++ decode.
- `EXPR_NODES.md` §D boxes flipped; compile/format clean; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| Shape ops lowered | ✅ 10 ops | reshape/squeeze/transpose/as_type/bitcast/broadcast/pad/reverse/concatenate/stack in `emlx/lib/emlx/native/expr.ex` |
| iattrs encoding documented | ✅ | Opcode table in `EMLX.Native.Expr` moduledoc; encoding mirrored in `emlx_compiler.cpp` comment block |
| Variadic (concat/stack) | ✅ | All tensor refs emitted as `operands`; axis in `attrs[0]`; negative axis normalised in Elixir |
| Equivalence tests | ✅ 31 tests | All §D ops tested vs eager `EMLX.Backend` across f32/s32/bf16/u8; negative axes, rank-changing, broadcasting edge cases; concat/stack with 3+ tensors; 3 Interpreter↔C++ parity tests (reshape, broadcast, concatenate); squeeze with no explicit axes |

**Notes:**
- `pad` raises for `interior > 0` or negative `lo`/`hi` (not yet lowered; raises with clear message).
- `as_type` reuses Stage 02's `:astype` opcode; `emit_cast_to` always emits the cast for explicit `:as_type` nodes.
- `broadcast` mirrors `EMLX.Backend.maybe_reshape` + `broadcast_to` exactly: builds intermediate all-1s shape then places input dims at axis positions.
- All 79 tests pass (31 new Stage 03 + 48 previous). `mix compile --warnings-as-errors` and `mix format --check-formatted` clean.
