# Stage 03 — Shape / movement ops

Status: not started

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
| Shape ops lowered | | |
| iattrs encoding documented | | |
| Variadic (concat/stack) | | |
| Equivalence tests | | |
