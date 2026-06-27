# Stage 05 — Indexing / selection

Status: not started

## Why this stage exists

Indexing ops mix tensor and integer/tensor-index operands (e.g. `slice` start
indices can be scalars *or* tensors — see the `apply_args` special cases), and
are essential for KV-cache updates and gather/scatter in transformers. They
stress the operand-classification path (which args are refs vs inline ints).

## Procedure

1. Lower (`EXPR_NODES.md` §F): select, clip, slice, put_slice, gather, take,
   take_along_axis, indexed_add, indexed_put.
2. Handle `slice`/`put_slice` mixed start indices (integers stay inline in
   `iattrs`; tensor starts become operand refs) per `Nx.Defn.Tree.apply_args`'s
   special handling.
3. Encode axis / slice-size / index-vector-axis metadata into `iattrs`.
4. Add opcodes + C++ replay (reuse `emlx_nif.cpp`); parity test; equivalence
   tests vs eager `EMLX.Backend` (static + dynamic indices); flip §F boxes.

## Acceptance

- All §F ops lower and replay correctly within tolerance vs eager
  `EMLX.Backend`, including dynamic (tensor) start indices for slice/put_slice.
- Mixed inline-int / tensor-ref operand handling correct and tested.
- `EXPR_NODES.md` §F boxes flipped; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| select/clip/slice/put_slice | | |
| gather/take/take_along_axis | | |
| indexed_add/put | | |
| dynamic-index tests | | |
