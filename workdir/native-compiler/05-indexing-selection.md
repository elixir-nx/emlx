# Stage 05 — Indexing / selection

Status: complete

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
| select | pass | `mlx::core::where`; 3-way parity (NIF / interpreter / EMLX.Backend) |
| clip | pass | `mlx::core::clip` with optional min/max sentinels; iattrs carry has_min/has_max bitmask |
| slice | pass | Static dims → `mlx::core::slice`; dynamic tensor dims → `arange * stride + clamped_start` via `mlx::core::take`. Mixed static+dynamic works. |
| put_slice | pass | Dynamic starts assembled into a 1-D `int32` array and passed to the `mlx::core::slice_update(tensor, update, starts, axes)` overload. Clamped to `[0, shape[i] − len[i]]`. |
| gather | pass | Indices split along last axis; `mlx::core::gather`; output reshaped to match Nx convention |
| take | pass | `mlx::core::take` with axis from iattrs |
| take_along_axis | pass | `mlx::core::take_along_axis` with axis from iattrs |
| indexed_add | pass | Indices split along last axis; `mlx::core::scatter_add` |
| indexed_put | pass | Indices split along last axis; `mlx::core::scatter` |
| dynamic-index tests | pass | KV-cache pattern (`put_slice` with runtime row index), mixed-index `slice`; 27 tests total |
| iattrs encoding | complete | `dynamic_mask` bitmask distinguishes per-dim static vs tensor starts for slice/put_slice; documented in `expr.ex` moduledoc |

27 Stage 05 tests pass (`mix test --only stage05`).
