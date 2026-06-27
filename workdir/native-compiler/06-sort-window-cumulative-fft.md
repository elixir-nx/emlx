# Stage 06 — Sort / window / cumulative / FFT

Status: not started

## Why this stage exists

This batch rounds out the array-op surface. Several of these arrive as
`Nx.Block.*` nodes (CumulativeSum/Product/Min/Max, FFT2/IFFT2, TopK,
Take/TakeAlongAxis) rather than raw primitives, so this is the first stage to
exercise the **block-recognition lowering path** (README "Lowering control")
alongside primitive lowering.

## Procedure

1. Lower (`EXPR_NODES.md` §G, §H):
   - sort, argsort;
   - window_sum/max/min/product (+ window_scatter_max/min);
   - cumulative_sum/product/min/max (last-axis fast path + interior axis);
   - fft, ifft, fft2/ifft2, rfft/irfft.
2. For ops that surface as `Nx.Block.*`: implement the **recognize-struct**
   path (emit the native op from the block struct + args), with **descend into
   `default_expr`** as the fallback for any block variant not yet special-cased.
3. Add opcodes + C++ replay; parity test; equivalence tests vs eager
   `EMLX.Backend`; flip §G/§H boxes.

## Acceptance

- Sort/window/cumulative/FFT ops lower and replay correctly within tolerance
  vs eager `EMLX.Backend`, including the interior-axis cumulative case.
- At least one `Nx.Block.*` node lowered via the recognize-struct path, with
  `default_expr` descent demonstrated as the fallback.
- `EXPR_NODES.md` §G/§H boxes flipped; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| sort/argsort | | |
| window reductions | | |
| cumulative (incl. interior axis) | | |
| fft family | | |
| block recognize-struct path | | |
