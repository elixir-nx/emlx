# Stage 06 — Sort / window / cumulative / FFT

Status: done

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
| sort / argsort | ✅ | NaN-aware (matches EMLX.Backend); `iattrs = [axis, asc_int]`; asc and desc tested; NaN ordering verified |
| window_sum/max/min/product | ✅ | `iattrs = [n_dims, op_int, lo/hi pairs, strides, window, dilations]`; sliding-window-view replicated in C++; padding=:same and strides tested; 1D and 2D |
| window_scatter_max/min | ✅ | `iattrs = [n_dims, lo/hi pairs, strides, window]`; window_scatter_impl_compiler in emlx_compiler.cpp mirrors emlx_nif.cpp |
| cumulative_sum/product/min/max | ✅ | Via `Nx.Block.Cumulative*` recognize-struct; `iattrs = [axis, reverse_int]`; `mlx::core::cumsum/cumprod/cummin/cummax`; all four tested with s32 and f32 |
| fft / ifft | ✅ | Raw Expr ops; `iattrs = [axis, n]`; `mlx::core::fft::fft/ifft`; 1D + explicit length tested |
| fft2 / ifft2 | ✅ | Via `Nx.Block.FFT2/IFFT2`; `iattrs = [ax0, ax1, n0, n1]`; `mlx::core::fft::fft2/ifft2`; 2D tested |
| rfft / irfft | ✅ | Via `expand_block_via_default` fallback; descends into default_expr (fft+slice decomposition); rfft tested |
| block recognize-struct path | ✅ | `Nx.Block.Cumulative*`, `Nx.Block.FFT2/IFFT2` recognized natively; unrecognized blocks descend into `default_expr` |
| `expand_block_via_default` | ✅ | Inner :parameter nodes mapped to parent-scope refs; inner scope expanded inline; demonstrated by rfft via Nx.Block.RFFT |
| Tests | 24/24 ✅ | All Stage 06 tests pass; all previous 115 tests (stages 01–05) unchanged |
| compile/format clean | ✅ | `mix compile --warnings-as-errors` + `mix format --check-formatted` both clean |
