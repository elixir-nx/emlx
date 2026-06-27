# Stage 10 — Fast kernels (`EMLX.Fast`)

Status: not started

## Why this stage exists

By this stage every model lowers fully via primitives. This stage is the
performance pass: recognize the lowered patterns for RMSNorm, LayerNorm, RoPE,
and scaled dot-product attention and **route them to the fused `EMLX.Fast`
Metal kernels** instead of the primitive expansion — the fused-kernel advantage
the `EMLX.Fast` kernels provide for LLM inference.

## Procedure

1. Identify how these surface in a lowered graph: as `Nx.Block.*` /
   `EMLX.Fast.*` blocks (preferred — recognize the struct, §6 path) or as
   recognizable primitive subgraphs (pattern-match in the lowerer).
2. Implement recognize-and-route lowering for (`EXPR_NODES.md` §L): rms_norm,
   layer_norm, rope (+ with_positions / with_freqs), scaled_dot_product_attention
   (+ causal / key-masked variants), swiglu.
3. Add fused opcodes + C++ replay calling the existing `emlx_fast.cpp`
   implementations; parity test.
4. Equivalence tests vs eager `EMLX.Backend` (and vs the primitive lowering,
   within fused-kernel tolerance); flip §L boxes. Benchmark the fused path on a
   decode-shaped transformer block vs the primitive replay.

## Acceptance

- The §L fused kernels lower via recognition and replay correctly within
  fused-kernel tolerance vs eager `EMLX.Backend` and vs the primitive lowering.
- A decode-shaped benchmark shows the fused path improving over the primitive
  replay (numbers recorded).
- `EXPR_NODES.md` §L boxes flipped; CI green.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| rms_norm / layer_norm | | |
| rope variants | | |
| sdpa variants | | |
| swiglu | | |
| fused vs primitive benchmark | | |
