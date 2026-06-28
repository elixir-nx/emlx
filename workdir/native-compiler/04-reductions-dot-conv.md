# Stage 04 — Reductions + dot + conv

Status: done

## Why this stage exists

Reductions and contraction (`dot`, `conv`) are the compute-heavy core of real
models and the first ops with rich keyword options (axes, keep_axes, contraction
/ batch axes, strides, padding, dilations). Lowering them correctly proves the
`iattrs` channel scales to multi-list option payloads.

## Procedure

1. Lower (`EXPR_NODES.md` §E): sum, product, all, any, reduce_max, reduce_min,
   argmax, argmin.
2. Lower `dot` — encode contraction + batch axes (the `[a, ca, ba, b, cb, bb]`
   arg shape) into `iattrs`.
3. Lower `conv` — strides, padding, input/kernel dilations, feature/batch
   groups; port `EMLX.Backend`'s option handling.
4. Note: custom-fun `reduce`/`window_reduce` are deferred (they wrap an inner
   `fun` scope) — either lower via child program later (Stage 08-adjacent) or
   leave raising. Record the choice.
5. Add opcodes + C++ replay (reuse `emlx_nif.cpp`); parity test; equivalence
   tests vs eager `EMLX.Backend`; flip `EXPR_NODES.md` §E boxes.

## Acceptance

- Reductions + argmax/argmin + dot + conv lower and replay correctly within
  tolerance vs eager `EMLX.Backend`, across representative axis/keep_axes and
  conv option combinations.
- Multi-list `iattrs` payloads (dot axes, conv options) parity-tested vs C++.
- Decision on custom-fun `reduce` recorded (lowered vs deferred).
- `EXPR_NODES.md` §E boxes flipped (except any explicitly deferred); CI green.

## Results

| Item | Outcome | Notes |
|------|---------|-------|
| Reductions lowered | ✅ | sum, product, all, any, reduce_max, reduce_min; `iattrs = [keep_axes_int, a0, a1, …]`; all/any always emit astype (MLX returns bool_) |
| argmax / argmin | ✅ | `iattrs = [axis, keep_axis_int]`; axis = −1 encodes global; always cast to out_type (MLX returns uint32) |
| dot lowered | ✅ | Elixir upcasts to computation_type; four length-delimited axis lists in iattrs; C++ uses tensordot (non-batched) or reconstructed einsum spec (batched) |
| conv lowered | ✅ | Expanded in Elixir into astype + transpose + :conv_general op; C++ calls mlx::core::conv_general; output re-transposed to canonical layout |
| custom-fun reduce | deferred | Raises ArgumentError at lower time; requires child-program support (Stage 08) |
| Tests | 112/112 ✅ | 33 new Stage 04 tests: reductions, argmax/argmin, dot (batched + non-batched), conv 1D/2D, interpreter↔C++ parity, E2E jit smoke |
