# Stage 22 — fast-kernel & quantization surface parity (sinks, microscaled quant, einsum)

Status: not started. Emily M25/M26/M27 parity (see Stage 20).

## Why this stage exists

Three contained, independent Emily-parity items that extend existing EMLX
surfaces rather than introduce new architecture. Grouped into one stage
because each is small; split into separate PRs/tackle-steps if any turns out
bigger than expected.

## Procedure

1. **SDPA attention sinks (M26).** Thread `mlx::core::fast::
   scaled_dot_product_attention`'s `sinks` parameter through:
   - `emlx_fast.cpp`'s SDPA NIFs (masked + causal + causal-key-masked
     variants) — a variadic-length-0-or-1 `sinks_arrs` param, `std::nullopt`
     when absent, mirroring the existing `mask_arrs` plumbing.
   - `EMLX.Fast`'s Elixir wrappers — new optional `:sinks` opt, default
     absent (source-compatible with every existing call site).
   - The Stage-10 native compiled path (`fast_kernel_dispatch/2` +
     `EMLX.Native.Expr`'s C++ opcode), so the compiled replay lane supports
     sinks too, not just the eager `EMLX.Fast` calls.
   Equivalence-test against the fallback softmax-with-sinks math (row_max
   over both logits and sinks — see `emily/PLAN.md` M26 for the exact
   formula) at f32 tolerance.
2. **Microscaled quantization modes (M25).** Thread a `:mode` string
   (`"affine"` default / `"mxfp4"` / `"mxfp8"` / `"nvfp4"`) through
   `EMLX.Quantization.quantize/2`, `dequantize/1`, `quantized_matmul/2`, the
   NIF layer, and `EMLX.Quantization.Config` — biases become optional for
   microscaled modes since `mx::fp_quantize` returns only `(wq, scales)` for
   them. `dequantize/1`'s defn-callable path should raise a clear
   `ArgumentError` on non-affine modes if a full dense reconstruction isn't
   feasible, pointing callers at the eager-only path (mirrors Emily M25's
   `dequantize_defn/1` behavior). Equivalence-test per-mode round-trip +
   `quantized_matmul` vs `Nx.dot(x, Nx.transpose(dense))`.
3. **Public `einsum` helper (M27).** Expose the existing internal
   `EMLX.einsum` NIF (already used by `dot_spec_to_einsum_spec/…` in
   `backend.ex`) as a public eager helper on `EMLX.Fast` (or a suitable
   existing module), raising a clear `ArgumentError` for non-`EMLX.Backend`
   operands ("transfer with `Nx.backend_transfer/2` first"). Tests:
   2-operand (`"ij,jk->ik"`), batched (`"bij,bjk->bik"`), attention-style
   (`"bhid,bhjd->bhij"`), 3-operand (`"ij,jk,kl->il"`) contractions, and the
   non-EMLX-backend error path.

## Acceptance

- SDPA sinks work in both the eager `EMLX.Fast` path and the Stage-10
  compiled opcode, with equivalence tests; `EXPR_NODES.md` section L updated.
- Microscaled quantization modes round-trip and matmul correctly, per mode,
  with tests.
- A public eager `einsum` helper ships with tests across operand arities.
