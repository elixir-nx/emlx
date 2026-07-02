# Stage 22 — fast-kernel & quantization surface parity (sinks, microscaled quant)

Status: done. Emily M25/M26 parity (see Stage 20).

## Why this stage exists

Two contained, independent Emily-parity items that extend existing EMLX
surfaces rather than introduce new architecture.

> **Scope correction (advisor sign-off, before starting).** This doc
> originally bundled a third item — a public `einsum` helper (M27) — into
> this stage. The advisor flagged that the existing `EMLX.einsum` NIF is
> fixed arity-2 (`einsum_async`, registered `{"einsum", 5, ...}`, decodes
> exactly two `TENSOR_PARAM`s calling `mlx::core::einsum(spec, {*a, *b},
> device)`), so a 3-operand contraction test (part of this item's own
> acceptance criteria) needs a genuine variadic-tensor NIF signature change
> — bigger than "expose an existing NIF," per this doc's own escape valve
> ("split into separate PRs/tackle-steps if any turns out bigger than
> expected"). Split out to
> [`27-public-einsum-helper`](27-public-einsum-helper.md) (renumbered from
> its original 26 when Stage 25 was inserted as `25-quantized-dot-full-fix`
> and the burndown shifted down one — see that stage doc's header for the
> numbering history). This stage now covers only items 1 and 2 below.

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

   Per the advisor's confirmed boundary: this item threads `:mode` through
   the eager NIF/Elixir quantization surface only — it must not touch
   `expr.ex`/`emlx_compiler.cpp` (the native-compiler lowering path).
   Stage 24's already-deferred quantized-dot compiler-visibility gap
   (a quantized `Nx.dot` operand is invisible to the native compiler at
   trace time) stays out of scope here; tests call the NIF/eager path
   directly, never via `defn`/`compiler: EMLX`.

## Acceptance

- SDPA sinks work in both the eager `EMLX.Fast` path and the Stage-10
  compiled opcode, with equivalence tests; `EXPR_NODES.md` section L updated.
- Microscaled quantization modes round-trip and matmul correctly, per mode,
  with tests.

## Results

### SDPA attention sinks (M26)

- **`emlx_nif_shared.hpp`**: new `OPTIONAL_TENSOR_PARAM` macro — decodes an
  Elixir `nil` to `nullptr`/`std::nullopt` instead of requiring a real tensor
  resource, mirroring `TENSOR_PARAM` otherwise. Used for `sinks` (this stage)
  and `biases` (microscaled quant, below) so both stay fully backward
  compatible without a second NIF entry point per optional-arg combination.
- **`emlx_fast.cpp`**: all four SDPA NIFs (`fast_sdpa`, `fast_sdpa_masked`,
  `fast_sdpa_causal`, `fast_sdpa_causal_key_masked`) take a trailing optional
  `sinks` tensor and forward it to `mlx::core::fast::
  scaled_dot_product_attention`'s `sinks` param (a plain `std::optional<array>`,
  not the variadic-length param the doc's procedure anticipated — MLX's
  signature only ever takes 0-or-1 sinks tensors, so a fixed optional slot was
  simpler and sufficient).
- **`EMLX.Fast`**: new optional `:sinks` opt on all four SDPA wrappers,
  arity-disambiguated from the mask arg via `when is_list(opts)` guards.
  Source-compatible with every pre-existing call site.
- **Stage-10 compiled path**: four new `_sinks`-suffixed opcodes
  (`fast_sdpa_sinks`, `fast_sdpa_masked_sinks`, `fast_sdpa_causal_sinks`,
  `fast_sdpa_causal_key_masked_sinks`) rather than overloading the existing
  opcodes with a variable operand count — the masked (no-sinks) and sinks (no
  mask) variants would otherwise collide on the same 4-operand shape.
  `fast_kernel_dispatch/2` recognizes the four new `EMLX.Fast.sdpa_*_sinks_callback`
  captures; `emlx_compiler.cpp`'s `op_registry` gained matching entries (the
  causal-key-masked+sinks one duplicates the existing in-graph causal/key_mask
  composition, then passes the extra `sinks` operand through). The Layer-B
  Elixir interpreter (`dispatch/3`, used as the oracle in `expr_test.exs`) got
  matching clauses so both replay lanes stay covered by the same tests.
- **Tests**: `sdpa_sinks_test.exs` (eager) checks all four variants against a
  from-scratch softmax-with-sinks reference implementation (row-max over both
  logits and sinks) at f32 tolerance, plus a GQA shape check and
  omitted-`:sinks` backward-compatibility checks. `expr_test.exs` adds sinks
  cases to the "lowers to a single fused opcode" table and four
  compiled-vs-eager equivalence tests (`compiler: EMLX` vs
  `compiler: Nx.Defn.Evaluator`) in the Metal describe block, covering the
  same four SDPA variants end-to-end through the real NIF replay.
- **Fixed along the way (not scoped, but blocking)**: `Nx.with_default_backend/2`
  was needed in `sdpa_sinks_test.exs` — a binary op between an
  `Nx.BinaryBackend` tensor and a bare number (`Nx.divide(t, 37)`) silently
  promotes to the process's default backend (`EMLX.Backend` here), so
  reference-math tensors kept round-tripping through MLX and colliding with
  the `EMLX.Backend`-only tensors used for the real NIF calls
  (`Tensor has been deallocated`). Solved by overriding
  `EMLX.Case`'s default-backend `setup` to `Nx.BinaryBackend` for this test
  module, so all plain (non-`gpu/1`-transferred) tensor construction and math
  stays off MLX.

### Microscaled quantization modes (M25)

- **NIFs** (`quantize`, `dequantize`, `quantized_matmul`): thread a `mode`
  string through to `mx::quantize`/`mx::dequantize`/`mx::quantized_matmul`;
  `biases` becomes optional (`OPTIONAL_TENSOR_PARAM`) since `mx::fp_quantize`
  returns only `(wq, scales)` for microscaled modes. `mode` is decoded via
  `nx::nif::get(env, argv[N], mode)` (a real string binary), not `ATOM_PARAM`
  (which expects an Elixir atom) — Elixir passes `mode` as a string.
- **`EMLX.Quantization.Config`**: gained a `:mode` field (default `"affine"`);
  `:biases` is now `nil`-able.
- **`EMLX.Quantization`**: `quantize/2` accepts `:mode` with validation
  (`valid_quantization_modes`, `microscaled_constraints` — group_size/bits
  combinations differ per mode); `dequantize/1` and `quantized_matmul/2`
  correctly infer their `deftransform`-time output `Nx.Type` per mode
  (scales' dtype for `"affine"`; `:bf16`/activation's dtype for microscaled,
  since microscaled scales are packed `:u8` exponent bytes, not a float
  dtype).
- **Scope deviation from the doc's procedure**: the doc anticipated
  `dequantize/1`'s defn-callable path needing to raise `ArgumentError` on
  non-affine modes "if a full dense reconstruction isn't feasible." It turned
  out to be feasible — `mx::dequantize` reconstructs a dense float array for
  every mode uniformly — so no raise was needed; `dequantize/1` just works for
  all four modes.
- **Discovered (documented, not fixed — out of scope, same shape as Stage
  24's gap)**: `deftransform`-based output-type inference (`dequantize/1`,
  `quantized_matmul/2`) is trace-blind to runtime `EMLX.Backend` metadata —
  during `defn` tracing, `qw.data` is an `Nx.Defn.Expr`, not the real
  `EMLX.Backend` struct with `quantization_config`, so the `case` clauses
  above only match at eager-call time. Nested `defn` calls (e.g. a `defn`
  calling `EMLX.Quantization.dequantize/1` on a param) fall through to the
  generic branch. Doesn't block this stage's tests (they call the NIF/eager
  path directly, per the advisor-confirmed boundary), but is a latent gap for
  future JIT-traced callers.
- **Tests**: `microscaled_quantization_test.exs` — per-mode
  (`"affine"`/`"mxfp4"`/`"mxfp8"`/`"nvfp4"`) round-trip, mode/constraint
  validation, `quantized_matmul` vs `Nx.dot(x, Nx.transpose(dense))`
  equivalence, and transparent `Nx.dot` dispatch checks.
