# Stage 10 — Fast kernels (`EMLX.Fast`)

Status: done

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

### Surfacing (corrected premise, advisor-reviewed)

The stage doc assumed `EMLX.Fast.*` would surface as `Nx.Block.*` /
`EMLX.Fast.*` blocks or recognizable primitive subgraphs. **It does neither.**
Each `EMLX.Fast.*` function is a `deftransform` that emits a single
`Nx.runtime_call(out, container, opts, &EMLX.Fast.<name>_callback/2)`. So inside
a `compiler: EMLX` defn they surface as `:runtime_call` nodes (previously
unhandled → raised). The recognition seam is therefore the **captured callback
function**, matched by module+name+arity (`fast_kernel_dispatch/2`), not a block
struct. Advisor confirmed scope (a): route the single-NIF (decode/T=1)
callbacks to fused opcodes; the per-token prefill RoPE callbacks
(`rope_with_positions_callback` / `rope_with_freqs_callback`) are host-side Nx
compositions over eager NIFs (not one kernel, not traceable) → raise a
fallback-eligible `does not yet lower op` so the seam delegates to the
Evaluator. No host-split (option b) built.

### Mechanism

- **Lowerer** (`expr.ex`): one `:runtime_call` `expand_node` clause; operands
  come from the call's tensor container via `Composite.flatten_list` (order
  matches the C++ positional `ops[]`); `fast_kernel_dispatch/2` maps the
  callback → `{opcode, attrs}`.
- **Float attrs** (eps/scale/base): the IR attr channel is int64-only, so each
  float is reinterpreted to its IEEE-754 double bits (`f64_bits/1` ↔
  `bits_to_f64/1`; C++ `attr_to_float` via `memcpy`). Integer opts (dims,
  traditional 0/1, offset, kv_offset) pass directly. No parallel float wire
  format added.
- **C++** (`emlx_compiler.cpp`): fused opcodes in `op_registry` call
  `mlx::core::fast::*` on the worker's default stream (Metal). `fast_rope_ids` /
  `fast_rope_with_freqs` extract `position_ids[:, 0]` in-graph. The
  causal-key-masked opcode always builds the combined causal+key_mask additive
  mask in-graph (the eager NIF's `all(key_mask).item<bool>()` host branch can't
  live inside `detail::compile`); correctness preserved, the no-padding
  micro-opt dropped.
- **Interpreter** (Layer B oracle): fused-opcode dispatch calls the same eager
  `EMLX.fast_*` NIFs the C++ opcodes wrap.

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| rms_norm / layer_norm | native fused (`:fast_rms_norm`, `:fast_layer_norm`, `:fast_layer_norm_no_bias`) | vs eager + hand-written primitive within 1e-3 |
| rope variants | native fused for decode/T=1 (`:fast_rope`, `:fast_rope_ids`, `:fast_rope_with_freqs`); prefill T>1 lowers via an in-graph cos/sin/rotate composition (Stage 15) | vs eager |
| sdpa variants | native fused (`:fast_sdpa`, `:fast_sdpa_masked`, `:fast_sdpa_causal`, `:fast_sdpa_causal_key_masked`) | causal-key-masked builds mask in-graph (no `.item()`); vs eager + softmax(QKᵀ)·V primitive |
| swiglu | native fused (`:fast_swiglu`) | vs hand-written `silu(gate)*up` |
| fused vs primitive benchmark | fused faster | decode block (causal SDPA → reshape → RMSNorm): ~300 µs/call fused vs ~400 µs/call primitive replay (~1.3–1.4× on this machine) |
| tests | 15 Stage 10 tests (4 pure lowering/round-trip/fallback, 11 `:metal` E2E + benchmark); eager-parity for every kernel + primitive-parity for rms_norm/layer_norm/swiglu/sdpa(+causal+masked); causal-key-masked tested padded **and** all-present (covers the always-build-mask divergence). Full native suite 216 passing; full EMLX suite 2509 passing, no regressions | `test/emlx/native/expr_test.exs` |
