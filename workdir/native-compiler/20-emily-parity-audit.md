# Stage 20 — Emily backend-parity gap audit

Status: not started. Docs-only; produces no code, scopes Stages 21–23.

## Why this stage exists

`~/coding/emily` (github.com/ausimian/emily) is the explicit successor
lineage to EMLX — its own README benchmarks against "EMLX, the older
MLX-backed Nx backend," and its `PLAN.md`/`ROADMAP.md` (milestones M0–M27)
describe a considerably larger shipped feature set than EMLX currently has:
native single-NIF compilation with a fallback+telemetry gate, `mlx::compile`
fusion, native linalg, quantized inference (incl. microscaled modes), SDPA
attention sinks, mixed-precision training, zero-copy `to_binary`,
observability/telemetry, compile-time debug assertions, and more. Emily
itself doesn't coordinate with EMLX ("EMLX coordination: none — quiet ship" —
`emily/PLAN.md`), so no assumption should be made that EMLX already tracks
Emily's decisions; this audit exists to check, not assume.

Before opening new implementation stages, cross-reference exactly what's
already at parity (possibly under a different name/API) vs genuinely
missing — duplicating already-equivalent work wastes effort, and several
items below turned out to already exist in EMLX once checked.

## Procedure

Cross-reference `emily/PLAN.md`'s milestones M0–M27 against EMLX's current
`lib/`, `c_src/`, and `test/` trees, one milestone at a time, and record
status in the Results table below. Seed findings from this planning pass
(re-verify each — this is a snapshot, not a substitute for the full pass):

**Already at parity — no new stage needed:**
- Native `mlx::fast::*` fused kernels (rms_norm / layer_norm / rope / sdpa /
  swiglu — `EMLX.Fast`, Stage 10) ~ Emily M11.
- Native `mlx::linalg::*` (cholesky / qr / eigh / svd / lu / solve — Stage 09)
  ~ Emily M15.
- Zero-copy `to_binary` (`to_blob_term`'s `row_contiguous` fast path via
  `enif_make_resource_binary`, `emlx_nif.cpp:152-163`) ~ Emily M12.
- Affine int2/4/8 quantization + transparent `Nx.dot` → `quantized_matmul`
  dispatch (`EMLX.Quantization`) ~ Emily M10/M10.5.
- Per-process command-queue / worker-thread model (`EMLX.CommandQueue`) ~
  Emily's `Emily.Stream` (M14).
- Per-op hard-raise instead of a silent `Nx.BinaryBackend` fallback
  (`EMLX.Backend`: `"#{op} not supported in EMLX"`) — this is *stricter*
  than Emily's per-op fallback-with-telemetry model, so there's no gap to
  close here; it's a philosophy difference EMLX already resolved in the
  stricter direction (consistent with Stage 19's zero-fallback goal).
- Whole-graph `mlx::core::detail::compile` wrapping (Stage 01) already gives
  EMLX's native lane the fusion Emily's opt-in `:fuse` mode provides — no
  separate opt-in mode needed unless a future measurement shows otherwise.

**Confirmed genuinely missing — scoped into Stages 21–23:**
- `:telemetry` events/spans (M18) and compile-time debug-assertion flags
  (M22) — zero `:telemetry` usage anywhere in `emlx/lib` or `emlx/mix.exs`.
  → Stage 21.
- SDPA `:sinks` (M26) — `mlx::core::fast::scaled_dot_product_attention`'s C++
  signature already accepts a `sinks` param (see comment,
  `emlx_fast.cpp:39`) but `EMLX.Fast` never plumbs it. → Stage 22.
- Microscaled quantization modes (M25: `mxfp4`/`mxfp8`/`nvfp4`) —
  `EMLX.Quantization` only covers the classical affine scheme. → Stage 22.
- Public eager `EMLX.Fast.einsum/2`-equivalent helper (M27) — an internal
  `EMLX.einsum` NIF exists (used by `dot_spec_to_einsum_spec/…` in
  `backend.ex`) but isn't exposed as a public, directly-callable helper. →
  Stage 22.
- Grad / mixed-precision / conv-pool training conformance (M9/M13/M16/M17) —
  no grad-specific test files exist in `emlx/test` at all, vs Emily's
  dedicated grad-equivalence, bf16, and MNIST-convergence suites. → Stage 23.

**Explicitly out of scope — Emily hasn't shipped these either:**
- M19 (typed exception hierarchy), M20 (GPU interop pointers), M21
  (`mix emily.doctor`) are all listed in Emily's own `ROADMAP.md` as
  "Deferred to post-1.0" — Emily itself hasn't shipped them, so there is no
  gap to chase.

## Acceptance

A filled-in gap table (mirroring Emily's own `ROADMAP.md` capability table)
checked into this doc's Results section, confirming or correcting every claim
above against the actual current state of both repos, and finalizing Stages
21–23's exact scope before they're tackled.
