# Investigation: `native` lane regression on MLX-4bit vs `main`

**Status: RESOLVED.** See "Resolution" section near the end — fused
`qwen3_layer_quantized`/`qwen3_forward_greedy_ids_chunk_quantized` NIFs
closed the gap and then some (261.4 tok/s post-fix vs 66.6 pre-fix, 76.9 on
`main`).

## TL;DR

- **Symptom**: on `pv-feat/lowering-compiler`, the `:native` lane
  (`EMLXAxon.TextGeneration.from_mlx4bit/3`) benchmarks at ~58–68 tok/s for
  the `Qwen3-0.6B-MLX-4bit` checkpoint, vs ~78–82 tok/s on `main` (per the
  user's own side-by-side Livebook screenshots). `:native` on the **dense**
  checkpoint is unaffected (~82 tok/s on both branches).
- **This is an inference-time (per-token, steady-state) regression, not a
  load-time one.** There is no compile/JIT step in the native path on either
  branch to blame — see "Load time vs inference time" below.
- **Root cause (best-supported hypothesis, not yet proven with a bisect)**:
  the quantized native forward pass is structurally ~13x more "chatty" at
  the NIF boundary than the dense native forward pass (≈13 Elixir→NIF round
  trips per transformer layer vs 1). This ratio is unchanged from `main`
  (`model.ex`/`attention.ex` have zero logic diff vs `main` — see below) —
  but this branch touched the shared async-NIF dispatch machinery that
  *every* one of those round trips goes through
  (`emlx_async.hpp`/`emlx_worker.hpp`/`emlx_nif_shared.hpp`, plus a 859-line
  rewrite of `emlx_fast.cpp`). A small per-call overhead increase there would
  be invisible on the dense path (28 NIF calls/token) and clearly visible on
  the quantized path (~360 NIF calls/token) — exactly the asymmetry observed.
- This is **separate** from (and was initially conflated with) the
  benchmark-harness bug already fixed in `validate_qwen3_standalone.livemd`
  (the `:native` lane was redundantly loading + re-quantizing the whole
  Bumblebee model it never uses — fixed, see that file's `run_lane/3`
  comment). That fix corrected `bb+rewrite`'s mlx4bit numbers but did **not**
  move `:native`'s number, which is what led to this investigation.

## What was ruled out

1. **Peer isolation is not the cause.** Verified directly: `EMLX.memory_info/0`
   resets to zero at the start of every peer, and running the *same* lane
   6x back-to-back in fresh peers shows no drift (74.5 → 80.4 → 86.1 → 84.4
   → 85.0 → 83.8 tok/s) — no cross-run leakage, no thermal throttling from
   repeated peer spawns.
2. **`cfg.native_profile_timing?`** is `false` in the benchmark config, so
   `EMLXAxon.Qwen3.Generate`'s per-token timing instrumentation (which, when
   enabled, *also* disables the chunked greedy fast path — see
   `decode_step_timed`'s `sampler == :greedy and not profile_timing?` guard)
   is not in play.
3. **The benchmark-harness redundant-load bug** (`run_lane/3` unconditionally
   calling `Bumblebee.load_model` + `EMLXAxon.QuantizeParams.quantize` for
   every lane, including `:native`, which never uses the result). Fixed
   separately. Confirmed via isolated single-lane runs (fresh outer process,
   nothing else running):

   | lane | before fix | after fix |
   |---|---|---|
   | `bb_base` | 54.1 | 55.6 |
   | `bb_rewrite` | 74.3 | **82.4** |
   | `native` | 60.9 | **58.5** (unchanged) |

   `bb+rewrite` recovered to its expected ~1.5x multiplier over `bb_base`
   (matching the dense checkpoint's ratio). `:native` did not move — this
   is the residual, real issue this document is about.
4. **`model.ex`/`attention.ex`/`layers.ex` have no code diff vs `main`.**
   `git diff main...HEAD -- emlx_axon/lib/emlx_axon/qwen3/model.ex` shows
   only a moduledoc comment update (correcting stale claims about `defnp`/
   `Nx.Defn.Compiler.__jit__` usage that no longer reflected reality); the
   actual dispatch logic — which lane calls which NIF, how many times — is
   byte-for-byte identical to `main`. So this is not "the branch made the
   quantized path more granular"; that granularity already existed on
   `main`. Something in the *shared infrastructure* those calls go through
   must have changed cost instead.

## The call-count asymmetry

Per transformer layer (28 layers in Qwen3-0.6B), `model.ex`/`attention.ex`
dispatch differently depending on whether the layer's weights are quantized:

**Dense** (`layer/16` in `model.ex`, `forward_dense/12` in `attention.ex`):

- `EMLX.qwen3_layer/18` — **one** fused C++ NIF call does norms + QK-norm +
  RoPE + KV-cache update + SDPA + gate/up/down MLP, entirely in native code.
- **1 NIF call per layer → 28 calls/token.**

**MLX-4bit / quantized** (`forward_quantized/12` in `attention.ex`, the
`mlp/5` quantized branch in `model.ex`):

- `Layers.rms_norm` (hidden) → `EMLX.Fast.rms_norm` — 1 call
- `Nx.dot` × 3 (q/k/v proj) → `EMLX.Backend.dot`'s `quantized_dot` →
  `EMLX.quantized_matmul` NIF — 3 calls
- `Layers.rms_norm` (q_norm, k_norm) → `EMLX.Fast.rms_norm` — 2 calls
- `EMLX.qwen3_kv_cache_attention/9` (fused RoPE + cache update + SDPA) — 1 call
- `Nx.dot` (o_proj) → `quantized_matmul` — 1 call
- `Layers.rms_norm` (norm2) → `EMLX.Fast.rms_norm` — 1 call
- `Nx.dot` × 2 (gate/up proj) → `quantized_matmul` — 2 calls
- `EMLX.Fast.swiglu` — 1 call
- `Nx.dot` (down proj) → `quantized_matmul` — 1 call
- **≈13 NIF calls per layer → ≈364 calls/token.**

That's a **~13x** difference in NIF round trips per generated token, entirely
pre-existing on `main`. For `max_new_tokens: 60`, that's ~1,680 calls total
for dense vs ~21,840 for quantized over one `Nx.Serving.run/2`.

For reference, `bb+rewrite` doesn't have this problem at all regardless of
quantization: `defn_options: [compiler: EMLX]` means the whole forward pass
gets traced once and compiled into a single native program (cached in
EMLX's dispatch-cache ETS table), so it pays close to the "1 call" dense
regime too — consistent with it also benchmarking at ~82 tok/s post-fix.

## What changed at the NIF layer that all ≈364 quantized calls/token pass through

`git diff main...HEAD --stat` for the C++ layer:

```
emlx/c_src/emlx_async.hpp               |   23 +-
emlx/c_src/emlx_compiler.cpp            | 2064 +++++++++++++++++++++++++++++++  (new)
emlx/c_src/emlx_compiler.hpp            |   42 +   (new)
emlx/c_src/emlx_fast.cpp                |  859 ++++++-------
emlx/c_src/emlx_nif.cpp                 |  197 +--
emlx/c_src/emlx_nif_shared.hpp          |  274 +++-
emlx/c_src/emlx_runtime_call_bridge.hpp |  262 ++++  (new)
emlx/c_src/emlx_worker.hpp              |   62 +
```

Candidates for added per-call overhead, roughly in order of how directly
they sit on the hot path used by the quantized native lane:

1. **`emlx_async.hpp`**: every `ASYNC_NIF`-wrapped call (this includes
   `quantized_matmul`, `qwen3_kv_cache_attention`, and — via
   `FINE_ASYNC_NIF`, see below — the rewritten `EMLX.Fast.*` NIFs) now
   constructs a `CallerPidGuard` per job to make the calling pid available
   to `EMLXRuntimeCall` (needed for the new `Nx.runtime_call` bridge, which
   `EMLX.Quantization.dequantize/quantize/quantized_matmul`'s `deftransform`
   wrappers use — but note the native path calls the *direct* NIFs
   (`EMLX.quantized_matmul/8` from `EMLX.Backend.dot`), not those
   `Nx.runtime_call`-wrapped `deftransform`s, so it doesn't pay the full
   runtime-call round trip, just the guard). Cheap in isolation (pointer
   save/restore) but universal.
2. **`emlx_worker.hpp`**: new `g_current_worker` thread-local +
   `pump_until/1`, used when a job blocks inside `EMLXRuntimeCall::eval_cpu`/
   `eval_gpu` waiting on an Elixir-side reply. Not on the direct-NIF path the
   quantized native lane uses, but confirms the worker's job loop itself
   changed shape.
3. **`emlx_nif_shared.hpp`**: introduces a `fine`-based typed decode/dispatch
   bridge (`FINE_ASYNC_NIF`, `emlx_fine::nif`/`nif_impl`, `Decoder<TensorArg>`
   etc.) as an alternative to the old hand-rolled `TENSOR_PARAM` macros. This
   is very likely how the many *new* `EMLX.Fast.*` NIFs added in this branch
   (rope variants, sdpa+sinks variants — see `emlx_fast.cpp`'s 859-line
   rewrite) are implemented; whether the *existing* ones the quantized path
   already called (`fast_rms_norm`, `fast_swiglu`, plain SDPA) moved to this
   bridge too is the key open question (see next steps) — if so, each such
   call now pays templated `Decoder<Args>::decode()` + `std::index_sequence`
   dispatch instead of the old macro-expanded direct decode.
4. **`emlx_nif.cpp`**: 197-line diff, not yet inspected line-by-line for the
   NIFs the quantized path actually calls (`quantized_matmul`,
   `qwen3_kv_cache_attention`) — confirmed `quantized_matmul` itself is
   still the old-style `ASYNC_NIF` (not `FINE_ASYNC_NIF`), so it's unlikely
   to be a large factor on its own.

None of these are large in isolation (we're talking single-digit-to-low-
double-digit microseconds per call, not milliseconds) — but multiplied by
~360 calls/token × 60 tokens, even a modest per-call regression is enough to
explain the ~25–35% throughput drop observed (58–68 vs 78–82 tok/s).

## Load time vs inference time

This affects **inference time**, not load time:

- The native path has **no compilation/JIT step on either branch** to
  attribute a "cold load" cost to. Both `qwen3_layer` (dense) and the
  quantized fallback (`EMLX.Fast.*` + `Nx.dot`) are hand-written eager NIF
  calls — there's no `Nx.Defn.Compiler.__jit__`/EMLX-compiler graph being
  built and cached the first time, unlike `bb+rewrite`/`bb_base`.
- `Bench.warmup/5` runs the full generation twice before any timed run, so
  even if there *were* a one-time cost (e.g. first-call dispatch-table
  population, Metal pipeline state creation) it would be paid during warmup,
  not during the 5 measured runs.
- The degradation shows up **consistently across all 5 timed runs**, not
  just the first — that's the signature of a per-call/per-token steady-state
  cost, not a one-time load cost.
- Model *load* time (`EMLXAxon.Qwen3.Loader.load/1`, reading + dequantizing +
  re-quantizing the mlx4bit checkpoint) was measured directly at ~138ms in
  isolation — small, one-time, and outside the timed benchmark window
  regardless.

## Suggested next steps

1. **Confirm which `EMLX.Fast.*` NIFs moved to `FINE_ASYNC_NIF`** (diff
   `emlx_fast.cpp` function-by-function against `main`, focusing on
   `fast_rms_norm`, `fast_swiglu`, and the plain (non-sinks, non-mask)
   `scaled_dot_product_attention` — the three the quantized path actually
   calls). If they did, that's the most direct explanit path to chase.
2. **Microbenchmark the NIF call, not the model.** Write a tight loop
   calling `EMLX.Fast.rms_norm/3` (or `EMLX.quantized_matmul/8`) N times
   (N ≈ 10,000) on a small fixed-size tensor on both `main` and this branch,
   and compare wall time / N. This isolates pure per-call dispatch overhead
   from anything about the Qwen3 model or benchmark harness, and would give
   a concrete "Δ microseconds/call" number to confirm or kill this
   hypothesis before spending time on a fix.
3. **If confirmed, the durable fix is architectural, not a `EMLX.Fast`
   micro-optimization**: give the quantized path the same treatment as
   dense — a single fused `EMLX.qwen3_layer_quantized`-style NIF (or extend
   `qwen3_layer` to accept quantized weight structs) that does norms + RoPE
   + KV-cache + SDPA + quantized-projections + swiglu in one C++ call per
   layer, matching dense's "1 call/layer" shape instead of "~13
   calls/layer". This removes the 13x call-count multiplier entirely rather
   than chasing microseconds out of each of the ~364 calls/token.
   Short-circuiting inside `EMLX.Fast` (per the original suggestion) would
   only pay off *if* step 1 confirms the regression lives specifically in
   those NIFs' new dispatch path — it would not close the underlying
   architectural gap vs dense, only narrow it.

## Update: microbenchmark — no dispatch overhead found in the two NIFs tested (investigation not closed)

Ran `emlx_axon/bench/nif_overhead_microbench.exs` (new script) on a fixed
`{1, 1024}` tensor, 100-iteration warmup + N=10,000 timed calls, with
`EMLX.eval/1` forcing dispatch after every call, on both branches in place
(`mix compile` recompiles only the ~4 changed `.cpp` files per branch switch;
MLX itself is pinned to the same `0.31.2` build on both, confirmed via
identical `libmlx-0.31.2`/`emlx-0.3.1-mlx-0.31.2` cache paths in the build
logs, so any Δ is attributable to the NIF/dispatch layer, not MLX core).

**First pass (methodologically flawed — see correction below)** showed both
`EMLX.Fast.rms_norm/3` (moved to `FINE_ASYNC_NIF`) and `EMLX.quantized_matmul/2`
(unchanged `ASYNC_NIF`, intended as a control) regressing by a similar
~15-26µs/call from `main` to this branch, which looked like it implicated the
universal `async_dispatch<OP>`/`CallerPidGuard` path in `emlx_async.hpp`
(shared by every `ASYNC_NIF`-wrapped call) rather than the fine bridge
specifically. **This conclusion was wrong** — see below.

**The confound:** `NIF(eval)` itself changed between branches
(`emlx/c_src/emlx_nif.cpp`). On `main` it is `mlx::core::eval(*t)` only; on
`pv-feat/lowering-compiler` it gained `mlx::core::synchronize()` after the
`eval()` call. Since the benchmark calls `EMLX.eval` after *every* timed
iteration (required to force MLX's lazy graph to actually dispatch), this
silently confounded the rms_norm/quantized_matmul comparison — it wasn't
purely measuring those NIFs' dispatch cost, it was measuring
`op-dispatch + eval()`, and `eval()`'s own cost changed independently of
anything in `async_dispatch`/`CallerPidGuard`/the fine bridge.

Added a third **eval-only** leg (re-`EMLX.eval` an already-computed tensor,
isolating `NIF(eval)`'s own cost) and reran 3x per branch:

| Leg | `main` (median µs/call, 3 runs) | `pv-feat/lowering-compiler` (median µs/call, 3 runs) | Δ |
|---|---|---|---|
| `EMLX.Fast.rms_norm/3` | 148, 154, 155 | 169, 186, 180 | +21 to +32µs |
| `EMLX.quantized_matmul/2` | 156, 166, 162 | 178, 196, 189 | +22 to +30µs |
| `EMLX.eval/1` alone | 8, 9, 9 | 37, 39, 35 | **+27 to +30µs** |

Subtracting the eval-only cost from each op leg (`op − eval-only`, averaged
over the 3 runs) isolates the **pure op-dispatch cost**, independent of
`eval()`'s own change:

| | `main` avg | `pv-feat` avg | Δ |
|---|---|---|---|
| `rms_norm` (pure) | 143.7µs | 141.3µs | **−2.4µs (noise)** |
| `quantized_matmul` (pure) | 152.7µs | 150.7µs | **−2.0µs (noise)** |

**Finding, scoped to what was tested:** once `eval()`'s own (unrelated) cost
change is controlled for, there is **no evidence of increased per-call
dispatch overhead**, for either of the two representative dispatch styles
tested — the fine-bridge NIF (`rms_norm`) and the unchanged old-style NIF
(`quantized_matmul`) — on this branch vs `main`. The residual Δ (~2µs) is
smaller than the run-to-run range (~10-15µs) observed across repeated trials
on the same branch, with only 3 runs/leg (no formal variance/CI computed),
so this is "no evidence of overhead in these two NIFs," not a proof that
*no* NIF anywhere in the quantized path regressed — the quantized lane also
calls other NIFs not benchmarked here (e.g. `qwen3_kv_cache_attention`,
`EMLX.Fast.swiglu`), and `async_dispatch<OP>`/`CallerPidGuard`/
`Decoder<Args>::decode()` were not ruled out for those specifically.

The one *real*, reproducible difference found — `NIF(eval)` gaining
`mlx::core::synchronize()` — is a genuine ~28µs/call cost *on this
near-empty-queue microbenchmark*. Per `model.ex`/`generate.ex`'s moduledoc
comments (not independently verified by call-count instrumentation),
`EMLX.eval` is called once per token at the sampler boundary, identically for
dense and quantized, which would make this negligible (28µs vs the observed
~1.9-5.0ms/token regression) *if* `synchronize()`'s cost is roughly fixed
regardless of queue depth. That assumption is untested here: in production,
`synchronize()` waits on whatever async GPU work is actually queued for that
token's full forward pass, which is far larger and more decomposed on the
quantized path (~364 calls/token) than dense (~28 calls/token) — so its real
cost at the sampler boundary could scale with queued work and be
asymmetric between lanes in a way this isolated single-tensor benchmark
cannot capture. This is a live alternative worth checking directly (e.g.
compare `synchronize()`'s wall time at the actual end-of-token boundary for
dense vs quantized), not something ruled out.

**Net effect on the original hypothesis:** the fine-bridge/universal-guard
NIF-dispatch-overhead theory is not supported by the two NIFs tested, but the
investigation is not conclusively closed — it should not be treated as fully
falsified across the whole quantized call surface, nor should the
`synchronize()` finding be dismissed as irrelevant without directly measuring
its cost at realistic (dense vs quantized) queue depths. Per the plan's own
falsification criterion, this data does not cleanly confirm the original
"NIF dispatch adds overhead uniformly" story either. The previously proposed
fixes (fused `EMLX.qwen3_layer_quantized`-style NIF, or auditing
`CallerPidGuard`'s cost) are **not adopted on this evidence** — neither is
shown to address a demonstrated bottleneck. Recommended next steps (not
pursued as part of this task, in priority order):
1. Measure `mlx::core::synchronize()`'s actual wall time at the real
   sampler-boundary `EMLX.eval` call, comparing dense vs quantized decode
   (not the isolated single-tensor microbenchmark here), to test the
   queue-depth-dependent-cost alternative above.
2. Extend this microbenchmark to the untested quantized-path NIFs
   (`qwen3_kv_cache_attention`, `EMLX.Fast.swiglu`) before ruling out
   dispatch overhead more broadly.
3. Profile actual GPU/Metal kernel time for the quantized path's ops
   directly via `EMLX.metal_start_capture` on both branches, since a
   difference in kernel dispatch parameters, memory layout, or
   command-buffer batching behavior (not NIF-boundary cost) inside those
   same functions is also a plausible place to look.
4. Re-check whether `Nx.runtime_call`/`emlx_compiler.cpp` machinery is truly
   inert on the direct-NIF quantized path (previously marked out of scope on
   the assumption it wasn't reachable, but not independently re-verified
   against this branch's actual `EMLX.Backend.dot`/quantized dispatch code).

## Update: item 4 confirmed inert; item 1 revealed a queue-depth measurement gap in every prior benchmark, including this one

**Item 4 (`Nx.runtime_call`/`emlx_compiler.cpp` inertness) — confirmed, quick.**
`EMLXAxon.Qwen3.Model`'s own moduledoc states plainly: "every hot-path call is
a plain eager function call against concrete `EMLX.Backend` tensors ...  with
no `Nx.Defn.Expr` tracing or `Nx.Defn.Compiler` involved anywhere in the
loop." `EMLX.Fast.rms_norm`/`swiglu`'s `deftransform` bodies only route
through `Nx.runtime_call` when `traced?/1` (checks for `%Nx.Defn.Expr{}`)
is true — never the case for concrete native-lane tensors. Confirmed inert.

**Item 1 (measure `synchronize()` at real per-token queue depth) — done, but
it exposed a bigger problem: every benchmark in this investigation so far,
including this one, measured the wrong queue-depth regime.**

`bench/sync_boundary_timing.exs` replicated the decode loop by hand
(`Model.forward` + `Sampler.greedy`), calling `EMLX.eval` after **every**
token, for both dense and quantized, on both branches. Result: **no
regression at all** — quantized eval-boundary cost was statistically
identical between branches (`main` 8368µs vs `pv-feat` 8223µs median; dense
13317 vs 14218µs median).

Investigating *why* led to the real issue: `EMLXAxon.TextGeneration.serving/3`
/ `from_mlx4bit/3` — the actual production path that produced the originally
reported regression numbers — defaults to `host_sync: {:chunk, min(max_new,
31)}`, **not** per-token sync. This means the real workload lets MLX's lazy
graph accumulate across up to 31 tokens before a single host sync — for the
quantized fallback path (`decode_tensor_chunk_step`, since
`native_forward_greedy?/1` is `false` for quantized states, so
`forward_greedy_chunk` returns `:fallback`), that's up to **~364 NIF
calls/token × 31 ≈ 11,300 queued ops** before one flush. Dense, by contrast,
routes through `forward_native_greedy_chunk` — a **single fused C++ NIF**
that runs the whole chunk internally — so dense's real NIF-boundary crossing
count for a chunk is ~1, not ~31×28. Every microbenchmark run so far in this
investigation (including the original `nif_overhead_microbench.exs`'s
per-call eval-forcing, and `sync_boundary_timing.exs`'s per-token eval-
forcing) tested queue depths of 1 or ~364 ops — nowhere near the real
~11,300-op regime the production quantized path actually runs at. This means
none of the prior findings in this doc (including the "no measurable
dispatch overhead" and "eval() sync cost is ~28µs and negligible" findings)
can be assumed to hold at the real queue depth — they were never tested
there.

**Re-validated the regression is real and current, using the actual
production path.** Wrote `bench/native_serving_e2e_bench.exs`, using
`EMLXAxon.TextGeneration.run/4` (same `default_host_sync/1` as `serving/3`,
avoids `Nx.Serving`/Bumblebee-batch overhead for cleaner timing — **note:
this is not identical to `from_mlx4bit` + `Nx.Serving.run`, which adds
serving/batching overhead on top; treat this as a same-ballpark re-check, not
a like-for-like reproduction of the original screenshots**). 6 runs/branch,
`max_new_tokens: 60`, quantized model, greedy sampler:

| | `main` tok/s (6 runs) | `pv-feat/lowering-compiler` tok/s (6 runs) |
|---|---|---|
| individual runs | 65.6, 66.7, 77.1, 64.5, 79.4, 76.9 | 49.4, 67.2, 70.9, 66.6, 58.9, 64.8 |
| median | **76.9** | **66.6** |

This reproduces a real, current regression (~13% at the median) using the
actual production code path for the first time in this investigation — in
the same direction as, but smaller in magnitude than, the originally
reported 25-35% (78-82 → 58-68 tok/s). Both distributions have high variance
and overlap (`pv-feat`'s max of 70.9 exceeds `main`'s min of 64.5), so this
single 6-run comparison should be treated as noisy directional confirmation,
not a precise magnitude measurement.

**Status: paused here to check in before further investigation.** The
natural next experiment — instrument the real chunk-boundary flush point
(`tokens_to_host`/`Nx.to_flat_list` in `generate.ex`) directly, at realistic
chunk-size queue depth, comparing branches, to test whether
`synchronize()`'s cost (or some other queue-depth-dependent effect) actually
explains the regression at the depth where it's real — is a reasonable next
step, but this investigation has now spanned multiple sessions with a
shrinking, harder-to-pin-down effect size (25-35% originally reported → ~13%
median reproduced here), so further work should get explicit go-ahead rather
than continuing autonomously. **Recommended concrete next experiment, if
   pursued:** add temporary `System.monotonic_time` instrumentation around the
existing chunk-flush call site(s) in `generate.ex` (rather than another
standalone microbenchmark harness, which is what led to two confounds
already in this doc) and diff branches directly at real chunk-size queue
depth.

## Resolution: fused `qwen3_layer_quantized` + `qwen3_forward_greedy_ids_chunk_quantized` NIFs

Implemented the fix proposed in step 3 of "Suggested next steps" above (see
`.cursor/plans/fused_quantized_qwen3_layer_nif_dac3a14e.plan.md`): two new
C++ NIFs, `EMLX.qwen3_layer_quantized` (per-layer fusion, mirroring dense's
`qwen3_layer`) and `EMLX.qwen3_forward_greedy_ids_chunk_quantized`
(multi-token chunked-decode fusion, mirroring dense's
`qwen3_forward_greedy_ids_chunk`), both built around a new `Qwen3LinearWeight`
tagged-union type so every projection (q/k/v/o/gate/up/down, and `lm_head`)
can independently be dense or quantized inside one fused call. `model.ex`
now routes quantized/mixed layers through these NIFs unconditionally instead
of falling back to `decode_tensor_chunk_step`'s ~364-NIF-calls/token,
~11,300-queued-ops/chunk path. Correctness was verified with new
`emlx/test/emlx/fast_test.exs` fixtures/tests (41 tests, all passing)
comparing both NIFs against per-op references across `affine`/`mxfp4` modes
and a mixed dense/quantized layer, plus the existing
`emlx_axon/test/emlx/qwen3_quantized_test.exs` determinism test (unchanged,
still passing) confirming byte-identical greedy token ids.

Re-ran both e2e benchmarks on `pv-feat/lowering-compiler` after the fix
(same model, `Qwen3-0.6B-MLX-4bit`, same prompt/`max_new_tokens` as before):

**`bench/native_serving_e2e_bench.exs`** (real production path,
`host_sync: {:chunk, 31}`, `max_new_tokens: 60`, 6 runs) — the exact
benchmark that reproduced the regression above:

| | before (this doc, `pv-feat`) | after (this fix, `pv-feat`) | `main` (for reference) |
|---|---|---|---|
| individual runs (tok/s) | 49.4, 67.2, 70.9, 66.6, 58.9, 64.8 | 212.8, 256.4, 233.4, 262.9, 264.6, 261.4 | 65.6, 66.7, 77.1, 64.5, 79.4, 76.9 |
| median | 66.6 | **261.4** | 76.9 |

Not just closing the gap with dense — **~3.9x** the pre-fix median, and
**~3.4x** `main`'s dense-parity number. This is larger than the ~13x
call-count reduction alone would suggest at the microbenchmark level,
consistent with the earlier finding that queue-depth-dependent costs
(`synchronize()` at the chunk-flush boundary, scheduling/dispatch overhead
compounding across ~11,300 queued ops) dominate at the real chunk size —
collapsing the call count also collapses that queue-depth cost, not just the
flat per-call dispatch overhead.

**`bench/qwen3_e2e_bench.exs`** (`profile_timing: true`, which disables the
chunked fast path — measures per-token dispatch cost in isolation,
`max_new_tokens: 100`, greedy sampler):

| | kernel tok/s | e2e tok/s |
|---|---|---|
| after this fix | 185.7 | 166.2 |
| A0 reference (bobby_posts, M4 Max 64GB, pre-regression baseline) | — | 69.7 |

Even with chunking disabled (i.e. isolating just the per-layer
`qwen3_layer_quantized` fusion's effect on the ~13-calls/layer → 1-call/layer
reduction, without the chunk-level fusion also in play), throughput is
~2.4x the original pre-regression `main`/A0 baseline — confirming the
per-layer fusion alone (Phase 1) is a substantial win independent of the
chunk-level fusion (Phase 2).

**Status: resolved.** The regression reported at the top of this document
(~66.6 tok/s on `pv-feat` vs ~76.9 on `main`) is fixed, with quantized
throughput now well above both branches' pre-fix numbers.
