# Stage 20 — Emily backend-parity gap audit

Status: done. Docs-only; produced no code, finalized Stages 21–23 scope.

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
- Grad / mixed-precision / conv-pool training parity (M9/M13/M16/M17) —
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

## Results

**Advisor sign-off (before starting).** Confirmed docs-only scope is correct
(producing code here would pre-empt the scoping Stages 21–23 exist to
finalize); confirmed the seed list's claims are hypotheses to verify, not
facts ("re-verify each" is in the doc itself); flagged the real risk of
trusting Emily's own `PLAN.md`/`ROADMAP.md` as ground truth for Emily's
*shipped* state, symmetric to EMLX's own README-vs-code gap that Stage 19
closed — advisor's instruction: verify Emily's actual code
(`~/coding/emily/lib`, `~/coding/emily/c_src`), not just its docs, wherever a
claim is load-bearing. Also flagged that `~/.cursor/plans/native-compiler_emlxnativeexpr.plan.md`'s
`todos` list is stale (only covers Stages 00–19, missing 20–24 even though
they exist as docs and 24 is already `[x]` in `README.md`) — reconciled as a
housekeeping step below, kept separate from this stage's docs-only scope per
advisor guidance.

**Methodology.** Every claim below was checked against actual source —
`rg`/`grep` against `emlx/lib`, `emlx/c_src`, `emlx/test`, `emlx_axon/test`
on the EMLX side, and `~/coding/emily/lib`, `~/coding/emily/c_src` (not just
`PLAN.md`/`ROADMAP.md` prose) on the Emily side — not inferred from either
project's planning docs alone.

### Gap table (Emily M0–M27 + B-series, vs EMLX's current tree)

| Milestone | Capability | EMLX status | Evidence |
|---|---|---|---|
| M0–M2 | Scaffold, native op inventory, `Nx.Backend` impl | At parity (predates Emily; EMLX is the older project) | `EMLX.Backend` (`lib/emlx/backend.ex`) implements the full `@behaviour Nx.Backend` surface. |
| M3/M4/M7 | Bumblebee parity breadth (DistilBERT / Qwen3 / ViT / Whisper) | **Narrower on EMLX, but out of this audit's charter.** Only a Qwen3 parity suite exists (`emlx_axon/test/emlx/qwen3_quantized_test.exs`); no DistilBERT/ViT/Whisper-equivalent suite found. | `find emlx_axon/test -iname '*.exs'` → `axon_test.exs` (Axon-graph-rewrite unit tests, no model parity), `qwen3_quantized_test.exs`, `test_helper.exs`. Noted, not scoped into 21–23: the README's own framing of "Stages 20–23 extend this plan's charter" names four specific areas (observability, SDPA sinks, microscaled quantization, mixed-precision training) — model-breadth parity isn't one of them, and adding it would be new scope creep beyond what was asked. |
| M5 | `Nx.Defn.Compiler` impl | **Ahead.** Emily's M5 is a thin Elixir walk that dispatches one `Emily.Backend` NIF call per `Expr` node (same call-count as `Nx.Defn.Evaluator`) — no call-count optimization. EMLX's entire native-compiler project (this planning directory) *is* that optimization. | `emily/PLAN.md` M5: "In practice this is what `Nx.Defn.Evaluator` already does." |
| M6 | `mlx::core::compile` wrapping | **Diverged, not contradictory — see note below.** Emily measured and dropped it (transformer-block win <20% GPU, regression on CPU); EMLX ships it (`c_src/emlx_compiler.cpp:1804`, `mlx::core::detail::compile`). | See "M6 vs EMLX's Layer C" note below — these measure different optimization axes, not the same question with opposite answers. |
| M8 | Native `conv` | At parity. | `backend.ex:1017,1058` — `def conv` dispatches to native MLX, not a `via_binary`-style fallback. |
| M9 (primitives) | `indexed_add`/`indexed_put`/`gather` off `via_binary` | At parity — already native. | `backend.ex:1931` (`gather`), `1966`/`1971` (`indexed_add`/`indexed_put` via shared `indexed_op/6`). |
| M9 (testing) / M13 / M16 / M17 | Grad-equivalence suite, EXLA gradient oracle, `MixedPrecision` (bf16 + loss scaling), conv-pool training parity | **Confirmed genuinely missing**, exactly as seeded — and this is the real gap, not the primitives (see M9 row above). | Zero `*grad*`-named files under `emlx/test` or `emlx_axon/test`; no `MixedPrecision`/`mixed_precision` module or bf16-tagged test anywhere. **Side finding for Stage 23's triage:** `window_sum`/`window_max`/`window_min`/`window_product` are native *only* inside the compiler's IR (`lib/emlx/native/expr.ex:1169-1181`, Stage 06/13) — the eager `EMLX.Backend.window_reduce/6` hard-raises (`backend.ex:2256-2267`, `"window_reduce not supported in EMLX"`). Grad of a windowed op under `compiler: EMLX` is untested territory Stage 23 should include explicitly. |
| M10/M10.5 | Quantized inference + transparent `Nx.dot` dispatch | At parity, **arguably ahead.** Emily's M10 hit a real blocker: `Nx.dot/2`'s `Nx.LazyContainer.traverse/3` expects a single `%T{}`, so a 3-tensor `%QuantizedWeight{}` container raises before `Backend.dot/7` — Emily needed a whole Axon-layer rewrite (`quantized_dense`) + graph-rewriter (`Transform`, M10.5) to work around it. EMLX stores `quantization_config` *inline* on the `%EMLX.Backend{}` struct (still one `%Nx.Tensor{}` at the Nx layer), so `Nx.dot/2` never chokes — `Backend.dot/7` branches on `quantization_config` directly. | `backend.ex:106` (`defstruct [..., :quantization_config]`), `backend.ex:1236` (`dot/7` reading `cfg` off the weight tensor's `.data`). Also confirmed in the same-repo Stage 24 finding (`24-quantized-dot-compiler-gap.md`) that the *compiled* lane (not eager) still has a real, separate quantized-dot gap — unrelated to this M10/M10.5 comparison, which is about the eager path. |
| M11 | Fast fused kernels (`rms_norm`/`layer_norm`/`rope`/`sdpa`/`swiglu`) | At parity, **arguably ahead** on SDPA variant breadth. | `lib/emlx/fast.ex`: `rms_norm_callback`, `layer_norm_callback`(+`_no_bias`), 4 `rope_*_callback` variants, `swiglu_callback`, plus 4 SDPA variants including `scaled_dot_product_attention_causal_key_masked`; `lib/emlx.ex:924` additionally has a fused `kv_cache_sdpa_update` (donation-optimised KV-cache update + SDPA) that Emily's M11 doesn't have an equivalent for. |
| M12/M12.5 | Zero-copy `to_binary`; `from_binary` keeps memcpy (by design) | At parity, **same design call on both sides**, not just superficially similar. | `c_src/emlx_nif.cpp:152-163` (`to_blob_term`, `enif_make_resource_binary`, `row_contiguous` fast path). `from_binary` still `memcpy`s (`emlx_nif.cpp:223`) — matches Emily's own M12/M12.5 conclusion that page-aligned zero-copy `from_binary` isn't worth it for real-world (non-page-aligned) model checkpoints. |
| M14/M14.5 | Concurrency model (per-process/per-queue stream) | At parity on the *design intent* (per-execution-context stream ownership); **the "ahead" claim from an earlier draft of this row is weaker than stated and not established — corrected below.** Emily's stream-per-process model (M14) *also* gave each process its own stream, and still hit a real post-ship bug: concurrent `mx::eval` from multiple OS threads SIGABRT/SIGSEGVs, because the actual root cause is shared Metal `CommandEncoder` state, not stream ownership (`emily/PLAN.md:669-676`). EMLX's `EMLX.CommandQueue` gives each queue its own OS worker thread + stream (`c_src/emlx_worker.hpp`) the same way Emily's per-process streams did — but nothing in this audit rules out the identical bug class if two `CommandQueue`s dispatch `mx::eval` concurrently from their respective worker threads, since `mutex_`/`cv_` in `emlx_worker.hpp` guard *queue submission*, not a shared Metal encoder across queues. | `command_queue.ex` moduledoc; `emlx_worker.hpp:29,77,96,129,162`. **Not independently stress-tested as part of this audit** — a soak test mirroring Emily's `backend_concurrency_test.exs` (multiple concurrent queues/processes hammering `eval`) is needed before claiming parity or advantage either way; this is a real open question, not a resolved one, and shouldn't be cited as an "EMLX is ahead" data point until measured. |
| M15 | Native `mlx::linalg::*` | At parity. | `backend.ex:2164` (cholesky), `2177` (solve), `2188` (qr), `2209` (eigh), `2229` (svd), `2240` (lu); `triangular_solve` at `2024` (non-default variants are the Stage-17-accepted permanent hard-raise, unrelated to this parity check). |
| M18 | `:telemetry` events/spans | **Confirmed genuinely missing**, exactly as seeded. | Zero `telemetry` hits in `emlx/lib` or `emlx/mix.exs`; no `{:telemetry, ...}` dep. |
| M19 | Typed exception hierarchy | Out of scope on both sides — no correction needed. | EMLX ships one generic `EMLX.NIFError` (`lib/emlx.ex:1-3`); Emily itself defers a typed hierarchy to its 2.x line (`ROADMAP.md`). Neither project is behind the other here. |
| M20 | GPU interop pointers (`from_pointer`/`to_pointer`) | Out of scope on both sides — no correction needed. | Not implemented in EMLX; Emily defers to "a concrete downstream consumer asks" (`ROADMAP.md`). |
| M21 | `mix <lib>.doctor` | **CORRECTION — this is a real, unscoped gap, not "out of scope on both sides."** Emily actually ships `mix emily.doctor` today (`~/coding/emily/lib/mix/tasks/emily.doctor.ex`, 421 lines + `test/mix/tasks/emily_doctor_test.exs`): platform/macOS-version checks, active-variant detection, required `priv/` artifact checks, NIF-loadability check, and a live `Emily.Backend` smoke test, with short-circuiting so a failed prerequisite reports dependent checks as `[skip]` instead of cascading noise. Only the *narrower add-on* PLAN.md M21 describes (Xcode CLT / CMake / MLX-source-tree-state probes for a from-source build) is deferred — the shipped diagnostic task itself is not. EMLX has no equivalent Mix task at all. **Not folded into Stages 21–23** (none of those three stages' charters cover tooling/diagnostics) — flagged here as a finding needing its own scoping decision, not silently dropped. | `find ~/coding/emily -iname '*doctor*'` → `lib/mix/tasks/emily.doctor.ex`, `test/mix/tasks/emily_doctor_test.exs`. This was caught only on reviewer spot-check, not the initial pass — see "Reviewer sign-off" below; recorded here as a reminder that Emily's own `ROADMAP.md`/`PLAN.md` framing ("deferred") describes only the *unshipped increment*, not the whole milestone, and a `find`-the-code check would have caught it the first time. |
| M22 | Compile-time debug-assertion flags (`:debug_bounds_check`, `:debug_detect_nan_inf`) | **CORRECTION to the seed list — already substantially at parity, not "confirmed genuinely missing."** EMLX already ships `@enable_bounds_check` and `@detect_non_finites` (`backend.ex:6-94`), same `Application.compile_env/3`-gated, default-`false`, dead-code-eliminated-when-off design as Emily's M22. `@enable_bounds_check` already covers **100%** of Emily's M22 target op list: `gather` (`1934`), `indexed_add`/`indexed_put` (`1978`, shared `indexed_op/6`), `take`/`take_along_axis` (`2111`, `2121`). `@detect_non_finites` covers `dot` (`1313`) but **not yet** `conv` or the `EMLX.Fast` kernels — a real, narrower gap than the seed claimed. | See exact line numbers above. This is the single biggest correction this audit produced — it changes Stage 21's actual remaining scope materially (see "Stage 21 rescoped" below). |
| M23 | Docs/examples pass | Not audited in depth — out of this stage's charter (the README's four named areas don't include a docs pass), and EMLX already has per-module docs; a dedicated audit would be its own stage if ever prioritized. | — |
| M24 | 1.0 / Hex release | Not applicable as a gap — EMLX already ships versioned Hex releases (`mix.exs:5`, `@version "0.3.1"`, `package:` config present) predating Emily's whole M24 milestone. | `mix.exs:5,16,47`. |
| M25 | Microscaled quantization (`mxfp4`/`mxfp8`/`nvfp4`) | **Confirmed genuinely missing**, exactly as seeded. | Zero `mxfp4`/`mxfp8`/`nvfp4`/`microscal` hits anywhere in `emlx/lib` or `emlx/c_src`; `EMLX.Quantization` only carries `scales`/`biases`/`group_size`/`bits` (classical affine). |
| M26 | SDPA attention sinks | **Confirmed genuinely missing, and genuinely unplumbed** (not just "present but unused"). | `c_src/emlx_fast.cpp:39` has only a *comment* showing MLX's C++ signature includes `sinks`; none of the four `fast_sdpa*` NIFs (`fast_sdpa`, `fast_sdpa_masked`, `fast_sdpa_causal`, `fast_sdpa_causal_key_masked`) take a sinks parameter. |
| M27 | Public `einsum` helper | **Confirmed genuinely missing**, exactly as seeded. | `EMLX.einsum/3` (`lib/emlx.ex:223`) is a raw, undocumented 2-operand `deftensor`-generated NIF wrapper operating on raw refs, used only internally by `backend.ex`'s `dot_spec_to_einsum_spec` path — no public `%Nx.Tensor{}`-in/out helper, no non-EMLX-backend error path, no 3+-operand support demonstrated, unlike Emily's `Emily.Fast.einsum/2`. |
| B3 | Sparse/MoE matmuls (`gather_qmm`, `gather_mm`, `block_masked_mm`, `segmented_mm`) | Missing on both sides — no correction needed, not scoped in. | Zero hits in EMLX; Emily itself defers ("first MoE model target"). No MoE model is in either project's current charter. |
| B4b | FP8 dtype (`to_fp8`/`from_fp8`) | Missing on both sides — no correction needed, not scoped in. | Zero hits in EMLX; Emily itself is blocked on Nx upstream FP8 support. |
| B5 | `ThreadLocalStream`/`new_thread_local_stream` | N/A to EMLX as a gap. | EMLX's worker-per-queue design (see M14/M14.5 row) already gets the "each execution context owns its own stream" property Emily's B5 is investigating as a possible *simplification* of its own different (shared-thread) starting point. Nothing to adopt here. |

### Note: M6 vs EMLX's Layer C — divergence, not contradiction

At first glance, "Emily measured `mlx::core::compile` and dropped it; EMLX
ships `mlx::core::compile` as Layer C" looks like the two projects reaching
opposite conclusions on the same question. They didn't — they measured
different things:

- **Emily's M6** measured `mlx::core::compile`'s *kernel-fusion* win (RMSNorm
  chains, softmax neighborhoods, SwiGLU's `silu × up`) on top of a backend
  that *already* issues one NIF call per `Nx.Defn.Expr` node (Emily's M5
  Compiler is explicitly "what `Nx.Defn.Evaluator` already does" — no
  call-count change). The fusion win alone was <20% on transformer shapes
  (matmul-dominated, not fusion-bound) and a regression on CPU — not worth
  the integration cost *for that specific win*.
- **EMLX's whole native-compiler project's thesis** (this planning
  directory, README "Goal") is a *different* axis: collapsing **N
  BEAM↔NIF round-trips into 1 per invocation** — a dispatch-cost problem
  Emily's M5/M6 never measured or addressed, since Emily still pays N NIF
  calls per `defn` today (same call-count as EMLX's pre-Stage-01 Evaluator
  baseline). EMLX's Layer C happens to *also* wrap the replay in
  `mlx::core::detail::compile` (getting Emily's measured ≤20% fusion bonus
  "for free" during replay), but that's not EMLX's central bet — the
  call-count collapse is, and Stage 01's own perf gate ("single-NIF replay
  must beat op-by-op Evaluator") already validated it independently of
  Emily's M6 finding.

Conclusion: Emily's M6 result doesn't undermine EMLX's Stage 01 perf gate,
and EMLX's Layer C doesn't mean Emily's M6 drop was wrong — they answered
different questions. No action item; recorded so a future reader doesn't
mistake this for an unresolved architectural disagreement.

### Stage 21 rescoped (per the M22 correction above)

`21-observability.md`'s Procedure step 3 currently reads as "build two debug
flags from scratch, mirroring Emily M22" — that's now known to be wrong.
Corrected in that doc directly (see diff): `:debug_bounds_check`-equivalent
coverage already exists and needs no new work; `:debug_detect_nan_inf`-equivalent
coverage exists for `dot` only and needs **extending to `conv` and the
`EMLX.Fast` kernels**, not building fresh. The `:telemetry` half of Stage 21
is untouched — confirmed genuinely absent.

### Stage 22 and 23 — no scope correction needed

All three Stage 22 items (SDPA sinks, microscaled quantization, public
`einsum`) and Stage 23's framing (training *primitives* already native,
*parity testing* is the actual gap) were independently re-verified
against code, not just the seed list, and confirmed accurate as written.
Stage 23's doc gets one addition: the window-reduction eager-vs-compiled
asymmetry found above, as an explicit triage item.

### Housekeeping (flagged by advisor, done separately from this stage's docs-only scope)

`~/.cursor/plans/native-compiler_emlxnativeexpr.plan.md`'s `todos` list
reconciled to include Stages 20–24 (20 → `completed`, 21–23 → `pending`, 24 →
`completed`, matching `README.md`'s existing checklist state).

**Reviewer sign-off (round 1 — blockers found and fixed).** A fresh reviewer
subagent (fed only the Acceptance criteria + this doc + the three referenced
stage docs + a spot-check mandate against source, no reasoning) found two
real blockers and two gaps:

1. **M21 row was wrong** — Emily actually ships `mix emily.doctor`
   (`~/coding/emily/lib/mix/tasks/emily.doctor.ex`, 421 lines + a test file):
   platform/variant/artifact/NIF-loadability checks + a live Backend smoke
   test. Only a narrower source-build-diagnostics *increment* is deferred
   per `PLAN.md`, not the whole milestone — the initial pass over-trusted
   `ROADMAP.md`'s "Deferred to post-1.0" bullet without following it to the
   actual `lib/mix/tasks/` tree, the exact docs-vs-code trap this audit's
   own methodology claimed to guard against. **Fixed**: M21 row corrected
   to reflect the real gap (EMLX has no `mix emlx.doctor` equivalent at
   all); explicitly left unscoped (no Stage 21–23 charter covers tooling)
   rather than silently folded in.
2. **`README.md`'s Stage 21 checklist bullet still referenced the old,
   wrong (Emily-native) flag names** (`:debug_bounds_check`/
   `:debug_detect_nan_inf`) one line below the Stage 20 bullet describing
   the correction to EMLX's actual flag names
   (`:enable_bounds_check`/`:detect_non_finites`). **Fixed**: Stage 21's
   README bullet rewritten to use the correct names and framing.
3. **Gap**: the M14/M14.5 "arguably ahead" concurrency claim overstated what
   was established — Emily's own per-process-stream design *also* had its
   own stream per context and still hit the shared-`CommandEncoder` bug;
   nothing in this audit rules out the same class of bug across concurrent
   `EMLX.CommandQueue`s. **Fixed**: row rewritten to state this as parity on
   design intent only, explicitly un-established as "ahead," and flagged as
   needing a soak test before any stronger claim.
4. **Gap**: `kv_cache_sdpa_update` was cited as living in `fast.ex`; it's
   actually in `lib/emlx.ex:924`. **Fixed**: citation corrected.

All four addressed directly in this doc and `README.md` (see edits above).

**Reviewer sign-off (round 2, clean).** A fresh reviewer subagent (no
`resume`, same clean-context discipline) independently re-verified all four
round-1 fixes against source (M21 row, `README.md`'s Stage 21 bullet, the
M14 row's un-overclaimed framing, the `kv_cache_sdpa_update` citation) and
did a full fresh re-scan of every load-bearing citation in the gap table —
all checked out. Verdict: **pass**, no blockers remaining. Two non-blocking
observations: (a) `~/.cursor/plans/native-compiler_emlxnativeexpr.plan.md`'s
pre-existing staleness is broader than this stage's housekeeping fix
described (Stages 03–08 still show `pending` there despite being `[x]` in
`README.md`, and Stages 11–18 are missing entirely, not just 20–24) — a
pre-existing side issue predating this stage, explicitly out of this stage's
docs-only charter, not part of the M0–M27 gap table itself, and not
addressed further here; (b) this doc's structure, not its content.
