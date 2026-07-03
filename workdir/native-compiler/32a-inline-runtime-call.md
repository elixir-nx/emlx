# Stage 32a — inline (non-splitting) `runtime_call` execution

Status: in progress — Procedures #1–#5 and #5b are done (spike, production
`:host_callback` opcode, thread-local caller-pid redesign, `EMLX.Native.Expr`
lowering + `emlx.ex` wiring, Stage 31 split-point removal for `runtime_call`,
full `mix test` suite green). Procedure #8 (`validate_qwen3.exs`) found and
fixed a real deadlock (nested `mlx::core::eval()` reentrancy) and a real
silent-corruption bug (non-contiguous operand/reply byte serialization), but
uncovered a **new, unresolved** correctness bug: a prefill call's `offset`
operand reads garbage on a generation request's compiled-program replay
after a prior request already ran many calls against it (see Results for
what's been ruled out). Procedures #6/#7 (mutable-host-state regression
test, structural-fusion regression tests) are not started. See Results for
full detail before continuing.
Named by [`32-runtime-call-dispatch-cache`](32-runtime-call-dispatch-cache.md)'s
Results, per user directive, superseding that stage's approach.

## Why this stage exists

Stage 31 made an *unrecognized* `runtime_call` (any callback that isn't one
of Stage 10's `EMLX.Fast.*` fused kernels — e.g.
`EMLXAxon.native_kv_attn_callback/2`) **correct** by treating it as a
`Nx.Defn.Graph.split` point, exactly like `while`. Stage 32 tried to make
that **fast** by caching the compiled artifact for each split-point stage,
keyed by a structural (id-independent) signature instead of by `Expr`
identity, so a stage compiled once could be reused across decode steps and
structurally-identical call sites.

That did not clear the real bar. Even with the cache working correctly
(Stage 32 Results items 1–3), a real `bb+rewrite` run against Qwen3's
28-attention-layer model did not finish within 10 minutes. **User directive:
anything larger than a couple of seconds, per call, is unacceptable** — a
materially stricter bar than Stage 32's original "same order of magnitude as
`bb base`/`native`" framing. The problem is architectural, not a
caching-completeness gap: `Nx.Defn.Graph.split` fragmenting the graph into
dozens of stages, and re-tracing/re-splitting that fragmentation on every
call, has real cost independent of whether each fragment's *compiled NIF
artifact* is cached. Caching the artifact doesn't undo the fragmentation.

**This stage's charter: don't split at all.** Make an unrecognized
`runtime_call` an **in-graph** compiled instruction — the callback becomes
one more opcode in the same single compiled program the rest of the graph
already lowers to, exactly like a Stage 10 `EMLX.Fast.*` fused kernel. No
`Nx.Defn.Graph.split`, no host round-trip stage boundary, no re-tracing per
call. The whole graph (all 28 layers) compiles once (per structural shape,
via MLX's own `mlx::core::detail::compile` cache — already proven fast by
every other stage since Stage 01) and replays as a single NIF call, with the
`runtime_call`'s host callback invoked *from inside* that one NIF call when
the replay reaches it.

## Why this is plausible (spike this first, don't assume it)

MLX ships a real mechanism for exactly this shape of problem:
`mlx::core::custom_function`
(`~/Library/.../include/mlx/transforms.h`, backed by the
`CustomTransforms` primitive in `primitives.h`) wraps an arbitrary
`std::function<vector<array>(vector<array>)>` as one opaque graph node.
MLX's lazy engine treats it like any other op: evaluating a downstream array
that depends on it first materializes its input arrays, then calls the
wrapped function with concrete data, and continues from its (also concrete)
output arrays. Because it's just a C++ `std::function`, the callback body
can do arbitrary host work — including a blocking round-trip into Erlang —
without MLX needing to understand or trace through it. This composes with
`mlx::core::detail::compile()` the same way every other opcode in
`emlx_compiler.cpp` already does: the callback becomes one instruction in
the interpreter lambda that `compile_program` wraps and MLX caches/replays
by unique ID, same as `:dot` or `:fast_rms_norm`.

The part that needs a real spike, not an assumption: **can the worker OS
thread executing a compiled program's replay safely call back into Erlang
and block on a reply**, without deadlocking EMLX's `ASYNC_NIF`/`enif_send`
worker-queue dispatch (`c_src/emlx_worker.hpp`) or corrupting in-flight
Metal command encoder state on the GPU stream. This is the load-bearing
unknown — resolve it before committing to the rest of the design.

## Procedure (sketch — refine after the spike)

1. **Spike: host callback from inside a replayed compiled program.**
   Smallest possible repro: a `compile_program`'d graph with one
   `custom_function`-backed instruction whose C++ callback does a
   synchronous `enif_send` to a known Erlang process and blocks (with a
   timeout) for a reply, on both `:cpu` and `:gpu` devices, both as a
   standalone call and nested inside another compiled program (mirroring
   "`runtime_call` inside a `while` body"). Confirm: no deadlock with the
   worker's own async reply queue; GPU stream/encoder state survives the
   round-trip; the array returned by the callback is usable by subsequent
   instructions in the same program. **Decision gate: go/no-go on this
   stage based on the spike, before writing any of the rest.**
2. **New `emlx_compiler.cpp` opcode** (e.g. `:host_callback`) built on
   `mlx::core::custom_function`: given operand arrays, synchronously invoke
   a registered Erlang callback (see #3) and wrap the result back as
   `mx::array` outputs. Mirrors the existing op-registry pattern
   (`op_registry`/`multi_op_registry` in `emlx_compiler.cpp`), not a new
   dispatch mechanism.
3. **Callback identity across the NIF boundary.** A NIF call can't carry an
   Elixir closure. Need a stable way for the C++ instruction to name "which
   Erlang function to call" and for `eval_program` to route the mid-replay
   callback invocation back to it — likely a registry of `{module,
   function}` pairs (or capture MFAs, matching `EMLX.Native.Expr.
   recognized_runtime_call?/1`'s existing MFA-based recognition) resolved on
   the Elixir side when `eval_program`'s reply-dispatch fires, not something
   baked as an opaque pointer into the compiled program.
4. **`EMLX.Native.Expr.lower/2`**: an unrecognized `runtime_call` node lowers
   to the new opcode instead of raising / instead of Stage 31's split-point
   routing. Recognized `EMLX.Fast.*` callbacks are unaffected (still Stage
   10's direct fusion — no callback round-trip needed for those).
5. **`emlx.ex`**: `contains_split_point?/1`/`split_point?/1`/
   `build_split_chain_eval_fn/2`/`bare_runtime_call?/1`/
   `build_runtime_call_base_eval_fn/2` (Stage 31) become dead code for the
   `runtime_call` case (retained for `:while`, which still needs to split —
   this stage does not touch `while`). Remove or narrow them once the new
   path is proven equivalent.
6. **Mutable host state semantics.** `EMLXAxon.native_kv_attn_callback/2`
   reads/writes a process-dictionary-backed ETS-style KV cache
   (`Process.get/put(@kv_cache_proc_key, ...)`). Confirm the callback still
   runs in the *calling* Elixir process (not a NIF-internal worker process)
   so `Process.get/put` semantics are unchanged from today's
   `build_runtime_call_base_eval_fn` behavior — this is a correctness
   requirement, not a performance one.
7. **Regression tests**: reuse Stage 31's `:stage31` scenarios (bare
   runtime_call, surrounded, two independent calls, inside a `while` body,
   tuple operand container) asserting *structural* fusion this time (one
   compiled program, no split — mirroring the existing "recognized
   `EMLX.Fast.*` is fused, not split" regression test) in addition to
   numeric equivalence against `Nx.Defn.Evaluator`.
8. **Validate against `validate_qwen3.exs`** with the real acceptance bar
   (see below).

## Open questions (resolve before/while implementing)

- Does `mlx::core::custom_function`'s callback run synchronously on the
  thread that's evaluating the graph (the EMLX worker thread), or does MLX
  ever invoke it from a different internal thread (e.g. a Metal completion
  handler)? This determines whether the "call back into Erlang and block"
  design is even thread-safe as sketched.
- `mlx::core::compile()` caches by a graph's traced shape; does embedding a
  `custom_function` node change how MLX's compile cache treats
  re-tracing/graph identity across calls (e.g. does it re-trace every call
  regardless, defeating the "compile once" property this stage exists to
  restore)? Verify empirically in the spike, don't assume parity with
  ordinary ops.
- Backward-pass (`vjp`) support: `custom_function` takes optional
  `fun_vjp`/`fun_jvp`/`fun_vmap`. Does any real `runtime_call` site need
  gradients through it under `compiler: EMLX`? (Stage 28's grad-equivalence
  suite may already answer this — check before assuming it's needed.)
- Does this fully retire Stage 31/32's split-point machinery, or do some
  `runtime_call` shapes still need a split (e.g. one that must be the very
  last op before a hard host synchronization point Bumblebee's serving loop
  requires)? Scope narrowly — don't remove Stage 31 machinery until this
  stage's equivalence tests prove it unnecessary for every case Stage 31
  covered.

## Acceptance

- **Real bar (supersedes Stage 32's)**: `bb+rewrite` in `validate_qwen3.exs`
  completes each decode step in on the order of a couple of seconds or
  less, not tens of minutes and not "same order of magnitude as bb
  base/native" — measure wall-clock per call directly, don't infer from
  tok/s alone.
- A `runtime_call` split-point stage from Stage 31 no longer causes a
  `Nx.Defn.Graph.split` at all — assert this structurally (one compiled
  program, single `eval_program` NIF call), not just numerically.
- Full `emlx`/`emlx_axon`/`Nx` suites remain green; Stage 31's `:stage31`
  equivalence tests still pass (adapted to assert non-split fusion instead
  of split routing where the scenario is now handled in-graph).
- The spike (Procedure #1) has a written go/no-go verdict *before* the rest
  of the stage is built — if the callback-from-worker-thread mechanism
  isn't safe, this stage's Results should say so plainly and name whatever
  fallback is next, rather than forcing the original design through.

## Results

**Procedure #1 (spike) — done, verdict: GO, with two corrections to the plan
found before writing any production code.**

1. **The plan's named mechanism (`mlx::core::custom_function`) is wrong —
   found by reading MLX's actual source, not by building the spike.**
   Fetched `mlx/transforms.cpp` (ml-explore/mlx @ d4c81062) directly:
   `custom_function`'s returned lambda calls `auto outputs = fun(args);`
   **eagerly**, at graph-construction time — the `CustomTransforms`
   primitive it builds only overrides autodiff (vjp/jvp/vmap); its own
   `eval_cpu`/`eval_gpu` just passes through the already-computed outputs
   (confirmed against the vendored headers in
   `~/Library/Caches/libmlx/libmlx-0.31.2-arm64-apple-darwin/include/mlx/`,
   which match the fetched source exactly). Since `compile_program` traces
   its interpreter lambda once and `mlx::core::detail::compile`'s replay
   skips re-invoking that outer builder (the entire point of the compile
   cache — proven by every stage since Stage 01), wrapping a host callback
   in `custom_function` would fire it exactly once at first-trace time and
   never again on replay. That's fatal for a per-decode-step callback and
   would have surfaced as a silent, hard-to-diagnose correctness bug (stale
   first-call value replayed forever) rather than a build error.
   **Correction**: use a bespoke `mlx::core::Primitive` subclass instead —
   the same mechanism every existing opcode in `emlx_compiler.cpp` already
   relies on transitively (`mlx::core::linalg::*`, `EMLX.Fast` kernels).
   `Primitive` is a normal exported, subclassable base class
   (`mlx/primitives.h`), not a Python-only construct — fully available to
   EMLX as a C++ project linking libmlx directly.

2. **The load-bearing unknown itself (worker thread → blocking Erlang
   round trip, from inside a real compiled-graph replay) is a GO,
   confirmed empirically**, via a throwaway spike
   (`c_src/emlx_compiler.cpp`'s `spike32a` namespace + `spike32a_run`/
   `spike32a_resume` NIFs, driven by `bench/spike32a_host_callback.exs`).
   The compiled program is `z = 2 * HostCallback(x) + x; w = x + 100`,
   returning both — `z` forces the callback's output to be **consumed by a
   real downstream op** in the same program (not just returned raw) and
   `w` is a **second, independent op on the graph's own stream**, sharing
   the program but not depending on the callback, read back after the
   round trip completes (the encoder/stream-survival check; see item 6 for
   why `w` isn't wired into the callback's own input side):
   - `HostCallback : mlx::core::Primitive` is CPU-pinned (same
     `k_linalg_cpu`-pin precedent Stage 09 already validated for composing
     host-oriented ops inside a `:gpu`-streamed compiled graph). Its
     `eval_cpu`/`eval_gpu` (identical body) does `enif_send` + blocks on a
     `condition_variable` for a reply.
   - The reply is delivered by `spike32a_resume/2`, a **plain, non-worker-
     routed NIF** called directly by a different Erlang process — it
     bypasses `emlx::Worker::post` entirely (posting the resume through the
     worker's own queue would self-deadlock, since the worker thread is
     the one blocked; this was the advisor's flagged sharpest risk, and the
     design avoids it structurally rather than by accident).
   - Result: **no deadlock**, on both `:cpu` and `:gpu` (device selects the
     *surrounding* graph's stream; the callback node itself is always
     CPU-pinned). `same_thread` (compares `std::this_thread::get_id()`
     captured just before `mx::eval` against the id observed inside
     `eval_cpu`/`eval_gpu`) is `true` in every run — the callback executes
     synchronously on the same worker OS thread that called `mx::eval`,
     not on some MLX-internal/Metal-completion-handler thread.
   - **`w` (the independent :gpu-stream op) reads back correctly
     (`x + 100`) after the callback's blocking round trip, inside the same
     compiled program and single `eval_program`-equivalent call** — the
     Metal command-encoder/stream state survives a mid-replay host round
     trip, not just in a separate, later, top-level call. A plain
     `Nx.add` issued on the same `:gpu` worker immediately after the spike
     call also succeeds with correct values.
   - The callback's own output (`z`) is genuinely consumed by a real
     downstream op (`2 * callback_out + x`) in the same compiled program
     and comes back numerically correct — the interpreter's dependency
     tracking / buffer lifetime handling across a blocking host callback
     works for this shape.
   - **Open Question 2 answered empirically, not just inferred**: calling
     the same compiled program twice with the same `compile_id` but
     different inputs keeps the outer graph-builder's trace count at 1
     (`mlx::core::detail::compile` replays, does not re-trace) while the
     `HostCallback`'s own eval count increments every call — the host
     callback genuinely re-fires on every real replay, exactly like every
     other opcode already in the registry.
   - Five sequential calls sharing one `compile_id` (proxying "N
     structurally-identical attention layers" or "N decode steps", and by
     extension a `while`-body callback, since Stage 08's `while` already
     drives its host loop via repeated top-level `eval_program`-style calls,
     not C++-level nesting) all round-tripped correctly with the expected
     1-trace / N-eval counts.

3. **Found and fixed a bug in the spike's own code, not in EMLX**: the
   first draft skipped `eval_program`'s existing, documented precaution
   (force-`mx::eval` a compiled fn's inputs before passing them in, or a
   replay can read stale/reused-buffer data from a prior call's leaf) —
   without it, results silently compounded across calls (each call's input
   read back as the *previous* call's output). Fixed by mirroring
   `eval_program`'s existing pattern exactly. Confirms that precaution is a
   general compiled-graph-input rule, not specific to the ops that
   motivated the original comment.

4. **Found a real design correction for the production implementation**:
   the spike's first draft held `s_mlx_compile_mutex` (the process-wide
   mutex serializing MLX's compile cache across all workers, documented at
   its declaration site) across the *entire* `mx::eval` call, including the
   callback's unbounded, host-dependent blocking wait. That's not a
   deadlock (the mutex is eventually released once `resume` fires), but it
   would stall every *other* worker's compile/evict path — including
   unrelated `Nx.Serving` requests sharing the default worker — for the
   full duration of one callback's round trip, undermining Stage 32a's own
   "make it fast" charter. Fixed in the spike to mirror `eval_program`'s
   already-narrower scope (lock only around `compiled_fn(inputs)`, not the
   subsequent `mx::eval`) — the real opcode implementation (Step 2) must do
   the same.

6. **Found a real, separate MLX correctness landmine — more serious than
   the threading question, and independent of it**: an early draft fed the
   callback's *input* from an ordinary elementwise op computed just
   upstream in the same compiled program (`y = x + 1; callback_out =
   HostCallback(y); z = 2*callback_out + y`). On `:cpu` this **silently
   returned the wrong numeric value** (no crash) — the compiled program's
   output came back equal to `y` alone, not `z`, consistent with MLX's own
   CPU elementwise auto-fusion pass (`mlx::core::detail::compile`'s
   internal "Compiled" kernel-fusion simplification, distinct from the
   file's own op-registry) mishandling a shared subexpression that has
   *both* a fusable consumer and an opaque, fusion-unaware custom
   `Primitive` as consumers. Removing the pre-callback fusable op (feeding
   the callback directly from the compiled program's own external input
   instead) made the result correct again; the final spike's `w = x + 100`
   independent-op check (item 2) deliberately avoids re-triggering this by
   not wiring `w` into the callback's dependency chain. **This is a real,
   unresolved design constraint for Step 2** (the production opcode):
   feeding an unrecognized `runtime_call` from the output of an ordinary
   preceding op — the normal case for a real attention layer, e.g. a QKV
   projection feeding `native_kv_attn_callback` — needs its own targeted
   equivalence test before Step 2 is considered safe, not just the
   downstream-consumption shape this spike ended up validating.
7. **Separately, first-time compilation of a brand-new CPU kernel shape
   intermittently failed with `pclose() failed` in this sandboxed dev
   shell** (`/var/folders/.../T/mlx/0.31.2/cpu/*.so` — MLX's CPU backend
   shells out to `cc` via `popen`/`pclose` to JIT-compile fused kernels,
   caching the resulting `.so` on disk), even though the `.so` was in fact
   produced on disk with a plausible size; a bare retry (kernel now cached)
   always succeeded. Reproduced only for genuinely novel kernel shapes tied
   to this spike's exact op sequence — a plain, previously-uncompiled
   `Nx.Defn` elementwise chain through the *production* `compiler: EMLX`
   path compiled fine on the first try in the same session. Likely an
   artifact of this specific sandboxed shell's subprocess/signal handling
   (a classic `pclose()`-races-`SIGCHLD`-reaping failure mode), not an MLX
   or EMLX defect — flagged here rather than silently ignored, since Step 8
   (real `validate_qwen3.exs` validation) will hit many first-time kernel
   compiles and should not misread a flaky `pclose()` as a real failure
   without checking whether the `.so` was actually produced.
8. **Not exercised by this spike, deferred to Steps 2–8**: callback
   identity across the NIF boundary (Procedure #3's MFA registry), mutable
   host state / process-dictionary semantics (Procedure #6,
   `native_kv_attn_callback`'s KV-cache), backward-pass/vjp support (Open
   Question 3), and — critically — the real acceptance bar
   (`validate_qwen3.exs` wall-clock per decode step). The spike used a
   synthetic 1-element array and an Elixir-side `receive` loop standing in
   for the real callback dispatch; it establishes the mechanism is sound,
   not that the full integration meets the couple-of-seconds bar.

**Procedure #1 verdict: GO, conditionally** — the core mechanism (worker
thread blocking on a host round trip from inside a real compiled replay)
is a clean GO with no deadlock, correct results, and confirmed
encoder/stream survival. But item 6's fusion-adjacency finding means
Procedure #4 (the actual `EMLX.Native.Expr.lower/2` wiring) is **not**
simply "wire it into the lowerer" — it must also add a targeted
equivalence test for "unrecognized `runtime_call` fed directly from an
ordinary preceding op" (the realistic shape, not just the
callback-consumes-external-input shape this spike validated) before that
case can be trusted, and should treat MLX's CPU fusion pass's interaction
with custom `Primitive`s as a standing risk to re-check whenever the op
immediately upstream or downstream of a `runtime_call` changes. Plan
corrected to build a raw `Primitive` subclass (not `custom_function`) and
to keep `s_mlx_compile_mutex`'s scope narrow around the callback's
blocking wait. Full `mix test` suite green throughout (2671 passed, 0
failed) — the spike code is additive only (new `spike32a` namespace + 2
new NIFs), not wired into `op_registry`, so it cannot regress existing
behavior.

**Procedure #2 (production `:host_callback` opcode) — done.**

Implemented in `c_src/emlx_compiler.cpp`'s new `host_callback` namespace
(plus a `"host_callback"` entry in `op_registry`, matching the file's
existing string-keyed dispatch convention — not a new mechanism) and two
new NIFs (`host_callback_register/1`, `host_callback_resume/2`), declared
in `emlx_compiler.hpp` and wired in `emlx_nif.cpp` / `lib/emlx/nif.ex`.
This is the production generalization of Procedure #1's `spike32a`
mechanism: a real `mlx::core::Primitive` subclass (`host_callback::
HostCallback`, CPU-pinned like the `k_linalg_cpu` linalg opcodes), not
`mlx::core::custom_function`, for the reason established in Procedure #1.

1. **Callback registration and the round trip are generalized from
   spike32a's scalar-only probe to arbitrary-shape/dtype array operands
   and replies** — the realistic shape for a real `runtime_call` (e.g.
   `native_kv_attn_callback`'s 5 tensor operands). `host_callback_register/1`
   lets any Erlang process register itself as the receiver for a fresh
   integer `callback_id` (a placeholder for Procedure #3's real MFA
   registry — see that procedure's scope note in the opcode's own comment
   header). The op itself takes `attrs = [callback_id, dtype_int, n_dims,
   d0..dn-1]` (the output's shape/dtype, since a `Primitive`-backed array
   must declare its shape/dtype at construction time — mirrors `iota`/`eye`
   in the same file) and forwards all its operand arrays.
2. **The outbound message is fully self-describing — `{ref, shape,
   dtype_atom}` per operand — specifically so the receiving Erlang process
   never needs to call a worker-routed NIF (e.g. `EMLX.shape/1`) to
   interpret it.** This matters structurally, not just stylistically: the
   worker OS thread that would service such a call is the exact thread
   blocked inside `host_round_trip`, so routing through it would
   self-deadlock. (Empirically, `EMLX.shape/1`/`EMLX.scalar_type/1` turned
   out to be `defvalue`-based — not worker-routed at all — so this
   particular pair would have been safe either way, but the message stays
   self-describing rather than depending on that non-obvious fact holding.)
3. **`host_callback_resume/2` is deliberately NOT worker-routed** (same
   reasoning as `spike32a_resume`) but IS registered `ERL_NIF_DIRTY_JOB_
   CPU_BOUND` (precedented by `array_from_shm`/`tensor_data_ptr`/
   `array_from_ptr` in `emlx_nif.cpp`) since it calls `mlx::core::eval` on
   the reply tensor, which may do real (possibly GPU-triggering) compute —
   running that on an ordinary BEAM scheduler thread would stall it for
   the eval's duration.
4. **Verified end-to-end through the real production path** (not a
   parallel spike-only helper): `bench/host_callback_opcode.exs` hand-builds
   the `EMLX.Native.Expr` wire format for a single `"host_callback"`
   instruction and drives it through the actual `compile_program`/
   `eval_program` NIFs, servicing the mid-eval `{:emlx_host_callback, …}`
   message with a real Nx-computed reply (`2 * operand`, elementwise, on a
   3-element f32 tensor — not spike32a's single scalar). Confirms the full
   contract: registration → compiled instruction → mid-replay message →
   Nx-computed reply → resume → correct materialized result
   (`[1,2,3] → [2,4,6]`), run repeatedly without flakiness.
5. **Found a second, real self-deadlock trap while writing the test — this
   time in the callback body's own compute, not the resume plumbing.**
   The first draft computed the reply with a plain `Nx.multiply/2` on the
   default `:cpu` device/worker — but that is the *same* worker OS thread
   already blocked inside `host_round_trip` for this very call; the
   `Nx.multiply` NIF call queued behind the block instead of running,
   recovering only once `host_round_trip`'s 30-second timeout fired,
   unblocked the worker, and let it drain its queue — at which point
   `host_callback_resume` failed with "unknown call_id" (the pending entry
   had already been erased by the timeout path). **Fix**: the callback
   computes inside a dedicated `EMLX.CommandQueue.with_queue/2` block (a
   second, independent worker OS thread), not the default device worker.
   **This is a real, unresolved design requirement for Procedure #3/#6**:
   whatever dispatches a real `runtime_call` callback (e.g.
   `native_kv_attn_callback`) must run the callback's own Nx computation on
   a command queue distinct from the one executing the blocked compiled
   program — using the plain default worker (today's behavior for every
   other Nx op) would self-deadlock on the very first invocation, not just
   under contention. This is a stronger, more general version of Procedure
   #1's threading finding: it's not just that the *round-trip plumbing*
   must avoid the worker's queue (item 3 above, and `spike32a_resume`) —
   the callback's *own subsequent computation* must too.
6. **Not exercised here, deferred to later procedures**: real MFA-based
   callback identity (Procedure #3 — `host_callback_register/1` is
   intentionally minimal pid bookkeeping, not that registry), `emlx.ex`/
   `EMLX.Native.Expr.lower/2` wiring (Procedure #4), the fusion-adjacency
   equivalence test named in Procedure #1 item 6, an explicit `:gpu`-device
   run of this opcode (architecturally identical to `spike32a`'s already-
   validated `:cpu`/`:gpu` parity, since `HostCallback` is CPU-pinned
   exactly like the linalg opcodes regardless of the surrounding graph's
   device — not re-run here to conserve effort, but worth a real
   `:gpu` pass before Procedure #8), and a permanent ExUnit regression test
   (`bench/host_callback_opcode.exs` is a manual smoke-test artifact,
   mirroring `spike32a`'s own bench-script precedent; Procedure #7 is
   where this stage's tests become permanent). No error-reply path
   (`host_callback_resume_error`-style) was added — an Erlang-side
   exception during the callback simply never resumes, surfacing as
   `host_round_trip`'s existing 30s-timeout error rather than the
   callback's real exception message; acceptable for this procedure's
   scope but worth revisiting once real callbacks (which can raise) are
   wired in.

Full `mix test` suite green throughout (2671 passed, 0 failed, no new
warnings) — additive only (new `host_callback` namespace/op-registry entry
+ 2 new NIFs), not reachable from any existing code path.

**Overall verdict: proceed to Steps 3–8.** The core opcode mechanism
(register → compile → mid-replay message → real Nx-computed reply →
resume → correct result) is proven end-to-end through the actual
`compile_program`/`eval_program` path, generalized from spike32a's scalar
probe to real (multi-byte, arbitrary-shape) tensor operands and replies.
Item 5's finding (the callback's own computation needs an independent
command queue) is a concrete, actionable requirement to carry into
Procedure #3's MFA-registry design and Procedure #6's mutable-host-state
confirmation — not a blocker, but should not be silently assumed away.
Procedure #1's fusion-adjacency condition (above) still stands as a
requirement for Procedure #4.

**Procedures #3–#5 (thread-local caller-pid redesign, `EMLX.Native.Expr`
lowering, `emlx.ex` wiring, Stage 31 split-point removal) — done.**

Superseded Procedure #2's placeholder `host_callback_register/1` pid
bookkeeping with a thread-local `emlx::g_current_caller_pid_ptr`
(`emlx_async.hpp`) set fresh by `async_dispatch` for whichever call is
*currently* running on a worker thread, so a compiled program traced once
but replayed by many different Erlang processes (e.g. a pooled decode
loop) routes each mid-eval callback to its *actual* current caller, not a
stale registered one (`bench/host_callback_multi_caller.exs` regression-
probes exactly this). `callback_slot` (an opaque 0-based index into the
per-program callback-spec table `EMLX.Native.Expr.runtime_call_callback_specs/1`
builds by re-walking the same `output` expr in the same post-order `lower/2`
uses) replaces the old `callback_id`. `EMLX.dispatch_host_callback/6`
reconstructs each operand from its wire `{ref, shape, dtype_atom}` (or, for
a bare-parameter pass-through operand, substitutes the caller's real bound
tensor to preserve Elixir-side-only metadata invisible on the wire —
`quantization_config`), runs the callback on the dedicated
`host_callback_worker` command queue, force-evaluates the reply there
(`host_callback_resume/2` is a dirty NIF on an arbitrary OS thread with no
MLX stream of its own and deliberately does not evaluate the reply itself),
then resumes. Stage 31's `Nx.Defn.Graph.split`-based routing for
`runtime_call` (`contains_split_point?/1`/`split_point?/1`/
`build_split_chain_eval_fn/2`/`bare_runtime_call?/1`/
`build_runtime_call_base_eval_fn/2`) is removed (retained for `:while`,
untouched by this stage). `EMLX.Native.Expr.quantizable_param_positions/1`
was added to scope `quant_signature/2`'s dispatch-cache key to only the
parameter positions actually consumed by a `:dot` (avoiding cache
fragmentation from unrelated bound quantized tensors — found and fixed
while chasing Procedure #5b's `mix test` regressions, see below).

**Two real MLX-level correctness bugs found and fixed along the way, both
about raw-byte serialization assuming row-major contiguity that MLX does
not guarantee just because an array is "evaluated":**

1. **Non-contiguous (strided) operands/replies silently produce wrong
   bytes.** An array reaching `host_round_trip`'s operand-serialization
   loop, or the callback's reply in `HostCallback::run`, can still be a
   lazy strided *view* even once evaluated (e.g. `transpose` — pervasive in
   `native_kv_attn_callback`'s Q/K/V handling and its `Nx.transpose`
   output) — MLX's `eval()` materializes the computation but does not
   itself force row-major layout. `create_tensor_resource`'s NIF resource
   is read back byte-for-byte on the Elixir side (`EMLX.Backend.to_nx/2`)
   assuming row-major contiguity, and `HostCallback::run`'s reply-copy is a
   raw `memcpy`; both silently shuffle data instead of crashing. **Fix**:
   force contiguity before either boundary.
2. **The naive fix (`mlx::core::eval(mlx::core::contiguous(x))`)
   self-deadlocks.** `host_round_trip`/`HostCallback::run` execute *from
   inside* `eval_cpu`/`eval_gpu`, themselves invoked by a thread already
   inside `mlx::core::eval()`'s own dependency walk for the outer compiled
   graph. Calling `mlx::core::eval()` again on that same thread reenters
   MLX's scheduler and hangs on a plain, non-recursive mutex — confirmed
   empirically (0% CPU, no progress, indefinitely) via a real
   `validate_qwen3.exs` run, not inferred from docs. **Fix**: a hand-rolled
   `make_row_major_contiguous` helper (`emlx_compiler.cpp`, `host_callback`
   namespace) that does a pure host-side strided byte copy with no MLX
   graph/`eval()` involvement at all — safe to call reentrantly. Applied at
   both the operand-send side (`host_round_trip`) and the reply-receive
   side (`HostCallback::run`).

Both fixes verified: `bench/host_callback_opcode.exs` and
`bench/host_callback_multi_caller.exs` still pass; full `mix test` (2671
tests) green throughout.

**Procedure #5b (fix `mix test` regressions from the Stage 31 removal +
`expr.ex` changes) — done, full suite green (2671 passed, 0 failed).**
Fixed, in order: a `MatchError`/segfault chain from the C++ contiguity
fixes above; `FunctionClauseError`/`ArgumentError` in `dequantize`/
`quantized_matmul` from pass-through quantized operands losing
`quantization_config` on the wire (fixed via `operand_param_positions`
substitution); dispatch-cache fragmentation from `quant_signature`
including irrelevant bound quantized tensors (fixed via
`quantizable_param_positions/1`); a `Shape mismatch` regression in the
`runtime_call_inside_while` test from coupling output-side pass-through
reconstruction to the now-narrower `quant_signature` (fixed by checking the
original bound tensor's own `quantization_config` directly, independent of
`quant_signature`'s scoping).

**Procedure #8 (validate against `validate_qwen3.exs`) — partially done,
one bug fixed, one NEW bug found and NOT yet fixed.**

The deadlock from Procedure #5's fix #2 above was first discovered here
(real GPU run hung indefinitely on `[bb+rewrite] Warmup`) and is now fixed.
With both C++ contiguity fixes in place, `[bb base]` (no rewrite) runs
correctly throughout, and `[bb+rewrite]`'s **first** generation call
(warmup 1) now completes without crashing or hanging — a real improvement
over the pre-fix state (indefinite hang) and the state before *that*
(immediate `MatchError`/segfault). However, numeric correctness is still
broken: warmup 1 already produces garbage tokens (wrong output, not a
crash), and a **second** generation call (observed at the boundary between
one benchmark run and the next) reliably crashes with `` (ArgumentError)
length at axis 1 must be less than axis size of 1084, got: <garbage> `` in
`EMLXAxon.native_kv_decode/7`'s `Nx.slice_non_scalar/6`, sourced from a
corrupted `offset` scalar.

**Root-caused as far as: the corruption is specific to prefill (t_new>1)
calls on the second-and-later generation requests against the same
dispatch-cached compiled program — not decode-step offset reads, not
callback dispatch ordering, not the wire round-trip mechanism in general.**
Debug instrumentation (temporarily added and removed) established, in
order:
- `dispatch_host_callback`'s 28 `host_callback` invocations per
  eval_program call fire in strict, non-interleaved `slot=0,1,2,...,27`
  order every time (ruling out an MLX-internal-scheduler-parallelism theory
  — the decoder blocks' genuine sequential data dependency holds as
  expected).
- Two isolated repros (a single `runtime_call` replayed across many
  separate `Nx.Defn.jit` calls with a fresh scalar operand each time; a
  chain of 6 *different* `callback_slot`s in one compiled program, with
  `Process`-dictionary caching logic mirroring `get_step_offset`'s "seen
  layer_key" scheme) both round-trip scalar operands correctly across many
  calls, on both `:cpu` and `:gpu` — ruling out a generic scalar-wire-
  reconstruction bug.
- `get_step_offset`'s decode-path offset tracking is verified correct for
  an entire 60-token generation run (offset increments 1024→1025→...→1082
  exactly as expected, 28 calls per value). The corruption instead hits the
  **other** branch: a *prefill* call's `Nx.to_number(offset_tensor)` (the
  `t_new > 1` branch in `native_kv_attn_callback`, which reads offset
  directly, bypassing `get_step_offset`'s cache) returns garbage on the
  **second** generation request's prefill, where it should always read `0`
  (verified correct — exactly `0` — on the very first prefill call of the
  whole process).

**Follow-up session narrowed this substantially further: confirmed to be a
sporadic race condition (not a deterministic logic bug), specific to the
host_callback/`native_kv_attn_callback` path (not a generic Nx.Serving
issue), with the exact corrupted quantity pinpointed.**

- The C++ contiguity fix (Procedure #5's fix #1 above) was independently
  verified to be a real bug fix but **not the cause of this remaining
  corruption**: instrumented `make_row_major_contiguous` with a debug
  counter and ran the full `validate_qwen3.exs` benchmark — **zero** of the
  ~30k operand/reply arrays crossing the host_callback boundary were ever
  non-contiguous. The fix is harmless dead weight for this specific model's
  op shapes (kept regardless, since it's still correct and cheap, and the
  underlying MLX behavior it guards against is real — just not what's
  causing this).
- Instrumented both `register_prefill_layer`'s and `get_step_offset`'s
  `Nx.to_number(offset_tensor)` reads directly. Confirmed exactly which
  value is corrupted and did the arithmetic: the crash's reported garbage
  (e.g. `1156191537`) is precisely `corrupted_offset + t_new` (e.g.
  `1156190513 + 1024`), i.e. `native_kv_attn_callback`'s **prefill-path**
  offset read (`t_new > 1` branch, always-fresh — bypasses
  `get_step_offset`'s cache entirely) is the corrupted value; the rest of
  that call's arithmetic is completely consistent and correct given the
  bad input.
- **The corruption is specific to a *second-or-later* top-level
  `Nx.Serving.run` call's prefill** — never the first. Repeated runs are
  **non-deterministic**: sometimes the crash fires on the very next call
  after a clean first generation, sometimes correct prefill/decode
  behavior continues for 60+ tokens and multiple full generations before a
  later call's prefill corrupts. This rules out a deterministic
  reconstruction/ordering bug (already independently confirmed earlier:
  callback dispatch order is always strictly sequential, and isolated
  scalar-round-trip repros never reproduce it) and instead points squarely
  at a **timing-dependent hazard** — most likely a GPU buffer/resource
  still in flight from one generation call's tail end being reused before
  it's truly safe to, specific to whatever `native_kv_attn_callback`'s
  Process-dictionary-backed KV-cache and/or the host_callback's own
  independent command queue introduce that the old Stage 31 split-based
  design didn't (Stage 31 forced an explicit `eval_program`/await boundary
  at *every* split point; Stage 32a's inline design has only one such
  boundary per whole `Nx.Serving.run` call).
- **Decisive corroborating experiment**: inserted `:erlang.
  garbage_collect(); Process.sleep(500)` between each call in
  `validate_qwen3.exs`'s warmup loop only (temporary, not committed — see
  below). With the pause, 3 consecutive full generation calls (2 warmups +
  the first benchmark run, which has no pause guarding it) all completed
  correctly; the very next *unpaused* call (benchmark run 2) crashed with
  the same corrupted-offset signature. A softer, non-deterministic but
  clearly timing-sensitive bug reproducing on the unpaused boundary and
  disappearing (at least for the paused calls) with an inserted delay is
  strong evidence for a race, not a logic error. **`[bb base]` (no rewrite,
  no host_callback at all) runs 7 consecutive `Nx.Serving.run` calls with
  zero pause and zero corruption** — confirming this is not a generic
  "back-to-back Serving calls" hazard, but specific to code this stage's
  host_callback mechanism enables.
- **Not yet resolved.** The exact synchronization gap (what, specifically,
  `EMLX.dispatch_host_callback/6` or `native_kv_attn_callback`'s
  Process-dictionary KV-cache handling fails to wait for before a
  generation call is considered "done" and the next one's tensors start
  getting allocated/computed) has not been identified — only that pausing
  long enough between calls avoids it. Candidate next steps: check whether
  `dispatch_host_callback` force-evaluating only the *reply* tensor on
  `host_callback_worker` (not also any other still-queued work on that same
  command queue, e.g. `native_kv_prefill`'s own KV-cache-storing side
  effects) leaves a gap; check whether `Nx.Serving.run`'s return to Elixir
  genuinely implies the underlying MLX command queue/stream has been
  drained, or only that the specific output tensor's own dependency chain
  has. All temporary debug instrumentation (offset value prints in
  `lib/emlx.ex` and `emlx_axon/lib/emlx_axon.ex`, the C++ contiguity
  counter, the GC+sleep probe in `bench/validate_qwen3.exs`) has been
  removed; none of it is in the current diff. This is new information, not
  present in this doc's original plan, and needs its own investigation
  before Procedure #8 can be called done.

**Procedures #6, #7 — not started.** Procedure #6 (confirm
`Process.get/put` semantics empirically through the real `defn`/
`EMLX.__compile__` path) is implicitly exercised by `native_kv_attn_callback`
itself running correctly *within* a single generation request (the
decode-offset-cache trace above), but has no dedicated regression test.
Procedure #7 (adapt Stage 31's `:stage31` scenarios to assert structural
fusion, not just numeric equivalence) has not been started — Stage 31's
existing tests pass (proving numeric equivalence for their scenarios) but
none of them assert "one compiled program, no split" structurally.
