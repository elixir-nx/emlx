# Stage 26 — refactor NIF plumbing onto `fine` (scoping + spike)

Status: done. Not an Emily-parity item — a maintainability investment
in EMLX's own `c_src/` tree.

## Why this stage exists

`emlx/c_src/` is ~5.4k lines of hand-rolled `erl_nif.h` C++
(`emlx_nif.cpp` 1984, `emlx_compiler.cpp` 1862, `emlx_fast.cpp` 642,
`nx_nif_utils.hpp` 401, plus `emlx_worker.hpp`/`emlx_async.hpp`/
`emlx_nif_shared.hpp`) built on manual `enif_get_*`/`enif_make_*` calls, a
custom macro layer (`nx_nif_utils.hpp`'s `PARAM`/`TUPLE_PARAM`/`ATOM_PARAM`/
`GET`/`CATCH`, `emlx_nif_shared.hpp`'s `TENSOR_PARAM`/`TENSOR`/`NIF`), and a
bespoke atomic-refcounted resource wrapper (`TensorP` in
`emlx_nif_shared.hpp`) — the exact kind of boilerplate
[`fine`](https://github.com/elixir-nx/fine) (the Nx team's C++ NIF-ergonomics
library, Apache-2.0, `{:fine, "~> 0.1"}`) exists to eliminate: automatic
argument/return encoding-decoding inferred from function signatures, RAII
resource management via `fine::ResourcePtr<T>`, and C++-exception→Elixir
propagation. Dashbit's own writeup reports removing "over 1k LOC" refactoring
EXLA's NIFs onto it. This stage exists to find out how much of that applies
to EMLX's tree specifically, and whether EMLX's two non-stock wrinkles —
the manual atomic-refcounted `TensorP`/`create_tensor_resource` scheme and
the `EMLX.CommandQueue` async-dispatch model (`ASYNC_NIF`/
`emlx::async_dispatch`, argv[0]-is-a-queue-ref convention) — compose cleanly
with `fine`'s `ResourcePtr`/RAII model or need a bridging layer.

This is explicitly **not** a rewrite-for-its-own-sake: no behavior, no public
Elixir API, and no perf characteristic may change. The sole goal is reducing
`c_src/` boilerplate and making future stages (26+, or any op-coverage work)
cheaper to extend.

## What was done

Advisor sign-off (before starting) flagged the load-bearing risk correctly:
*"every `emlx_fast.cpp` NIF is `ASYNC_NIF`-wrapped ... the highest-risk
unknown [is] whether fine exposes the typed inner function (pre-wrapper) so
you can pass that into `async_dispatch`."* Verified against `fine`'s actual
source (`c_include/fine.hpp` in `elixir-nx/fine`, not its README/docs), not
assumed:

1. **`ASYNC_NIF` compatibility — resolved, with a bridging layer, not a
   mechanical macro swap.** `fine::nif()`/`FINE_NIF` are usable as raw
   `ERL_NIF_TERM(ErlNifEnv*, int, const ERL_NIF_TERM*)` functions (matching
   `emlx::async_dispatch<SyncOp>`'s template parameter exactly), **but**
   `fine::nif()`'s internal `nif_impl` translates a caught C++ exception
   into a *raised* Elixir exception via `enif_raise_exception` — a return
   value that is only meaningful as the reply of a live NIF call serviced
   directly by the BEAM scheduler. EMLX's `ASYNC_NIF` convention instead
   runs the sync NIF body on a worker thread and ships its `{:ok, _}` /
   `{:error, _}` tagged tuple back over `enif_send` (`emlx_async.hpp`) —
   `enif_raise_exception`'s sentinel return is not a valid message payload
   there, so `fine::nif()`/`FINE_NIF`/`FINE_INIT` cannot be used verbatim.
   **Fix**: reuse `fine`'s `Decoder<Args>...`/`Encoder<Return>` typed
   marshalling (the actual value-add), but drive it through a ~15-line
   custom dispatcher (`emlx_fine::nif`, in `emlx_nif_shared.hpp`) that
   catches exceptions and returns EMLX's own `nx::nif::error(env, msg)`
   tuple instead of raising. This also resolves the stage's "mixed
   old/new NIF registration in one `.so`" open question: since we never
   call `FINE_NIF`/`FINE_INIT` (which populate `fine`'s own global
   registration vector and expect to own `ERL_NIF_INIT`), there is no
   dual-registration conflict — `fine` is used purely as a decode/encode
   template library, and EMLX's existing hand-written `nif_funcs[]` +
   `ERL_NIF_INIT` (in `emlx_nif.cpp`) is completely untouched. The
   generated symbol names/arities (`NAME`, `NAME_async`) are identical to
   before, so **zero changes were needed to `emlx_nif.cpp`'s registration
   table or forward declarations.**

2. **`TensorP`/resource-refcount question — resolved: not subsumed,
   confirmed a real second layer, kept as a bridged custom type.**
   `fine::ResourcePtr<T>` only wraps ERTS's own
   `enif_keep_resource`/`enif_release_resource` refcounting (see
   `fine.hpp`'s `ResourcePtr` copy/move ctors and `Registration::resources`
   `enif_open_resource_type` call). EMLX's `TensorP` adds a **second,
   independent** atomic refcount + `deleted` flag on top, allocated inline
   in the same resource block — used by the explicit `deallocate` NIF
   (`emlx_nif.cpp:75`, registered as `EMLX.NIF.deallocate/1`) to eagerly
   free GPU memory ahead of BEAM GC, a facility `fine::ResourcePtr`'s plain
   ERTS-refcount wrapping does not provide. This is a real semantic layer,
   not accidental duplication — `TensorP` could not be a mechanical
   `fine::ResourcePtr<mlx::core::array>` swap without also giving up the
   early-deallocate facility (out of scope to redesign here). Resolution:
   `TensorP` stays exactly as-is; a new `TensorArg` wrapper (owning a
   `TensorP` + the raw `array*`) bridges it into `fine`'s `Decoder`/`Encoder`
   traits, so NIF bodies get ergonomic `*x`/`x->...` tensor access without
   touching the resource type, `create_tensor_resource`, or the
   `deallocate` NIF.

3. **`ASYNC_NIF`/command-queue interaction (step 3) — no conflict found.**
   `fine`'s macros never assume a synchronous call-and-return NIF shape;
   `fine::nif()`/`Decoder`/`Encoder` are free functions with no dependency
   on how their result reaches the caller. The queue-per-process dispatch
   model (`emlx_worker.hpp`) and the `_async`/`nif_funcs[]` registration
   convention were left completely unmodified.

4. **Spike executed on the full pilot file** (not a subset): every NIF in
   `emlx_fast.cpp` (`fast_rms_norm`, `fast_rope`, `fast_sdpa`,
   `fast_sdpa_masked`, `fast_layer_norm`, `fast_layer_norm_no_bias`,
   `fast_rope_ids`, `fast_rope_with_freqs`, `fast_rope_positions`,
   `fast_sdpa_causal_key_masked`, `fast_swiglu`, `fast_sdpa_causal`,
   `kv_cache_attention`, `kv_cache_attention_masked`,
   `kv_cache_sdpa_update`) was rewritten from the `NIF(...)`/
   `TENSOR_PARAM`/`PARAM`/`DEVICE_PARAM`/`OPTIONAL_TENSOR_PARAM`/`TENSOR`/
   `CATCH()` macro style to a typed `NAME##_impl(ErlNifEnv*, TensorArg...,
   int/double/bool/mlx::core::Device...) -> mlx::core::array` (or
   `std::tuple<array, array, array>` for the 3-output KV-cache fusions)
   function plus a one-line `FINE_ASYNC_NIF(NAME)`. Manual validation
   raises (`fast_rope_positions`'s shape/dims checks) became
   `throw std::invalid_argument(...)`, caught uniformly by the new
   dispatcher. `mix.exs`/`Makefile` wired `{:fine, "~> 0.1", runtime:
   false}` + `FINE_INCLUDE_DIR` (mirroring the exact pattern already used
   by the sibling project `~/coding/emily`'s `mix.exs`, confirmed by
   reading its source directly).

## Results (filled in after execution)

- **Compiles clean**, first try, both `dev` and `test` `MIX_ENV`.
- **`mix test`: identical pass/fail set before and after** —
  `2629/2647 passed (826/826 doctests, 1803/1821 tests), 5 excluded`,
  same 18 pre-existing `EMLX.FastTest` qwen3-helper failures
  (`:nif_not_loaded`, unrelated to this file/stage — a pre-existing
  baseline issue in `emlx_fast/qwen3.cpp`'s own NIF registration, confirmed
  present before touching any code in this stage). Diffed the two 18-line
  failure sets textually — identical.
- **No public API change**: same NIF names/arities registered in
  `emlx_nif.cpp`'s `nif_funcs[]` (that file was not touched); same Elixir
  call sites in `lib/emlx.ex`/`lib/emlx/fast.ex` (also untouched).
- **No perf regression**: micro-benchmarked `EMLX.Fast.rms_norm_callback/2`
  and `EMLX.Fast.swiglu_callback/2` (5000 warm iterations each) against a
  `git stash`-restored pre-migration build of the same file on the same
  machine: `fast_rms_norm` 14.49 µs/call (before) vs 14.74 µs/call (after);
  `fast_swiglu` 8.89 µs/call (before) vs 8.83 µs/call (after) — within
  run-to-run noise, no measurable regression. (The suite's own `[Stage 10]`
  compiled-graph decode-block micro-bench, a different call path through
  `emlx_compiler.cpp`'s opcode registry rather than these NIFs directly,
  also stayed in the same 1.26–1.39× band across runs.)

## Verdict: **go**, fan out to `emlx_nif.cpp` next; `emlx_compiler.cpp` sized separately

- `emlx_nif.cpp` (1984 lines, the bulk of the boilerplate) is a **go**:
  same bridging pattern (`TensorArg`/`FINE_ASYNC_NIF`/`emlx_fine::nif`)
  applies directly — it's mostly single-tensor-in/single-tensor-out or
  small-tuple-out ops like this file, just ~15× more of them. Recommend a
  follow-on stage sized at "convert `emlx_nif.cpp` mechanically, one
  commit-sized chunk at a time (e.g. by op-family), re-running `mix test`
  after each chunk" rather than one giant diff.
- `emlx_compiler.cpp` (1862 lines) is **not** a 1:1 NIF-per-Elixir-call
  file — its IR-opcode dispatch table means the `fine::Decoder`/`Encoder`
  win is smaller (most args are already decoded once into the IR, not
  per-NIF-call), so per Stage 26's own scoping note it should stay a
  separate, independently-sized follow-on stage, not folded into the
  `emlx_nif.cpp` fan-out.
- `deallocate`'s `TensorP` early-free semantics are unaffected either way
  (this stage never touched them) and don't need to be "resolved" further
  before fanning out — the bridging pattern (custom `Decoder`/`Encoder`
  over the *existing* resource type, no `fine::ResourcePtr` swap) is
  already proven and directly reusable.

## Procedure (scoping — expect a spike before committing to full migration)

1. **Spike: port one small, self-contained NIF file first.** `emlx_fast.cpp`
   (642 lines, no async command-queue involvement beyond the existing
   `ASYNC_NIF` wrapper, a bounded set of fused-kernel NIFs) is the natural
   pilot — small enough to fully convert in one pass, large enough to
   exercise tensor-resource passing, optional params, and error propagation.
   Add `{:fine, "~> 0.1", runtime: false}` to `mix.exs`, wire
   `FINE_INCLUDE_DIR` into `make_env` (per `fine`'s `elixir_make`
   integration), and rewrite `emlx_fast.cpp`'s NIFs with `FINE_NIF`/
   `FINE_RESOURCE`/`FINE_INIT`, keeping `emlx_nif.cpp`/`emlx_compiler.cpp` on
   the old macros meanwhile (mixed old/new NIF registration in one `.so` is
   expected to coexist during migration — confirm this explicitly, since
   Emily/EXLA precedent doesn't establish it either way for a two-registration-style
   split). **Done — with one correction: `FINE_NIF`/`FINE_INIT` themselves
   were not used (see "What was done" #1); `fine`'s `Decoder`/`Encoder`
   were used directly via a small custom dispatcher, so there is no
   dual-registration question in practice — `emlx_nif.cpp`'s own
   `nif_funcs[]`/`ERL_NIF_INIT` remains the sole registration path.**
2. **Resolve the `TensorP`/resource-refcount question before going further.**
   EMLX's `TensorP` does manual atomic refcounting with a raw
   `mlx::core::array *` behind the resource (`emlx_nif_shared.hpp:43-101`),
   not a plain RAII `std::shared_ptr`-style resource — check whether
   `fine::ResourcePtr<mlx::core::array>` can subsume this directly (MLX's
   `array` is already refcounted internally via its own `std::shared_ptr`
   array-data, so the outer `TensorP` layer may turn out to be redundant
   under `fine`, not just portable) or whether the existing scheme must stay
   as a wrapped resource type. This determines whether the migration is a
   mechanical macro swap or a real resource-model change — size the
   remaining stages accordingly once known. **Done — see "What was done"
   #2: not redundant, kept as a bridged custom type (`TensorArg`).**
3. **Resolve the `ASYNC_NIF`/command-queue interaction.** `emlx_worker.hpp`'s
   queue-per-process dispatch model expects NIFs to hand off work and return
   a job ref; confirm `fine`'s NIF-registration macros don't assume a
   synchronous call-and-return NIF shape that conflicts with this (`fine`'s
   docs/examples are single-call-return oriented — verify against its actual
   source, don't assume compatibility either way, per this plan's existing
   "verify against code, not docs" discipline from Stage 20). **Done — see
   "What was done" #1 and #3.**
4. **If the spike is clean**, fan out to `emlx_nif.cpp` then
   `emlx_compiler.cpp` (largest, most structurally distinct — its IR-opcode
   dispatch table is not a simple 1:1 NIF-per-Elixir-call mapping, so treat
   it as its own follow-on stage rather than folding it into the same pass
   as the other two files) as separate, independently-sized follow-on
   stages, each gated on: identical `mix test` pass/fail set before and
   after, no public `EMLX`/`EMLX.Fast`/`EMLX.Native.Expr` API change, and no
   measurable perf regression on the existing `bench/` suite. **Spike is
   clean — see Verdict above; `emlx_nif.cpp` fan-out is a go, not yet
   started/named as its own stage number.**
5. **If the spike surfaces a hard incompatibility** (e.g. `TensorP`'s
   refcount scheme or the async-queue handoff genuinely can't sit under
   `fine`'s macros without fighting them), stop and record the specific
   blocker — a partial/no-go outcome (mirroring Stage 12/14's precedent) is
   an acceptable result of this stage, not a failure to avoid. **Not
   triggered — no hard incompatibility found, once `fine`'s own
   registration/exception macros were bypassed in favor of its decode/encode
   primitives directly.**

## Acceptance (for *this* scoping + spike stage)

- `emlx_fast.cpp` ported to `fine` (or a documented, specific reason it
  can't be, per step 5), with `mix test` green (identical pass/fail set to
  the pre-migration baseline) and no public API change. **Met — see Results.**
- A written verdict on the `TensorP`-refcount and `ASYNC_NIF`-command-queue
  compatibility questions (steps 2–3), with follow-on stages for
  `emlx_nif.cpp` and `emlx_compiler.cpp` named and sized only if the verdict
  is go. **Met — see Verdict. Follow-on `emlx_nif.cpp` fan-out not yet
  assigned a stage number (next available: 33); left for the user to
  schedule.**
- No perf regression vs the existing `bench/` baseline on the ported file's
  NIFs. **Met — see Results (micro-bench, git-stash before/after
  comparison).**
