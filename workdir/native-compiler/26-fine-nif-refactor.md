# Stage 26 — refactor NIF plumbing onto `fine` (scoping + spike)

Status: not started. Not an Emily-parity item — a maintainability investment
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
   split).
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
   remaining stages accordingly once known.
3. **Resolve the `ASYNC_NIF`/command-queue interaction.** `emlx_worker.hpp`'s
   queue-per-process dispatch model expects NIFs to hand off work and return
   a job ref; confirm `fine`'s NIF-registration macros don't assume a
   synchronous call-and-return NIF shape that conflicts with this (`fine`'s
   docs/examples are single-call-return oriented — verify against its actual
   source, don't assume compatibility either way, per this plan's existing
   "verify against code, not docs" discipline from Stage 20).
4. **If the spike is clean**, fan out to `emlx_nif.cpp` then
   `emlx_compiler.cpp` (largest, most structurally distinct — its IR-opcode
   dispatch table is not a simple 1:1 NIF-per-Elixir-call mapping, so treat
   it as its own follow-on stage rather than folding it into the same pass
   as the other two files) as separate, independently-sized follow-on
   stages, each gated on: identical `mix test` pass/fail set before and
   after, no public `EMLX`/`EMLX.Fast`/`EMLX.Native.Expr` API change, and no
   measurable perf regression on the existing `bench/` suite.
5. **If the spike surfaces a hard incompatibility** (e.g. `TensorP`'s
   refcount scheme or the async-queue handoff genuinely can't sit under
   `fine`'s macros without fighting them), stop and record the specific
   blocker — a partial/no-go outcome (mirroring Stage 12/14's precedent) is
   an acceptable result of this stage, not a failure to avoid.

## Acceptance (for *this* scoping + spike stage)

- `emlx_fast.cpp` ported to `fine` (or a documented, specific reason it
  can't be, per step 5), with `mix test` green (identical pass/fail set to
  the pre-migration baseline) and no public API change.
- A written verdict on the `TensorP`-refcount and `ASYNC_NIF`-command-queue
  compatibility questions (steps 2–3), with follow-on stages for
  `emlx_nif.cpp` and `emlx_compiler.cpp` named and sized only if the verdict
  is go.
- No perf regression vs the existing `bench/` baseline on the ported file's
  NIFs.
