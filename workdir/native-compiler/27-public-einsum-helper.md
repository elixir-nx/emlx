# Stage 27 — public `einsum` helper (variadic operands)

Status: done. Emily M27 parity (see Stage 20). Split out of
[`22-fast-kernel-quant-parity`](22-fast-kernel-quant-parity.md) by advisor
sign-off before that stage started (see its "Scope correction" note).
Originally numbered 26 (not 25) because Stage 25 (`25-fine-nif-refactor`, at
the time) was claimed concurrently by another session while this split was in
flight; renumbered to 27 when Stage 25 was inserted as
`25-quantized-dot-full-fix` and the rest of the burndown shifted down one
(`25-fine-nif-refactor` → `26-fine-nif-refactor`, this stage 26 → 27, and so
on through Stage 30).

## Why this stage exists

Expose EMLX's einsum capability as a public eager helper, matching Emily's
M27. Originally scoped as a small addendum to Stage 22 ("expose the existing
internal `EMLX.einsum` NIF"), but the existing NIF is fixed arity-2 (see git
history for the pre-split Stage 22 doc). Concretely, `emlx_nif.cpp`'s
`einsum` NIF decodes exactly two `TENSOR_PARAM`s and calls
`mlx::core::einsum(spec_string, {*a, *b}, device)`; it is registered as
`{"einsum", 5, einsum_async}` (2 tensors + spec + device + queue). A
3-operand contraction (`"ij,jk,kl->il"`) — required by this stage's own
acceptance criteria — cannot be expressed through that signature. This is a
real NIF-level arity change (variadic tensor-list decode), not just a thin
Elixir wrapper around an existing call — bigger than "expose an existing
NIF," hence its own stage.

`mlx::core::einsum` itself already accepts `std::vector<array>` (see
`mlx/ops.h`), so the C++ side of a variadic NIF is a call-site-only change;
the work is in the NIF argument decoding (Erlang list-of-tensor-resources →
`std::vector<mlx::core::array>`) and the registration/arity plumbing.

## Procedure

1. **Variadic-tensor NIF.** Change (or add a new) `einsum` NIF in
   `emlx_nif.cpp` to accept a spec string plus an Erlang list of tensor
   resources (`LIST_PARAM`-style decode, or a hand-rolled
   `enif_get_list_cell` loop building `std::vector<mlx::core::array>`),
   calling `mlx::core::einsum(spec_string, operand_arrays, device)`. Decide
   whether to replace the existing 2-operand `einsum` NIF in place (updating
   `EMLX.einsum/…`'s one call site in `backend.ex`'s
   `dot_spec_to_einsum_spec/…`) or add a new variadic entry point alongside
   it — prefer replacing in place if the 2-operand call site can trivially
   pass a 2-element list, to avoid two parallel NIFs doing the same thing.
2. **Public eager helper.** Expose a public function (`EMLX.Fast.einsum/2`
   or another suitable existing module — decide based on where it reads most
   naturally; `EMLX.Fast` already hosts other public eager `mlx::core::fast`
   wrappers, but plain `einsum` is `mlx::core::einsum`, not `mlx::core::fast`,
   so consider `EMLX` itself or a new small module instead) accepting a spec
   string and a variadic/list of `Nx.Tensor.t()`. Raise a clear
   `ArgumentError` for any non-`EMLX.Backend` operand ("transfer with
   `Nx.backend_transfer/2` first").
3. **Tests:** 2-operand (`"ij,jk->ik"`), batched (`"bij,bjk->bik"`),
   attention-style (`"bhid,bhjd->bhij"`), 3-operand (`"ij,jk,kl->il"`)
   contractions (each checked against `Nx.dot`/manual contraction or a
   known-good tensor), and the non-`EMLX.Backend`-operand error path.

## Acceptance

- A public eager `einsum` helper ships, accepting 2+ operands.
- Tests cover 2-operand, batched, attention-style, and 3-operand
  contractions, plus the non-`EMLX.Backend` error path.
- The pre-existing internal einsum call site (`backend.ex`'s
  `dot_spec_to_einsum_spec/…`) continues to work unchanged (either untouched,
  or migrated to the new variadic NIF with no behavior change).

## Results

Implemented by directly mirroring `~/coding/emily`'s already-shipped `Emily.Fast.einsum/2` (the M27 parity source of truth), rather than designing from scratch:

- **NIF (`emlx_nif.cpp`)**: `einsum` NIF changed from two fixed `TENSOR_PARAM`s to `LIST_PARAM(0, std::vector<mlx::core::array>, arrays)` — the exact list-of-tensor-resources decode pattern already proven by `stack`/`concatenate` in the same file (and already backed by an existing `nx::nif::get_list` overload for `std::vector<mlx::core::array>` in `nx_nif_utils.hpp`, so no new C++ decode infrastructure was needed). Registration arity dropped from 5 (worker + 2 tensors + spec + device) to 4 (worker + list + spec + device).
- **Elixir NIF-level wrapper (`emlx.ex`)**: `deftensor einsum(tensorA, tensorB, spec_string)` → `deftensor einsum(tensors, spec_string)`. The existing `deftensor`/`prepare_tensors!/1` machinery already transparently supports a *list* of `{device, ref}` tensor tuples as a "tensor" arg (same mechanism backing `stack(tensors, axis)`/`concatenate(tensors, axis)`), so no macro changes were needed.
- **Internal call site (`backend.ex`)**: `dot`'s batched-axes path (`dot_spec_to_einsum_spec`-adjacent code) migrated in place from `EMLX.einsum(ref_a, ref_b, spec)` to `EMLX.einsum([ref_a, ref_b], spec)` — same semantics, new argument shape, single call site, no behavior change.
- **Public eager helper**: `EMLX.Fast.einsum(subscripts, operands)`, `operands` a list of 2+ `Nx.Tensor.t()`. Named `EMLX.Fast.einsum/2` rather than a bare `EMLX.einsum/2` because the `EMLX` module already has the NIF-level `einsum/2` from the point above — same arity, incompatible argument types (raw `{device,ref}` tuples vs `Nx.Tensor.t()`), so colocating both in `EMLX` would create a confusing same-arity same-module overload. `EMLX.Fast` already hosts other public eager helpers, and Emily's own parity source resolved the identical collision the same way (`Emily.Fast.einsum/2` alongside a NIF-level `einsum/2`), so this directly mirrors the parity target rather than inventing a new split. Eager-only / not defn-callable (no `Nx.runtime_call`, unlike every other `EMLX.Fast` member) — raises `ArgumentError` with a "transfer with `Nx.backend_transfer/2` first" message for any non-`EMLX.Backend` operand (an explicit per-operand check, deliberately not `EMLX.Backend.from_nx/1`'s existing silent auto-transfer, per this stage's acceptance criteria). `EMLX.Fast`'s moduledoc updated to carve out `einsum/2` as the one documented exception to "every function is defn-safe" (advisor-flagged: Emily's own moduledoc has the identical carve-out for its `einsum/2`).
- **Tests**: new `emlx/test/emlx/fast/einsum_test.exs` (file-for-file mirror of Emily's `test/emily/fast/einsum_test.exs`) covering 2-operand (`"ij,jk->ik"` vs `Nx.dot`), batched (`"bij,bjk->bik"` vs `Nx.dot` with explicit batch axes), attention-style (`"bhid,bhjd->bhij"` vs `Nx.dot` with explicit batch axes), 3-operand (`"ij,jk,kl->il"` vs both left-to-right and right-to-left hand-chosen `Nx.dot` contraction orders, exploiting associativity), and the non-`EMLX.Backend` operand error path (`Nx.BinaryBackend` input raises the transfer-first `ArgumentError`). Plus a doctest (`doctest EMLX.Fast, only: [einsum: 2]`).
- **Verification**: `mix compile` clean (C++ + Elixir); new test file green (6/6: 5 tests + 1 doctest); full `mix test` green — **2653 passed (827 doctests, 1826 tests), 5 excluded, 0 failed** (large_memory/debug_flags_functional tags excluded as usual) — confirming the `backend.ex` `dot` migration and the `deftensor`/NIF arity change introduced no regressions anywhere else in the suite.
