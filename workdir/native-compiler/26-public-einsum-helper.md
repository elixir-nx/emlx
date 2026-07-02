# Stage 26 — public `einsum` helper (variadic operands)

Status: not started. Emily M27 parity (see Stage 20). Split out of
[`22-fast-kernel-quant-parity`](22-fast-kernel-quant-parity.md) by advisor
sign-off before that stage started (see its "Scope correction" note).
Numbered 26 (not 25) because Stage 25 (`25-fine-nif-refactor`) was claimed
concurrently by another session while this split was in flight.

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

_(fill in when tackled)_
