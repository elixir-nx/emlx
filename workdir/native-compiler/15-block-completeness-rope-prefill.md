# Stage 15 — Block-descent completeness + prefill RoPE

Status: done. Independent of the Stage 12 spike.

## Why this stage exists

Closes the two remaining `[~]`/`[ ]` gaps that are not custom-fun reductions:

- `block` (`EXPR_NODES.md` line 37, `[~]`) is the catch-all decomposition lever;
  its completeness is bounded by what its `default_expr` descent reaches.
- `runtime_call` (line 40, `[~]`) fuses every decode/T=1 `EMLX.Fast.*` callback
  but raises on per-token **prefill** RoPE.

## Part A — block-descent completeness

`expand_block_via_default` already descends RFFT/IRFFT/AllClose/Phase/
Determinant/TopK and unrecognized blocks, but line 156's helpers are not
equivalence-tested/flipped.

1. Add equivalence tests (vs eager `EMLX.Backend` / Evaluator) for AllClose,
   Phase, TopK, and the Determinant descent paths.
2. Flip `EXPR_NODES.md` line 156. Document the remaining structural boundary:
   a `while` (or, until Stage 13, a custom-fun reduce) reached *inside* a
   block's `default_expr` still raises, because `while` is handled at
   `build_eval_fn` level, not in `expand_node`.

## Part B — prefill RoPE (`runtime_call` completion)

`fast_kernel_dispatch/2` raises for `rope_with_positions_callback` /
`rope_with_freqs_callback` when T>1 (the prefill path is a host-side Nx
composition over eager NIFs, not one `mlx::core::fast::rope` call —
`expr.ex:1808`).

1. Lower prefill RoPE as an in-graph primitive subgraph (gather freqs by
   `position_ids`, build cos/sin, rotate), no new C++ kernel.
2. Equivalence vs eager `EMLX.Fast` prefill on a `:metal`/GPU worker.
3. Close line 40's remaining gap. Leave non-`EMLX.Fast` `runtime_call`s as a
   deliberate hard raise (genuine host side effects).

## Acceptance

- Line 156 flipped with tests; block structural boundary documented.
- Prefill RoPE lowers natively and matches eager; line 40 gap closed.

## Results

**Part A.** `expand_block_via_default` already descended AllClose/Phase/
Determinant correctly; TopK crashed because its `default_expr` is a raw
Elixir tuple of `%T{}` nodes (not a single tensor with `.data.id`) — fixed by
storing a list of `flat_refs` for tuple-output `default_expr`s, mirroring the
existing multi-output linalg convention (`elem` already supports list-valued
`node_to_ref` entries). Added equivalence tests (vs `Nx.Defn.Evaluator`) for
`Nx.all_close` (close/not-close), `Nx.phase` (complex), and `Nx.top_k`
(values+indices, batched, tuple-output path). `EXPR_NODES.md` line 156
flipped to `[x]`; `block`'s remaining structural boundary (`while` reached
*inside* a `default_expr` descent — not reachable via `build_eval_fn`'s
top-level `:while` split) documented in place, `block` stays `[~]` for that
narrow case.

**Part B.** Added two C++ `op_registry` opcodes (`fast_rope_positions`,
`fast_rope_with_freqs_positions`), both a hand-written cos/sin/rotate
composition over plain `mlx::core` primitives (factored into a shared
`rope_rotate_from_angles` helper) — no new C++ kernel, matching the spec.
Wired `fast_kernel_dispatch/2` to route `rope_with_positions_callback` /
`rope_with_freqs_callback` (T>1) to these opcodes instead of raising.
Equivalence-tested on `:metal`/GPU against eager `EMLX.Fast`, including
left-padded/non-sequential positions and a `dims < D` pass-through-tail case.
`EXPR_NODES.md` line 40 (`runtime_call`) flipped to `[x]`.

**Unplanned finding, filed separately (not fixed — out of scope):** while
building the `rope_with_freqs` equivalence tests, found that
`mlx::core::fast::rope` (all overloads) miscomputes non-head-0 rotations for
any multi-head (`H>1`) tensor in EMLX/Bumblebee's non-transposed `{B,T,H,D}`
layout — a pre-existing bug in the already-shipped decode/T=1
`EMLX.Fast.rope*` fast paths, invisible to the existing Stage 10 tests because
both the compiled opcode and the eager NIF call the same buggy primitive and
so trivially agree with each other. Confirmed via advisor consultation
(agent `d24c8caa-f6a9-4d8c-92f8-c5a0c6357a7b`) to be real, in-scope-supported
usage, and high-severity but out of Stage 15's charter. Root-caused and filed
as `mlx-fast-rope-multihead-bugreport.md`. Because the new
`fast_rope_with_freqs_positions` opcode never calls `fast::rope` (it uses the
same manual formula as the already-trusted `fast_rope_positions` opcode), its
H>1 equivalence tests were switched to a hand-written pure-Nx primitive
reference instead of the (for H>1) unreliable eager `EMLX.Fast.rope_with_freqs`
reference; an H=1 case still validates directly against eager. `rope_with_positions_callback`'s eager reference (`fast_rope_positions`, a
hand-written NIF that never calls `fast::rope`) is unaffected, so its tests
were left comparing against eager throughout.

**Verification:** `mix test` — 2545 passed (825 doctests, 1720 tests), 1
excluded, 0 failures. `mix test --only stage15` — 14 passed (Part A: 5, Part B
lowering-shape: 2, Part B `:metal` equivalence: 7).
