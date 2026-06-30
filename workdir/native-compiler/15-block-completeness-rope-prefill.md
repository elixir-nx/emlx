# Stage 15 — Block-descent completeness + prefill RoPE

Status: open (planned). Independent of the Stage 12 spike.

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
