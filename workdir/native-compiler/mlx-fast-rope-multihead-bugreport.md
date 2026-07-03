# Bug report — `mlx::core::fast::rope` miscomputes non-head-0 rotations for multi-head, non-transposed (`{B, T, H, D}`) tensors

**Component:** MLX 0.31.2, `mlx::core::fast::rope` (all overloads: scalar offset,
array offset, and `freqs`), specifically `mlx/backend/metal/rope.cpp`'s
row-contiguous dispatch branch (and its CPU fallback in `mlx/backend/common` —
reproduced on both `:cpu` and `:gpu`/Metal).
**Affected EMLX surface:** `EMLX.Fast.rope/6`, `EMLX.Fast.rope_with_positions/6`
(T=1 decode fast path), `EMLX.Fast.rope_with_freqs/6` (T=1 decode fast path) —
i.e. every `EMLX.Fast` RoPE entry point that calls `mlx::core::fast::rope`
directly, for **any multi-head input (H>1)** in EMLX/Bumblebee's
"heads-not-yet-transposed" `{B, T, H, D}` layout.
**Severity:** high — silently wrong attention rotations for every head beyond
head 0, on an already-shipped, production decode path. Not a hang or a crash;
numbers are just wrong.
**Found via:** Stage 15 Part B (native-compiler prefill-RoPE lowering)
equivalence tests. The new prefill (T>1) opcodes don't call `fast::rope` at
all (hand-written cos/sin/rotate composition, matching the existing
`fast_rope_positions` eager NIF), so their equivalence tests against
`EMLX.Fast.rope_with_freqs`'s T>1 host loop (which *does* call `fast::rope`
per token) exposed the discrepancy.

## Symptom

For a Bumblebee-layout tensor `a` of shape `{B, T, H, D}` with `H > 1`, calling
any `EMLX.Fast.rope*` variant returns **correct** results for `a[.., .., 0, ..]`
(head 0) but **incorrect** results for every other head. The error grows with
the position/offset. This reproduces:
- On a freshly-allocated, non-sliced, contiguous tensor (not a slicing/view
  artifact).
- On both `:cpu` and `:gpu` (Metal) devices.
- For `EMLX.fast_rope_ids` (scalar offset), `EMLX.fast_rope` (array offset),
  and `EMLX.fast_rope_with_freqs` (freqs tensor) alike — i.e. every
  `fast::rope` overload.

## Minimal repro

```elixir
a = Nx.iota({1, 1, 2, 64}, type: :f32) |> Nx.divide(100)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

joint = EMLX.fast_rope(EMLX.Backend.from_nx(a), 64, false, 10_000.0, 1.0, 6)
        |> EMLX.Backend.to_nx()

head1_alone = a[[.., .., 1..1, ..]]
alone = EMLX.fast_rope(EMLX.Backend.from_nx(head1_alone), 64, false, 10_000.0, 1.0, 6)
        |> EMLX.Backend.to_nx()

# joint's head 1 slice != alone, even though both represent "head 1 at the
# same position/offset". `alone` matches the textbook RoPE formula; `joint`
# does not.
Nx.to_flat_list(joint[[.., .., 1..1, ..]]) |> Enum.take(4)
# => [-0.148..., 1.166..., 0.237..., -0.845...]
Nx.to_flat_list(alone) |> Enum.take(4)
# => [0.883..., 0.811..., -0.416..., -1.117...]
```

Reproduces identically on `:cpu`; reproduces for `fast_rope_ids`/`fast_rope`
(offset-based) as well as `fast_rope_with_freqs`.

## Root cause

`mlx::core::fast::rope` is written for a canonical `(B, *, T, D)` layout where
the rotated sequence axis is the array's second-to-last dimension. EMLX (via
Bumblebee) calls it on `{B, T, H, D}` tensors instead — heads not yet
transposed to the front, T (not H) is the intended rotated axis. MLX's rope
implementation has a `head_seq_transpose` stride-detection special case
(`mlx/backend/metal/rope.cpp`) apparently meant to recognize exactly this
transposed convention, **but it only triggers for a specific non-row-contiguous
stride signature** (`strides[0] == T*N*D && strides[1] == D && strides[2] ==
N*D`, using MLX's own axis naming where "T" = `shape(-2)`, i.e. our H). A plain,
freshly-allocated `{B, T, H, D}` tensor is row-contiguous, so
`strides[1] == H*D`, not `D` — the `head_seq_transpose` branch's guard is
never satisfied for the common case.

Because `dims_ == D` here (no `dims_ < D` early-out) and the array *is*
row-contiguous, execution instead falls into the `in.flags().row_contiguous`
branch, which sets `strides[0] = mat_size = shape(-2)*D` and iterates the
kernel's own "T" (= MLX's rotated axis) over `shape(-2)` — **which is our H
axis, not our T axis**. The kernel ends up applying rotation angle
`position + head_index` to each head instead of `position` — head 0 gets the
right angle (offset `+0`), head 1 gets `position+1`'s angle, etc. This is
consistent with the observed symptom (head 0 always right; head *n* wrong by
an amount that grows with `n` and with the position offset).

## Why existing EMLX tests don't catch this

The Stage 10 (`10-fast-kernels.md`) equivalence tests for `rope`,
`rope_with_positions` (decode/T=1), and `rope_with_freqs` (decode/T=1) all
compare the **compiled-graph opcode** against the **eager `EMLX.Fast` NIF**.
Both call the identical `mlx::core::fast::rope` primitive under the hood, so
they trivially agree with each other while both are wrong relative to the
textbook RoPE formula — the bug is invisible to a "compiled vs eager" reference
that shares the same buggy primitive on both sides. It only surfaces when
compared against an independent, hand-written primitive formula (as Stage 15's
new prefill lowering — which does *not* call `fast::rope` — incidentally is).

`EMLX.Fast.rope_with_positions_callback`'s existing T>1/high-base fallback
(`EMLX.fast_rope_positions`, a hand-written cos/sin/rotate NIF in
`emlx_fast.cpp` that never calls `mlx::core::fast::rope`) is **not** affected —
this is why Stage 15's `fast_rope_positions` opcode's equivalence tests (which
mirror that same hand-written formula) passed cleanly even for H>1, while the
`fast_rope_with_freqs_positions` opcode's tests (compared against
`rope_with_freqs_callback`, whose T>1 loop calls `fast::rope` per token) did
not.

## Practical impact

Any real transformer decode step calling `EMLX.Fast.rope`/`rope_with_positions`
/`rope_with_freqs` on a genuine multi-head `{B, 1, H, D}` Q/K tensor (H>1 is
universal) is silently rotating every head but the first by the wrong angle.
`validate_qwen3.exs` (Stage 11) does not appear to hit this in practice because
Qwen3's RoPE base (1e6) routes decode through the hand-written
`fast_rope_positions` path rather than `fast::rope` directly (per
`emlx_axon.ex`'s dispatch); models/configs that do route decode through
`fast::rope` directly would be affected. Not independently verified against a
production model in this investigation — flagging for follow-up.

## Suggested fix (upstream MLX or EMLX-side workaround)

Either:
- Fix `mlx::core::fast::rope`'s `head_seq_transpose` detection to also trigger
  for the row-contiguous `{B, T, H, D}` case (broaden the stride-signature
  check), or
- Transpose Q/K to `{B, H, T, D}` before calling `fast::rope` and transpose
  back after (extra copy, but correct), or
- Route all multi-head `EMLX.Fast.rope*` calls through the hand-written
  cos/sin/rotate composition (`fast_rope_positions`-style) instead of
  `mlx::core::fast::rope`, eliminating the dependency on this primitive
  entirely.

## Status — open, unfixed

Out of scope for Stage 15 (block-descent completeness + prefill RoPE), which
only adds new T>1 lowering paths that do not call `mlx::core::fast::rope` and
are therefore unaffected. Filed here rather than silently worked around, per
the stage's advisor consultation. Needs a dedicated follow-up stage/issue to
fix `EMLX.Fast`'s decode-path RoPE kernels.
