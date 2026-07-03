# Stage 33 — `:pad` with interior padding (needed for strided `window_sum`/`window_reduce` backward)

Status: done. Named by
[`28-grad-equivalence-suite`](28-grad-equivalence-suite.md)'s Results.

## Why this stage exists

Stage 28's widened grad-equivalence suite added a `window_sum` scenario with
non-unit strides (`strides: [2, 1]`), going beyond Stage 23's original
`window_sum` grad scenario (which used default/unit strides). It found a
genuine, reproducible gap: `Nx.Defn.Grad`'s backward for `window_sum`
(`grad(:window_sum, [x, window_dimensions, opts], _, g, _batch_count)` in
`deps/nx/nx/lib/nx/defn/grad.ex`) un-strides the cotangent via `Nx.pad` with
an **interior** padding value whenever `strides != [1, 1, …]` (to reinsert
the zero gaps a stride skips over). `:pad` with interior padding (or negative
lo/hi) is a **pre-existing, deliberate, documented not-yet-lowered gap** in
the native compiler:

```181:184:workdir/native-compiler/EXPR_NODES.md
- [x] as_type
- [x] bitcast
- [x] pad (simple: non-negative lo/hi, interior=0; interior/negative raises — not yet lowered)
```

```595:604:emlx/lib/emlx/native/expr.ex
  # pad: raises for interior > 0 or negative lo/hi (not yet lowered).
  # iattrs = [n_dims, lo0, hi0, int0, lo1, hi1, int1, …].
  defp expand_node(
         %T{data: %Nx.Defn.Expr{id: id, op: :pad, args: [tensor, pad_value, config]}},
         ...
       ) do
    if Enum.any?(config, fn {lo, hi, interior} -> lo < 0 or hi < 0 or interior > 0 end) do
      raise ArgumentError,
            "does not yet lower op :pad with interior padding or negative lo/hi values"
    end
```

Nothing in Stages 06/13 (window ops, custom-fun reductions) exercised this
because their equivalence tests used default strides for `window_sum`/
`window_max`. Stage 28's non-default-stride scenario is the first to reach
it. Interior padding is also needed generally by `Nx.Defn.grad`'s
`window_reduce`-family backward (not just `window_sum`) whenever a caller
supplies non-unit strides — this is a real, if narrow, coverage gap, not a
correctness bug (the forward ops and unit-stride backward are unaffected).

## Procedure (sketch — refine at stage start)

1. **Confirm scope.** Grep `deps/nx/nx/lib/nx/defn/grad.ex` for every
   backward path that can synthesize an interior-padded or negative-lo/hi
   `Nx.pad` (start with `grad(:window_sum, …)`; check `window_max`/`window_min`/
   `window_product`'s `:window_scatter_*` backward and any `conv`/pooling
   backward with strides too, per Stage 20's window-ops audit) — this
   determines whether "interior pad" alone closes the gap or whether
   negative lo/hi also needs to be handled for some path.
2. **Implement `:pad` with interior padding in `emlx/lib/emlx/native/expr.ex`
   + `emlx/c_src/emlx_compiler.cpp`.** The compiler's own `window_reduce`
   custom-fun lowering (Stage 13) already worked around `mlx::core::pad`'s
   lack of interior-padding support with a reshape/broadcast/slice trick
   (see the comment at `emlx_compiler.cpp:206-213`, "Alternatively: …
   reshape+broadcast trick … We'll do this per axis sequentially") — reuse
   or generalize that trick for the general `:pad` opcode rather than
   inventing a second implementation.
3. **Equivalence tests.** Extend `EMLX.GradEquivalenceTest`'s
   `"windowed ops grad with non-default strides/padding"` describe block:
   flip the `window_sum`-with-strides scenario from "asserts the known raise"
   back to "asserts grad equivalence vs the Evaluator reference" once `:pad`
   supports interior padding, across the shapes already in that test
   (`{4, 4}`, `{3, 5}`) plus at least one 3D case.
4. **Flip `EXPR_NODES.md`'s `:pad` line** once interior padding (and,
   if step 1 finds it's needed, negative lo/hi) is fully lowered and tested.

## Acceptance

- `Nx.pad` with interior padding (and negative lo/hi, if step 1's audit
  finds a real caller) lowers correctly in the native compiler, validated
  against eager `EMLX.Backend` directly (Layer B reference, per this project's
  per-layer-reference testing philosophy) and via the widened `window_sum`
  strided-grad scenario from Stage 28.
- `EMLX.GradEquivalenceTest`'s `window_sum`-with-strides test no longer
  needs to assert a raise — it asserts grad equivalence like every other
  scenario in that suite.
- `EXPR_NODES.md`'s `:pad` line flipped to fully-closed (no more
  interior/negative carve-out), or narrowed precisely if only interior
  padding (not negative lo/hi) turns out to be needed.

## Results

**Scope correction (advisor sign-off before starting):** the audit in Procedure
step 1 needed to be broader than "window backward paths." `grad.ex` has two
*more common* `:pad`-generating backward paths, neither gated on window ops
at all:

- `grad(:pad, …)` (`deps/nx/nx/lib/nx/defn/grad.ex:535`) un-pads the cotangent
  via **negative** lo/hi whenever the forward `Nx.pad` had positive lo/hi —
  unconditionally, no strides required.
- `grad(:slice, …)` (`deps/nx/nx/lib/nx/defn/grad.ex:549`) re-inserts
  **interior** padding into the cotangent whenever the forward `Nx.slice`
  used non-unit strides — very common, unrelated to window ops.
- `grad(:window_sum, …)` (the path that originally surfaced the gap, via
  Stage 28) combines both: `conv_lhs_padding`-derived lo/hi (which can go
  negative for atypical `padding:` configs) *and* interior (from `strides`)
  on the same call.

All three are closed by the same general `:pad` decomposition (below) — no
per-path special-casing was needed.

**Implementation.** Negative lo/hi is *not* a variant of interior padding
(MLX's pad primitive can only grow a tensor, never crop it) — treating them
as one mechanism was the advisor's second correction. The general `:pad`
opcode now decomposes, entirely in Elixir (`EMLX.Native.Expr.expand_pad_general/5`,
`emit_interior_padding/5`, `emit_negative_crop/4`), into three sequential
steps mirroring `EMLX.Backend.pad/4`'s own eager algorithm exactly:

1. **Interior padding** — reuses `EMLX.Backend`'s reshape/pad/slice trick
   (append a size-1 trailing spacer dim, then for each axis in turn pad the
   *next* axis by `next_axis_size * interior` and reshape/slice to fold that
   into the current axis; the spacer role rotates forward one axis per
   step). This generalizes Stage 13's `window_reduce`-specific pad-with-acc
   trick to arbitrary axes/interior amounts/runtime pad-value operands (Stage
   13's version assumed a compile-time-scalar acc; this one takes any scalar
   ref).
2. **Negative-lo/hi crop** — a plain static `:slice`, not run through the
   interior-pad machinery (advisor's correction #2).
3. **Plain non-negative pad** — for whatever `max(lo,0)`/`max(hi,0)` remains,
   reusing the existing `emit_pad_with/5` helper unchanged.

The wire `:pad` opcode itself is untouched — it still only ever carries
non-negative lo/hi with interior=0 on the wire; **zero C++ changes**, per the
stage doc's original direction to reuse rather than invent a second
implementation.

**Validation.**
- Layer B reference (`EMLX.Native.ExprTest`, `describe "Stage 03 — pad"`, new
  `:stage33`-tagged tests): interior-only (1D/2D), negative-only, mixed
  positive/negative/interior, and a 3D case, each checked against
  `Nx.Defn.Evaluator` on `EMLX.Backend`-tagged inputs.
- Grad equivalence (`EMLX.GradEquivalenceTest`): the `window_sum`-with-strides
  scenario flipped from asserting the known raise to asserting equivalence
  (2D, plus a new 3D case); two new scenarios added per the broadened audit —
  direct `Nx.pad`-forward grad (negative-lo/hi backward) and
  `Nx.slice`-with-strides-forward grad (interior backward).
- Full suite: 2679/2679 passed (827 doctests, 1852 tests), 0 regressions —
  one pre-existing test (`EMLX.Native.ExprTest` "unknown op raises...")
  used interior `:pad` as its generic-catch-all-error sentinel; since that's
  no longer a raise, the sentinel was swapped to `triangular_solve`'s
  permanent non-default-variant hard-raise (Stage 17/19), which is unrelated
  to this stage's scope and still guaranteed to raise.

**`EXPR_NODES.md`** flipped: `:pad` is now fully closed (no interior/negative
carve-out remains) — negative lo/hi turned out to be needed (found by the
broadened audit above), so both capabilities are closed together rather than
narrowed to interior-only.
