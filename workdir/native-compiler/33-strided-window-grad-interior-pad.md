# Stage 33 — `:pad` with interior padding (needed for strided `window_sum`/`window_reduce` backward)

Status: not started. Named by
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
   back to "asserts grad equivalence vs the Evaluator oracle" once `:pad`
   supports interior padding, across the shapes already in that test
   (`{4, 4}`, `{3, 5}`) plus at least one 3D case.
4. **Flip `EXPR_NODES.md`'s `:pad` line** once interior padding (and,
   if step 1 finds it's needed, negative lo/hi) is fully lowered and tested.

## Acceptance

- `Nx.pad` with interior padding (and negative lo/hi, if step 1's audit
  finds a real caller) lowers correctly in the native compiler, validated
  against eager `EMLX.Backend` directly (Layer B oracle, per this project's
  per-layer-oracle testing philosophy) and via the widened `window_sum`
  strided-grad scenario from Stage 28.
- `EMLX.GradEquivalenceTest`'s `window_sum`-with-strides test no longer
  needs to assert a raise — it asserts grad equivalence like every other
  scenario in that suite.
- `EXPR_NODES.md`'s `:pad` line flipped to fully-closed (no more
  interior/negative carve-out), or narrowed precisely if only interior
  padding (not negative lo/hi) turns out to be needed.

## Results

(not started)
