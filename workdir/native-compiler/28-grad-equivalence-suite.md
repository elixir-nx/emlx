# Stage 28 — grad-equivalence regression suite

Status: done. Emily M9 (testing half) parity, sized by Stage 23's
triage.

> **Plan adjustment (before starting, user directive + advisor sign-off).**
> Two changes to the original "Why this stage exists" / Procedure below:
>
> 1. **No `StreamData`.** Instead of a property test over generated
>    shapes/dtypes, this stage widens `grad_triage_test.exs` into a
>    **table-driven fixed zoo**: explicit `for` loops over scenario × shape ×
>    dtype combinations generating ExUnit test cases, so breadth is still
>    "materially more than Stage 23's 8 scenarios" without random generation.
> 2. **No non-differentiable-op exclusion list.** `argmax`/`argmin`/`floor`/
>    `sign`/comparisons are *included* in the zoo rather than excluded — Nx's
>    `Nx.Defn.grad` already implements forcing/stop-gradient rules that make
>    grad well-defined (typically zero) at these ops, so the real test is
>    **does EMLX's native backward lowering apply the same stop-gradient rule
>    as the Evaluator**, not whether grad exists at all. Per advisor guidance,
>    the finite-difference reference (Procedure item 2) is restricted to the
>    smooth subset only, and test points for non-diff ops are chosen away
>    from discontinuities (no exact argmax ties, no `x = 0` for `sign`) since
>    FD is meaningless exactly at those boundaries — the Evaluator-vs-native
>    equivalence check (not FD) is what covers the non-diff ops.

## Why this stage exists

Stage 23's triage (`emlx/test/emlx/grad_triage_test.exs`) ran an 8-scenario
zoo of `Nx.Defn.grad`-wrapped functions (elementwise, reduction, dot, both
`cond` branches, a counted `while`, `window_sum`, `window_max`) through
`compiler: EMLX` and found all 8 pass unmodified against a
`Nx.BinaryBackend`/`Nx.Defn.Evaluator` reference. That triage zoo is deliberately
narrow (one shape, one dtype, one representative case per op class) — this
stage widens it into a permanent, broader regression suite, mirroring Emily's
own M9 harness design (`~/coding/emily/PLAN.md`'s "Testing — Layers 4 (Grad)
and 5 (Training)" section):

1. A `StreamData`-based property test: for a larger zoo of `defn`-expressible
   functions (excluding non-differentiable ops per Emily's own exclusion
   list — `argmax`, `argmin`, `floor`, `sign`, comparisons), assert
   `Nx.Defn.grad(f)` under `compiler: EMLX` matches the same grad under
   `Nx.BinaryBackend`/`Nx.Defn.Evaluator`, across generated shapes/dtypes.
2. A numerical finite-difference reference for the differentiable subset:
   `(f(x+ε) - f(x-ε)) / 2ε ≈ grad(f)(x)`, with per-op documented tolerance
   (f32 central differences bottom out ~1e-3 relative — Emily's own finding,
   re-verify against EMLX's actual float precision, don't just copy the
   number).
3. Explicitly widen the control-flow subset beyond Stage 23's single counted
   `while` and two-branch `cond`: nested `cond`/`while`, multi-output `while`
   carries, and a `while` whose body itself contains a `cond`.

This is **not** a new compiler-code stage — Stage 23 already established the
mechanism works. This stage is breadth of coverage, catching the *next*
untested op-class combination before a real user hits it, not a bug fix.

## Procedure

1. Extend `grad_triage_test.exs` (or promote it to a differently-named
   permanent suite file — naming call is part of this stage, not decided
   here) with the StreamData property test and finite-difference reference
   described above.
2. Run the widened zoo; record any genuine failures (expect none, per Stage
   23's finding, but this stage exists specifically to falsify that
   expectation at greater breadth).
3. If a genuine gap surfaces, it gets its own follow-on stage (do not fix
   compiler bugs inline in a "testing" stage — name and size a new stage,
   same discipline as Stage 12's spike → Stages 13/14 split).

## Acceptance

A permanent grad-equivalence regression suite checked in, covering
materially more op-class combinations and shapes/dtypes than Stage 23's
8-scenario triage, with a Results section here confirming pass/fail and
naming any newly-discovered follow-on stages.

## Results

**Suite:** `emlx/test/emlx/grad_equivalence_test.exs` (new, permanent file —
`grad_triage_test.exs` kept as-is, unmodified, as Stage 23's historical
triage record). 14 tests, each internally table-driven over shape × dtype
(and, for control-flow scenarios, branch/carry combinations), covering 10
scenario groups materially beyond Stage 23's 8 single-shape/single-dtype
scenarios:

1. Smooth elementwise chain (`sin`/`cos`/`log1p`/`tanh`/`sigmoid`/`sqrt`/`abs`
   composed) — 4 shapes × 2 dtypes.
2. Reduction chain (`sum`/`mean`/`reduce_max`/`reduce_min` composed) — 3
   shapes × 2 dtypes.
3. Dot chain (`dot` → `tanh` → `sum`) — 3 shape pairs × 2 dtypes.
4. Nested `cond` (`cond` inside `cond`, all 4 leaf branches) — 4 cases.
5. `while` whose body contains a `cond` — 2 dtypes (see bug finding below).
6. Multi-output `while` carries (3 carried tensors) — 2 dtypes.
7. Nested `while` (`while` inside `while`) — 2 dtypes.
8. Windowed ops with non-default strides/padding (`window_sum` w/ strides,
   `window_max` w/ padding) — 2 shapes × 2 dtypes each (see gap finding
   below).
9. Non-differentiable ops used as an operand — stop-gradient boundary parity
   for `sign`, `floor`, a comparison feeding `select`, and `argmax` feeding
   `take_along_axis` (max-pooling-style pattern) — 4 shapes × 2 dtypes each.
10. Finite-difference reference for the smooth-unary subset (`sin`/`cos`/`exp`/
    `tanh`/`sigmoid`/`sqrt`/`log`/`cbrt`/`expm1`/`log1p`) at a fixed
    away-from-discontinuity vector point, `eps = 1.0e-3`, tolerance `5.0e-3`
    (re-verified empirically here, not copied from Emily's number).

**All 14 tests pass** (full suite: `mix test` → 2667 passed, 0 failed — no
regressions).

**Plan adjustments applied (user directive + advisor sign-off, recorded
in-file above):** no `StreamData` (table-driven fixed zoo instead); no
non-differentiable-op exclusion list (included, with the FD reference correctly
restricted to the smooth subset per advisor guidance).

**Test-harness bug found and fixed (not a compiler bug):** the `reference/2`
helper originally didn't isolate itself from `EMLX.Case`'s process-global
`Nx.default_backend(EMLX.Backend)` setup. `Nx.Defn.Evaluator` uses the
*current default backend* for any tensor it synthesizes internally (e.g.
`window_max`'s min-value padding fill) rather than matching the explicit
backend of the args passed in — so the "pure `Nx.BinaryBackend`/Evaluator"
reference was silently mixing in `EMLX.Backend` for exactly the scenario
(`window_max` with explicit `:padding`) that first triggered an internal
constant synthesis. Fixed by scoping `Nx.default_backend/1` around the
reference call (save/restore). This is a latent risk in `grad_triage_test.exs`'s
identical-pattern reference helper too, though none of its 8 scenarios happen to
trigger it (no explicit non-default padding) — left as-is since that file is
Stage 23's closed historical record, not touched here.

**Genuine gap #1 (EMLX, real, named as Stage 33):** `window_sum` grad with
non-unit strides (`strides: [2, 1]`) raises `does not yet lower op :pad with
interior padding or negative lo/hi values` — `Nx.Defn.Grad`'s backward for
strided `window_sum` un-strides the cotangent via an interior-padded
`Nx.pad`, and interior-padding `:pad` is a pre-existing, deliberately
not-yet-lowered native-compiler gap (`EXPR_NODES.md`). Stage 23's
`window_sum` scenario used default (unit) strides and never reached this
path. Test asserts the known raise (regression-pins the exact gap) rather
than asserting equivalence. Follow-on:
[`33-strided-window-grad-interior-pad`](33-strided-window-grad-interior-pad.md).

**Genuine gap #2 (Nx, not EMLX — filed as a bug report, not a follow-on
stage):** a `while` whose body contains a data-dependent `cond` (predicate
`sum(out) > 0`, which is true on every iteration for the test's inputs)
produces a **wrong** gradient under `Nx.Defn.Evaluator` (pure
`Nx.BinaryBackend`, no EMLX involved) —
`[3.4130693e-20, 1.9330125, 3.4130693e-20]` — while **EMLX's native compiler
produces the analytically- and finite-difference-correct**
`[1.4641001, 1.4641001, 1.4641001]` (verified independently by both a
closed-form derivation — the loop is a pure ×1.1 scale repeated 4 times, so
the gradient must be uniform `1.1^4` — and central-difference finite
differences on the pure-`Nx.BinaryBackend` forward pass). This is a bug in
`Nx.Defn.Grad`'s backward `:while` construction (specifically its handling
of a nested `cond` inside the derived backward body), reproducible with zero
EMLX involvement — filed as
[`nx-grad-while-cond-bugreport.md`](nx-grad-while-cond-bugreport.md), same
pattern as this project's prior `Nx`/`Nx.Defn.Graph` bug reports. The test
scenario is checked against a finite-difference reference instead of the
known-broken `Nx.Defn.Evaluator` path, so it still pins EMLX's correctness
without asserting against a broken reference. No EMLX follow-on stage is
needed — this is out of scope for this project (upstream `Nx` bug), noted
here for visibility.

**Follow-on stages named:** only
[`33-strided-window-grad-interior-pad`](33-strided-window-grad-interior-pad.md)
(the one genuine EMLX-side gap). No other follow-on stages needed — every
other scenario across all 10 groups passed unmodified, extending Stage 23's
"native compiler already handles grad'd expressions cleanly" finding to
materially more control-flow nesting, windowed-op parameterizations, and
non-differentiable-op-as-operand patterns.
