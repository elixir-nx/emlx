# Stage 28 — grad-equivalence regression suite

Status: not started. Emily M9 (testing half) parity, sized by Stage 23's
triage.

## Why this stage exists

Stage 23's triage (`emlx/test/emlx/grad_triage_test.exs`) ran an 8-scenario
zoo of `Nx.Defn.grad`-wrapped functions (elementwise, reduction, dot, both
`cond` branches, a counted `while`, `window_sum`, `window_max`) through
`compiler: EMLX` and found all 8 pass unmodified against a
`Nx.BinaryBackend`/`Nx.Defn.Evaluator` oracle. That triage zoo is deliberately
narrow (one shape, one dtype, one representative case per op class) — this
stage widens it into a permanent, broader regression suite, mirroring Emily's
own M9 harness design (`~/coding/emily/PLAN.md`'s "Testing — Layers 4 (Grad)
and 5 (Training)" section):

1. A `StreamData`-based property test: for a larger zoo of `defn`-expressible
   functions (excluding non-differentiable ops per Emily's own exclusion
   list — `argmax`, `argmin`, `floor`, `sign`, comparisons), assert
   `Nx.Defn.grad(f)` under `compiler: EMLX` matches the same grad under
   `Nx.BinaryBackend`/`Nx.Defn.Evaluator`, across generated shapes/dtypes.
2. A numerical finite-difference oracle for the differentiable subset:
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
   here) with the StreamData property test and finite-difference oracle
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

(not started)
