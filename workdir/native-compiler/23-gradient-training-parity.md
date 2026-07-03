# Stage 23 — gradient & training parity epic (scoping only)

Status: done. Triage (`emlx/test/emlx/grad_triage_test.exs`, 8 scenarios)
found the native compiler already handles grad'd expressions cleanly across
elementwise/reduction/dot/`cond`/`while`/windowed-op scenarios — no compiler
bugs found. Named and sized three follow-on stages: Stages 27 (testing
breadth, small), 28 (real missing feature, M16), 29 (M17, rescoped smaller
after finding the primitive lift is already done). Emily M9/M13/M16/M17
parity (see Stage 20).

## Why this stage exists

Emily's M9/M13/M16/M17 collectively represent a substantial grad/training
parity investment: grad-equivalence property tests vs `Nx.BinaryBackend`
grad + EXLA reference, `Emily.MixedPrecision` (bf16 forward + f32 master weights
+ dynamic loss scaling), MNIST convergence canaries, and conv-pool training
parity. EMLX has **zero** grad-specific tests today (no `*grad*` files
anywhere in `emlx/test`).

`Nx.Defn.grad` differentiates the *traced expression* before the EMLX
compiler ever sees it, so in principle every already-lowered op "just works"
under grad as long as its forward semantics are correct — but that's an
untested assumption, and specifically: the native compiler's whole value
proposition (single-NIF replay) is completely unvalidated for training-shaped
graphs (multi-output backward graphs, accumulation patterns, `while`-based
training loops backpropagated through). This stage exists to find out where
reality diverges from that assumption before committing to a training
feature-parity roadmap.

## Procedure (scoping — expect this to fan out into 3–4 follow-on stages once
triaged, the same way Stage 12's spike fanned into Stages 13/14)

1. **Triage.** Run a small zoo of `Nx.Defn.grad`-wrapped functions (mirroring
   Emily M9's zoo — a representative spread of elementwise, reduction, dot,
   and control-flow-bearing functions) through `compiler: EMLX` and record
   what breaks. Likely candidates: `while`-in-backward (backprop through a
   training loop), multi-output `elem` handling under grad, any op whose
   *backward* pass composes ops not yet native-lowered even though the
   *forward* op is. **From Stage 20's audit**: explicitly include a windowed
   op (`window_sum`/`window_max`/etc.) in the zoo — `window_sum` and friends
   are native only inside the compiler's IR (Stage 06/13), while the eager
   `EMLX.Backend.window_reduce/6` hard-raises, so grad-of-windowed-op under
   `compiler: EMLX` is untested territory distinct from the other candidates
   above.
2. **If the native compiler already handles grad'd expressions cleanly**:
   this becomes "just add the test suite" — a materially smaller task than
   Emily's M9 was for them (they built the compiler and the backend
   together; EMLX's backend already exists and presumably works correctly
   under ordinary `Nx.Defn.Evaluator` grad — the open question is
   specifically the *native* single-NIF lane's behavior, not correctness of
   the underlying ops).
3. **If real gaps surface**, split into dedicated follow-on stages, each
   sized only after step 1's findings are in hand:
   - Grad-equivalence test suite (vs `Nx.BinaryBackend` grad + finite
     differences + EXLA reference, per Emily M9/M13's harness design).
   - `EMLX.MixedPrecision` module mirroring Emily M16's
     `cast_params/2` / `accumulate_grad/2` / `loss_scale/1` / `scale_loss/2`
     / `unscale/2` / `update/2` / `has_overflow?/1`, with a bf16-tolerance
     grad-equivalence suite and an MNIST-style bf16 convergence canary.
   - Conv-pool training parity (Emily M17).
4. Do not block Stages 16–22 on this epic — they ship independently. This
   stage's only deliverable right now is the triage report and named,
   sized follow-on stages.

## Acceptance (for *this* scoping doc)

A triage report (Results table) stating exactly which grad/training
scenarios pass today unmodified under `compiler: EMLX`, which don't, and
naming the follow-on stages needed to close each real gap — with those
follow-on stages stubbed as new numbered docs in this directory once sized.

## Results

**Advisor sign-off (before starting).** Confirmed the triage-script approach;
flagged four adjustments, all applied: (1) new follow-on stage numbers must
start at 27 (24/25/26 already taken); (2) the triage instrument should be a
real ExUnit test file (`emlx/test/emlx/grad_triage_test.exs`), not a
throwaway script, mirroring `sdpa_sinks_test.exs`'s structure, so the Results
below are reproducible; (3) reference is `Nx.BinaryBackend` via
`compiler: Nx.Defn.Evaluator` only — no EXLA dependency added just for this
triage; (4) for `while`, confirm empirically (not assumed) whether
`Nx.Defn.grad`'s reverse-mode transform reaches the compiler's `while`-lowering
machinery at all.

**Finding on (4), load-bearing for the whole triage:** `Nx.Defn.grad`'s
`:while` case (`deps/nx/nx/lib/nx/defn/grad.ex:322`, `update_grads/5`) builds
a **new `:while` `Expr` node** for the backward pass (reverse-mode
accumulation over the same iteration count) — it does not unroll or otherwise
avoid emitting `:while`. So a grad'd while-containing function hits the exact
same `Nx.Defn.Graph.split` + host-loop machinery Stage 08/12/14 built for a
*forward* `while`, just applied to a second, compiler-synthesized `while`
node. There is only one "direction" to test (grad is a defn-body macro, not
an outside-compile wrapper) — the doc's premise of "two directions" collapsed
to this one question, which the triage answers directly below.

**Triage zoo (`emlx/test/emlx/grad_triage_test.exs`, 8 scenarios, referenced
against `Nx.BinaryBackend` via `compiler: Nx.Defn.Evaluator`, `assert_all_close`
default tolerance):**

| Scenario | Forward ops | Backward path exercised | Result |
|---|---|---|---|
| Elementwise | `sin`, `cos`, `multiply`, `sum` | chain-rule elementwise + reduction-sum broadcast | **pass** |
| Reduction | `sum` (axis), `mean` | reduction backward (broadcast-and-scale) | **pass** |
| Dot | `dot`, `sum` | `dot` backward (`dot` w/ transposed operand) | **pass** |
| `cond` (branch A) | `cond`, `multiply`, `sum` | inline `:select`-based backward through the taken branch | **pass** |
| `cond` (branch B) | `cond`, `abs`, `sum` | same, other branch taken | **pass** |
| `while` | 3-iteration counted `while`, `multiply` | compiler-synthesized backward `:while` (see finding above) | **pass** |
| `window_sum` | `window_sum`, `sum` | `window_sum` again (grad.ex's own backward for sum is pad + `window_sum`, not a scatter op) | **pass** |
| `window_max` | `window_max`, `sum` | `:window_scatter_max` (distinct opcode from `window_sum`'s backward — Stage 06/13 coverage) | **pass** |

**Headline result: all 8 scenarios pass unmodified today.** The doc's
hypothesis in "Why this stage exists" — that `Nx.Defn.grad` differentiates
before the EMLX compiler ever sees the graph, so already-lowered forward ops
"just work" under grad — holds for this zoo, including the two scenarios
flagged as highest-risk going in (`while`-in-backward, windowed-op grad).
Per the doc's own step 2 branch: this collapses to "just add the test suite"
rather than opening compiler-fix stages.

**Additional finding (M17, conv-pool training primitives) — Stage 20's audit
call needs a correction, recorded here rather than silently left stale.**
Stage 20 lumped M17 into "confirmed genuinely missing… → Stage 23" without
separating the *primitive* claim from the *testing* claim (unlike its M9 row,
which did separate them). Checked directly against Emily's own M17
description (`~/coding/emily/PLAN.md:829` — "Lift window reductions
(`window_sum`, `window_max`, `window_min`, `window_product`,
`window_scatter_max`, `window_scatter_min`) off `via_binary` onto their
native MLX counterparts"): EMLX's eager `EMLX.Backend` **already** implements
all six ops natively via a real MLX sliding-window view + reduction/scatter
(`lib/emlx/backend.ex:1728` `window_op/5`, `1835` `window_scatter_function/7`)
— none of them fall through to `via_binary`. `pooling_test.exs` already
exercises `Nx.Defn.grad` (default `Nx.Defn.Evaluator` compiler, eager
`EMLX.Backend` tensors) for `window_scatter_max`/`window_scatter_min` today.
So M17's primitive-lift half is **already at parity**, same conclusion as
M9's primitives row — the real remaining M17 gap is narrower than the seed
implied: a training-curve-matching canary (small CNN/pool model converges),
not a primitive port.

**Genuinely missing, independent of this triage's pass result:**
`EMLX.MixedPrecision` (M16) does not exist in any form — zero
`MixedPrecision`/`mixed_precision`/`loss_scal` hits in `emlx/lib`. Unlike the
grad-compiler question above, this isn't conditional on triage findings: it's
a feature that must be built regardless, mirroring Emily's
`cast_params/2` / `accumulate_grad/2` / `loss_scale/1` / `scale_loss/2` /
`unscale/2` / `update/2` / `has_overflow?/1` surface (M16).

**Follow-on stages named and sized** (stubbed as new docs in this directory,
originally starting at 27 per advisor guidance — 24/25/26 already taken;
renumbered to 28/29/30 when Stage 25 was inserted as
`25-quantized-dot-full-fix` and the burndown shifted down one):

- [`28-grad-equivalence-suite`](28-grad-equivalence-suite.md) — formalize
  this stage's triage zoo into a permanent property-based grad-equivalence
  regression suite (StreamData harness, mirroring Emily's M9 design), run
  under `compiler: EMLX`. Small — the zoo above already proves the mechanism
  works; this stage is breadth (more ops, more shapes, a finite-difference
  reference for the differentiable-op subset), not new compiler code.
- [`29-mixed-precision`](29-mixed-precision.md) — build `EMLX.MixedPrecision`
  (Emily M16 parity) from scratch: bf16-forward + f32-master-weights +
  dynamic loss scaling, with its own bf16-tolerance grad-equivalence suite
  and an MNIST-style bf16 convergence canary. Medium — genuinely new module,
  not gap-closing.
- [`30-conv-pool-training-curve-canary`](30-conv-pool-training-curve-canary.md)
  — Emily M17 parity, rescoped per the finding above: primitives are already
  native, so this is a training-curve-matching canary (small CNN/pool model
  trains and converges under `compiler: EMLX`), not a primitive port. Small.

**Reviewer sign-off.** A fresh reviewer subagent (no `resume`, outcome
artifacts only — the triage test file, the Results section above, the three
follow-on stage docs, and the README/plan-file updates) independently
verified every claim above against source (line numbers, test-run counts,
`via_binary` absence, `EMLX.MixedPrecision` absence) and reproduced both the
triage suite's "8 passed" and the full suite's
"2613 passed (826 doctests, 1787 tests), 5 excluded." Verdict: **pass**, no
blockers.
