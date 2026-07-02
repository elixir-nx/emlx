# Stage 28 — `EMLX.MixedPrecision` module

Status: not started. Emily M16 parity, named by Stage 23's triage.

## Why this stage exists

Unlike Stage 27 (a testing-breadth stage — the mechanism it tests already
works), this is a genuinely missing feature: `EMLX.MixedPrecision` does not
exist in any form (zero `MixedPrecision`/`mixed_precision`/`loss_scal` hits
anywhere in `emlx/lib`, confirmed directly during Stage 23's triage). Emily's
M16 (`~/coding/emily/PLAN.md`) ships bf16-forward training with f32 master
weights and dynamic loss scaling — a real training-recipe feature, not a
compiler-coverage gap. Nothing about Stage 23's clean grad-triage result
implies this exists; it must be built from scratch.

## Procedure

1. Mirror Emily's `Emily.MixedPrecision` module surface:
   `cast_params/2`, `accumulate_grad/2`, `loss_scale/1`, `scale_loss/2`,
   `unscale/2`, `update/2`, `has_overflow?/1`. Read Emily's actual
   implementation (`~/coding/emily/lib`), not just its `PLAN.md` prose, per
   the docs-vs-code discipline Stage 20 established.
2. Design decision (make explicitly, don't default silently): does this
   module target the native `compiler: EMLX` lane specifically, the eager
   `EMLX.Backend` lane, or both? Emily's M16 is scoped against its single
   compiled lane; EMLX has both an eager backend and this planning
   directory's native compiler — pick and record which (or both, sized
   separately) before implementing.
3. bf16-tolerance grad-equivalence suite: verify gradients survive the
   bf16-forward / f32-master-weight round trip within documented tolerance.
4. MNIST-style bf16 convergence canary: a small model actually trains to a
   reasonable accuracy under this module, not just "the API doesn't crash."

## Acceptance

`EMLX.MixedPrecision` shipped with the seven-function surface above, a
bf16-tolerance grad-equivalence suite, and a convergence canary that
demonstrably trains (accuracy improves over epochs, not just "runs without
raising").

## Results

(not started)
