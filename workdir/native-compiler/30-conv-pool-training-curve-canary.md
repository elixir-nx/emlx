# Stage 30 — conv-pool training curve-matching canary

Status: not started. Emily M17 parity, rescoped by Stage 23's triage.

## Why this stage exists

Emily's M17 (`~/coding/emily/PLAN.md:829`, "Conv-pool training") is scoped
there as "lift window reductions (`window_sum`, `window_max`, `window_min`,
`window_product`, `window_scatter_max`, `window_scatter_min`) off
`via_binary` onto their native MLX counterparts." Stage 23's triage checked
this directly against EMLX's actual code (not just Stage 20's seed claim,
which had lumped M17 in with M16 as "confirmed genuinely missing" without
checking the primitive claim separately) and found **EMLX already does
this**: `EMLX.Backend`'s `window_op/5` and `window_scatter_function/7`
(`lib/emlx/backend.ex:1728`, `1835`) implement all six ops via a real MLX
sliding-window view, not `via_binary` — and `pooling_test.exs` already
grad-tests `window_scatter_max`/`window_scatter_min` against
`Nx.BinaryBackend`. So the primitive-lift half of M17 is closed already; this
stage's scope is narrower than the original M17 charter.

## Procedure

1. Confirm (re-verify, don't just cite Stage 23's finding — same
   docs/citations-need-re-checking discipline as Stage 20) that no
   window-reduction op still routes through `via_binary` anywhere reachable
   from a training loop, on both the eager `EMLX.Backend` lane and, if in
   scope for this stage, the native `compiler: EMLX` lane (Stage 23's
   `window_sum`/`window_max` grad triage covered the *compiler* lane
   already — cite it, don't re-derive it).
2. Build a small training-curve-matching canary: a handwritten small
   CNN/pool classifier (mirrors Emily's own "handwritten MLP and handwritten
   [CNN]" M17 testing plan) trains for N steps/epochs and its loss curve is
   compared against a `Nx.BinaryBackend` reference run — not just "does it
   run," but "does it converge the same way."
3. Decide scope: does the canary run under the eager backend, the native
   `compiler: EMLX` lane, or both? Record the decision explicitly.

## Acceptance

A training-curve-matching canary test checked in and passing, demonstrating
a small conv/pool model trains equivalently (within documented tolerance) to
a `Nx.BinaryBackend` reference — closing Emily's M17 on the *testing* axis,
since the *primitive* axis is already closed per the finding above.

## Results

(not started)
