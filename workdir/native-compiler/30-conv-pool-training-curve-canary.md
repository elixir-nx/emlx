# Stage 30 — conv-pool training curve-matching canary

Status: done. Emily M17 parity, rescoped by Stage 23's triage.

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

**Re-verification (procedure item 1):** re-grepped `lib/emlx/backend.ex` directly
(not just cited Stage 23) — `window_op/5` (~line 1735) and
`window_scatter_function/7` (~line 1837) remain the sole implementations of
`window_sum`/`window_max`/`window_min`/`window_product`/`window_scatter_max`/
`window_scatter_min`, no `via_binary` anywhere in the file. Stage 23's grad
triage and Stage 28's grad-equivalence suite already cover the `compiler:
EMLX` lane for `window_sum`/`window_max` grad (8/8 and 14/14 passing,
respectively) — cited, not re-derived.

**Canary:** `emlx/test/emlx/conv_pool_training_canary_test.exs` (new). A
handwritten conv(4×1×3×3, valid) → relu → `Nx.window_max` (2×2, stride 2,
valid) → flatten → dense(36→3) classifier, hand-rolled SGD (`lr = 0.05`, no
optimizer library), trained for 20 steps over a fixed 3-batch deterministic
dataset (values generated via a seeded `sin`-based formula, not `Nx.Random`,
to keep the comparison independent of cross-backend RNG parity — an
orthogonal, already-settled concern per `grad_equivalence_test.exs`). The
per-step loss curve (not just the final loss) is asserted equivalent
(`atol = 1.0e-3`) against a `Nx.BinaryBackend`/`Nx.Defn.Evaluator` reference,
plus a coarse "did it actually converge" assertion (final loss < 50% of
initial loss) on the reference curve itself, so the test would fail if the
model were accidentally not learning.

**Scope decision (procedure item 3, per advisor sign-off — both, not
either/or):** two tests, both against the same oracle —

1. Eager `EMLX.Backend` (params/data transferred to `EMLX.Backend`, trained
   via `Nx.Defn.jit_apply(compiler: Nx.Defn.Evaluator)`) — matches the
   reference curve exactly within tolerance.
2. Native `compiler: EMLX` (same `Nx.BinaryBackend`-resident params/data
   passed straight to `Nx.Defn.jit_apply(compiler: EMLX)`, which handles the
   cross-backend hand-off internally, same pattern as
   `grad_equivalence_test.exs`) — also matches within tolerance.

**Advisor-flagged landmine avoided:** pooling uses only `Nx.window_max`
(max-pool), not strided `Nx.window_sum`/`window_mean` (avg-pool) — Stage 28
found strided `window_sum`'s *backward* pass hits the pre-existing
interior-padding `:pad` gap (Stage 33, not yet started). Max-pool grad is
already at parity (Stage 23/28), so this canary doesn't collide with that
open gap.

**Full suite:** `mix test` → 2669 passed, 0 failed (2667 prior + this
stage's 2 new tests), no regressions.

Closes Emily's M17 on the testing axis — the primitive axis was already
closed per Stage 23's finding, re-confirmed above.
