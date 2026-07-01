# Stage 23 — gradient & training conformance epic (scoping only)

Status: not started — largest, least-scoped item. This doc defines the
triage/sub-plan, not the implementation. Emily M9/M13/M16/M17 parity (see
Stage 20).

## Why this stage exists

Emily's M9/M13/M16/M17 collectively represent a substantial grad/training
conformance investment: grad-equivalence property tests vs `Nx.BinaryBackend`
grad + EXLA golden, `Emily.MixedPrecision` (bf16 forward + f32 master weights
+ dynamic loss scaling), MNIST convergence canaries, and conv-pool training
conformance. EMLX has **zero** grad-specific tests today (no `*grad*` files
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
     differences + EXLA golden, per Emily M9/M13's harness design).
   - `EMLX.MixedPrecision` module mirroring Emily M16's
     `cast_params/2` / `accumulate_grad/2` / `loss_scale/1` / `scale_loss/2`
     / `unscale/2` / `update/2` / `has_overflow?/1`, with a bf16-tolerance
     grad-equivalence suite and an MNIST-style bf16 convergence canary.
   - Conv-pool training conformance (Emily M17).
4. Do not block Stages 16–22 on this epic — they ship independently. This
   stage's only deliverable right now is the triage report and named,
   sized follow-on stages.

## Acceptance (for *this* scoping doc)

A triage report (Results table) stating exactly which grad/training
scenarios pass today unmodified under `compiler: EMLX`, which don't, and
naming the follow-on stages needed to close each real gap — with those
follow-on stages stubbed as new numbered docs in this directory once sized.
