# Bug report — `Nx.Defn.Grad`'s backward `:while` mishandles a data-dependent `cond` nested in the loop body

**Component:** `Nx.Defn.Grad` (`lib/nx/defn/grad.ex`, `update_grads(:while, …)`)
**Affected APIs:** `Nx.Defn.grad/2` (any caller — `Nx.Defn.Evaluator`, `EXLA`,
`EMLX`, or any other `Nx.Defn.Compiler`, since the bug is in the *expression*
`Nx.Defn.grad/2` builds, before any backend/compiler ever sees it)
**Severity:** medium — silently wrong (not near-zero-everywhere, not a crash)
gradient for a specific, plausible control-flow shape; does not affect
`while`-without-nested-`cond` or `cond`-without-`while` (both independently
correct — see Stage 23's triage).
**Found via:** Stage 28 (`workdir/native-compiler/28-grad-equivalence-suite.md`)
widening Stage 23's grad triage to cover "a `while` whose body itself contains
a `cond`" (a scenario Stage 23 explicitly deferred as new coverage). **EMLX's
native compiler is not at fault here** — its native-lowered result is the one
that's finite-difference-correct; `Nx.Defn.Evaluator`'s result (computed from
the same `Nx.Defn.grad`-built expression, no EMLX involved) is the one that's
wrong. This is an `Nx.Defn.Grad` bug, not an EMLX lowering bug.

---

## Symptom

```elixir
defn loss(x) do
  {out, _i} =
    while {out = x, i = 0}, Nx.less(i, 4) do
      out =
        cond do
          Nx.all(Nx.greater(Nx.sum(out), 0)) -> Nx.multiply(out, 1.1)
          true -> Nx.add(out, 0.05)
        end

      {out, i + 1}
    end

  Nx.sum(out)
end

defn grad_fn(x), do: grad(x, &loss/1)
```

For `x = Nx.tensor([0.045, 0.415, 0.785])`, the `cond` predicate
(`sum(out) > 0`) is true on every one of the 4 iterations (the loop only ever
scales `out` up by ×1.1, so its sum stays positive throughout) — confirmed by
comparing the *forward* pass between `Nx.Defn.Evaluator` and `EMLX` (both
agree: `1.8228046` / `1.8228047`, rounding only). So the analytically-correct
gradient is uniform: `d(sum(x * 1.1^4))/dx_i = 1.1^4 = 1.4641` for every `i`.

* **`Nx.Defn.Evaluator`** (`compiler: Nx.Defn.Evaluator`, pure
  `Nx.BinaryBackend`, no EMLX in the loop at all) returns
  `[3.4130693e-20, 1.9330125, 3.4130693e-20]` — wrong on all three elements,
  and not just "slightly off": two of the three components are ~20 orders of
  magnitude too small, the third is ~32% too high.
* **`EMLX` native compiler** (`compiler: EMLX`) returns
  `[1.4641001, 1.4641001, 1.4641001]` — matches the analytic value.
* **Finite differences** (central difference, `eps = 1.0e-4`, pure
  `Nx.BinaryBackend`, no `Nx.Defn.grad` involved at all — perturb each
  coordinate of `x` independently and re-run the *forward* `loss` function)
  give `[1.46409996, 1.46409996, 1.46409996]` — confirms `EMLX` is right and
  `Nx.Defn.Evaluator` (i.e. `Nx.Defn.Grad`'s expression) is wrong.

## Reproduction

```
cd emlx && mix run -e '
import Nx.Defn

defmodule Repro do
  defn loss(x) do
    {out, _i} =
      while {out = x, i = 0}, Nx.less(i, 4) do
        out =
          cond do
            Nx.all(Nx.greater(Nx.sum(out), 0)) -> Nx.multiply(out, 1.1)
            true -> Nx.add(out, 0.05)
          end

        {out, i + 1}
      end

    Nx.sum(out)
  end

  defn grad_fn(x), do: grad(x, &loss/1)
end

x = Nx.tensor([0.045, 0.415, 0.785], type: {:f, 64}, backend: Nx.BinaryBackend)
IO.inspect(Nx.Defn.jit_apply(&Repro.grad_fn/1, [x], compiler: Nx.Defn.Evaluator))
'
```

No EMLX dependency is required to reproduce this — it is reproducible on plain
`Nx.BinaryBackend` with `compiler: Nx.Defn.Evaluator`, confirming the bug lives
in `Nx.Defn.Grad`'s expression construction, not in any backend/compiler.

## Suspected root cause (not fully bisected — flagging for whoever picks this up)

`update_grads(:while, [initial, arg, condition, body], …)`
(`deps/nx/nx/lib/nx/defn/grad.ex:322`) builds a **new `:while` node** for the
backward pass, differentiating the loop `body` once (via `to_grad/5` over
`parents_tree(body, %{})`) and re-running that single derived backward step
`n` times via the new while's own iteration count. That's correct when the
body's *local Jacobian* is the same on every iteration (e.g. a fixed
elementwise op). It is not obviously correct when the body itself branches on
a runtime condition via `cond` (as here, or any data-dependent `cond` whose
predicate can vary iteration-to-iteration): `update_grads(:cond, …)`
(same file, line 369) builds its own backward `cond` with a **freshly-derived
predicate**, sharing the "the checks are cheap and shared between the
original cond and the graded cond" assumption (see the comment at line 425).
Nesting that inside the backward `:while`'s single derived body means the
per-iteration predicate reevaluation and the per-iteration state threading
between the forward carry and the backward cotangent don't obviously line up
one-to-one — worth checking whether the backward `cond`'s branches are being
selected using the *correct* per-iteration carry value, or a stale/aliased
one from a different point in the loop (the exponentially-small
`3.4130693e-20` values look like an accumulated `0` from repeatedly taking a
zero-gradient path, consistent with the backward `cond` picking the wrong —
or a mismatched-magnitude — branch on some iterations).

## Scope / non-fix

Per this project's discipline (a testing stage does not fix compiler bugs
inline, and this isn't even an EMLX bug), this is filed as a bug report, not
fixed here. `EMLX.GradEquivalenceTest`'s
`"while-body-contains-cond grad"` scenario is tested against a
finite-difference reference instead of `Nx.Defn.Evaluator` for this specific
scenario, with this bug report cited inline, so the suite still pins EMLX's
(correct) behavior without depending on the known-broken `Evaluator` path.
