# Stage 14 — `while` via C++ child program (contingent)

Status: open (contingent). **Only pursued if Stage 12's gate greenlights it.**

## Why this stage might exist

Stage 08 runs `while` host-side: `Nx.Defn.Graph.split` isolates each loop and
the trip count is driven from Elixir, recompiling stages by re-entering this
compiler. That works but pays BEAM↔NIF crossings per iteration. If Stage 12's
spike shows a held pred/body subprogram driven by an eval-per-iteration C++ loop
(on the worker thread) is meaningfully faster, re-express `while` that way.

## The constraint

MLX 0.31.2 has no lazy data-dependent loop primitive, so a C++ `while` is
**not** a single traced graph: it must `mlx::core::eval` the predicate scalar
each iteration and replay the body subprogram. The only win over the host loop
is staying off the BEAM — this must be measured, not assumed.

## Procedure (if greenlit)

1. Emit a `:while` instruction with held pred/body subprograms (Stage 12's
   sub-IR channel) instead of `Graph.split`.
2. C++: loop — `eval(pred(carry))`, read bool, replay `body(carry)` until false;
   return the carry. Refcount-hold the subprograms on the parent `Expr`.
3. Equivalence vs eager + vs the current host-loop path (counted loop,
   carried-state loop, nested while, while-as-input — the Stage 08 cases).
4. Benchmark a decode-shaped loop vs the host loop; keep whichever wins.

## Acceptance

- C++ `while` matches the host-loop path on all Stage 08 cases and shows a
  measured improvement, **or** the stage is explicitly dropped with the host
  loop retained and the decision recorded here.
