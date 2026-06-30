# Stage 14 — `while` via C++ child program (contingent)

Status: **dropped (gated no-go by Stage 12).** Not pursued now; re-open only on a
concrete triggering workload (see "Revisit triggers" below). The Stage-08
`Graph.split` host loop is retained.

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

## Stage 12 gate outcome + analysis (decision: dropped)

Stage 12 verified (Task 1) against the real MLX 0.31.2 headers that the public
core has **no** lazy/data-dependent control flow (`compile.h` / `transforms.h` /
`ops.h` / `compile_impl.h` — no `while_loop`/`fori_loop`/`scan`/`cond`); only
`eval(std::vector<array>)` + `array::item<T>()` exist. This sets a hard ceiling
on what a C++ `while` can buy, which drove the no-go.

### The hard ceiling

A C++ loop driving a dynamic `while` must `mlx::core::eval` the predicate every
iteration to decide whether to continue — a materialization barrier per
iteration. Moving the loop into C++ removes the **BEAM↔NIF crossing** but
**cannot** add cross-iteration graph fusion. The two costs are independent: C++
replay attacks only the round-trip, never the per-iteration eval barrier. No
amount of C++ plumbing changes this under 0.31.2.

### "Represent reduce as a `while` and lower it like one" — collapses to known options

A static `reduce` rewritten as a counted `while` does **not** help:

- **+ current host-loop `while` lowering** → N BEAM↔NIF round-trips for N scalar
  reducer applications. Strictly worse than baking it into one graph; for a
  static count + trivial body it is a pure regression.
- **+ C++-replay `while`** → for a *static* count there is no predicate to eval,
  so the C++ loop either (a) builds the body graph N times into one lazy graph =
  **the unroll, constructed in C++** = the C++ `:fold` Stage 12 already measured
  as negligibly different from the Elixir unroll (≤176 KB wire once, ≤2 ms
  build); or (b) evals each iteration = N serialized kernel launches, slower than
  the lazy unroll. Neither beats routing associative reducers (`add`/`mul`/
  `max`/`min`) to a single native `sum`/`product`/`reduce_max`/`reduce_min`.

So "reduce-as-while + C++ replay" is the C++ `:fold` by another name and does not
reopen the Stage 12 gate.

### Where C++-`while`-replay *would* genuinely win

The win exists only for **many iterations × a body light enough that the
per-iteration BEAM↔NIF + queue-dispatch + scalar-marshalling overhead is
comparable to the body's own work.** The actual target — autoregressive
**decode** loops — has a *heavy* body (a full transformer layer) and a moderate
iteration count, so the round-trip is already noise there. That mismatch is why
the host loop is good enough and Stage 14 is dropped.

### The one angle that could flip it

When a `defn` contains a large reduce with an **arbitrary, non-associative**
reducer, the current alternative is Evaluator fallback, which de-fuses the
**entire** surrounding `defn` to op-by-op (loses single-NIF for everything, not
just the reduce). A C++ eval-per-iteration loop would keep the whole `defn` as
one NIF call (slow reduce, rest stays fused). That is the only scenario where a
C++ held-body loop has a real edge — single-NIF-but-slow vs
Evaluator-fallback-and-de-fused. It is rare and costs the full subprogram-channel
+ held-body + worker-loop plumbing, so it does not justify the stage on its own.

### Revisit triggers (re-open only if one is observed)

1. A concrete `while` workload with **many iterations and a light body** where
   profiling shows host-loop overhead (BEAM crossing + queue dispatch + scalar
   marshalling) dominates per-iteration time.
2. A real model where a **large custom (non-associative) reduce** forces an
   Evaluator de-fusion that measurably hurts the surrounding `defn`, and keeping
   it single-NIF via a C++ loop recovers it.
3. An MLX upgrade that adds a lazy in-trace loop/scan primitive — which would
   remove the per-iteration eval barrier and change this analysis entirely.
