# Stage 14 — `while` via C++ child program (contingent)

Status: **dropped — no-go RE-AFFIRMED by measurement (2026-06-30 revisit).** The
user re-opened this on a `while`-prevalence argument; a zero-C++ benchmark
(`emlx/bench/while_dispatch_bench.exs`) refuted it (see "Revisit measurement"
below): a C++ in-worker `while` saves ≤30 % per iteration for convergent loops
(shrinking with body weight) and is a **regression** for counted loops (the host
loop already fuses). The Stage-08 host loop is retained. A separate correctness
bug in the counter-only bare-while path was found and filed as follow-up.

> **Correction (advisor, 2026-06-30):** the original framing below ("recompiling
> stages by re-entering this compiler … pays BEAM↔NIF crossings per iteration")
> overstated the per-iteration cost. `build_while_base_eval_fn` jits `cond_fn`/
> `body_fn` **once** at compile time; `run_while_loop` only *replays* per
> iteration. The `Graph.split` + re-jit is a **per-invocation fixed cost**, not a
> per-iteration one. The only per-iteration removable cost is: 2 jit-dispatched
> `eval_program` NIF calls (cond + body) + 1 `Nx.to_number` scalar pull. That is
> the load-bearing quantity being measured.

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

## Revisit measurement (2026-06-30) — decision: RE-AFFIRM no-go

Advisor gated the revisit on measuring the **removable-overhead fraction** per
iteration before writing any C++. Built `emlx/bench/while_dispatch_bench.exs`
(zero C++) which decomposes the real Stage-08 host-loop path. Key insight that
reframed everything: `Nx.to_number(cond)` in `run_while_loop` only forces what
the **condition** reads, so there are two structurally different regimes the
original analysis conflated:

- **counted** — cond reads only a loop counter. The body's lazy MLX graph
  accumulates across iterations and **fuses**; only the tiny counter is forced
  each step. Models fixed trip-count loops.
- **convergent** — cond reads the carry (Newton/fixed-point). Every iteration
  forces a full worker eval barrier. Models data-dependent loops.

### Numbers (median, N=500 iters/run)

Removable per iter = `2·jit_dispatch_floor + to_number_floor` — the max a C++
in-worker `while` can save (it collapses the loop into one NIF call, removing
the per-iteration BEAM↔NIF crossings + scalar pull; it CANNOT remove the worker
eval barrier under MLX 0.31.2).

| device | dispatch floor | to_number floor | removable/iter |
|--------|---------------:|----------------:|---------------:|
| CPU    | ~40 µs         | ~25 µs          | ~106 µs        |
| GPU    | ~51 µs         | ~27 µs          | ~130 µs        |

**Convergent regime — removable fraction shrinks as body grows** (GPU):

| body            | µs/iter | removable % |
|-----------------|--------:|------------:|
| cos scalar `{}` | 434     | 29.9 %      |
| cos vec `{1024}`| 460     | 28.2 %      |
| cos `{256,256}` | 514     | 25.3 %      |
| dot `{64,64}`   | 541     | 24.0 %      |
| dot `{256,256}` | 538     | 24.1 %      |
| dot `{512,512}` | 748     | 17.3 %      |

CPU is worse for C++ (14–22 %). Even the lightest possible real body caps the
C++ win at ~30 % (GPU) / ~22 % (CPU), and it falls as the body gets heavier —
the exact opposite of what a "prevalence" argument needs.

**Counted regime — the host loop already fuses; a C++ `while` would be SLOWER.**
Lazy (host counted-while) vs forced (what a C++ eval-per-iteration while pays,
minus one crossing) per iter, GPU:

| body            | lazy µs/it | C++-while ceiling µs/it | verdict          |
|-----------------|-----------:|------------------------:|------------------|
| cos vec `{1024}`| 150        | 181                     | host **beats** C++|
| cos `{256,256}` | 149        | 177                     | host **beats** C++|
| dot `{64,64}`   | 214        | 216                     | host **beats** C++|
| dot `{256,256}` | 255        | 262                     | host **beats** C++|
| dot `{512,512}` | 396        | 414                     | host **beats** C++|

For **every** body weight the fused host loop beats the C++-while ceiling,
because a C++ eval-per-iteration loop breaks the cross-iteration fusion the
host loop gets for free. So for the very common fixed-trip-count case, a C++
`while` is a **regression**, not a win.

**`Graph.split` fragmentation (#2)** is a per-invocation fixed cost (~0–0.75 ms,
within noise at 20 iters), independent of trip count → amortized to near-zero for
real loops. Not a per-iteration cost; does not move the decision.

### Decision

**RE-AFFIRM no-go.** The revisit trigger (`while` prevalence ⇒ avoid round-trips
& nested compilations) does not survive measurement:
1. Convergent loops: C++ saves ≤30 % (GPU) and shrinks with body weight —
   marginal, and the eval barrier (the real cost) is irreducible under MLX
   0.31.2.
2. Counted loops: the host loop already fuses; C++ would be **slower**.
3. "Nested compilations" is a fixed per-invocation cost (advisor-corrected: the
   loop does **not** recompile per iteration), amortized to noise.

The host loop stays. Revisit triggers below unchanged, plus the new hard datum:
a C++ `while` only helps convergent loops, only marginally, and never counted
ones — so it needs a specific profiled convergent workload dominated by BEAM
crossings to justify the full subprogram-channel build.

### Follow-up bug found (separate from this gate) — FIXED

The measurement surfaced a **real correctness bug** in the existing Stage-08
host-loop path (`build_while_base_eval_fn`): a bare-tail `while` whose condition
does **not** reference every carry element (e.g. a counter-only `Nx.less(i, n)`
with the payload carried but unread by the cond) either ran **0 iterations**
(scalar carry) or **crashed** with a shape mismatch (non-scalar carry) —
`EMLX.Backend.check_shape_and_type!/3` via `run_while_loop/3`. Reproduced for
`{}`, `{1024}`, `{256,256}`, matmul carries. The tested cases (`count_to_10`,
`while_two_carry`) all have carry-reading conditions, which is why this slipped
through.

**Root cause:** `EMLX.Native.Expr.lower/1` built its wire input list only from
parameter positions actually *referenced* inside the given sub-expression,
sorted-then-compacted (dropping unreferenced positions rather than reserving
their slot). This is safe for a fresh top-level `defn` trace or a
`Graph.split` stage (both are dense by construction — every position crossing
that boundary is, by definition, used downstream). It is **not** safe for
`while`'s `cond_fn`/`body_fn`, which reuse a pre-built sub-scope expression
standalone: a `while` condition legitimately ignores carry slots it doesn't
read (the body still threads them through), so its embedded parameter
positions can be sparse — while the host loop's runtime dispatch
(`run_while_loop/3`) always supplies the *full*, dense carry tuple. The
compaction silently shifted every position after a gap, binding wrong values
(or, for shape-incompatible neighbors, crashing).

**Fix:** `EMLX.Native.Expr.lower/2` now takes an optional `num_inputs` arity
hint and densifies the wire input list to `0..max(num_inputs, max_referenced_pos)`,
filling any gap with a placeholder ref no instruction ever reads (so wire
index == original parameter position, always). `EMLX.__compile__` threads the
true call arity (`length(Composite.flatten_list(vars))`, already on hand in
`try_native_compile/3`) into the one call site that needed it — the flat/no-while
branch of `build_eval_fn/4` (used by `cond_fn`/`body_fn` when they re-enter the
compiler). No other call site needed a change (`bare_while?`/`Graph.split`
branches don't lower directly and are dense by construction). Zero-cost: the
extra wire slots are unreferenced by any instruction, so no data crosses the
NIF boundary for them beyond what the caller already sends.

Regression tests added (`while_counter_only_cond/3`, Stage 08 describe block,
`emlx/test/emlx/native/expr_test.exs`): scalar and non-scalar payload, both vs
the Evaluator oracle. Full suite green (2532 tests).

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
