# Stage 08 — Control flow (`cond`, `while`)

Status: complete

## Why this stage exists

`cond` and `while` are the first nodes with **child scopes**, so this stage
resolves the Stage-00 open question (who recurses into sub-scopes) and
introduces **child programs** — sub-IRs compiled and held by the parent
program, replayed under host control. This is the load-bearing capability for
autoregressive decode loops (`Bumblebee.Text.generation`-style `defn while`).

## Procedure

1. Finalize the child-scope mechanism (minimal: lowerer extracts each
   control-flow node's inner root(s) from `args` and calls
   `EMLX.Defn.Tree.post_order/1` + `lower` recursively). Update README decision
   gate and Stage 00 Results accordingly.
2. Lower `cond`: each clause predicate + body becomes a child program; combine
   via native select / a native cond opcode. (`apply_args` already traverses
   cond predicates in-scope; bodies are child scopes.)
3. Lower `while`: `[initial, arg, pred, body]` → initial as loop-carried inputs,
   `pred` and `body` as child programs; drive the loop **host-controlled** from
   the worker (replay body per iteration until pred is false).
4. Extend `compile_program`/`eval_program` to accept and hold child program
   handles by refcount; the recursion stays in Elixir (NIF receives built
   handles). Add `:async`/`:build` eval modes if needed for an overlapped loop.
5. Equivalence tests vs eager `EMLX.Backend` (a small `cond`, a counted
   `while`, and a carried-state `while`); flip §A control-flow boxes.

## Acceptance

- `cond` and `while` lower to child programs and replay correctly within
  tolerance vs eager `EMLX.Backend`, including loop-carried state and a
  data-dependent trip count.
- Child program handles are held by the parent program across evals (weights /
  sub-IR captured once).
- Stage-00 child-scope decision finalized and documented.
- `EXPR_NODES.md` control-flow boxes flipped; CI green.

## Results

The implementation deviates from the original procedure: rather than teaching
the NIF about child programs, control flow is resolved entirely in Elixir.
`cond` lowers inline; `while` is handled structurally via `Nx.Defn.Graph` so the
loop runs host-side while every straight-line segment stays a single-NIF
program. The compiler (`EMLX.build_eval_fn/3`) is recursive and re-enters itself
through `Nx.Defn.jit`/`Nx.Defn.Graph.run` (with `compiler: EMLX`), which makes
non-tail and nested `while`s — including `while`-as-input to later computation —
compile natively without falling back to the Evaluator.

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| child-scope mechanism finalized | ✓ | No NIF child programs. `cond` stays in the parent scope (lowered inline). `while` sub-scopes are isolated by `Nx.Defn.Graph.split/2` (split `:both` on `:while`) and each stage is recompiled by re-entering this compiler. |
| cond lowered | ✓ | Right-folded nested `:select` ops. All branches already in `node_to_ref` by the time the `:cond` node is processed. Tuple-output cond stores a list of refs; `:elem` picks from that list. |
| while lowered (Graph split + host loop) | ✓ | `build_eval_fn/3` routes: no while → flat program; bare tail while (initial carry == params) → host loop; while + surrounding work → `Graph.split` replayed by `Graph.run(…, compiler: EMLX)`. The base case compiles the condition/body via `Nx.Defn.jit` (recursing for nested whiles) and drives iterations from Elixir (`run_while_loop/3`). |
| input ordering | ✓ | Stage inputs arrive in stage-argument order, which need not match the carry/sub-scope parameter order; `build_while_base_eval_fn` reorders inputs by each `initial` parameter's position before binding the condition/body. Required for nested whiles (e.g. threefry). |
| capture backends | ✓ | `defn`-embedded constant tensors (e.g. RNG algorithm constants) are traced on the default backend; `compile_native_program/3` copies any non-EMLX capture onto the device before `to_wire`. |
| validated against | ✓ | User `cond`/`while` equivalence tests plus Nx threefry RNG (`Nx.Random.uniform`/`normal`) which is a nested `while`-as-input — now native, previously Evaluator fallback. No C++ changes required. |
