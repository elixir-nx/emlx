# Stage 08 — Control flow (`cond`, `while`)

Status: not started

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

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| child-scope mechanism finalized | | |
| cond lowered | | |
| while lowered (host loop) | | |
| child-program lifetime correct | | |
