# Stage 11 — Investigation: `validate_qwen3` benchmark regression

Status: in progress — **BLOCKING**. Do this before Stage 12+. The end-to-end
benchmark is the perf oracle for the whole compiler effort (README decision
gate "After 01"); while it is broken we cannot validate any further stage.

## Symptom

`emlx_axon/bench/validate_qwen3.exs` stopped working "after the last couple
commits". Two faces observed (may be one root cause):

- **A — hang:** `bb base` (stock Bumblebee graph, `defn_options: [compiler:
  EMLX]`, no `EMLXAxon.rewrite`) hangs indefinitely on the first warmup run.
- **B — crash:** with `bb base` commented out, `bb+rewrite` warmup raises on the
  **Evaluator fallback** path:

  ```
  ** (Enum.OutOfBoundsError) out of bounds error at position 308 when
     traversing enumerable [ …17 × Nx.LazyContainer.Nx.Tensor.traverse/3… ]
      (elixir) lib/enum.ex:1080: Enum.fetch!/2
      (nx 0.12.1) lib/nx/defn/evaluator.ex:282: Nx.Defn.Evaluator.eval_apply/5
      (nx 0.12.1) lib/nx/defn/evaluator.ex:234: Nx.Defn.Evaluator.eval/3
  ```

  Position 308 looked up in a 17-element arg list ⇒ a **parameter-index vs
  args-length mismatch**: a node carries a `:parameter` index from a different
  scope than the args it is being applied to. It surfaces inside
  `Nx.Defn.Evaluator`, i.e. native lowering raised `does not yet lower op` and
  the seam fell back to the whole-defn Evaluator (`emlx.ex` `try_native_compile/3`
  rescue, ~line 1394), which then itself fails.

(Ignore the param **shape-mismatch** warnings printed earlier in the run —
`expected {1024, 2048}, got {128, 2048}` etc. Those are the MLX-4bit packed/
quantized shapes from param loading and predate the regression; rule them out
but they are almost certainly not the cause.)

## Likely culprits (last two commits)

- `ababb5f feat: while graph chain compilation and control flow` — **+298 lines
  in `emlx.ex`**: `build_eval_fn/3` routing (`bare_while?`/`contains_while?`),
  `build_while_chain_eval_fn`, `build_while_base_eval_fn`, `run_while_loop`,
  `Nx.Defn.Graph.split` + `Graph.run(compiler: EMLX)`, and the
  input-reordering-by-parameter-position logic.
- `6fe0d47 block lowering` — +53 in `emlx.ex`, +279 in `expr.ex`: block
  recognize-struct + `expand_block_via_default` descent.

Symptom A (hang) smells like a non-terminating host loop in `run_while_loop`
(predicate never goes false) or a `Graph.split` stage that re-enters the
compiler without making progress. Symptom B (param-index mismatch) smells like
the wrong `vars`/scope being threaded into a sub-expression (while body / block
`default_expr` / `fun`) or into the Evaluator fallback.

## Procedure

1. **Reproduce + classify.** Run the bench with `bb base` only, then
   `bb+rewrite` only. For each, log in `build_eval_fn/3` which branch is taken
   (flat / bare-while / while-chain) and instrument the `try_native_compile/3`
   rescue to print the op that raised `does not yet lower op` (so we know
   whether/why the fallback fires).
2. **Bisect against the suspects.** Stash the working-tree edits, then run the
   bench at `1936e76` (the commit before both suspects) to confirm it worked,
   then at `ababb5f`, then `6fe0d47`, to localize the break to a single commit.
3. **Hang (A).** Instrument `run_while_loop`: confirm termination, trip count,
   and the input-reorder-by-`initial`-parameter-position step. Verify a bare
   tail-`while` base case actually advances its carry each iteration.
4. **Crash (B).** Confirm independently that `compiler: Nx.Defn.Evaluator` (no
   EMLX) runs this exact model to isolate compiler-seam vs model/graph. Then
   trace the position-308 node: which scope's `:parameter` index 308 is it, and
   which 17-element arg list is it indexed against. Check whether the fallback
   hands `Nx.Defn.Evaluator.__compile__/4` the correct `key`/`vars`/`fun`.
5. **Fix + guard.** Land the fix at the identified seam. Add a CI-sized
   regression test reproducing the failing routing path (e.g. a `while` +
   surrounding work defn for A, and a defn that forces the Evaluator fallback
   for B) so neither symptom can silently return.

## Acceptance

- `validate_qwen3.exs` runs end-to-end again for `bb base`, `bb+rewrite`, and
  `native` (no hang, no crash); numbers recorded.
- Root cause documented here (which commit, which seam, why).
- Regression test(s) added; full native + EMLX suites green.
