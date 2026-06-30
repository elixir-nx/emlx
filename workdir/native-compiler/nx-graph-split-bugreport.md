# Bug report — `Nx.Defn.Graph.split/run` mishandles shared DAGs, `runtime_call` operands, and non-tuple final outputs

**Component:** `Nx.Defn.Graph` (`lib/nx/defn/graph.ex`)
**Affected APIs:** `Nx.Defn.Graph.split/2,3`, `Nx.Defn.Graph.run/3`
**Severity:** high — hangs and silently-malformed stages on real graphs
**Found via:** compiling a Qwen3 generation graph through a `Graph.split`-based
custom compiler (a `while`-chain native compiler). All three bugs are in nx
itself and are compiler-agnostic; a custom compiler is only needed to *exercise*
`Graph.split` on a large, shared graph.

Three independent bugs, all surfaced by the same workload. They are described
separately because each can be reproduced and fixed on its own.

---

## Bug 1 — `rewrite_subtree` is exponential on shared DAGs (hang)

### Symptom
`Nx.Defn.Graph.split/2` never returns (CPU-bound, ~10⁹ reductions) on a graph
with heavy structural sharing — e.g. a multi-layer transformer, or any graph
where one node feeds many consumers. The hang is inside the second rewrite pass,
in `rewrite_subtree/3` → `composite_rewrite_subtree/3`.

### Root cause
After a split, the post-split subgraph is rewritten by `rewrite_subtree/3`, which
recurses into each node's args. It has **no memoization across shared edges**, so
a node reachable by *k* distinct paths is rewritten *k* times. For a chain of
doublings (`add(x, x)` repeated *n* times) that is `O(2ⁿ)`. `defn` expression
graphs are DAGs, not trees, so this blows up on any realistic model.

### Minimal repro
```elixir
# A DAG that is n nodes but 2^n tree-paths.
shared = Enum.reduce(1..28, Nx.template({2}, :f32), fn _, acc -> Nx.add(acc, acc) end)
# Build an expr with a split point before `shared` (e.g. a `while`) and call
# Nx.Defn.Graph.split/2 — it hangs without memoization, returns instantly with.
```

### Fix
Memoize `rewrite_subtree/3` per node `id` within a single rewrite pass. The
rewrite is pure given a fixed `nodes_to_replace` (constant within one pass), and
`used_args` is id-keyed, so collecting a shared node's parameters once is
sufficient. Implemented by splitting `rewrite_subtree` into a memoizing wrapper +
`do_rewrite_subtree/3` clauses, threading a `cache` map through
`composite_rewrite_subtree`'s accumulator.

```elixir
defp composite_rewrite_subtree(container, state, acc \\ %{used_args: %{}, cache: %{}})

defp rewrite_subtree(%T{data: %Expr{id: id}} = expr, state, acc) do
  case acc.cache do
    %{^id => cached} -> {cached, acc}
    _ ->
      {res, acc} = do_rewrite_subtree(expr, state, acc)
      {res, %{acc | cache: Map.put(acc.cache, id, res)}}
  end
end

defp rewrite_subtree(other, state, acc), do: do_rewrite_subtree(other, state, acc)

# existing clauses renamed rewrite_subtree -> do_rewrite_subtree
```

---

## Bug 2 — `runtime_call` operands are dropped during rewrite (param-index overflow / crash)

### Symptom
When a `runtime_call` node sits in a stage that gets split (e.g. its result feeds
a `while` carry), the resulting stage is **malformed**: its argument list is
shorter than the highest `:parameter` index its expression references. Downstream
this manifests as either:
- `Enum.OutOfBoundsError` (e.g. "position 308 … 17-element enumerable") if the
  stage falls back to `Nx.Defn.Evaluator`, or
- `KeyError` on the param-index map if a native compiler consumes the stage.

### Root cause
A `:runtime_call` node stores its operands as an **Nx container** (typically a
tuple) inside `args` — `args = [tensor_expr, callback, out, opts]`. The generic
`rewrite_subtree` clause walks `args` with the list handler, which only recurses
into `%T{}` elements and *skips other container elements* (the operand tuple).
The operands' `:parameter` nodes are therefore never collected into `used_args`,
so `arg_remapping` under-counts the stage's inputs while the expression still
references the (un-remapped) higher indices. This mirrors the special-casing that
already exists in `Nx.Defn.Tree.apply_args/4` for `:runtime_call`, which the
splitter was missing.

### Minimal repro
A `runtime_call` whose operands are a tuple, feeding a split point:
```elixir
# pseudo: out = runtime_call({x, weight}, cb); carry = {out + k, k}; while(...)
# split before the while -> before-stage drops x/weight params -> malformed stage
```
(In our setup this is `EMLX.Fast.rms_norm/3`, which lowers to
`Nx.runtime_call(out, {x, weight}, [eps: eps], &cb/2)`.)

### Fix
Add a dedicated `:runtime_call` clause that traverses the operand container and
leaves `callback`/`out`/`opts` opaque:

```elixir
defp do_rewrite_subtree(
       %T{data: %Expr{op: :runtime_call, id: id, args: [tensor_expr, callback, out, opts]}} = expr,
       state,
       acc
     ) do
  case state.nodes_to_replace do
    %{^id => res} -> {res, put_in(acc.used_args[id], res)}
    _ ->
      {tensor_expr, acc} = composite_rewrite_subtree(tensor_expr, state, acc)
      {put_in(expr.data.args, [tensor_expr, callback, out, opts]), acc}
  end
end
```

---

## Bug 3 — `Graph.run/3` cannot return a non-tuple (map/struct) final output

### Symptom
`Nx.Defn.Graph.run/3` raises when the chain's final stage returns a **map** (or
any non-tuple container) — e.g. a generation defn returning
`%{token_ids: ..., length: ...}`. The `tuple` branch tries to `Tuple.to_list/1`
the map.

### Root cause
The stage-output handling in `run/3` only matched `%T{}` (single tensor) and
`tuple` (assumed `is_tuple`). Intermediate stages are always tuples of tensors,
but the **final** stage's output is the chain's return value and can be an
arbitrary container.

### Fix
Guard the tuple clause with `is_tuple/1` and add a pass-through clause for other
containers (only ever the final stage, which has no downstream consumers):

```elixir
case Nx.Defn.jit_apply(fn _ -> expr end, [List.to_tuple(args)], opts) do
  %T{} = tensor ->
    {tensor, Map.put(scope, {id, 0}, tensor)}

  tuple when is_tuple(tuple) ->
    # ... existing index-into-scope logic ...

  other ->
    {other, scope}
end
```

---

## Suggested upstream test coverage
- A `split` over a doubling-chain DAG (Bug 1) — assert it completes (timeout-guarded).
- A `split` with a `runtime_call` whose operands are a tuple feeding a split
  point (Bug 2) — assert the produced stages' arg counts cover all referenced
  param indices, and that `run/3` matches the unsplit Evaluator result.
- A chain whose final stage returns a map container (Bug 3) — assert `run/3`
  returns it.

## Status — resolved upstream
Fixed in the nx fork in commits `7290b7fa` / `1316bb74` (`fix: keep subscopes
hermetic`) and `631afbf5` (`fix: handle generic containers`). The landed fix is
broader than the three bugs above: it also makes `while`/`cond`/`fun` sub-scopes
hermetic in `Graph.split` (dedicated `eval_while`/`eval_cond`/`eval_fun` traversal
+ a `force_none` flag so conditionally-executed computation is never hoisted out
of a `cond` branch and sub-scope parameter indices never leak into the parent
scope). Bug 2 (runtime_call operands) was one surface symptom of that missing
hermeticity. This report is retained as the historical root-cause analysis.
