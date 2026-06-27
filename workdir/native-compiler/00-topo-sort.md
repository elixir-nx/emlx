# Stage 00 — `EMLX.Defn.Tree.post_order/1` (Layer A)

Status: not started

## Why this stage exists

The lowerer (Layer B) needs a **dependency-respecting linear order** of an
`Nx.Defn.Expr` DAG, restricted to one scope. `Nx.Defn.Expr` has structural
sharing (dedup by `Expr.id`), and `while`/`fun`/`block` bodies are separate
scopes with their own parameters. Nx provides `apply_args(_, :scope, _, _)`
(scope-correct traversal) and `scope_ids/1` (the in-scope *set*) but **no
ordering**. This stage produces that ordering as an isolated, upstreamable
module — the foundation every later stage builds on. It is pure Elixir with no
C++/MLX dependency, so it can land and be proven first.

## Procedure

1. Create `emlx/lib/emlx/defn/tree.ex` defining `EMLX.Defn.Tree`, with **zero
   EMLX dependencies** (only `Nx.Defn.{Tree, Composite, Expr}`, `Nx.Tensor`).
   Module doc must state it is an upstream candidate for `Nx.Defn.Tree`.
2. Implement `post_order/1`:
   - `@spec post_order(Nx.Container.t()) :: [Nx.Tensor.t()]`
   - Flatten the output container to leaves via `Nx.Defn.Composite.flatten_list/1`.
   - Iterative DFS over `Nx.Defn.Tree.apply_args(node, :scope, acc, fun)`,
     visited-set keyed by `node.data.id`, emit each node **on exit**
     (post-order) so every node appears after all its same-scope operands.
   - Return the **same `%Nx.Tensor{}` structs** received, reordered. No
     rewriting, no new node types.
   - `cond` is traversed in-scope by `apply_args`; `while`/`fun`/`block` are
     returned as opaque single nodes (their inner scopes are NOT expanded here).
3. Add `test/emlx/defn/tree_test.exs` covering:
   - linear chain; diamond (shared subexpression appears once, after operands);
   - multi-output container (tuple) ordering;
   - constants / parameters / `tensor` leaves;
   - scope boundary: a `while`/`fun` node appears as a single node and its body
     nodes do NOT leak into the parent ordering;
   - property (StreamData if convenient): for the returned order, every node's
     same-scope operands have a strictly smaller index.
4. Record the §5.5 open question outcome (minimal vs richer return shape) in the
   Results table and in README's "Decision gates".

## Acceptance

- `EMLX.Defn.Tree.post_order/1` exists in `emlx/lib/emlx/defn/tree.ex`, pure
  (no EMLX deps), returning the input `%Nx.Tensor{}` nodes in dependency-first
  order, deduped by `Expr.id`.
- Control-flow/composite nodes are returned opaque (inner-scope nodes do not
  leak into the parent ordering).
- Tests in `test/emlx/defn/tree_test.exs` pass, including the "every operand
  before its consumer" property and the scope-boundary case.
- `mix compile --warnings-as-errors` and `mix format --check-formatted` clean.
- README stage box `00-topo-sort` flipped to `[x]`; decision-gate note on the
  return shape recorded.

## Results

| Item | Outcome | Notes / artifacts |
|------|---------|-------------------|
| `post_order/1` implemented | | |
| Return shape (minimal vs richer) | | |
| Tests passing | | |
| compile/format clean | | |
