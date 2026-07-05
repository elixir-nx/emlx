defmodule EMLX.Defn.Tree do
  @moduledoc """
  Topological traversal of `Nx.Defn.Expr` DAGs.

  This module is an upstream candidate for `Nx.Defn.Tree`; the only change
  needed to upstream it is a rename. It has **zero EMLX dependencies** — it
  only uses `Nx.Defn.{Tree, Composite}` and `Nx.Tensor`. Callers needing
  compiler-specific scope handling (e.g. EMLX's `:__EMLX__` metadata nodes,
  see `EMLX.Native.Expr.scope_dependencies/1`) inject it via `post_order/2`'s
  `scope_dependencies` argument instead of this module knowing about it.

  ## Scope model

  An `Nx.Defn.Expr` DAG may contain scope boundaries: `while`/`fun`/`block`
  nodes each introduce an inner scope whose sub-graphs must not be mixed with
  the parent ordering. `cond` clauses share the parent scope and are traversed
  normally.

  `post_order/2` respects these boundaries: `while`, `fun`, and `block` nodes
  are treated as **opaque leaves** in the parent ordering — their inner-scope
  sub-graphs do not appear in the result.
  """

  alias Nx.Defn.Composite
  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T

  @doc """
  Returns the `%Nx.Tensor{}` nodes reachable from `output` in post-order.

  `output` may be any `Nx.Container.t()` (a tensor, a tuple, or any struct
  implementing `Nx.Container`). The result is a flat list of the same
  `%Nx.Tensor{}` structs that make up the DAG (no rewriting), ordered so that
  every node's same-scope operands appear at a strictly smaller index than the
  node itself.

  Each node appears **exactly once**, deduplicated by `node.data.id`.

  `scope_dependencies` lets a caller override which nodes count as a given node's
  same-scope dependencies, without this module needing to know about
  compiler-specific node shapes. For each visited node it is called as
  `scope_dependencies.(node)` and must return either:

    * `{:ok, deps}` — treat `deps` (a list of `%Nx.Tensor{}`) as the node's
      *only* same-scope dependencies, recursing into them instead of the
      node's own `args`. Use this to redirect traversal away from a node's
      literal `args` (e.g. a node wrapping an inner sub-expression that
      should stay out of the ordering entirely).
    * `:default` — fall back to the default traversal below.

  ## Sub-scope handling

  `while`, `fun`, and `block` nodes are returned **opaque**: their inner-scope
  sub-graphs are not traversed and do not appear in the result. The caller is
  responsible for recurring into sub-scopes as needed. For each such node, the
  relevant inner scope can be accessed from its `args`:

    * `:while` — `args = [initial, arg, condition, body]`; `arg`, `condition`,
      and `body` belong to the while scope. Call `post_order/2` on `condition`
      or `body` to obtain their orderings independently.
    * `:fun` — `args = [params, body, mfa]`; `body` is the function scope.
    * `:block` — `args = [struct, in_args, default_expr, callback]`;
      `default_expr` is the block's inner scope.
  """
  def post_order(output, scope_dependencies \\ fn _node -> :default end) do
    roots = Composite.flatten_list([output])
    {_visited, rev} = Enum.reduce(roots, {MapSet.new(), []}, &visit(&1, &2, scope_dependencies))
    Enum.reverse(rev)
  end

  # --- recursive post-order DFS ---

  defp visit(%T{data: %Nx.Defn.Expr{id: id}} = node, {visited, output}, scope_dependencies) do
    if MapSet.member?(visited, id) do
      {visited, output}
    else
      # Mark visited before recursing to handle shared subexpressions correctly.
      visited = MapSet.put(visited, id)
      {visited, output} = visit_scope_deps(node, {visited, output}, scope_dependencies)
      {visited, [node | output]}
    end
  end

  # fun's args[0] contains inner-scope parameter templates — no parent-scope deps.
  defp visit_scope_deps(%T{data: %Nx.Defn.Expr{op: :fun}}, acc, _scope_dependencies), do: acc

  defp visit_scope_deps(node, acc, scope_dependencies) do
    case scope_dependencies.(node) do
      {:ok, deps} ->
        Enum.reduce(deps, acc, &visit(&1, &2, scope_dependencies))

      :default ->
        # For all other ops (including while and block, which expose parent-scope
        # deps via apply_args :scope), recurse into same-scope operands before emitting.
        {_, acc} =
          Tree.apply_args(node, :scope, acc, fn dep, a ->
            {dep, visit(dep, a, scope_dependencies)}
          end)

        acc
    end
  end
end
