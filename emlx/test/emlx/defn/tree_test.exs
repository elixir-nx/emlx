defmodule EMLX.Defn.TreeTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias EMLX.Defn.Tree

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  # A defn function that produces a :while node; used to verify scope-boundary.
  defn count_up(n) do
    while n, Nx.less(n, 10) do
      Nx.add(n, 1)
    end
  end

  # Verifies the core post-order invariant:
  # For every node in `order`, each of its same-scope tensor operands (if
  # present in the ordering) must appear at a strictly smaller index.
  #
  # Deps that are *absent* from the index are inner-scope nodes (e.g. fun
  # parameter templates) and are expected — they are silently skipped.
  defp assert_post_order(order) do
    id_to_idx = Map.new(Enum.with_index(order), fn {t, i} -> {t.data.id, i} end)

    for {node, node_idx} <- Enum.with_index(order) do
      Nx.Defn.Tree.apply_args(node, :scope, :ok, fn dep, :ok ->
        case Map.fetch(id_to_idx, dep.data.id) do
          {:ok, dep_idx} ->
            assert dep_idx < node_idx,
                   "#{inspect(dep.data.op)} (idx #{dep_idx}) is not before " <>
                     "its consumer #{inspect(node.data.op)} (idx #{node_idx})"

          :error ->
            # inner-scope dep (e.g. fun parameter template) — not in order, expected
            :ok
        end

        {dep, :ok}
      end)
    end
  end

  describe "post_order/1" do
    test "linear chain: nodes appear in dependency-first order" do
      expr =
        Nx.Defn.debug_expr_apply(
          fn x ->
            y = Nx.negate(x)
            Nx.abs(y)
          end,
          [Nx.tensor(1.0)]
        )

      order = Tree.post_order(expr)

      assert length(order) == 3
      ops = Enum.map(order, & &1.data.op)
      assert [:parameter, :negate, :abs] == ops
      assert_post_order(order)
    end

    test "diamond: shared subexpression appears exactly once, after its operands" do
      expr =
        Nx.Defn.debug_expr_apply(
          fn x ->
            y = Nx.negate(x)
            # y is shared: should appear once
            Nx.add(y, y)
          end,
          [Nx.tensor(1.0)]
        )

      order = Tree.post_order(expr)

      assert length(order) == 3
      ops = Enum.map(order, & &1.data.op)
      assert [:parameter, :negate, :add] == ops
      assert_post_order(order)
    end

    test "multi-output container: all roots are collected, shared param appears once" do
      {e1, e2} =
        Nx.Defn.debug_expr_apply(
          fn x -> {Nx.negate(x), Nx.abs(x)} end,
          [Nx.tensor(1.0)]
        )

      order = Tree.post_order({e1, e2})
      ops = Enum.map(order, & &1.data.op)
      assert [:parameter, :negate, :abs] == ops
      assert_post_order(order)

      # reordering the output tuple should cause the order to change
      order = Tree.post_order({e2, e1})
      ops = Enum.map(order, & &1.data.op)
      assert [:parameter, :abs, :negate] == ops
      assert_post_order(order)
    end

    test "constant and parameter leaves appear before their consumers" do
      expr =
        Nx.Defn.debug_expr_apply(
          fn x -> Nx.add(x, Nx.tensor(1.0)) end,
          [Nx.tensor(2.0)]
        )

      order = Tree.post_order(expr)

      # It doesn't really matter if constants appear closer to invocation
      # than parameters as they are decoupled, so we don't assert on them here.
      ops = Enum.map(order, & &1.data.op)
      assert :parameter in ops
      assert :constant in ops
      # add must be last
      assert Enum.find_index(order, &(&1.data.op == :add)) == 2
      assert_post_order(order)
    end

    test "while node is opaque: appears as a single node; inner body does not leak" do
      expr = Nx.Defn.debug_expr_apply(&count_up/1, [Nx.tensor(0)])

      order = Tree.post_order(expr)

      # There is exactly one while node
      assert {while, not_while} = Enum.split_with(order, &(&1.data.op == :while))

      assert [%T{data: %Expr{op: :while, args: [_, _, pred, body]}}] = while

      # Inner-body ops (:less, :add at inner scope and hoisted :constant) must NOT appear
      not_while = Enum.map(not_while, & &1.data.op)
      assert [:parameter] == not_while, "inner-scope nodes leaked into parent ordering: #{inspect(not_while)}"

      assert_post_order(order)

      pred_order = Tree.post_order(pred)
      assert [:parameter, :constant, :less] == Enum.map(pred_order, & &1.data.op)
      assert_post_order(pred_order)

      body_order = Tree.post_order(body)
      assert [:constant, :parameter, :add] == Enum.map(body_order, & &1.data.op)
      assert_post_order(body_order)
    end

    test "post-order invariant holds on a diamond through a shared interior node" do
      expr =
        Nx.Defn.debug_expr_apply(
          fn x, y ->
            a = Nx.add(x, y)
            b = Nx.multiply(a, x)
            c = Nx.subtract(b, y)
            # 'a' is shared between b and this add
            Nx.add(a, c)
          end,
          [Nx.tensor(1.0), Nx.tensor(2.0)]
        )

      order = Tree.post_order(expr)

      # :multiply and :subtract could appear in different others, this doesn't really impact the test
      assert [:parameter, :parameter, :add, :multiply, :subtract, :add] == Enum.map(order, & &1.data.op)

      # Every operand must precede its consumer
      assert_post_order(order)
      # Sanity-check that both :parameter nodes are distinct as well as  both :add nodes
      ids = Enum.map(order, & &1.data.id)
      assert ids == Enum.uniq(ids)
    end
  end
end
