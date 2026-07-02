defmodule EMLX.GradTriageTest do
  @moduledoc """
  Stage 23 (gradient-training-parity, scoping-only): triage of
  `Nx.Defn.grad`-wrapped functions run through `compiler: EMLX`, oracled
  against `Nx.BinaryBackend` via `compiler: Nx.Defn.Evaluator`.

  This is the triage instrument the stage doc's Procedure calls for — its
  pass/fail results feed the Results table in
  `workdir/native-compiler/23-gradient-training-parity.md`, not a permanent
  regression suite for a shipped feature. Scenarios that pass stay here as
  regression coverage; scenarios that fail are the seeds for follow-on
  stages (27+).
  """
  use EMLX.Case, async: false
  import Nx.Defn

  # ── forward functions (the zoo) ───────────────────────────────────────────

  defn elementwise_loss(x), do: Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x)))
  defn elementwise_grad(x), do: grad(x, &elementwise_loss/1)

  defn reduction_loss(x), do: Nx.mean(Nx.sum(x, axes: [1]))
  defn reduction_grad(x), do: grad(x, &reduction_loss/1)

  defn dot_loss(a, b), do: Nx.sum(Nx.dot(a, b))
  defn dot_grad(a, b), do: grad(a, fn a -> dot_loss(a, b) end)

  defn cond_loss(x) do
    cond do
      Nx.all(Nx.greater(x, 0)) -> Nx.sum(Nx.multiply(x, x))
      true -> Nx.sum(Nx.abs(x))
    end
  end

  defn cond_grad(x), do: grad(x, &cond_loss/1)

  defn while_loss(x) do
    {out, _i} =
      while {out = x, i = 0}, Nx.less(i, 3) do
        {Nx.multiply(out, out), Nx.add(i, 1)}
      end

    Nx.sum(out)
  end

  defn while_grad(x), do: grad(x, &while_loss/1)

  defn window_loss(x), do: Nx.sum(Nx.window_sum(x, {2, 2}))
  defn window_grad(x), do: grad(x, &window_loss/1)

  # window_max's backward hits :window_scatter_max (not just window_sum
  # again, unlike window_sum's own backward) — a distinct opcode, tested
  # separately per the Stage 20 finding this stage's doc calls out.
  defn window_max_loss(x), do: Nx.sum(Nx.window_max(x, {2, 2}))
  defn window_max_grad(x), do: grad(x, &window_max_loss/1)

  # ── oracle helper ──────────────────────────────────────────────────────────

  # Oracle runs the same defn through the plain Evaluator on BinaryBackend
  # tensors (no EMLX involved at all) — independent of whatever the process
  # default backend/compiler happens to be.
  defp oracle(fun, args) do
    Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)
  end

  defp native(fun, args) do
    Nx.Defn.jit_apply(fun, args, compiler: EMLX)
  end

  defp bin(list_or_tensor, opts \\ []) do
    Nx.tensor(list_or_tensor, [backend: Nx.BinaryBackend] ++ opts)
  end

  describe "elementwise grad" do
    test "matches the Evaluator oracle under compiler: EMLX" do
      x = bin([0.1, 0.5, -0.3, 1.2])

      assert_all_close(
        native(&elementwise_grad/1, [x]),
        oracle(&elementwise_grad/1, [x])
      )
    end
  end

  describe "reduction grad" do
    test "matches the Evaluator oracle under compiler: EMLX" do
      x = bin([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      assert_all_close(
        native(&reduction_grad/1, [x]),
        oracle(&reduction_grad/1, [x])
      )
    end
  end

  describe "dot grad" do
    test "matches the Evaluator oracle under compiler: EMLX" do
      a = bin([[1.0, 2.0], [3.0, 4.0]])
      b = bin([[5.0, 6.0], [7.0, 8.0]])

      assert_all_close(
        native(&dot_grad/2, [a, b]),
        oracle(&dot_grad/2, [a, b])
      )
    end
  end

  describe "cond grad" do
    test "matches the Evaluator oracle under compiler: EMLX (branch taken: true)" do
      x = bin([1.0, 2.0, 3.0])

      assert_all_close(
        native(&cond_grad/1, [x]),
        oracle(&cond_grad/1, [x])
      )
    end

    test "matches the Evaluator oracle under compiler: EMLX (branch taken: false)" do
      x = bin([-1.0, -2.0, 3.0])

      assert_all_close(
        native(&cond_grad/1, [x]),
        oracle(&cond_grad/1, [x])
      )
    end
  end

  describe "while grad (backward pass builds its own :while node — Nx.Defn.Graph splits it same as a forward while)" do
    test "matches the Evaluator oracle under compiler: EMLX" do
      x = bin([0.5, 0.6, 0.7])

      assert_all_close(
        native(&while_grad/1, [x]),
        oracle(&while_grad/1, [x])
      )
    end
  end

  describe "windowed-reduce grad (eager EMLX.Backend.window_reduce/6 hard-raises; window_sum forward is native-only in the compiler's IR — Stage 20 finding)" do
    test "window_sum grad matches the Evaluator oracle under compiler: EMLX" do
      x = bin([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

      assert_all_close(
        native(&window_grad/1, [x]),
        oracle(&window_grad/1, [x])
      )
    end

    test "window_max grad (backward hits :window_scatter_max) matches the Evaluator oracle under compiler: EMLX" do
      x = bin([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

      assert_all_close(
        native(&window_max_grad/1, [x]),
        oracle(&window_max_grad/1, [x])
      )
    end
  end
end
