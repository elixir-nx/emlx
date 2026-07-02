defmodule EMLX.Fast.EinsumTest do
  @moduledoc """
  Tests for `EMLX.Fast.einsum/2` — the eager-only helper that dispatches
  to `mlx::core::einsum`.

  Covered:

    * Two-operand contractions (matmul, batched matmul, attention-style
      QKᵀ fold).
    * Three-operand chain — sanity check that MLX's internal path
      optimisation produces the same result as both left-to-right and
      right-to-left hand contractions.
    * Error path — a non-EMLX backend raises a clear transfer-first
      message.
  """

  use EMLX.Case, async: true

  doctest EMLX.Fast, only: [einsum: 2]

  @f32_tol 1.0e-5

  describe "two-operand contractions" do
    test "\"ij,jk->ik\" matches Nx.dot" do
      a = Nx.iota({3, 4}, backend: EMLX.Backend, type: :f32)
      b = Nx.iota({4, 5}, backend: EMLX.Backend, type: :f32)

      got = EMLX.Fast.einsum("ij,jk->ik", [a, b])
      expected = Nx.dot(a, b)

      assert Nx.shape(got) == {3, 5}
      assert_all_close(got, expected, atol: @f32_tol, rtol: @f32_tol)
    end

    test "\"bij,bjk->bik\" matches Nx.dot with explicit batch axes" do
      a = Nx.iota({2, 3, 4}, backend: EMLX.Backend, type: :f32)
      b = Nx.iota({2, 4, 5}, backend: EMLX.Backend, type: :f32)

      got = EMLX.Fast.einsum("bij,bjk->bik", [a, b])
      # Contract a's last axis with b's second-to-last; keep batch 0 as
      # a batch dim on both sides.
      expected = Nx.dot(a, [2], [0], b, [1], [0])

      assert Nx.shape(got) == {2, 3, 5}
      assert_all_close(got, expected, atol: @f32_tol, rtol: @f32_tol)
    end

    test "attention-style \"bhid,bhjd->bhij\"" do
      q = Nx.iota({2, 3, 4, 5}, backend: EMLX.Backend, type: :f32)
      k = Nx.iota({2, 3, 6, 5}, backend: EMLX.Backend, type: :f32)

      got = EMLX.Fast.einsum("bhid,bhjd->bhij", [q, k])
      # Contract d (axis 3 on both); keep b, h as batch dims on both.
      expected = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])

      assert Nx.shape(got) == {2, 3, 4, 6}
      assert_all_close(got, expected, atol: @f32_tol, rtol: @f32_tol)
    end
  end

  describe "three-operand contractions" do
    test "\"ij,jk,kl->il\" matches both hand-chosen contraction orders" do
      a = Nx.iota({2, 3}, backend: EMLX.Backend, type: :f32)
      b = Nx.iota({3, 4}, backend: EMLX.Backend, type: :f32)
      c = Nx.iota({4, 5}, backend: EMLX.Backend, type: :f32)

      got = EMLX.Fast.einsum("ij,jk,kl->il", [a, b, c])

      left_first = Nx.dot(Nx.dot(a, b), c)
      right_first = Nx.dot(a, Nx.dot(b, c))

      assert Nx.shape(got) == {2, 5}
      # Associativity means both orders are mathematically identical;
      # any FP reordering across MLX's chosen path is still within
      # f32 tolerance on these small shapes.
      assert_all_close(got, left_first, atol: @f32_tol, rtol: @f32_tol)
      assert_all_close(got, right_first, atol: @f32_tol, rtol: @f32_tol)
    end
  end

  describe "error paths" do
    test "raises a transfer-first ArgumentError on non-EMLX operands" do
      a = Nx.iota({3, 4}, backend: Nx.BinaryBackend, type: :f32)
      b = Nx.iota({4, 5}, backend: EMLX.Backend, type: :f32)

      assert_raise ArgumentError, ~r/Nx\.BinaryBackend.*Nx\.backend_transfer/s, fn ->
        EMLX.Fast.einsum("ij,jk->ik", [a, b])
      end
    end
  end
end
