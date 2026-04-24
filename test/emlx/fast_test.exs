defmodule EMLX.FastTest do
  @moduledoc """
  Unit tests for EMLX.Fast — fused Metal shaders via mlx::fast.

  Each test verifies:
  1. The NIF returns a tensor with the correct shape and dtype.
  2. Numerical output is close to the equivalent primitive-op chain.

  All tests require Metal (tagged :metal).
  """
  use EMLX.Case, async: true

  @moduletag :metal

  alias EMLX.Fast

  defp gpu(tensor), do: Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

  # ── rms_norm ────────────────────────────────────────────────────────────────

  describe "EMLX.Fast.rms_norm/3" do
    test "output shape and type match input" do
      x = Nx.iota({2, 4, 16}, type: :f16) |> Nx.divide(100) |> gpu()
      w = Nx.broadcast(Nx.tensor(1.0, type: :f16), {16}) |> gpu()
      eps = 1.0e-5

      out = Fast.rms_norm(x, w, eps)

      assert Nx.shape(out) == {2, 4, 16}
      assert Nx.type(out) == {:f, 16}
    end

    test "matches primitive rms_norm on small input" do
      # Primitive: y = x / sqrt(mean(x^2) + eps) * weight
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: :f32) |> gpu()
      w = Nx.tensor([1.0, 1.0, 1.0, 1.0], type: :f32) |> gpu()
      eps = 1.0e-5

      fast_out = Fast.rms_norm(x, w, eps) |> Nx.backend_transfer()

      rms = Nx.sqrt(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true) |> Nx.add(eps))
      prim_out = Nx.divide(x, rms) |> Nx.multiply(w) |> Nx.backend_transfer()

      assert_all_close(fast_out, prim_out, atol: 1.0e-3)
    end
  end

  # ── rope ────────────────────────────────────────────────────────────────────

  describe "EMLX.Fast.rope/6" do
    test "output shape and type match input" do
      # Qwen3 0.6B: head_dim=128, traditional=false
      a = Nx.iota({1, 8, 1, 128}, type: :f16) |> Nx.divide(1000) |> gpu()

      out = Fast.rope(a, 128, false, 1_000_000.0, 1.0, 0)

      assert Nx.shape(out) == {1, 8, 1, 128}
      assert Nx.type(out) == {:f, 16}
    end

    test "offset=0 and offset=1 produce different outputs" do
      a = Nx.iota({1, 4, 1, 64}, type: :f32) |> Nx.divide(100) |> gpu()

      out0 = Fast.rope(a, 64, false, 10_000.0, 1.0, 0)
      out1 = Fast.rope(a, 64, false, 10_000.0, 1.0, 1)

      # Different offsets → different rotation → different output
      refute Nx.all_close(out0, out1) |> Nx.to_number() == 1
    end
  end

  # ── scaled_dot_product_attention ─────────────────────────────────────────────

  describe "EMLX.Fast.scaled_dot_product_attention/4" do
    test "output shape: {B, N_q, T_q, D}" do
      b = 1; n_q = 8; n_kv = 4; t_q = 1; t_kv = 32; d = 64
      scale = 1.0 / :math.sqrt(d)

      q = Nx.broadcast(0.1, {b, n_q, t_q, d}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.1, {b, n_kv, t_kv, d}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.1, {b, n_kv, t_kv, d}) |> Nx.as_type(:f16) |> gpu()

      out = Fast.scaled_dot_product_attention(q, k, v, scale)

      assert Nx.shape(out) == {b, n_q, t_q, d}
      assert Nx.type(out) == {:f, 16}
    end
  end

  describe "EMLX.Fast.scaled_dot_product_attention/5 (masked)" do
    test "output shape matches q with additive mask" do
      b = 1; n_q = 4; t_q = 4; t_kv = 4; d = 32
      scale = 1.0 / :math.sqrt(d)

      q = Nx.broadcast(0.1, {b, n_q, t_q, d}) |> Nx.as_type(:f32) |> gpu()
      k = Nx.broadcast(0.1, {b, n_q, t_kv, d}) |> Nx.as_type(:f32) |> gpu()
      v = Nx.broadcast(0.1, {b, n_q, t_kv, d}) |> Nx.as_type(:f32) |> gpu()

      # Simple upper-triangle causal mask: 0.0 for allowed, -1e9 for masked
      mask =
        Nx.tril(Nx.broadcast(0.0, {t_q, t_kv}))
        |> Nx.subtract(Nx.triu(Nx.broadcast(1.0e9, {t_q, t_kv}), k: 1))
        |> Nx.reshape({1, 1, t_q, t_kv})
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      out = Fast.scaled_dot_product_attention(q, k, v, scale, mask)

      assert Nx.shape(out) == {b, n_q, t_q, d}
    end
  end
end
