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

  describe "EMLX.Fast.rope_with_freqs/6" do
    test "T>1 prefill matches per-token T=1 stack (sequential position_ids)" do
      dims = 8
      half = div(dims, 2)

      freqs =
        Nx.iota({half}, type: :f32) |> Nx.add(0.1) |> Nx.divide(20) |> gpu()

      a = Nx.iota({1, 2, 1, dims}, type: :f32) |> Nx.divide(20) |> gpu()
      pos = Nx.tensor([[0, 1]], type: :s32) |> gpu()

      out2 = Fast.rope_with_freqs(a, pos, dims, false, 1.0, freqs) |> Nx.as_type(:f32)

      a0 = a[[0..0, 0..0, .., ..]]
      a1 = a[[0..0, 1..1, .., ..]]

      o0 =
        Fast.rope_with_freqs(a0, Nx.tensor([[0]], type: :s32) |> gpu(), dims, false, 1.0, freqs)

      o1 =
        Fast.rope_with_freqs(a1, Nx.tensor([[1]], type: :s32) |> gpu(), dims, false, 1.0, freqs)

      expected = Nx.concatenate([o0, o1], axis: 1) |> Nx.as_type(:f32) |> Nx.backend_transfer()
      out2 = out2 |> Nx.backend_transfer()

      assert_all_close(out2, expected, atol: 1.0e-4, rtol: 1.0e-4)
    end

    test "T>1 non-sequential position_ids: second token position is honored" do
      dims = 8
      half = div(dims, 2)

      freqs =
        Nx.iota({half}, type: :f32) |> Nx.add(0.1) |> Nx.divide(20) |> gpu()

      a = Nx.iota({1, 2, 1, dims}, type: :f32) |> Nx.divide(20) |> gpu()
      # First token 0, second 5 (not offset+1 from first)
      pos2 = Nx.tensor([[0, 5]], type: :s32) |> gpu()
      out2 = Fast.rope_with_freqs(a, pos2, dims, false, 1.0, freqs) |> Nx.as_type(:f32)

      a0 = a[[0..0, 0..0, .., ..]]
      a1 = a[[0..0, 1..1, .., ..]]

      o0 =
        Fast.rope_with_freqs(a0, Nx.tensor([[0]], type: :s32) |> gpu(), dims, false, 1.0, freqs)

      o1 =
        Fast.rope_with_freqs(a1, Nx.tensor([[5]], type: :s32) |> gpu(), dims, false, 1.0, freqs)

      expected = Nx.concatenate([o0, o1], axis: 1) |> Nx.as_type(:f32) |> Nx.backend_transfer()
      out2 = out2 |> Nx.backend_transfer()

      assert_all_close(out2, expected, atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  describe "EMLX.Fast.rope_with_positions/6" do
    test "high-base decode (T=1) matches per-token RoPE formula" do
      dims = 8
      base = 1_000_000.0
      a = Nx.iota({1, 1, 1, dims}, type: :f32) |> Nx.divide(50) |> gpu()
      pos = Nx.tensor([[7]], type: :s32) |> gpu()

      out = Fast.rope_with_positions(a, pos, dims, false, base, 1.0) |> Nx.backend_transfer()

      half = div(dims, 2)
      i = Nx.iota({half}, type: :f32) |> Nx.multiply(2) |> Nx.divide(dims)
      inv = Nx.divide(1.0, Nx.pow(Nx.tensor(base, type: :f32), i))
      angles = Nx.multiply(Nx.as_type(pos, :f32) |> Nx.new_axis(-1), inv)
      cos = angles |> Nx.cos() |> Nx.new_axis(-2)
      sin = angles |> Nx.sin() |> Nx.new_axis(-2)
      cos_full = Nx.concatenate([cos, cos], axis: -1)
      sin_full = Nx.concatenate([sin, sin], axis: -1)
      x1 = a[[.., .., .., 0..(half - 1)//1]]
      x2 = a[[.., .., .., half..(dims - 1)//1]]
      rotated = Nx.concatenate([Nx.negate(x2), x1], axis: -1)

      expected =
        Nx.add(Nx.multiply(a, cos_full), Nx.multiply(rotated, sin_full)) |> Nx.backend_transfer()

      assert_all_close(out, expected, atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  # ── scaled_dot_product_attention ─────────────────────────────────────────────

  describe "EMLX.Fast.scaled_dot_product_attention/4" do
    test "output shape: {B, N_q, T_q, D}" do
      b = 1
      n_q = 8
      n_kv = 4
      t_q = 1
      t_kv = 32
      d = 64
      scale = 1.0 / :math.sqrt(d)

      q = Nx.broadcast(0.1, {b, n_q, t_q, d}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.1, {b, n_kv, t_kv, d}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.1, {b, n_kv, t_kv, d}) |> Nx.as_type(:f16) |> gpu()

      out = Fast.scaled_dot_product_attention(q, k, v, scale)

      assert Nx.shape(out) == {b, n_q, t_q, d}
      assert Nx.type(out) == {:f, 16}
    end
  end

  # ── swiglu ──────────────────────────────────────────────────────────────────

  describe "EMLX.Fast.swiglu/2" do
    test "output shape and type match gate input" do
      gate = Nx.iota({1, 1, 64}, type: :f16) |> Nx.divide(100) |> gpu()
      up = Nx.iota({1, 1, 64}, type: :f16) |> Nx.divide(100) |> gpu()

      out = Fast.swiglu(gate, up)

      assert Nx.shape(out) == {1, 1, 64}
      assert Nx.type(out) == {:f, 16}
    end

    test "matches silu(gate) * up on f32 inputs" do
      gate = Nx.tensor([[[1.0, -1.0, 2.0, 0.5]]], type: :f32) |> gpu()
      up = Nx.tensor([[[2.0, 3.0, 1.0, 4.0]]], type: :f32) |> gpu()

      fast_out = Fast.swiglu(gate, up) |> Nx.backend_transfer()
      # silu(x) = x * sigmoid(x)
      prim_out = Nx.multiply(Nx.multiply(gate, Nx.sigmoid(gate)), up) |> Nx.backend_transfer()

      assert_all_close(fast_out, prim_out, atol: 1.0e-5)
    end

    test "matches silu(gate) * up on Qwen3 FFN decode shape (f16)" do
      # Qwen3-0.6B: ffn_intermediate_size = 1536 (ffn_size / 2 for gated FFN)
      gate = Nx.iota({1, 1, 1536}, type: :f16) |> Nx.divide(1000) |> gpu()
      up = Nx.iota({1, 1, 1536}, type: :f16) |> Nx.divide(1000) |> gpu()

      fast_out = Fast.swiglu(gate, up) |> Nx.as_type(:f32) |> Nx.backend_transfer()

      prim_out =
        Nx.multiply(Nx.multiply(gate, Nx.sigmoid(gate)), up)
        |> Nx.as_type(:f32)
        |> Nx.backend_transfer()

      assert_all_close(fast_out, prim_out, atol: 1.0e-2)
    end
  end

  describe "EMLX.Fast.scaled_dot_product_attention/5 (masked)" do
    test "output shape matches q with additive mask" do
      b = 1
      n_q = 4
      t_q = 4
      t_kv = 4
      d = 32
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

  # ── GQA SDPA (mismatched Q/KV heads) ─────────────────────────────────────────

  describe "EMLX.Fast.scaled_dot_product_attention_causal_key_masked/5 (GQA)" do
    @tag :metal
    test "GQA: Q {1,16,1,64} × K/V {1,8,10,64} produces {1,16,1,64}" do
      # Qwen3-0.6B: 16 query heads, 8 key/value heads (GQA groups=2).
      q = Nx.broadcast(0.1, {1, 16, 1, 64}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.1, {1, 8, 10, 64}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.1, {1, 8, 10, 64}) |> Nx.as_type(:f16) |> gpu()
      scale = 1.0 / :math.sqrt(64)
      key_mask = Nx.broadcast(1, {1, 10}) |> gpu()

      out = Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask)

      assert Nx.shape(out) == {1, 16, 1, 64}
      assert Nx.type(out) == {:f, 16}
    end
  end
end
