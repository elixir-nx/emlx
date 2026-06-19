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

  describe "EMLX.kv_cache_sdpa_update/7" do
    test "preserves BNHD value input and attention output contract" do
      q = Nx.iota({1, 2, 3, 4}, type: :f32) |> Nx.divide(100) |> gpu()
      new_k = Nx.iota({1, 1, 3, 4}, type: :f32) |> Nx.divide(200) |> gpu()
      new_v = Nx.iota({1, 1, 3, 4}, type: :f32) |> Nx.divide(300) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 5, 4}) |> Nx.as_type(:f32) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 5, 4}) |> Nx.as_type(:f32) |> gpu()

      {attn_ref, k_upd_ref, v_upd_ref} =
        EMLX.kv_cache_sdpa_update(
          EMLX.Backend.from_nx(q),
          EMLX.Backend.from_nx(new_k),
          EMLX.Backend.from_nx(new_v),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          0,
          1.0 / :math.sqrt(4)
        )

      attn = EMLX.Backend.to_nx(attn_ref)
      k_upd = EMLX.Backend.to_nx(k_upd_ref)
      v_upd = EMLX.Backend.to_nx(v_upd_ref)

      assert Nx.shape(attn) == {1, 2, 3, 4}
      assert Nx.shape(k_upd) == {1, 1, 5, 4}
      assert Nx.shape(v_upd) == {1, 1, 5, 4}

      assert_all_close(
        Nx.slice_along_axis(k_upd, 0, 3, axis: 2) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_k),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(v_upd, 0, 3, axis: 2) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_v),
        atol: 1.0e-5
      )
    end
  end

  describe "EMLX.causal_kv_attention/7" do
    test "prefill updates compact GQA cache and returns attention output" do
      {q_cpu, new_k_cpu, new_v_cpu, k_cache_cpu, v_cache_cpu, key_mask_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 3, 4, 8}, type: :f32) |> Nx.divide(100),
            Nx.iota({1, 3, 2, 8}, type: :f32) |> Nx.divide(200),
            Nx.iota({1, 3, 2, 8}, type: :f32) |> Nx.divide(300),
            Nx.broadcast(0.0, {1, 5, 2, 8}) |> Nx.as_type(:f32),
            Nx.broadcast(0.0, {1, 5, 2, 8}) |> Nx.as_type(:f32),
            Nx.tensor([[1, 1, 1]], type: :u8)
          }
        end)

      scale = 1.0 / :math.sqrt(8)

      expected_attn =
        causal_kv_attention_reference(
          q_cpu,
          new_k_cpu,
          new_v_cpu,
          k_cache_cpu,
          v_cache_cpu,
          0,
          scale,
          key_mask_cpu
        )

      q = gpu(q_cpu)
      new_k = gpu(new_k_cpu)
      new_v = gpu(new_v_cpu)
      k_cache = gpu(k_cache_cpu)
      v_cache = gpu(v_cache_cpu)
      key_mask = gpu(key_mask_cpu)

      {attn_ref, k_upd_ref, v_upd_ref} =
        EMLX.causal_kv_attention(
          EMLX.Backend.from_nx(q),
          EMLX.Backend.from_nx(new_k),
          EMLX.Backend.from_nx(new_v),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          0,
          scale: scale,
          key_mask: EMLX.Backend.from_nx(key_mask)
        )

      attn = EMLX.Backend.to_nx(attn_ref)
      k_upd = EMLX.Backend.to_nx(k_upd_ref)
      v_upd = EMLX.Backend.to_nx(v_upd_ref)

      assert Nx.shape(attn) == {1, 3, 4, 8}
      assert Nx.shape(k_upd) == {1, 5, 2, 8}
      assert Nx.shape(v_upd) == {1, 5, 2, 8}

      assert_all_close(
        Nx.backend_transfer(attn, Nx.BinaryBackend),
        expected_attn,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(k_upd, 0, 3, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_k),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(v_upd, 0, 3, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_v),
        atol: 1.0e-5
      )
    end

    test "decode appends one token after prefill and keeps output dtype" do
      {q_prefill_cpu, k_prefill_cpu, v_prefill_cpu, k_cache_cpu, v_cache_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 3, 4, 8}, type: :f32) |> Nx.divide(100),
            Nx.iota({1, 3, 2, 8}, type: :f32) |> Nx.divide(200),
            Nx.iota({1, 3, 2, 8}, type: :f32) |> Nx.divide(300),
            Nx.broadcast(0.0, {1, 5, 2, 8}) |> Nx.as_type(:f32),
            Nx.broadcast(0.0, {1, 5, 2, 8}) |> Nx.as_type(:f32)
          }
        end)

      scale = 1.0 / :math.sqrt(8)

      q_prefill = gpu(q_prefill_cpu)
      k_prefill = gpu(k_prefill_cpu)
      v_prefill = gpu(v_prefill_cpu)
      k_cache = gpu(k_cache_cpu)
      v_cache = gpu(v_cache_cpu)

      {_attn_ref, k_cache_ref, v_cache_ref} =
        EMLX.causal_kv_attention(
          EMLX.Backend.from_nx(q_prefill),
          EMLX.Backend.from_nx(k_prefill),
          EMLX.Backend.from_nx(v_prefill),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          0,
          scale: scale,
          key_mask: EMLX.Backend.from_nx(Nx.tensor([[1, 1, 1]], type: :u8) |> gpu())
        )

      {q_decode_cpu, k_decode_cpu, v_decode_cpu, decode_key_mask_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 1, 4, 8}, type: :f32) |> Nx.add(100) |> Nx.divide(100),
            Nx.iota({1, 1, 2, 8}, type: :f32) |> Nx.add(100) |> Nx.divide(200),
            Nx.iota({1, 1, 2, 8}, type: :f32) |> Nx.add(100) |> Nx.divide(300),
            Nx.tensor([[1, 1, 1, 1]], type: :u8)
          }
        end)

      k_cache_after_prefill =
        Nx.put_slice(k_cache_cpu, [0, 0, 0, 0], k_prefill_cpu)

      v_cache_after_prefill =
        Nx.put_slice(v_cache_cpu, [0, 0, 0, 0], v_prefill_cpu)

      expected =
        causal_kv_attention_reference(
          q_decode_cpu,
          k_decode_cpu,
          v_decode_cpu,
          k_cache_after_prefill,
          v_cache_after_prefill,
          3,
          scale,
          decode_key_mask_cpu
        )

      q_decode = gpu(q_decode_cpu)
      k_decode = gpu(k_decode_cpu)
      v_decode = gpu(v_decode_cpu)

      {attn_ref, k_upd_ref, v_upd_ref} =
        EMLX.causal_kv_attention(
          EMLX.Backend.from_nx(q_decode),
          EMLX.Backend.from_nx(k_decode),
          EMLX.Backend.from_nx(v_decode),
          k_cache_ref,
          v_cache_ref,
          3,
          scale: scale,
          key_mask: EMLX.Backend.from_nx(Nx.tensor([[1, 1, 1, 1]], type: :u8) |> gpu())
        )

      attn = EMLX.Backend.to_nx(attn_ref)
      k_upd = EMLX.Backend.to_nx(k_upd_ref)
      v_upd = EMLX.Backend.to_nx(v_upd_ref)

      assert Nx.shape(attn) == {1, 1, 4, 8}
      assert Nx.type(attn) == {:f, 32}

      assert_all_close(Nx.backend_transfer(attn, Nx.BinaryBackend), expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(k_upd, 3, 1, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(k_decode),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(v_upd, 3, 1, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(v_decode),
        atol: 1.0e-5
      )
    end
  end

  describe "EMLX.qwen3_kv_cache_attention/9 validation" do
    test "rejects malformed Q rank before reading shapes" do
      q = Nx.broadcast(0.0, {1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()

      assert_raise EMLX.NIFError, ~r/q expects rank 4/, fn ->
        EMLX.qwen3_kv_cache_attention(
          EMLX.Backend.from_nx(q),
          EMLX.Backend.from_nx(k),
          EMLX.Backend.from_nx(v),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          0,
          1.0 / :math.sqrt(4),
          4,
          10_000.0
        )
      end
    end

    test "rejects negative offsets" do
      q = Nx.broadcast(0.0, {1, 1, 2, 4}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()

      assert_raise EMLX.NIFError, ~r/offset must be non-negative/, fn ->
        EMLX.qwen3_kv_cache_attention(
          EMLX.Backend.from_nx(q),
          EMLX.Backend.from_nx(k),
          EMLX.Backend.from_nx(v),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          -1,
          1.0 / :math.sqrt(4),
          4,
          10_000.0
        )
      end
    end

    test "rejects cache capacity overflow" do
      q = Nx.broadcast(0.0, {1, 1, 2, 4}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()

      assert_raise EMLX.NIFError, ~r/KV cache capacity 4 is smaller than required length 5/, fn ->
        EMLX.qwen3_kv_cache_attention(
          EMLX.Backend.from_nx(q),
          EMLX.Backend.from_nx(k),
          EMLX.Backend.from_nx(v),
          EMLX.Backend.from_nx(k_cache),
          EMLX.Backend.from_nx(v_cache),
          4,
          1.0 / :math.sqrt(4),
          4,
          10_000.0
        )
      end
    end

    test "matches pure Nx reference for GQA prefill and updates caches" do
      {q_cpu, k_cpu, v_cpu, k_cache_cpu, v_cache_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 2, 4, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(100),
            Nx.iota({1, 2, 2, 4}, type: :f32) |> Nx.add(11) |> Nx.divide(110),
            Nx.iota({1, 2, 2, 4}, type: :f32) |> Nx.add(17) |> Nx.divide(120),
            Nx.iota({1, 2, 5, 4}, type: :f32) |> Nx.divide(1_000),
            Nx.iota({1, 2, 5, 4}, type: :f32) |> Nx.add(50) |> Nx.divide(1_000)
          }
        end)

      offset = 1
      head_dim = 4
      theta = 10_000.0
      scale = 1.0 / :math.sqrt(head_dim)

      {attn_ref, k_ref, v_ref} =
        EMLX.qwen3_kv_cache_attention(
          EMLX.Backend.from_nx(gpu(q_cpu)),
          EMLX.Backend.from_nx(gpu(k_cpu)),
          EMLX.Backend.from_nx(gpu(v_cpu)),
          EMLX.Backend.from_nx(gpu(k_cache_cpu)),
          EMLX.Backend.from_nx(gpu(v_cache_cpu)),
          offset,
          scale,
          head_dim,
          theta
        )

      {expected_attn, expected_k, expected_v} =
        qwen3_kv_cache_attention_reference(
          q_cpu,
          k_cpu,
          v_cpu,
          k_cache_cpu,
          v_cache_cpu,
          offset,
          scale,
          head_dim,
          theta
        )

      assert_all_close(
        attn_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_attn,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "matches pure Nx reference for batched GQA prefill with nonzero offset" do
      {q_cpu, k_cpu, v_cpu, k_cache_cpu, v_cache_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({2, 2, 4, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(100),
            Nx.iota({2, 2, 2, 4}, type: :f32) |> Nx.add(23) |> Nx.divide(130),
            Nx.iota({2, 2, 2, 4}, type: :f32) |> Nx.add(37) |> Nx.divide(140),
            Nx.iota({2, 2, 6, 4}, type: :f32) |> Nx.add(5) |> Nx.divide(1_000),
            Nx.iota({2, 2, 6, 4}, type: :f32) |> Nx.add(95) |> Nx.divide(1_000)
          }
        end)

      offset = 2
      head_dim = 4
      theta = 10_000.0
      scale = 1.0 / :math.sqrt(head_dim)

      {attn_ref, k_ref, v_ref} =
        EMLX.qwen3_kv_cache_attention(
          EMLX.Backend.from_nx(gpu(q_cpu)),
          EMLX.Backend.from_nx(gpu(k_cpu)),
          EMLX.Backend.from_nx(gpu(v_cpu)),
          EMLX.Backend.from_nx(gpu(k_cache_cpu)),
          EMLX.Backend.from_nx(gpu(v_cache_cpu)),
          offset,
          scale,
          head_dim,
          theta
        )

      {expected_attn, expected_k, expected_v} =
        qwen3_kv_cache_attention_reference(
          q_cpu,
          k_cpu,
          v_cpu,
          k_cache_cpu,
          v_cache_cpu,
          offset,
          scale,
          head_dim,
          theta
        )

      assert Nx.shape(expected_attn) == {2, 2, 16}

      assert_all_close(
        attn_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_attn,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end
  end

  describe "dense Qwen3 native helpers" do
    test "qwen3_mlp matches pure Nx reference" do
      {hidden_cpu, norm_cpu, gate_cpu, up_cpu, down_cpu} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(10),
            Nx.tensor([1.0, 1.1, 0.9, 1.2], type: :f32),
            Nx.iota({4, 6}, type: :f32) |> Nx.add(1) |> Nx.divide(50),
            Nx.iota({4, 6}, type: :f32) |> Nx.add(7) |> Nx.divide(60),
            Nx.iota({6, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(70)
          }
        end)

      eps = 1.0e-6

      out_ref =
        EMLX.qwen3_mlp(
          EMLX.Backend.from_nx(gpu(hidden_cpu)),
          EMLX.Backend.from_nx(gpu(norm_cpu)),
          EMLX.Backend.from_nx(gpu(gate_cpu)),
          EMLX.Backend.from_nx(gpu(up_cpu)),
          EMLX.Backend.from_nx(gpu(down_cpu)),
          eps
        )

      expected = qwen3_mlp_reference(hidden_cpu, norm_cpu, gate_cpu, up_cpu, down_cpu, eps)

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "qwen3_attention_block matches pure Nx reference" do
      fixtures = qwen3_dense_attention_fixtures()
      scale = 1.0 / :math.sqrt(fixtures.head_dim)

      {out_ref, k_ref, v_ref} =
        EMLX.qwen3_attention_block(
          EMLX.Backend.from_nx(gpu(fixtures.hidden)),
          EMLX.Backend.from_nx(gpu(fixtures.norm1)),
          EMLX.Backend.from_nx(gpu(fixtures.q_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.k_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.v_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.o_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.q_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
          EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
          fixtures.offset,
          scale,
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      {expected, expected_k, expected_v} = qwen3_attention_block_reference(fixtures)

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "qwen3_attention_block matches pure Nx reference for batched GQA" do
      fixtures = qwen3_batched_dense_attention_fixtures()
      scale = 1.0 / :math.sqrt(fixtures.head_dim)

      {out_ref, k_ref, v_ref} =
        EMLX.qwen3_attention_block(
          EMLX.Backend.from_nx(gpu(fixtures.hidden)),
          EMLX.Backend.from_nx(gpu(fixtures.norm1)),
          EMLX.Backend.from_nx(gpu(fixtures.q_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.k_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.v_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.o_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.q_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
          EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
          fixtures.offset,
          scale,
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      {expected, expected_k, expected_v} = qwen3_attention_block_reference(fixtures)

      assert Nx.shape(expected) == {2, 2, 4}

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "qwen3_layer matches pure Nx reference" do
      fixtures = qwen3_dense_attention_fixtures()
      scale = 1.0 / :math.sqrt(fixtures.head_dim)

      {out_ref, k_ref, v_ref} =
        EMLX.qwen3_layer(
          EMLX.Backend.from_nx(gpu(fixtures.hidden)),
          EMLX.Backend.from_nx(gpu(fixtures.norm1)),
          EMLX.Backend.from_nx(gpu(fixtures.q_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.k_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.v_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.o_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.q_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_norm)),
          EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
          EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
          EMLX.Backend.from_nx(gpu(fixtures.norm2)),
          EMLX.Backend.from_nx(gpu(fixtures.gate_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.up_proj)),
          EMLX.Backend.from_nx(gpu(fixtures.down_proj)),
          fixtures.offset,
          scale,
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      {attn_expected, expected_k, expected_v} = qwen3_attention_block_reference(fixtures)

      expected =
        qwen3_mlp_reference(
          attn_expected,
          fixtures.norm2,
          fixtures.gate_proj,
          fixtures.up_proj,
          fixtures.down_proj,
          fixtures.eps
        )

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end

    test "qwen3_attention_residual matches pure Nx reference" do
      {hidden, attn_out, o_proj} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(10),
            Nx.iota({1, 2, 3}, type: :f32) |> Nx.add(5) |> Nx.divide(20),
            Nx.iota({3, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(30)
          }
        end)

      out_ref =
        EMLX.qwen3_attention_residual(
          EMLX.Backend.from_nx(gpu(hidden)),
          EMLX.Backend.from_nx(gpu(attn_out)),
          EMLX.Backend.from_nx(gpu(o_proj))
        )

      expected =
        hidden
        |> Nx.add(Nx.dot(attn_out, [2], o_proj, [0]))
        |> Nx.backend_transfer(Nx.BinaryBackend)

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 1.0e-5,
        rtol: 1.0e-5
      )
    end
  end

  describe "EMLX.qwen3_attention_block/15 validation" do
    test "qwen3_kv_cache_attention rejects required cache length overflow before graph construction" do
      q = Nx.broadcast(0.0, {1, 1, 2, 2}) |> Nx.as_type(:f32) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 2}) |> Nx.as_type(:f32) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 2}) |> Nx.as_type(:f32) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32) |> gpu()

      assert_raise EMLX.NIFError,
                   ~r/KV cache capacity 4 is smaller than required length 2147483648/,
                   fn ->
                     EMLX.qwen3_kv_cache_attention(
                       EMLX.Backend.from_nx(q),
                       EMLX.Backend.from_nx(k),
                       EMLX.Backend.from_nx(v),
                       EMLX.Backend.from_nx(k_cache),
                       EMLX.Backend.from_nx(v_cache),
                       2_147_483_647,
                       1.0 / :math.sqrt(2),
                       2,
                       10_000.0
                     )
                   end
    end

    test "qwen3_layer rejects required cache length overflow before graph construction" do
      fixtures = qwen3_dense_attention_fixtures()

      assert_raise EMLX.NIFError,
                   ~r/KV cache capacity 4 is smaller than required length 2147483649/,
                   fn ->
                     EMLX.qwen3_layer(
                       EMLX.Backend.from_nx(gpu(fixtures.hidden)),
                       EMLX.Backend.from_nx(gpu(fixtures.norm1)),
                       EMLX.Backend.from_nx(gpu(fixtures.q_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.v_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.o_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.q_norm)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_norm)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
                       EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
                       EMLX.Backend.from_nx(gpu(fixtures.norm2)),
                       EMLX.Backend.from_nx(gpu(fixtures.gate_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.up_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.down_proj)),
                       2_147_483_647,
                       1.0 / :math.sqrt(fixtures.head_dim),
                       fixtures.head_dim,
                       fixtures.theta,
                       fixtures.eps
                     )
                   end
    end

    test "qwen3_attention_block rejects required cache length overflow before graph construction" do
      fixtures = qwen3_dense_attention_fixtures()

      assert_raise EMLX.NIFError,
                   ~r/KV cache capacity 4 is smaller than required length 2147483649/,
                   fn ->
                     EMLX.qwen3_attention_block(
                       EMLX.Backend.from_nx(gpu(fixtures.hidden)),
                       EMLX.Backend.from_nx(gpu(fixtures.norm1)),
                       EMLX.Backend.from_nx(gpu(fixtures.q_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.v_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.o_proj)),
                       EMLX.Backend.from_nx(gpu(fixtures.q_norm)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_norm)),
                       EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
                       EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
                       2_147_483_647,
                       1.0 / :math.sqrt(fixtures.head_dim),
                       fixtures.head_dim,
                       fixtures.theta,
                       fixtures.eps
                     )
                   end
    end

    test "rejects projection widths before deriving head counts" do
      hidden = Nx.broadcast(0.0, {1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      norm = Nx.broadcast(1.0, {4}) |> Nx.as_type(:f16) |> gpu()
      q_proj = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f16) |> gpu()
      k_proj = Nx.broadcast(0.1, {4, 1}) |> Nx.as_type(:f16) |> gpu()
      v_proj = Nx.broadcast(0.1, {4, 1}) |> Nx.as_type(:f16) |> gpu()
      o_proj = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f16) |> gpu()
      q_norm = Nx.broadcast(1.0, {2}) |> Nx.as_type(:f16) |> gpu()
      k_norm = Nx.broadcast(1.0, {2}) |> Nx.as_type(:f16) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f16) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f16) |> gpu()

      assert_raise EMLX.NIFError,
                   ~r/projection output widths must be divisible by head_dim/,
                   fn ->
                     EMLX.qwen3_attention_block(
                       EMLX.Backend.from_nx(hidden),
                       EMLX.Backend.from_nx(norm),
                       EMLX.Backend.from_nx(q_proj),
                       EMLX.Backend.from_nx(k_proj),
                       EMLX.Backend.from_nx(v_proj),
                       EMLX.Backend.from_nx(o_proj),
                       EMLX.Backend.from_nx(q_norm),
                       EMLX.Backend.from_nx(k_norm),
                       EMLX.Backend.from_nx(k_cache),
                       EMLX.Backend.from_nx(v_cache),
                       0,
                       1.0 / :math.sqrt(2),
                       2,
                       10_000.0,
                       1.0e-6
                     )
                   end
    end
  end

  defp qwen3_dense_attention_fixtures do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      %{
        hidden: Nx.iota({1, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(10),
        norm1: Nx.tensor([1.0, 1.1, 0.9, 1.2], type: :f32),
        norm2: Nx.tensor([0.9, 1.0, 1.1, 1.2], type: :f32),
        q_norm: Nx.tensor([1.0, 1.1], type: :f32),
        k_norm: Nx.tensor([0.9, 1.2], type: :f32),
        q_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(50),
        k_proj: Nx.iota({4, 2}, type: :f32) |> Nx.add(2) |> Nx.divide(60),
        v_proj: Nx.iota({4, 2}, type: :f32) |> Nx.add(3) |> Nx.divide(70),
        o_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(4) |> Nx.divide(80),
        gate_proj: Nx.iota({4, 6}, type: :f32) |> Nx.add(1) |> Nx.divide(50),
        up_proj: Nx.iota({4, 6}, type: :f32) |> Nx.add(7) |> Nx.divide(60),
        down_proj: Nx.iota({6, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(70),
        k_cache: Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32),
        v_cache: Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32),
        offset: 0,
        head_dim: 2,
        theta: 10_000.0,
        eps: 1.0e-6
      }
    end)
  end

  defp qwen3_batched_dense_attention_fixtures do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      %{
        hidden: Nx.iota({2, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(20),
        norm1: Nx.tensor([1.0, 1.1, 0.9, 1.2], type: :f32),
        q_norm: Nx.tensor([1.0, 1.1], type: :f32),
        k_norm: Nx.tensor([0.9, 1.2], type: :f32),
        q_proj: Nx.iota({4, 8}, type: :f32) |> Nx.add(1) |> Nx.divide(80),
        k_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(90),
        v_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(5) |> Nx.divide(100),
        o_proj: Nx.iota({8, 4}, type: :f32) |> Nx.add(7) |> Nx.divide(110),
        k_cache: Nx.iota({2, 2, 6, 2}, type: :f32) |> Nx.add(11) |> Nx.divide(1_000),
        v_cache: Nx.iota({2, 2, 6, 2}, type: :f32) |> Nx.add(101) |> Nx.divide(1_000),
        offset: 2,
        head_dim: 2,
        theta: 10_000.0,
        eps: 1.0e-6
      }
    end)
  end

  defp qwen3_kv_cache_attention_reference(
         q,
         new_k,
         new_v,
         k_cache,
         v_cache,
         offset,
         scale,
         head_dim,
         theta
       ) do
    {batch, seq_len, q_heads, _head_dim} = Nx.shape(q)
    {_batch, _seq_len, kv_heads, _head_dim} = Nx.shape(new_k)

    q_bn =
      q
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> qwen3_rope_reference(head_dim, theta, offset)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    k_bn =
      new_k
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> qwen3_rope_reference(head_dim, theta, offset)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    v_bn = new_v |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.backend_transfer(Nx.BinaryBackend)

    k_cache = Nx.put_slice(k_cache, [0, 0, offset, 0], k_bn)
    v_cache = Nx.put_slice(v_cache, [0, 0, offset, 0], v_bn)

    valid_len = offset + seq_len
    k_valid = Nx.slice_along_axis(k_cache, 0, valid_len, axis: 2)
    v_valid = Nx.slice_along_axis(v_cache, 0, valid_len, axis: 2)

    groups = div(q_heads, kv_heads)
    k_repeated = repeat_kv_heads_bn(k_valid, groups)
    v_repeated = repeat_kv_heads_bn(v_valid, groups)

    scores =
      q_bn
      |> Nx.new_axis(3)
      |> Nx.multiply(Nx.new_axis(k_repeated, 2))
      |> Nx.sum(axes: [4])
      |> Nx.multiply(scale)
      |> apply_causal_mask(offset, seq_len, valid_len)

    weights = softmax_reference(scores, 3)

    attn =
      weights
      |> Nx.new_axis(4)
      |> Nx.multiply(Nx.new_axis(v_repeated, 2))
      |> Nx.sum(axes: [3])

    attn_out =
      attn
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, q_heads * head_dim})

    {attn_out, k_cache, v_cache}
  end

  defp qwen3_mlp_reference(hidden, norm, gate_proj, up_proj, down_proj, eps) do
    xn = rms_norm_reference(hidden, norm, eps)
    gate = Nx.dot(xn, [2], gate_proj, [0])
    up = Nx.dot(xn, [2], up_proj, [0])
    mlp = Nx.multiply(Nx.multiply(gate, Nx.sigmoid(gate)), up)
    out = Nx.dot(mlp, [2], down_proj, [0])
    Nx.add(hidden, out)
  end

  defp qwen3_attention_block_reference(fixtures) do
    hidden = fixtures.hidden
    offset = fixtures.offset
    head_dim = fixtures.head_dim
    scale = 1.0 / :math.sqrt(head_dim)

    {batch, seq_len, _hidden_size} = Nx.shape(hidden)

    xn = rms_norm_reference(hidden, fixtures.norm1, fixtures.eps)

    q_flat = Nx.dot(xn, [2], fixtures.q_proj, [0])
    k_flat = Nx.dot(xn, [2], fixtures.k_proj, [0])
    v_flat = Nx.dot(xn, [2], fixtures.v_proj, [0])

    q_heads = div(elem(Nx.shape(fixtures.q_proj), 1), head_dim)
    kv_heads = div(elem(Nx.shape(fixtures.k_proj), 1), head_dim)

    q = Nx.reshape(q_flat, {batch, seq_len, q_heads, head_dim})
    k = Nx.reshape(k_flat, {batch, seq_len, kv_heads, head_dim})
    v = Nx.reshape(v_flat, {batch, seq_len, kv_heads, head_dim})

    q = rms_norm_reference(q, fixtures.q_norm, fixtures.eps)
    k = rms_norm_reference(k, fixtures.k_norm, fixtures.eps)

    q_bn =
      q
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> qwen3_rope_reference(head_dim, fixtures.theta, offset)

    k_bn =
      k
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> qwen3_rope_reference(head_dim, fixtures.theta, offset)

    q_bn = Nx.backend_transfer(q_bn, Nx.BinaryBackend)
    k_bn = Nx.backend_transfer(k_bn, Nx.BinaryBackend)
    v_bn = v |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.backend_transfer(Nx.BinaryBackend)

    k_cache = Nx.put_slice(fixtures.k_cache, [0, 0, offset, 0], k_bn)
    v_cache = Nx.put_slice(fixtures.v_cache, [0, 0, offset, 0], v_bn)

    valid_len = offset + seq_len
    k_valid = Nx.slice_along_axis(k_cache, 0, valid_len, axis: 2)
    v_valid = Nx.slice_along_axis(v_cache, 0, valid_len, axis: 2)

    groups = div(q_heads, kv_heads)
    k_repeated = repeat_kv_heads_bn(k_valid, groups)
    v_repeated = repeat_kv_heads_bn(v_valid, groups)

    scores =
      q_bn
      |> Nx.new_axis(3)
      |> Nx.multiply(Nx.new_axis(k_repeated, 2))
      |> Nx.sum(axes: [4])
      |> Nx.multiply(scale)

    scores = apply_causal_mask(scores, offset, seq_len, valid_len)
    weights = softmax_reference(scores, 3)

    attn =
      weights
      |> Nx.new_axis(4)
      |> Nx.multiply(Nx.new_axis(v_repeated, 2))
      |> Nx.sum(axes: [3])

    attn_out =
      attn
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, q_heads * head_dim})

    projected = Nx.dot(attn_out, [2], fixtures.o_proj, [0])

    {Nx.add(hidden, projected), k_cache, v_cache}
  end

  defp rms_norm_reference(tensor, weight, eps) do
    tensor
    |> Nx.pow(2)
    |> Nx.mean(axes: [-1], keep_axes: true)
    |> Nx.add(eps)
    |> Nx.sqrt()
    |> then(&Nx.divide(tensor, &1))
    |> Nx.multiply(weight)
  end

  defp qwen3_rope_reference(tensor, dims, theta, offset) do
    {_batch, _heads, seq_len, _dims} = Nx.shape(tensor)
    half = div(dims, 2)

    inv_freq =
      Nx.iota({half}, type: :f32)
      |> Nx.multiply(2)
      |> Nx.divide(dims)
      |> then(&Nx.pow(theta, &1))
      |> then(&Nx.divide(1.0, &1))

    positions =
      Nx.iota({seq_len}, type: :f32)
      |> Nx.add(offset)
      |> Nx.reshape({seq_len, 1})

    freqs = Nx.multiply(positions, Nx.reshape(inv_freq, {1, half}))

    cos =
      Nx.concatenate([Nx.cos(freqs), Nx.cos(freqs)], axis: 1)
      |> Nx.reshape({1, 1, seq_len, dims})

    sin =
      Nx.concatenate([Nx.sin(freqs), Nx.sin(freqs)], axis: 1)
      |> Nx.reshape({1, 1, seq_len, dims})

    first = tensor[[.., .., .., 0..(half - 1)//1]]
    second = tensor[[.., .., .., half..(dims - 1)//1]]
    rotated = Nx.concatenate([Nx.negate(second), first], axis: 3)

    Nx.add(Nx.multiply(tensor, cos), Nx.multiply(rotated, sin))
  end

  defp repeat_kv_heads_bn(tensor, 1), do: tensor

  defp repeat_kv_heads_bn(tensor, groups) do
    {batch, n_kv, t_kv, head_dim} = Nx.shape(tensor)

    tensor
    |> Nx.new_axis(2)
    |> Nx.broadcast({batch, n_kv, groups, t_kv, head_dim})
    |> Nx.reshape({batch, n_kv * groups, t_kv, head_dim})
  end

  defp apply_causal_mask(scores, offset, seq_len, valid_len) do
    query_positions =
      Nx.iota({seq_len}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.add(offset)
      |> Nx.reshape({1, 1, seq_len, 1})

    key_positions =
      Nx.iota({valid_len}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.reshape({1, 1, 1, valid_len})

    mask =
      key_positions
      |> Nx.less_equal(query_positions)
      |> Nx.broadcast(Nx.shape(scores))

    Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))
  end

  defp softmax_reference(tensor, axis) do
    shifted = Nx.subtract(tensor, Nx.reduce_max(tensor, axes: [axis], keep_axes: true))
    exp = Nx.exp(shifted)
    Nx.divide(exp, Nx.sum(exp, axes: [axis], keep_axes: true))
  end

  defp causal_kv_attention_reference(q, new_k, new_v, k_cache, v_cache, offset, scale, key_mask) do
    q = Nx.backend_transfer(q, Nx.BinaryBackend)
    new_k = Nx.backend_transfer(new_k, Nx.BinaryBackend)
    new_v = Nx.backend_transfer(new_v, Nx.BinaryBackend)
    k_cache = Nx.backend_transfer(k_cache, Nx.BinaryBackend)
    v_cache = Nx.backend_transfer(v_cache, Nx.BinaryBackend)
    key_mask = Nx.backend_transfer(key_mask, Nx.BinaryBackend)

    {_batch, t_q, n_q, _head_dim} = Nx.shape(q)
    {_batch, t_new, n_kv, _head_dim} = Nx.shape(new_k)
    t_kv = offset + t_new

    k_cache = Nx.put_slice(k_cache, [0, offset, 0, 0], new_k)
    v_cache = Nx.put_slice(v_cache, [0, offset, 0, 0], new_v)

    k_valid = Nx.slice_along_axis(k_cache, 0, t_kv, axis: 1)
    v_valid = Nx.slice_along_axis(v_cache, 0, t_kv, axis: 1)

    q_heads = Nx.transpose(q, axes: [0, 2, 1, 3])
    k_heads = k_valid |> Nx.transpose(axes: [0, 2, 1, 3]) |> repeat_kv_heads(div(n_q, n_kv))
    v_heads = v_valid |> Nx.transpose(axes: [0, 2, 1, 3]) |> repeat_kv_heads(div(n_q, n_kv))

    scores =
      q_heads
      |> Nx.new_axis(3)
      |> Nx.multiply(Nx.new_axis(k_heads, 2))
      |> Nx.sum(axes: [4])
      |> Nx.multiply(scale)

    query_positions =
      Nx.iota({t_q}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.add(offset)
      |> Nx.reshape({1, 1, t_q, 1})

    key_positions =
      Nx.iota({t_kv}, type: :s32, backend: Nx.BinaryBackend)
      |> Nx.reshape({1, 1, 1, t_kv})

    causal_mask = Nx.less_equal(key_positions, query_positions)

    valid_mask =
      key_mask
      |> Nx.equal(1)
      |> Nx.reshape({elem(Nx.shape(key_mask), 0), 1, 1, t_kv})

    mask =
      causal_mask
      |> Nx.logical_and(valid_mask)
      |> Nx.broadcast(Nx.shape(scores))

    scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    probs =
      scores
      |> Nx.subtract(Nx.reduce_max(scores, axes: [3], keep_axes: true))
      |> Nx.exp()

    probs = Nx.divide(probs, Nx.sum(probs, axes: [3], keep_axes: true))

    probs
    |> Nx.new_axis(4)
    |> Nx.multiply(Nx.new_axis(v_heads, 2))
    |> Nx.sum(axes: [3])
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp repeat_kv_heads(kv, 1), do: kv

  defp repeat_kv_heads(kv, groups) do
    {batch, n_kv, t_kv, head_dim} = Nx.shape(kv)

    kv
    |> Nx.new_axis(2)
    |> Nx.broadcast({batch, n_kv, groups, t_kv, head_dim})
    |> Nx.reshape({batch, n_kv * groups, t_kv, head_dim})
  end
end
