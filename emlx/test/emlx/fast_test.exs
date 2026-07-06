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

    # https://github.com/elixir-nx/emlx/issues/121 — mlx::core::fast::rope's
    # head_seq_transpose stride-detection guard doesn't trigger for a
    # freshly-allocated, contiguous {B,T,H,D} tensor, so its row_contiguous
    # fallback used to rotate head h at angle `position + h` instead of
    # `position`, for every h > 0. Every head, computed jointly, must equal
    # that same head computed alone (same position, no cross-head leakage).
    test "T=1 decode, H>1: every head matches its single-head computation" do
      dims = 8
      half = div(dims, 2)
      freqs = Nx.iota({half}, type: :f32) |> Nx.add(0.1) |> Nx.divide(20) |> gpu()

      a = Nx.iota({1, 1, 3, dims}, type: :f32) |> Nx.divide(20) |> gpu()
      pos = Nx.tensor([[6]], type: :s32) |> gpu()

      joint = Fast.rope_with_freqs(a, pos, dims, false, 1.0, freqs) |> Nx.backend_transfer()

      for head <- 0..2 do
        a_head = a[[.., .., head..head, ..]]

        alone =
          Fast.rope_with_freqs(a_head, pos, dims, false, 1.0, freqs) |> Nx.backend_transfer()

        assert_all_close(joint[[.., .., head..head, ..]], alone, atol: 1.0e-4, rtol: 1.0e-4)
      end
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

    # https://github.com/elixir-nx/emlx/issues/121 — low-base (< 1.0e5) T=1
    # decode routes through fast_rope_ids (mlx::core::fast::rope's
    # array-offset overload), which had the same multi-head bug as
    # fast_rope_with_freqs (see the regression test in the describe block
    # above). Every head, computed jointly, must equal that same head
    # computed alone.
    test "low-base decode (T=1), H>1: every head matches its single-head computation" do
      dims = 8
      base = 10_000.0

      a = Nx.iota({1, 1, 3, dims}, type: :f32) |> Nx.divide(50) |> gpu()
      pos = Nx.tensor([[7]], type: :s32) |> gpu()

      joint = Fast.rope_with_positions(a, pos, dims, false, base, 1.0) |> Nx.backend_transfer()

      for head <- 0..2 do
        a_head = a[[.., .., head..head, ..]]

        alone =
          Fast.rope_with_positions(a_head, pos, dims, false, base, 1.0)
          |> Nx.backend_transfer()

        assert_all_close(joint[[.., .., head..head, ..]], alone, atol: 1.0e-4, rtol: 1.0e-4)
      end
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

  describe "EMLX.Fast.kv_cache_sdpa_update/8 (eager)" do
    test "writes new_k/new_v at offset and matches put_slice+causal_key_masked reference" do
      # Bumblebee-native layout: {B, T, N, D} (heads NOT transposed).
      q = Nx.iota({1, 1, 2, 4}, type: :f32) |> Nx.divide(100) |> gpu()
      new_k = Nx.iota({1, 1, 1, 4}, type: :f32) |> Nx.divide(200) |> gpu()
      new_v = Nx.iota({1, 1, 1, 4}, type: :f32) |> Nx.divide(300) |> gpu()
      k_cache = Nx.iota({1, 5, 1, 4}, type: :f32) |> Nx.divide(50) |> gpu()
      v_cache = Nx.iota({1, 5, 1, 4}, type: :f32) |> Nx.divide(60) |> gpu()
      offset = Nx.tensor(2, type: :s32) |> gpu()
      key_mask = Nx.tensor([[1, 1, 1, 0, 0]], type: :s32) |> gpu()
      scale = 1.0 / :math.sqrt(4)

      {attn, k_upd, v_upd} =
        Fast.kv_cache_sdpa_update(q, new_k, new_v, k_cache, v_cache, offset, key_mask, scale)

      # attn_out is head-transposed ({B, N, T, D}), matching the unfused
      # metadata node's own output convention; k_upd/v_upd stay
      # Bumblebee-native ({B, T, N, D}), matching the cache's own shape.
      assert Nx.shape(attn) == {1, 2, 1, 4}
      assert Nx.shape(k_upd) == {1, 5, 1, 4}
      assert Nx.shape(v_upd) == {1, 5, 1, 4}

      # Compute the unfused reference (reusing new_k/new_v/k_cache/v_cache)
      # *before* any `Nx.backend_transfer/1` call below, since that deallocates
      # its (single) argument's underlying EMLX resource -- see
      # `EMLX.Backend.backend_transfer/3`.
      k_ref = Nx.put_slice(k_cache, [0, 2, 0, 0], new_k)
      v_ref = Nx.put_slice(v_cache, [0, 2, 0, 0], new_v)
      q_t = Nx.transpose(q, axes: [0, 2, 1, 3])
      k_ref_t = Nx.transpose(k_ref, axes: [0, 2, 1, 3])
      v_ref_t = Nx.transpose(v_ref, axes: [0, 2, 1, 3])

      attn_ref =
        Fast.scaled_dot_product_attention_causal_key_masked(q_t, k_ref_t, v_ref_t, scale, key_mask)

      assert_all_close(
        Nx.slice_along_axis(k_upd, 2, 1, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_k),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.slice_along_axis(v_upd, 2, 1, axis: 1) |> Nx.backend_transfer(),
        Nx.backend_transfer(new_v),
        atol: 1.0e-5
      )

      assert_all_close(Nx.backend_transfer(attn), Nx.backend_transfer(attn_ref), atol: 1.0e-3)
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
