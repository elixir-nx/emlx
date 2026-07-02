defmodule EMLX.SdpaSinksTest do
  @moduledoc """
  Stage 22 (fast-kernel-quant-parity): SDPA attention sinks (Emily M26
  parity). Equivalence-tests `EMLX.Fast.scaled_dot_product_attention*`'s
  `:sinks` opt against the fallback softmax-with-sinks math:

      row_max = max(reduce_max(logits), sinks_broadcast)
      probs = exp(logits - row_max) / (sum(exp(logits - row_max)) + exp(sinks - row_max))

  (same formula as Emily's M26 fallback — see `~/coding/emily/PLAN.md`).
  """
  use EMLX.Case, async: true

  @moduletag :metal

  alias EMLX.Fast

  # `EMLX.Case`'s setup defaults to `EMLX.Backend`; override it to
  # `Nx.BinaryBackend` here so all the plain (non-`gpu/1`) tensor construction
  # and reference math below runs on plain Nx instead of round-tripping
  # through MLX. `gpu/1` still explicitly transfers to `EMLX.Backend`
  # regardless of the default.
  setup do
    Nx.default_backend(Nx.BinaryBackend)
    :ok
  end

  defp gpu(tensor), do: Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

  # Reference (non-fused) softmax-with-sinks, operating on plain (BinaryBackend) Nx
  # tensors. `logits_mask` is an optional {t_q, t_kv} boolean keep-mask (broadcast
  # to {b, n, t_q, t_kv}); nil means no masking (full attention).
  defp reference_sdpa_sinks(q, k, v, scale, sinks, logits_mask \\ nil) do
    logits = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.multiply(scale)
    {b, n, t_q, t_kv} = Nx.shape(logits)

    logits =
      if logits_mask do
        keep = Nx.reshape(logits_mask, {1, 1, t_q, t_kv}) |> Nx.broadcast({b, n, t_q, t_kv})
        neg_inf = Nx.Constants.neg_infinity(Nx.type(logits)) |> Nx.broadcast(Nx.shape(logits))
        Nx.select(keep, logits, neg_inf)
      else
        logits
      end

    sinks_b = Nx.reshape(sinks, {1, n, 1, 1}) |> Nx.broadcast({b, n, t_q, 1})
    row_max = Nx.max(Nx.reduce_max(logits, axes: [3], keep_axes: true), sinks_b)
    exp_logits = Nx.exp(Nx.subtract(logits, row_max))
    exp_sinks = Nx.exp(Nx.subtract(sinks_b, row_max))
    denom = Nx.add(Nx.sum(exp_logits, axes: [3], keep_axes: true), exp_sinks)
    probs = Nx.divide(exp_logits, denom)

    Nx.dot(probs, [3], [0, 1], v, [2], [0, 1])
  end

  defp causal_keep_mask(t_q, t_kv) do
    row = Nx.iota({t_q, t_kv}, axis: 0)
    col = Nx.iota({t_q, t_kv}, axis: 1)
    Nx.less_equal(col, row)
  end

  # f32 tolerance matches Emily's own reported max-abs-diff (~2e-7) for this
  # exact fallback-vs-fused-kernel comparison.
  @tol 1.0e-4

  describe "EMLX.Fast.scaled_dot_product_attention/5 (opts, no mask) with :sinks" do
    test "matches the softmax-with-sinks fallback math" do
      b = 1
      n = 4
      t_q = 5
      t_kv = 5
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q_cpu = Nx.iota({b, n, t_q, d}, type: :f32) |> Nx.divide(37)
      k_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(41)
      v_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(53)
      sinks_cpu = Nx.tensor([0.1, -0.2, 0.5, 0.0], type: :f32)

      out_fast =
        Fast.scaled_dot_product_attention(gpu(q_cpu), gpu(k_cpu), gpu(v_cpu), scale,
          sinks: gpu(sinks_cpu)
        )
        |> Nx.backend_transfer()

      out_ref = reference_sdpa_sinks(q_cpu, k_cpu, v_cpu, scale, sinks_cpu)

      assert_all_close(out_fast, out_ref, atol: @tol)
    end

    test "GQA (N_q != N_kv): sinks shape follows N_q" do
      b = 1
      n_q = 8
      n_kv = 4
      t_q = 3
      t_kv = 6
      d = 16
      scale = 1.0 / :math.sqrt(d)

      q = Nx.iota({b, n_q, t_q, d}, type: :f32) |> Nx.divide(29) |> gpu()
      k = Nx.iota({b, n_kv, t_kv, d}, type: :f32) |> Nx.divide(31) |> gpu()
      v = Nx.iota({b, n_kv, t_kv, d}, type: :f32) |> Nx.divide(37) |> gpu()
      sinks = Nx.iota({n_q}, type: :f32) |> Nx.divide(10) |> gpu()

      out = Fast.scaled_dot_product_attention(q, k, v, scale, sinks: sinks)

      assert Nx.shape(out) == {b, n_q, t_q, d}
      assert Nx.type(out) == {:f, 32}
    end

    test "omitting :sinks is unchanged from the pre-sinks arity" do
      b = 1
      n = 4
      t_q = 4
      t_kv = 4
      d = 16
      scale = 1.0 / :math.sqrt(d)

      q = Nx.iota({b, n, t_q, d}, type: :f32) |> gpu()
      k = Nx.iota({b, n, t_kv, d}, type: :f32) |> gpu()
      v = Nx.iota({b, n, t_kv, d}, type: :f32) |> gpu()

      out_4arity = Fast.scaled_dot_product_attention(q, k, v, scale)
      out_empty_opts = Fast.scaled_dot_product_attention(q, k, v, scale, [])

      assert Nx.all_close(out_4arity, out_empty_opts) |> Nx.to_number() == 1
    end
  end

  describe "EMLX.Fast.scaled_dot_product_attention/6 (mask + :sinks)" do
    test "matches the softmax-with-sinks fallback math with an additive causal mask" do
      b = 1
      n = 4
      t_q = 5
      t_kv = 5
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q_cpu = Nx.iota({b, n, t_q, d}, type: :f32) |> Nx.divide(37)
      k_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(41)
      v_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(53)
      sinks_cpu = Nx.tensor([0.1, -0.2, 0.5, 0.0], type: :f32)

      mask =
        Nx.tril(Nx.broadcast(0.0, {t_q, t_kv}))
        |> Nx.subtract(Nx.triu(Nx.broadcast(1.0e9, {t_q, t_kv}), k: 1))
        |> Nx.reshape({1, 1, t_q, t_kv})

      out_fast =
        Fast.scaled_dot_product_attention(gpu(q_cpu), gpu(k_cpu), gpu(v_cpu), scale, gpu(mask),
          sinks: gpu(sinks_cpu)
        )
        |> Nx.backend_transfer()

      out_ref =
        reference_sdpa_sinks(q_cpu, k_cpu, v_cpu, scale, sinks_cpu, causal_keep_mask(t_q, t_kv))

      assert_all_close(out_fast, out_ref, atol: @tol)
    end
  end

  describe "EMLX.Fast.scaled_dot_product_attention_causal/5 with :sinks" do
    test "matches the softmax-with-sinks fallback math with an implicit causal mask" do
      b = 1
      n = 4
      t_q = 5
      t_kv = 5
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q_cpu = Nx.iota({b, n, t_q, d}, type: :f32) |> Nx.divide(37)
      k_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(41)
      v_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(53)
      sinks_cpu = Nx.tensor([0.1, -0.2, 0.5, 0.0], type: :f32)

      out_fast =
        Fast.scaled_dot_product_attention_causal(gpu(q_cpu), gpu(k_cpu), gpu(v_cpu), scale,
          sinks: gpu(sinks_cpu)
        )
        |> Nx.backend_transfer()

      out_ref =
        reference_sdpa_sinks(q_cpu, k_cpu, v_cpu, scale, sinks_cpu, causal_keep_mask(t_q, t_kv))

      assert_all_close(out_fast, out_ref, atol: @tol)
    end

    test "omitting :sinks is unchanged from the pre-sinks arity" do
      b = 1
      n = 2
      t_q = 3
      t_kv = 3
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q = Nx.iota({b, n, t_q, d}, type: :f32) |> gpu()
      k = Nx.iota({b, n, t_kv, d}, type: :f32) |> gpu()
      v = Nx.iota({b, n, t_kv, d}, type: :f32) |> gpu()

      out_4arity = Fast.scaled_dot_product_attention_causal(q, k, v, scale)
      out_empty_opts = Fast.scaled_dot_product_attention_causal(q, k, v, scale, [])

      assert Nx.all_close(out_4arity, out_empty_opts) |> Nx.to_number() == 1
    end
  end

  describe "EMLX.Fast.scaled_dot_product_attention_causal_key_masked/6 with :sinks" do
    test "trivial (all-ones) key_mask matches causal-with-sinks fallback math" do
      b = 1
      n = 4
      t_q = 1
      t_kv = 5
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q_cpu = Nx.iota({b, n, t_q, d}, type: :f32) |> Nx.divide(37)
      k_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(41)
      v_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(53)
      sinks_cpu = Nx.tensor([0.1, -0.2, 0.5, 0.0], type: :f32)
      key_mask_ones = Nx.broadcast(1, {b, t_kv})

      out_fast =
        Fast.scaled_dot_product_attention_causal_key_masked(
          gpu(q_cpu),
          gpu(k_cpu),
          gpu(v_cpu),
          scale,
          gpu(key_mask_ones),
          sinks: gpu(sinks_cpu)
        )
        |> Nx.backend_transfer()

      # decode (T_q=1): kv_offset = T_kv - 1, so every key position is causally visible.
      keep_mask = Nx.broadcast(1, {t_q, t_kv}) |> Nx.equal(1)
      out_ref = reference_sdpa_sinks(q_cpu, k_cpu, v_cpu, scale, sinks_cpu, keep_mask)

      assert_all_close(out_fast, out_ref, atol: @tol)
    end

    test "padded key_mask (non-trivial branch) matches combined causal+key_mask fallback math" do
      b = 1
      n = 4
      t_q = 5
      t_kv = 5
      d = 8
      scale = 1.0 / :math.sqrt(d)

      q_cpu = Nx.iota({b, n, t_q, d}, type: :f32) |> Nx.divide(37)
      k_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(41)
      v_cpu = Nx.iota({b, n, t_kv, d}, type: :f32) |> Nx.divide(53)
      sinks_cpu = Nx.tensor([0.1, -0.2, 0.5, 0.0], type: :f32)
      key_mask_padded = Nx.tensor([[0, 1, 1, 1, 1]])

      out_fast =
        Fast.scaled_dot_product_attention_causal_key_masked(
          gpu(q_cpu),
          gpu(k_cpu),
          gpu(v_cpu),
          scale,
          gpu(key_mask_padded),
          sinks: gpu(sinks_cpu)
        )
        |> Nx.backend_transfer()

      causal_keep = causal_keep_mask(t_q, t_kv)
      km_keep = Nx.equal(key_mask_padded, 1) |> Nx.reshape({1, t_kv}) |> Nx.broadcast({t_q, t_kv})
      keep_mask = Nx.logical_and(causal_keep, km_keep)

      out_ref = reference_sdpa_sinks(q_cpu, k_cpu, v_cpu, scale, sinks_cpu, keep_mask)

      assert_all_close(out_fast, out_ref, atol: @tol)
    end

    test "omitting :sinks is unchanged from the pre-sinks arity" do
      q = Nx.broadcast(0.1, {1, 16, 1, 64}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.1, {1, 8, 10, 64}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.1, {1, 8, 10, 64}) |> Nx.as_type(:f16) |> gpu()
      scale = 1.0 / :math.sqrt(64)
      key_mask = Nx.broadcast(1, {1, 10}) |> gpu()

      out_5arity = Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask)

      out_empty_opts =
        Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask, [])

      assert Nx.all_close(out_5arity, out_empty_opts) |> Nx.to_number() == 1
    end
  end
end
