defmodule EMLXAxon.Qwen3PluginTestAdapter do
  @moduledoc false

  alias EMLXAxon.Qwen3.Native

  def kv_cache_attention(q, k, v, k_cache, v_cache, offset, scale, head_dim, theta) do
    Native.kv_cache_attention(
      tensor(q),
      tensor(k),
      tensor(v),
      tensor(k_cache),
      tensor(v_cache),
      offset,
      scale,
      head_dim,
      theta
    )
    |> raw_tuple()
  end

  def mlp(hidden, norm, gate, up, down, eps) do
    Native.mlp(tensor(hidden), tensor(norm), tensor(gate), tensor(up), tensor(down), eps)
    |> raw()
  end

  def attention_residual(hidden, attention, projection) do
    Native.attention_residual(tensor(hidden), tensor(attention), tensor(projection)) |> raw()
  end

  def attention_block(
        hidden,
        norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        k_cache,
        v_cache,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    Native.attention_block(
      tensor(hidden),
      tensor(norm),
      tensor(q_proj),
      tensor(k_proj),
      tensor(v_proj),
      tensor(o_proj),
      tensor(q_norm),
      tensor(k_norm),
      tensor(k_cache),
      tensor(v_cache),
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_tuple()
  end

  def layer(
        hidden,
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        k_cache,
        v_cache,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    Native.layer_dense(
      tensor(hidden),
      tensor(norm1),
      tensor(q_proj),
      tensor(k_proj),
      tensor(v_proj),
      tensor(o_proj),
      tensor(q_norm),
      tensor(k_norm),
      tensor(k_cache),
      tensor(v_cache),
      tensor(norm2),
      tensor(gate_proj),
      tensor(up_proj),
      tensor(down_proj),
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_tuple()
  end

  def layer_quantized(
        hidden,
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        k_cache,
        v_cache,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    Native.layer_generalized(
      tensor(hidden),
      tensor(norm1),
      tensor(q_proj),
      tensor(k_proj),
      tensor(v_proj),
      tensor(o_proj),
      tensor(q_norm),
      tensor(k_norm),
      tensor(k_cache),
      tensor(v_cache),
      tensor(norm2),
      tensor(gate_proj),
      tensor(up_proj),
      tensor(down_proj),
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_tuple()
  end

  def final_greedy(hidden, norm, lm_head, eps) do
    Native.final_greedy(tensor(hidden), tensor(norm), tensor(lm_head), eps) |> raw()
  end

  def forward_greedy_ids(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    validate_single_batch!(tensor(input_ids), "forward_greedy_ids_token_id")
    embed_tokens = tensor(embed_tokens)
    hidden = embed(tensor(input_ids), embed_tokens)

    Native.forward_greedy_dense(
      hidden,
      tensor_layers(layers),
      tensor_cache(kv_cache),
      tensor(norm),
      tensor(lm_head),
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_forward()
  end

  def forward_greedy_ids_token_id(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    {token, cache} =
      forward_greedy_ids(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        scale,
        head_dim,
        theta,
        eps
      )

    {token |> tensor() |> Nx.squeeze() |> Nx.to_number(), cache}
  end

  def forward_greedy_token_id(
        token_id,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        scale,
        head_dim,
        theta,
        eps
      ) do
    embed = tensor(embed_tokens)
    ids = Nx.tensor([[token_id]], type: :s64, backend: backend(embed))

    forward_greedy_ids_token_id(
      ids,
      embed,
      layers,
      kv_cache,
      norm,
      lm_head,
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
  end

  def forward_greedy_ids_chunk(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        count,
        scale,
        head_dim,
        theta,
        eps
      ) do
    validate_decode_input!(tensor(input_ids), "forward_greedy_ids_chunk")

    Native.forward_greedy_chunk_dense(
      tensor(input_ids),
      tensor(embed_tokens),
      tensor_layers(layers),
      tensor_cache(kv_cache),
      tensor(norm),
      tensor(lm_head),
      offset,
      count,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_chunk()
  end

  def forward_greedy_ids_chunk_quantized(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        offset,
        count,
        scale,
        head_dim,
        theta,
        eps
      ) do
    validate_decode_input!(tensor(input_ids), "forward_greedy_ids_chunk_quantized")

    Native.forward_greedy_chunk_generalized(
      tensor(input_ids),
      tensor(embed_tokens),
      tensor_layers(layers),
      tensor_cache(kv_cache),
      tensor(norm),
      tensor(lm_head),
      offset,
      count,
      scale,
      head_dim,
      theta,
      eps
    )
    |> raw_chunk()
  end

  defp tensor(%Nx.Tensor{} = tensor), do: tensor
  defp tensor(ref), do: EMLX.Backend.to_nx(ref)

  defp tensor_layers(layers),
    do:
      Enum.map(
        layers,
        &(&1 |> Tuple.to_list() |> Enum.map(fn value -> tensor(value) end) |> List.to_tuple())
      )

  defp tensor_cache(cache), do: Enum.map(cache, fn {k, v} -> {tensor(k), tensor(v)} end)

  defp embed(input_ids, embed_tokens) do
    ids = Nx.squeeze(input_ids, axes: [0])
    embed_tokens |> Nx.take(ids, axis: 0) |> Nx.new_axis(0)
  end

  defp backend(%Nx.Tensor{data: %EMLX.Backend{ref: {device, _ref}}}),
    do: {EMLX.Backend, device: device}

  defp raw(%Nx.Tensor{} = tensor), do: EMLX.Backend.from_nx(tensor)
  defp raw_tuple(tuple), do: tuple |> Tuple.to_list() |> Enum.map(&raw/1) |> List.to_tuple()

  defp raw_forward({token, cache}) do
    {raw(token), cache |> Tuple.to_list() |> Enum.map(fn {k, v} -> {raw(k), raw(v)} end)}
  end

  defp raw_chunk({tokens, cache}) do
    token_refs = tokens |> Nx.to_batched(1) |> Enum.map(&raw/1)

    {token_refs, cache |> Tuple.to_list() |> Enum.map(fn {k, v} -> {raw(k), raw(v)} end)}
  end

  defp validate_single_batch!(input_ids, name) do
    case Nx.shape(input_ids) do
      {1, _sequence} ->
        :ok

      {batch, _sequence} ->
        raise ArgumentError, "#{name} requires batch size 1, got batch size #{batch}"

      shape ->
        raise ArgumentError, "#{name} expects rank 2 input_ids, got: #{inspect(shape)}"
    end
  end

  defp validate_decode_input!(input_ids, name) do
    validate_single_batch!(input_ids, name)

    case Nx.shape(input_ids) do
      {1, 1} ->
        :ok

      {1, sequence} ->
        raise ArgumentError, "#{name} requires sequence length 1, got sequence length #{sequence}"
    end
  end
end

defmodule EMLXAxon.Qwen3NativeTest do
  @moduledoc """
  Unit tests for the Qwen3 operations backed by the standalone compute plugin
  loaded from this project
  (see `c_src/qwen3_plugin.cpp` and `EMLXAxon.Application`).

  Each test verifies:
  1. The plugin operation returns a tensor with the correct shape and dtype.
  2. Numerical output is close to the equivalent pure-Nx reference.

  All tests require Metal (tagged :metal).
  """
  use ExUnit.Case, async: false
  import Nx.Testing

  alias EMLXAxon.Qwen3PluginTestAdapter, as: Qwen3Plugin

  @moduletag :metal

  defp gpu(tensor), do: Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})

  describe "Qwen3Plugin.kv_cache_attention/9 validation" do
    test "rejects malformed Q rank before reading shapes" do
      q = Nx.broadcast(0.0, {1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 4}) |> Nx.as_type(:f16) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 4}) |> Nx.as_type(:f16) |> gpu()

      assert_raise ArgumentError, ~r/query expects rank 4/, fn ->
        Qwen3Plugin.kv_cache_attention(
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
        Qwen3Plugin.kv_cache_attention(
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
        Qwen3Plugin.kv_cache_attention(
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
        Qwen3Plugin.kv_cache_attention(
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
        Qwen3Plugin.kv_cache_attention(
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
        Qwen3Plugin.mlp(
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
        Qwen3Plugin.attention_block(
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
        Qwen3Plugin.attention_block(
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
        Qwen3Plugin.layer(
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

    test "qwen3_layer matches pure Nx reference for batched GQA" do
      fixtures = qwen3_batched_dense_attention_fixtures()
      scale = 1.0 / :math.sqrt(fixtures.head_dim)

      {out_ref, k_ref, v_ref} =
        Qwen3Plugin.layer(
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

    # ── qwen3_layer_quantized (dense-or-quantized per-projection fusion) ──────

    @qwen3_layer_quantized_modes [
      {"affine", 32, 4},
      {"mxfp4", 32, 4}
    ]

    for {mode, group_size, bits} <- @qwen3_layer_quantized_modes do
      test "qwen3_layer_quantized (#{mode}) matches per-op quantized reference for prefill (T_new > 1)" do
        fixtures =
          qwen3_quantized_attention_fixtures(
            [:q_proj, :k_proj, :v_proj, :o_proj, :gate_proj, :up_proj, :down_proj],
            group_size: unquote(group_size),
            bits: unquote(bits),
            mode: unquote(mode)
          )

        {out_ref, k_ref, v_ref} = qwen3_layer_quantized_call(fixtures)
        {expected, expected_k, expected_v} = qwen3_layer_quantized_reference(fixtures)

        assert_all_close(
          out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
          expected,
          atol: 5.0e-2,
          rtol: 5.0e-2
        )

        assert_all_close(
          k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
          expected_k,
          atol: 5.0e-2,
          rtol: 5.0e-2
        )

        assert_all_close(
          v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
          expected_v,
          atol: 5.0e-2,
          rtol: 5.0e-2
        )
      end
    end

    test "qwen3_layer_quantized matches per-op quantized reference for decode (T_new == 1)" do
      fixtures =
        [:q_proj, :k_proj, :v_proj, :o_proj, :gate_proj, :up_proj, :down_proj]
        |> qwen3_quantized_attention_fixtures()
        |> Map.update!(:hidden, &Nx.slice_along_axis(&1, 0, 1, axis: 1))

      {out_ref, k_ref, v_ref} = qwen3_layer_quantized_call(fixtures)
      {expected, expected_k, expected_v} = qwen3_layer_quantized_reference(fixtures)

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 5.0e-2,
        rtol: 5.0e-2
      )

      assert_all_close(
        k_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_k,
        atol: 5.0e-2,
        rtol: 5.0e-2
      )

      assert_all_close(
        v_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected_v,
        atol: 5.0e-2,
        rtol: 5.0e-2
      )
    end

    test "qwen3_layer_quantized matches per-op reference for a mixed dense/quantized layer" do
      # Only the MLP projections are quantized; q/k/v/o stay dense — exercises
      # `Qwen3LinearWeight`/`qwen3_apply_linear` picking the right branch
      # independently per projection within one layer.
      fixtures = qwen3_quantized_attention_fixtures([:gate_proj, :up_proj, :down_proj])

      {out_ref, k_ref, v_ref} = qwen3_layer_quantized_call(fixtures)
      {expected, expected_k, expected_v} = qwen3_layer_quantized_reference(fixtures)

      assert_all_close(
        out_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected,
        atol: 5.0e-2,
        rtol: 5.0e-2
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

    # ── qwen3_forward_greedy_ids_chunk_quantized (multi-step chunk fusion) ────

    test "qwen3_forward_greedy_ids_chunk_quantized produces identical token ids to " <>
           "running the quantized single-layer NIF + manual final step N times" do
      # Each fused plugin call `std::move`s the k/v caches it receives (mirroring
      # `qwen3_layer`'s existing move-and-return-updated-cache contract), so
      # the two independent call paths below must not share the same
      # underlying cache resource — build separate fixture instances rather
      # than reusing one across both.
      count = 3

      chunk_tokens = qwen3_quantized_chunk_call(qwen3_quantized_chunk_fixtures(), count)
      manual_tokens = qwen3_quantized_chunk_manual_tokens(qwen3_quantized_chunk_fixtures(), count)

      assert chunk_tokens == manual_tokens
    end

    test "qwen3_forward_greedy_ids_chunk_quantized returns 1023 outputs without truncation" do
      count = 1023
      tokens = qwen3_quantized_chunk_call(qwen3_quantized_chunk_fixtures(count), count)

      expected_prefix =
        qwen3_quantized_chunk_manual_tokens(qwen3_quantized_chunk_fixtures(), 3)

      assert length(tokens) == count
      assert Enum.all?(tokens, &(length(&1) == 1))
      assert Enum.take(tokens, 3) == expected_prefix
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
        Qwen3Plugin.attention_residual(
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

    test "qwen3_final_greedy matches pure Nx reference" do
      {hidden, norm, lm_head} =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          {
            Nx.iota({1, 2, 4}, type: :f32) |> Nx.add(1) |> Nx.divide(10),
            Nx.tensor([1.0, 1.1, 0.9, 1.2], type: :f32),
            Nx.tensor(
              [
                [0.2, 0.1, 0.0, 0.3],
                [0.0, 0.5, 0.2, 0.1],
                [0.3, 0.1, 0.4, 0.2]
              ],
              type: :f32
            )
          }
        end)

      eps = 1.0e-6

      token_ref =
        Qwen3Plugin.final_greedy(
          EMLX.Backend.from_nx(gpu(hidden)),
          EMLX.Backend.from_nx(gpu(norm)),
          EMLX.Backend.from_nx(gpu(lm_head)),
          eps
        )

      expected =
        hidden
        |> qwen3_final_logits_reference(norm, lm_head, eps)
        |> Nx.argmax(axis: 1)

      assert_all_close(
        token_ref |> EMLX.Backend.to_nx() |> Nx.backend_transfer(Nx.BinaryBackend),
        expected
      )
    end

    test "qwen3_forward_greedy_token_id defaults to embedding tensor device" do
      fixtures = qwen3_dense_attention_fixtures()
      cpu = fn tensor -> Nx.backend_transfer(tensor, {EMLX.Backend, device: :cpu}) end

      layer = {
        cpu.(fixtures.norm1),
        cpu.(fixtures.norm2),
        cpu.(fixtures.q_norm),
        cpu.(fixtures.k_norm),
        cpu.(fixtures.q_proj),
        cpu.(fixtures.k_proj),
        cpu.(fixtures.v_proj),
        cpu.(fixtures.o_proj),
        cpu.(fixtures.gate_proj),
        cpu.(fixtures.up_proj),
        cpu.(fixtures.down_proj)
      }

      kv_cache =
        {EMLX.Backend.from_nx(cpu.(fixtures.k_cache)),
         EMLX.Backend.from_nx(cpu.(fixtures.v_cache))}

      embed_tokens =
        Nx.tensor(List.duplicate(List.duplicate(0.1, 4), 4),
          type: :f32,
          backend: {EMLX.Backend, device: :cpu}
        )

      norm = Nx.tensor(List.duplicate(1.0, 4), type: :f32, backend: {EMLX.Backend, device: :cpu})

      lm_head =
        Nx.tensor(List.duplicate(List.duplicate(0.1, 4), 4),
          type: :f32,
          backend: {EMLX.Backend, device: :cpu}
        )

      assert {:cpu, _ref} = EMLX.Backend.from_nx(embed_tokens)

      {_token_id, [{{k_device, _k_ref}, {v_device, _v_ref}}]} =
        Qwen3Plugin.forward_greedy_token_id(
          0,
          EMLX.Backend.from_nx(embed_tokens),
          [layer],
          [kv_cache],
          EMLX.Backend.from_nx(norm),
          EMLX.Backend.from_nx(lm_head),
          0,
          1.0 / :math.sqrt(fixtures.head_dim),
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      assert k_device == :cpu
      assert v_device == :cpu
    end

    test "qwen3_forward_greedy_ids returns tensor token matching token id path" do
      fixtures = qwen3_dense_attention_fixtures()
      input_ids = Nx.tensor([[0]], type: :s64) |> gpu()

      embed_tokens =
        Nx.tensor(List.duplicate(List.duplicate(0.1, 4), 4), type: :f32)
        |> gpu()

      norm = Nx.tensor(List.duplicate(1.0, 4), type: :f32) |> gpu()

      lm_head =
        Nx.tensor(List.duplicate(List.duplicate(0.1, 4), 4), type: :f32)
        |> gpu()

      layer = {
        gpu(fixtures.norm1),
        gpu(fixtures.norm2),
        gpu(fixtures.q_norm),
        gpu(fixtures.k_norm),
        gpu(fixtures.q_proj),
        gpu(fixtures.k_proj),
        gpu(fixtures.v_proj),
        gpu(fixtures.o_proj),
        gpu(fixtures.gate_proj),
        gpu(fixtures.up_proj),
        gpu(fixtures.down_proj)
      }

      {token_ref, _kv_cache} =
        Qwen3Plugin.forward_greedy_ids(
          EMLX.Backend.from_nx(input_ids),
          EMLX.Backend.from_nx(embed_tokens),
          [layer],
          [{gpu(fixtures.k_cache), gpu(fixtures.v_cache)}],
          EMLX.Backend.from_nx(norm),
          EMLX.Backend.from_nx(lm_head),
          0,
          1.0 / :math.sqrt(fixtures.head_dim),
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      {token_id, _kv_cache} =
        Qwen3Plugin.forward_greedy_token_id(
          0,
          EMLX.Backend.from_nx(embed_tokens),
          [layer],
          [{gpu(fixtures.k_cache), gpu(fixtures.v_cache)}],
          EMLX.Backend.from_nx(norm),
          EMLX.Backend.from_nx(lm_head),
          0,
          1.0 / :math.sqrt(fixtures.head_dim),
          fixtures.head_dim,
          fixtures.theta,
          fixtures.eps
        )

      token =
        token_ref
        |> EMLX.Backend.to_nx()
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.to_flat_list()
        |> hd()

      assert token == token_id
    end
  end

  describe "Qwen3Plugin.attention_block/15 validation" do
    test "qwen3_kv_cache_attention rejects required cache length overflow before graph construction" do
      q = Nx.broadcast(0.0, {1, 1, 2, 2}) |> Nx.as_type(:f32) |> gpu()
      k = Nx.broadcast(0.0, {1, 1, 1, 2}) |> Nx.as_type(:f32) |> gpu()
      v = Nx.broadcast(0.0, {1, 1, 1, 2}) |> Nx.as_type(:f32) |> gpu()
      k_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32) |> gpu()
      v_cache = Nx.broadcast(0.0, {1, 1, 4, 2}) |> Nx.as_type(:f32) |> gpu()

      assert_raise EMLX.NIFError,
                   ~r/KV cache capacity 4 is smaller than required length 2147483648/,
                   fn ->
                     Qwen3Plugin.kv_cache_attention(
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
                     Qwen3Plugin.layer(
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
                     Qwen3Plugin.attention_block(
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

    test "qwen3_forward_greedy_ids_token_id rejects batches before layer/cache normalization" do
      input_ids = Nx.tensor([[0], [1]], type: :s64) |> gpu()
      embed_tokens = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f32) |> gpu()
      norm = Nx.broadcast(1.0, {4}) |> Nx.as_type(:f32) |> gpu()
      lm_head = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f32) |> gpu()

      assert_raise ArgumentError,
                   ~r/forward_greedy_ids_token_id requires batch size 1, got batch size 2/,
                   fn ->
                     Qwen3Plugin.forward_greedy_ids_token_id(
                       EMLX.Backend.from_nx(input_ids),
                       EMLX.Backend.from_nx(embed_tokens),
                       [:invalid_layer],
                       [:invalid_cache],
                       EMLX.Backend.from_nx(norm),
                       EMLX.Backend.from_nx(lm_head),
                       0,
                       1.0 / :math.sqrt(2),
                       2,
                       10_000.0,
                       1.0e-6
                     )
                   end
    end

    test "qwen3_forward_greedy_ids_chunk rejects input that is not a decode step before layer/cache normalization" do
      input_ids = Nx.tensor([[0, 1]], type: :s64) |> gpu()
      embed_tokens = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f32) |> gpu()
      norm = Nx.broadcast(1.0, {4}) |> Nx.as_type(:f32) |> gpu()
      lm_head = Nx.broadcast(0.1, {4, 4}) |> Nx.as_type(:f32) |> gpu()

      assert_raise ArgumentError,
                   ~r/forward_greedy_ids_chunk requires sequence length 1, got sequence length 2/,
                   fn ->
                     Qwen3Plugin.forward_greedy_ids_chunk(
                       EMLX.Backend.from_nx(input_ids),
                       EMLX.Backend.from_nx(embed_tokens),
                       [:invalid_layer],
                       [:invalid_cache],
                       EMLX.Backend.from_nx(norm),
                       EMLX.Backend.from_nx(lm_head),
                       0,
                       1,
                       1.0 / :math.sqrt(2),
                       2,
                       10_000.0,
                       1.0e-6
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
                     Qwen3Plugin.attention_block(
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
        norm2: Nx.tensor([1.2, 0.8, 1.1, 0.95], type: :f32),
        q_norm: Nx.tensor([1.0, 1.1], type: :f32),
        k_norm: Nx.tensor([0.9, 1.2], type: :f32),
        q_proj: Nx.iota({4, 8}, type: :f32) |> Nx.add(1) |> Nx.divide(80),
        k_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(3) |> Nx.divide(90),
        v_proj: Nx.iota({4, 4}, type: :f32) |> Nx.add(5) |> Nx.divide(100),
        o_proj: Nx.iota({8, 4}, type: :f32) |> Nx.add(7) |> Nx.divide(110),
        gate_proj: Nx.iota({4, 6}, type: :f32) |> Nx.add(13) |> Nx.divide(120),
        up_proj: Nx.iota({4, 6}, type: :f32) |> Nx.add(29) |> Nx.divide(130),
        down_proj: Nx.iota({6, 4}, type: :f32) |> Nx.add(41) |> Nx.divide(140),
        k_cache: Nx.iota({2, 2, 6, 2}, type: :f32) |> Nx.add(11) |> Nx.divide(1_000),
        v_cache: Nx.iota({2, 2, 6, 2}, type: :f32) |> Nx.add(101) |> Nx.divide(1_000),
        offset: 2,
        head_dim: 2,
        theta: 10_000.0,
        eps: 1.0e-6
      }
    end)
  end

  # Passes a tensor through unchanged if it's already GPU/EMLX-backed
  # (quantized fixtures come back from `EMLX.quantize/2` that way); otherwise
  # transfers it, mirroring the dense `gpu/1` helper above.
  defp ensure_gpu(%Nx.Tensor{data: %EMLX.Backend{}} = tensor), do: tensor
  defp ensure_gpu(tensor), do: gpu(tensor)

  # Builds a Qwen3 single-layer fixture (H=32, head_dim=16, N_q=2, N_kv=1,
  # intermediate=32 — all input widths are 32, satisfying `EMLX.quantize/2`'s
  # supported group sizes {32, 64, 128} and every microscaled mode's fixed
  # group_size) and quantizes the projections listed in `quantized_slots` via
  # `EMLX.quantize/2`.
  #
  # For each quantized slot, also stashes a `<slot>_reference` dequantized
  # tensor in the *same* {in, out} convention as the dense fixtures, so
  # `qwen3_layer_quantized_reference/1` can reuse the existing dense-only
  # `qwen3_attention_block_reference/1`/`qwen3_mlp_reference/6` helpers
  # unchanged — isolating fusion correctness from quantization lossiness
  # (both sides of the comparison use the same quantized-then-dequantized
  # numbers).
  defp qwen3_quantized_attention_fixtures(quantized_slots, opts \\ []) do
    group_size = Keyword.get(opts, :group_size, 32)
    bits = Keyword.get(opts, :bits, 4)
    mode = Keyword.get(opts, :mode, "affine")

    base =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        %{
          hidden: Nx.iota({1, 2, 32}, type: :f32) |> Nx.add(1) |> Nx.divide(400),
          norm1: Nx.iota({32}, type: :f32) |> Nx.add(10) |> Nx.divide(10),
          norm2: Nx.iota({32}, type: :f32) |> Nx.add(20) |> Nx.divide(10),
          q_norm: Nx.iota({16}, type: :f32) |> Nx.add(30) |> Nx.divide(10),
          k_norm: Nx.iota({16}, type: :f32) |> Nx.add(40) |> Nx.divide(10),
          q_proj: Nx.iota({32, 32}, type: :f32) |> Nx.add(1) |> Nx.divide(900),
          k_proj: Nx.iota({32, 16}, type: :f32) |> Nx.add(2) |> Nx.divide(900),
          v_proj: Nx.iota({32, 16}, type: :f32) |> Nx.add(3) |> Nx.divide(900),
          o_proj: Nx.iota({32, 32}, type: :f32) |> Nx.add(4) |> Nx.divide(900),
          gate_proj: Nx.iota({32, 32}, type: :f32) |> Nx.add(5) |> Nx.divide(900),
          up_proj: Nx.iota({32, 32}, type: :f32) |> Nx.add(6) |> Nx.divide(900),
          down_proj: Nx.iota({32, 32}, type: :f32) |> Nx.add(7) |> Nx.divide(900),
          k_cache: Nx.broadcast(0.0, {1, 1, 4, 16}) |> Nx.as_type(:f32),
          v_cache: Nx.broadcast(0.0, {1, 1, 4, 16}) |> Nx.as_type(:f32),
          offset: 0,
          head_dim: 16,
          theta: 10_000.0,
          eps: 1.0e-6
        }
      end)

    Enum.reduce(quantized_slots, base, fn slot, acc ->
      dense_w = Map.fetch!(acc, slot)

      qw =
        dense_w
        |> gpu()
        |> Nx.transpose()
        |> EMLX.quantize(type: {:s, bits}, group_size: group_size, mode: mode)

      dequantized_in_out =
        qw
        |> EMLX.dequantize()
        |> Nx.transpose()
        |> Nx.backend_transfer(Nx.BinaryBackend)

      acc
      |> Map.put(slot, qw)
      |> Map.put(:"#{slot}_reference", dequantized_in_out)
    end)
  end

  defp qwen3_layer_quantized_call(fixtures) do
    scale = 1.0 / :math.sqrt(fixtures.head_dim)

    Qwen3Plugin.layer_quantized(
      EMLX.Backend.from_nx(gpu(fixtures.hidden)),
      ensure_gpu(fixtures.norm1),
      ensure_gpu(fixtures.q_proj),
      ensure_gpu(fixtures.k_proj),
      ensure_gpu(fixtures.v_proj),
      ensure_gpu(fixtures.o_proj),
      ensure_gpu(fixtures.q_norm),
      ensure_gpu(fixtures.k_norm),
      EMLX.Backend.from_nx(gpu(fixtures.k_cache)),
      EMLX.Backend.from_nx(gpu(fixtures.v_cache)),
      ensure_gpu(fixtures.norm2),
      ensure_gpu(fixtures.gate_proj),
      ensure_gpu(fixtures.up_proj),
      ensure_gpu(fixtures.down_proj),
      fixtures.offset,
      scale,
      fixtures.head_dim,
      fixtures.theta,
      fixtures.eps
    )
  end

  defp qwen3_layer_quantized_reference(fixtures) do
    dense_fixtures =
      Enum.reduce(
        [:q_proj, :k_proj, :v_proj, :o_proj, :gate_proj, :up_proj, :down_proj],
        fixtures,
        fn slot, acc ->
          case Map.fetch(fixtures, :"#{slot}_reference") do
            {:ok, dequantized} -> Map.put(acc, slot, dequantized)
            :error -> acc
          end
        end
      )

    {attn_expected, expected_k, expected_v} = qwen3_attention_block_reference(dense_fixtures)

    expected =
      qwen3_mlp_reference(
        attn_expected,
        dense_fixtures.norm2,
        dense_fixtures.gate_proj,
        dense_fixtures.up_proj,
        dense_fixtures.down_proj,
        dense_fixtures.eps
      )

    {expected, expected_k, expected_v}
  end

  # Single-layer fixture for `qwen3_forward_greedy_ids_chunk_quantized`
  # determinism testing: same projection shapes as
  # `qwen3_quantized_attention_fixtures/2`, plus a small embedding table and a
  # quantized `lm_head` (`{vocab, H}`, already in the `{out, in}` convention
  # `EMLX.quantize/2` expects — no transpose needed, unlike the projections).
  defp qwen3_quantized_chunk_fixtures(cache_capacity \\ 8) do
    base =
      qwen3_quantized_attention_fixtures([
        :q_proj,
        :k_proj,
        :v_proj,
        :o_proj,
        :gate_proj,
        :up_proj,
        :down_proj
      ])

    embed_tokens =
      Nx.iota({6, 32}, type: :f32) |> Nx.add(1) |> Nx.divide(900) |> gpu()

    lm_head =
      Nx.iota({6, 32}, type: :f32)
      |> Nx.add(2)
      |> Nx.divide(900)
      |> gpu()
      |> EMLX.quantize(type: {:s, 4}, group_size: 32, mode: "affine")

    norm = Nx.iota({32}, type: :f32) |> Nx.add(50) |> Nx.divide(10) |> gpu()

    k_cache =
      Nx.broadcast(0.0, {1, 1, cache_capacity, 16})
      |> Nx.as_type(:f32)
      |> gpu()

    v_cache =
      Nx.broadcast(0.0, {1, 1, cache_capacity, 16})
      |> Nx.as_type(:f32)
      |> gpu()

    %{
      base
      | k_cache: k_cache,
        v_cache: v_cache
    }
    |> Map.put(:embed_tokens, embed_tokens)
    |> Map.put(:lm_head, lm_head)
    |> Map.put(:norm, norm)
    |> Map.put(:input_id, 0)
  end

  defp qwen3_quantized_chunk_call(fixtures, count) do
    scale = 1.0 / :math.sqrt(fixtures.head_dim)
    input_ids = Nx.tensor([[fixtures.input_id]], type: :s64) |> gpu()

    layer = {
      ensure_gpu(fixtures.norm1),
      ensure_gpu(fixtures.norm2),
      ensure_gpu(fixtures.q_norm),
      ensure_gpu(fixtures.k_norm),
      ensure_gpu(fixtures.q_proj),
      ensure_gpu(fixtures.k_proj),
      ensure_gpu(fixtures.v_proj),
      ensure_gpu(fixtures.o_proj),
      ensure_gpu(fixtures.gate_proj),
      ensure_gpu(fixtures.up_proj),
      ensure_gpu(fixtures.down_proj)
    }

    {token_refs, _kv_cache} =
      Qwen3Plugin.forward_greedy_ids_chunk_quantized(
        EMLX.Backend.from_nx(input_ids),
        EMLX.Backend.from_nx(fixtures.embed_tokens),
        [layer],
        [{fixtures.k_cache, fixtures.v_cache}],
        EMLX.Backend.from_nx(fixtures.norm),
        fixtures.lm_head,
        fixtures.offset,
        count,
        scale,
        fixtures.head_dim,
        fixtures.theta,
        fixtures.eps
      )

    Enum.map(token_refs, &(&1 |> EMLX.Backend.to_nx() |> Nx.to_flat_list()))
  end

  # Manually replays the same `count` decode steps by calling the
  # already-verified single-layer `qwen3_layer_quantized` plugin operation plus a manual
  # final RMSNorm + `EMLX.quantized_matmul` lm_head + argmax step, once per
  # step. Both paths run the exact same MLX ops (just organized as 1 fused
  # plugin call vs `count` separate calls), so token ids should be bit-identical.
  defp qwen3_quantized_chunk_manual_tokens(fixtures, count) do
    scale = 1.0 / :math.sqrt(fixtures.head_dim)

    {_ids, _kv, tokens_rev} =
      Enum.reduce(1..count, {fixtures.input_id, {fixtures.k_cache, fixtures.v_cache}, []}, fn
        step, {token_id, {k_cache, v_cache}, acc} ->
          ids = Nx.tensor([token_id], type: :s64) |> gpu()

          hidden =
            fixtures.embed_tokens
            |> Nx.take(ids, axis: 0)
            |> Nx.new_axis(0)

          {hidden_ref, k_ref, v_ref} =
            Qwen3Plugin.layer_quantized(
              EMLX.Backend.from_nx(hidden),
              ensure_gpu(fixtures.norm1),
              ensure_gpu(fixtures.q_proj),
              ensure_gpu(fixtures.k_proj),
              ensure_gpu(fixtures.v_proj),
              ensure_gpu(fixtures.o_proj),
              ensure_gpu(fixtures.q_norm),
              ensure_gpu(fixtures.k_norm),
              EMLX.Backend.from_nx(k_cache),
              EMLX.Backend.from_nx(v_cache),
              ensure_gpu(fixtures.norm2),
              ensure_gpu(fixtures.gate_proj),
              ensure_gpu(fixtures.up_proj),
              ensure_gpu(fixtures.down_proj),
              fixtures.offset + step - 1,
              scale,
              fixtures.head_dim,
              fixtures.theta,
              fixtures.eps
            )

          hidden_out = EMLX.Backend.to_nx(hidden_ref)
          k_new = EMLX.Backend.to_nx(k_ref)
          v_new = EMLX.Backend.to_nx(v_ref)

          normed =
            hidden_out
            |> Nx.reshape({1, 32})
            |> EMLX.Fast.rms_norm(fixtures.norm, fixtures.eps)

          logits = EMLX.Quantization.quantized_matmul(normed, fixtures.lm_head)
          token = Nx.argmax(logits, axis: 1)
          token_id = token |> Nx.to_flat_list() |> hd()

          {token_id, {k_new, v_new}, [Nx.to_flat_list(token) | acc]}
      end)

    Enum.reverse(tokens_rev)
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

  defp qwen3_final_logits_reference(hidden, norm, lm_head, eps) do
    {_batch, seq_len, hidden_size} = Nx.shape(hidden)

    last =
      hidden
      |> Nx.slice([0, seq_len - 1, 0], [1, 1, hidden_size])
      |> Nx.reshape({1, hidden_size})

    last
    |> rms_norm_reference(norm, eps)
    |> Nx.dot([1], lm_head, [1])
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
end
