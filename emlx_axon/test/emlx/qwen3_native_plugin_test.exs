defmodule EMLXAxon.Qwen3NativePluginTest do
  use ExUnit.Case, async: false

  import Nx.Defn

  alias EMLXAxon.Qwen3.{Model, Native}

  defn compiled_mlp(hidden, norm, gate, up, down) do
    Native.mlp(hidden, norm, gate, up, down, 1.0e-5)
  end

  defn compiled_attention(query, key, value, k_cache, v_cache) do
    Native.kv_cache_attention(
      query,
      key,
      value,
      k_cache,
      v_cache,
      0,
      0.7071067811865475,
      2,
      10_000.0
    )
  end

  defn compiled_attention_loop(query, key, value, k_cache, v_cache, offset) do
    initial_attention = Nx.broadcast(Nx.tensor(0.0, type: :f32), {1, 1, 4})

    while {_attention = initial_attention, carried_query = query, carried_key = key,
           carried_value = value, current_k = k_cache, current_v = v_cache,
           current_offset = offset},
          current_offset < 2 do
      {next_attention, next_k, next_v} =
        Native.kv_cache_attention_tensor_offset(
          carried_query,
          carried_key,
          carried_value,
          current_k,
          current_v,
          current_offset,
          0.7071067811865475,
          2,
          10_000.0
        )

      {next_attention, carried_query, carried_key, carried_value, next_k, next_v,
       current_offset + 1}
    end
  end

  defn compiled_forward_dense(hidden, layer, k_cache, v_cache, norm, lm_head) do
    Native.forward_greedy_dense(
      hidden,
      [layer],
      [{k_cache, v_cache}],
      norm,
      lm_head,
      0,
      0.7071067811865475,
      2,
      10_000.0,
      1.0e-6
    )
  end

  defn compiled_chunk_dense(input_ids, embed, layer, k_cache, v_cache, norm, lm_head) do
    Native.forward_greedy_chunk_dense(
      input_ids,
      embed,
      [layer],
      [{k_cache, v_cache}],
      norm,
      lm_head,
      0,
      3,
      0.7071067811865475,
      2,
      10_000.0,
      1.0e-6
    )
  end

  defn compiled_chunk_generalized(input_ids, embed, layer, k_cache, v_cache, norm, lm_head) do
    Native.forward_greedy_chunk_generalized(
      input_ids,
      embed,
      [layer],
      [{k_cache, v_cache}],
      norm,
      lm_head,
      0,
      3,
      0.7071067811865475,
      2,
      10_000.0,
      1.0e-6
    )
  end

  test "dense MLP matches an Nx reference eagerly and under EMLX compilation" do
    hidden = emlx_tensor([[[0.2, -0.4, 0.8, 0.1], [0.5, 0.3, -0.2, 0.7]]])
    norm = emlx_tensor([1.0, 0.8, 1.2, 0.9])

    gate =
      emlx_tensor([
        [0.1, 0.2, -0.1, 0.3, 0.4, -0.2],
        [0.3, -0.2, 0.5, 0.1, -0.4, 0.2],
        [-0.1, 0.4, 0.2, -0.3, 0.1, 0.5],
        [0.2, 0.1, -0.4, 0.6, 0.3, -0.1]
      ])

    up =
      emlx_tensor([
        [0.2, -0.1, 0.3, 0.4, -0.2, 0.1],
        [-0.3, 0.5, 0.2, -0.1, 0.4, 0.3],
        [0.4, 0.1, -0.2, 0.2, 0.5, -0.4],
        [0.1, 0.3, 0.4, -0.2, 0.2, 0.6]
      ])

    down =
      emlx_tensor([
        [0.1, 0.2, -0.1, 0.3],
        [-0.2, 0.4, 0.3, 0.1],
        [0.5, -0.1, 0.2, -0.3],
        [0.3, 0.2, 0.1, 0.4],
        [-0.1, 0.3, 0.4, 0.2],
        [0.2, -0.4, 0.3, 0.5]
      ])

    expected = reference_mlp(hidden, norm, gate, up, down, 1.0e-5)
    eager = Native.mlp(hidden, norm, gate, up, down, 1.0e-5)

    compiled =
      Nx.Defn.jit(&compiled_mlp/5, compiler: EMLX).(hidden, norm, gate, up, down)

    assert_close(eager, expected)
    assert_close(compiled, expected)
  end

  test "dense MLP lowering emits one plugin instruction and no runtime call" do
    templates = [
      Nx.template({1, 2, 4}, :f32),
      Nx.template({4}, :f32),
      Nx.template({4, 6}, :f32),
      Nx.template({4, 6}, :f32),
      Nx.template({6, 4}, :f32)
    ]

    wire =
      Nx.Defn.debug_expr_apply(&compiled_mlp/5, templates)
      |> EMLX.Native.Expr.lower()
      |> EMLX.Native.Expr.to_native()

    assert [%EMLX.Native.Instruction{op: :plugin, attrs: attrs}] = wire.instructions
    assert Enum.at(attrs, 1) == "qwen3"
    assert Enum.at(attrs, 2) == "mlp"
    refute Enum.any?(wire.instructions, &(&1.op == :runtime_call))
  end

  test "multi-output KV attention preserves output order eagerly and when compiled" do
    query = emlx_tensor([[[[0.2, 0.4], [0.1, -0.3]], [[0.5, -0.2], [0.3, 0.6]]]])
    key = emlx_tensor([[[[0.7, 0.1]], [[-0.2, 0.5]]]])
    value = emlx_tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    k_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 4, 2})
    v_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 4, 2})
    scale = 1.0 / :math.sqrt(2.0)

    eager =
      Native.kv_cache_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        0,
        scale,
        2,
        10_000.0
      )

    compiled =
      Nx.Defn.jit(&compiled_attention/5, compiler: EMLX).(
        query,
        key,
        value,
        k_cache,
        v_cache
      )

    Enum.zip(Tuple.to_list(compiled), Tuple.to_list(eager))
    |> Enum.each(fn {actual, expected} -> assert_close(actual, expected) end)

    {_attention, updated_k, updated_v} = eager
    refute Nx.all_close(updated_k, updated_v) |> Nx.to_number() == 1
  end

  test "tensor offset changes across a compiled while loop without host extraction" do
    query = emlx_tensor([[[[0.2, 0.4], [0.1, -0.3]]]])
    key = emlx_tensor([[[[0.7, 0.1]]]])
    value = emlx_tensor([[[[1.0, 2.0]]]])
    k_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 3, 2})
    v_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 3, 2})
    offset = Nx.tensor(0, type: :s32, backend: EMLX.Backend)

    {attention, _query, _key, _value, updated_k, updated_v, final_offset} =
      Nx.Defn.jit(&compiled_attention_loop/6, compiler: EMLX).(
        query,
        key,
        value,
        k_cache,
        v_cache,
        offset
      )

    {_, expected_k, expected_v} =
      Native.kv_cache_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        0,
        0.7071067811865475,
        2,
        10_000.0
      )

    {expected_attention, expected_k, expected_v} =
      Native.kv_cache_attention(
        query,
        key,
        value,
        expected_k,
        expected_v,
        1,
        0.7071067811865475,
        2,
        10_000.0
      )

    assert_close(attention, expected_attention)
    assert_close(updated_k, expected_k)
    assert_close(updated_v, expected_v)
    assert Nx.to_number(final_offset) == 2
  end

  test "full forward preserves token and per-layer cache ordering when compiled" do
    fixture = dense_forward_fixture()

    eager =
      Native.forward_greedy_dense(
        fixture.hidden,
        [fixture.layer],
        [{fixture.k_cache, fixture.v_cache}],
        fixture.norm,
        fixture.lm_head,
        0,
        0.7071067811865475,
        2,
        10_000.0,
        1.0e-6
      )

    compiled =
      Nx.Defn.jit(&compiled_forward_dense/6, compiler: EMLX).(
        fixture.hidden,
        fixture.layer,
        fixture.k_cache,
        fixture.v_cache,
        fixture.norm,
        fixture.lm_head
      )

    assert_forward_close(compiled, eager, {1})
  end

  test "dense and generalized chunk callbacks preserve ordered tokens and caches when compiled" do
    fixture = dense_forward_fixture()

    for {eager_fun, compiled_fun} <- [
          {&Native.forward_greedy_chunk_dense/12, &compiled_chunk_dense/7},
          {&Native.forward_greedy_chunk_generalized/12, &compiled_chunk_generalized/7}
        ] do
      eager =
        eager_fun.(
          fixture.input_ids,
          fixture.embed,
          [fixture.layer],
          [{fixture.k_cache, fixture.v_cache}],
          fixture.norm,
          fixture.lm_head,
          0,
          3,
          0.7071067811865475,
          2,
          10_000.0,
          1.0e-6
        )

      compiled =
        Nx.Defn.jit(compiled_fun, compiler: EMLX).(
          fixture.input_ids,
          fixture.embed,
          fixture.layer,
          fixture.k_cache,
          fixture.v_cache,
          fixture.norm,
          fixture.lm_head
        )

      assert_forward_close(compiled, eager, {3})
    end
  end

  test "internal tensor offsets are safely clamped and validated" do
    query = emlx_tensor([[[[0.2, 0.4], [0.1, -0.3]]]])
    key = emlx_tensor([[[[0.7, 0.1]]]])
    value = emlx_tensor([[[[1.0, 2.0]]]])
    k_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 3, 2})
    v_cache = Nx.broadcast(emlx_tensor(0.0), {1, 1, 3, 2})

    call = fn offset ->
      Native.kv_cache_attention_tensor_offset(
        query,
        key,
        value,
        k_cache,
        v_cache,
        offset,
        0.7071067811865475,
        2,
        10_000.0
      )
    end

    low = call.(Nx.tensor(-50, type: :s32, backend: EMLX.Backend))
    high = call.(Nx.tensor(50, type: :s32, backend: EMLX.Backend))

    expected_low =
      Native.kv_cache_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        0,
        0.7071067811865475,
        2,
        10_000.0
      )

    expected_high =
      Native.kv_cache_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        2,
        0.7071067811865475,
        2,
        10_000.0
      )

    Enum.zip(Tuple.to_list(low), Tuple.to_list(expected_low))
    |> Enum.each(fn {actual, expected} -> assert_close(actual, expected) end)

    Enum.zip(Tuple.to_list(high), Tuple.to_list(expected_high))
    |> Enum.each(fn {actual, expected} -> assert_close(actual, expected) end)

    assert_raise ArgumentError, ~r/scalar int32 or int64/, fn ->
      call.(Nx.tensor([0, 1], type: :s32, backend: EMLX.Backend))
    end
  end

  @tag :metal
  test "quantized layer keeps physical weights and promoted output dtype when compiled" do
    hidden = gpu_iota({1, 1, 32}, 400, :bf16)
    norm1 = gpu_iota({32}, 10, :bf16)
    norm2 = gpu_iota({32}, 12, :bf16)
    q_norm = gpu_iota({16}, 20, :bf16)
    k_norm = gpu_iota({16}, 22, :bf16)
    q_proj = quantized_projection({32, 32}, 101, :f16)
    k_proj = quantized_projection({32, 16}, 102, :f16)
    v_proj = quantized_projection({32, 16}, 103, :f16)
    o_proj = quantized_projection({32, 32}, 104, :f16)
    gate_proj = quantized_projection({32, 32}, 105, :f16)
    up_proj = quantized_projection({32, 32}, 106, :f16)
    down_proj = quantized_projection({32, 32}, 107, :f16)
    zero = Nx.tensor(0.0, type: :f16, backend: {EMLX.Backend, device: :gpu})
    k_cache = Nx.broadcast(zero, {1, 1, 3, 16})
    v_cache = Nx.broadcast(zero, {1, 1, 3, 16})

    layer = fn input, key_cache, value_cache ->
      Native.layer_generalized(
        input,
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        key_cache,
        value_cache,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        0,
        0.25,
        16,
        10_000.0,
        1.0e-5
      )
    end

    eager = layer.(hidden, k_cache, v_cache)

    {operands, descriptors} =
      quantized_layer_operands(
        [hidden, norm1, q_norm, k_norm, k_cache, v_cache, norm2],
        [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
      )

    compiled_layer = fn flat_operands ->
      Native.layer_generalized_flat(
        Tuple.to_list(flat_operands),
        descriptors,
        0,
        0.25,
        16,
        10_000.0,
        1.0e-5
      )
    end

    compiled =
      Nx.Defn.jit(compiled_layer, compiler: EMLX, device: :gpu).(List.to_tuple(operands))

    assert Nx.type(elem(eager, 0)) == {:f, 32}
    assert Nx.type(elem(compiled, 0)) == {:f, 32}

    Enum.zip(Tuple.to_list(compiled), Tuple.to_list(eager))
    |> Enum.each(fn {actual, expected} ->
      assert Nx.all_close(actual, expected, atol: 5.0e-2, rtol: 5.0e-2)
             |> Nx.to_number() == 1
    end)

    wire =
      Nx.Defn.debug_expr_apply(compiled_layer, [
        operands |> Enum.map(&Nx.to_template/1) |> List.to_tuple()
      ])
      |> EMLX.Native.Expr.lower()
      |> EMLX.Native.Expr.to_native()

    assert [%EMLX.Native.Instruction{op: :plugin, attrs: attrs}] = wire.instructions
    assert Enum.at(attrs, 2) == "layer_generalized"
    refute Enum.any?(wire.instructions, &(&1.op == :runtime_call))
  end

  test "native preparation follows the same dense-only dispatch predicate" do
    dense = plan_state()

    assert %Model.State{native: nil} = Model.prepare_native(dense)

    head_quantized = %{dense | lm_head: quantized_projection({32, 32}, 201, :f16)}
    head_prepared = Model.prepare_native(head_quantized)
    assert_generalized_plan(head_prepared)
    assert Model.prepare_native(head_prepared) === head_prepared

    quantized = quantized_plan_state(dense)
    assert_generalized_plan(Model.prepare_native(quantized))

    [layer] = dense.layers
    {norm1, norm2, q_norm, k_norm, _q_proj, k_proj, v_proj, o_proj, gate, up, down} = layer

    mixed = %{
      dense
      | layers: [
          {norm1, norm2, q_norm, k_norm, quantized_projection({32, 32}, 202, :f16), k_proj,
           v_proj, o_proj, gate, up, down}
        ],
        config: %{dense.config | dense_layers?: false}
    }

    assert_generalized_plan(Model.prepare_native(mixed))
  end

  defp reference_mlp(hidden, norm, gate, up, down, eps) do
    squared = Nx.multiply(hidden, hidden)

    normalized =
      hidden
      |> Nx.multiply(Nx.rsqrt(Nx.add(Nx.mean(squared, axes: [2], keep_axes: true), eps)))
      |> Nx.multiply(norm)

    gate_value = Nx.dot(normalized, [2], gate, [0])
    up_value = Nx.dot(normalized, [2], up, [0])

    activated =
      gate_value
      |> Nx.multiply(Nx.sigmoid(gate_value))
      |> Nx.multiply(up_value)

    Nx.add(hidden, Nx.dot(activated, [2], down, [0]))
  end

  defp dense_forward_fixture do
    hidden = gpu_iota({1, 1, 4}, 20, :f32)
    norm1 = gpu_iota({4}, 10, :f32)
    norm2 = gpu_iota({4}, 12, :f32)
    q_norm = gpu_iota({2}, 14, :f32)
    k_norm = gpu_iota({2}, 16, :f32)

    layer = {
      norm1,
      norm2,
      q_norm,
      k_norm,
      gpu_iota({4, 4}, 100, :f32),
      gpu_iota({4, 2}, 110, :f32),
      gpu_iota({4, 2}, 120, :f32),
      gpu_iota({4, 4}, 130, :f32),
      gpu_iota({4, 6}, 140, :f32),
      gpu_iota({4, 6}, 150, :f32),
      gpu_iota({6, 4}, 160, :f32)
    }

    zero = Nx.tensor(0.0, type: :f32, backend: {EMLX.Backend, device: :gpu})

    %{
      hidden: hidden,
      input_ids: Nx.tensor([[1]], type: :s64, backend: {EMLX.Backend, device: :gpu}),
      embed: gpu_iota({8, 4}, 200, :f32),
      layer: layer,
      k_cache: Nx.broadcast(zero, {1, 1, 4, 2}),
      v_cache: Nx.broadcast(zero, {1, 1, 4, 2}),
      norm: gpu_iota({4}, 18, :f32),
      lm_head: gpu_iota({8, 4}, 190, :f32)
    }
  end

  defp plan_state do
    layer = {
      gpu_iota({32}, 10, :f16),
      gpu_iota({32}, 12, :f16),
      gpu_iota({16}, 14, :f16),
      gpu_iota({16}, 16, :f16),
      gpu_iota({32, 32}, 101, :f16),
      gpu_iota({32, 16}, 102, :f16),
      gpu_iota({32, 16}, 103, :f16),
      gpu_iota({32, 32}, 104, :f16),
      gpu_iota({32, 32}, 105, :f16),
      gpu_iota({32, 32}, 106, :f16),
      gpu_iota({32, 32}, 107, :f16)
    }

    %Model.State{
      embed_tokens: gpu_iota({32, 32}, 108, :f16),
      layers: [layer],
      norm: gpu_iota({32}, 109, :f16),
      lm_head: gpu_iota({32, 32}, 110, :f16),
      config: %{dense_layers?: true}
    }
  end

  defp quantized_plan_state(dense) do
    [layer] = dense.layers

    {norm1, norm2, q_norm, k_norm, _q_proj, _k_proj, _v_proj, _o_proj, _gate, _up, _down} =
      layer

    %{
      dense
      | layers: [
          {norm1, norm2, q_norm, k_norm, quantized_projection({32, 32}, 111, :f16),
           quantized_projection({32, 16}, 112, :f16), quantized_projection({32, 16}, 113, :f16),
           quantized_projection({32, 32}, 114, :f16), quantized_projection({32, 32}, 115, :f16),
           quantized_projection({32, 32}, 116, :f16), quantized_projection({32, 32}, 117, :f16)}
        ],
        lm_head: quantized_projection({32, 32}, 118, :f16),
        config: %{dense.config | dense_layers?: false}
    }
  end

  defp assert_generalized_plan(%Model.State{
         native: %{generalized_chunk: {:qwen3_generalized_chunk, [_layer], _head, descriptors}}
       }) do
    assert descriptors != []
  end

  defp assert_forward_close(
         {actual_token, {{actual_k, actual_v}}},
         {expected_token, {{expected_k, expected_v}}},
         token_shape
       ) do
    assert Nx.shape(actual_token) == token_shape
    assert Nx.to_flat_list(actual_token) == Nx.to_flat_list(expected_token)
    assert_close(actual_k, expected_k)
    assert_close(actual_v, expected_v)
  end

  defp emlx_tensor(value), do: Nx.tensor(value, type: :f32, backend: EMLX.Backend)

  defp gpu_iota(shape, divisor, type) do
    shape
    |> Nx.iota(type: type, backend: Nx.BinaryBackend)
    |> Nx.add(1)
    |> Nx.divide(divisor)
    |> Nx.backend_transfer({EMLX.Backend, device: :gpu})
  end

  defp quantized_projection({input, output}, divisor, type) do
    {input, output}
    |> gpu_iota(divisor, type)
    |> Nx.transpose()
    |> EMLX.quantize(type: {:s, 4}, group_size: 32, mode: "affine")
  end

  defp quantized_layer_operands(base, projections) do
    Enum.reduce(projections, {base, []}, fn
      %Nx.Tensor{
        data: %EMLX.Backend{
          ref: ref,
          quantization_config: %EMLX.Quantization.Config{} = config
        }
      },
      {operands, descriptors} ->
        weight_index = length(operands)
        physical = [EMLX.Backend.to_nx(ref), config.scales] ++ List.wrap(config.biases)
        biases_index = if config.biases, do: weight_index + 2, else: -1

        descriptor = [
          1,
          weight_index,
          weight_index + 1,
          biases_index,
          config.group_size,
          config.bits,
          0,
          1
        ]

        {operands ++ physical, descriptors ++ descriptor}
    end)
  end

  defp assert_close(left, right) do
    assert Nx.all_close(left, right, atol: 1.0e-4, rtol: 1.0e-4) |> Nx.to_number() == 1
  end
end
