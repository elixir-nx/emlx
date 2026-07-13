defmodule EMLXAxon.Qwen3NativePluginTest do
  use ExUnit.Case, async: false

  import Nx.Defn

  alias EMLXAxon.Qwen3.Native

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

    {old_attention, old_k, old_v} =
      EMLX.Native.Qwen3.kv_cache_attention(
        EMLX.Backend.from_nx(query),
        EMLX.Backend.from_nx(key),
        EMLX.Backend.from_nx(value),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        0,
        scale,
        2,
        10_000.0
      )
      |> then(fn {attention, updated_k, updated_v} ->
        {
          EMLX.Backend.to_nx(attention),
          EMLX.Backend.to_nx(updated_k),
          EMLX.Backend.to_nx(updated_v)
        }
      end)

    Enum.zip(Tuple.to_list(eager), [old_attention, old_k, old_v])
    |> Enum.each(fn {actual, expected} -> assert_close(actual, expected) end)

    Enum.zip(Tuple.to_list(compiled), [old_attention, old_k, old_v])
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
