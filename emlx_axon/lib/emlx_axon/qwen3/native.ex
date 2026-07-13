defmodule EMLXAxon.Qwen3.Native do
  @moduledoc """
  Compiler compatible Qwen3 operations backed by the EMLX native plugin.

  These functions keep eager and compiled execution on the same lazy MLX
  callback. Host synchronization remains the responsibility of higher level
  generation code.
  """

  import Nx.Defn

  alias EMLXAxon.Native.Plugin

  deftransform mlp(hidden, norm, gate_proj, up_proj, down_proj, eps) do
    tensors = [hidden, norm, gate_proj, up_proj, down_proj]
    attrs = [Plugin.f64_bits(eps)]

    if Plugin.traced?(tensors) do
      template = Nx.to_template(hidden)

      fused =
        Plugin.metadata(
          Nx.runtime_call(
            template,
            List.to_tuple(tensors),
            [eps: eps],
            &mlp_callback/2
          ),
          "qwen3",
          "mlp",
          tensors,
          attrs,
          template
        )

      Nx.Defn.Kernel.custom_grad(fused, tensors, fn _gradient ->
        raise ArgumentError, "Qwen3 native plugin operation mlp does not support gradients"
      end)
    else
      mlp_callback(List.to_tuple(tensors), eps: eps)
    end
  end

  deftransform kv_cache_attention(
                 query,
                 key,
                 value,
                 k_cache,
                 v_cache,
                 offset,
                 scale,
                 head_dim,
                 theta
               ) do
    {batch, tokens, heads, width} = Nx.shape(query)
    output = Nx.template({batch, tokens, heads * width}, Nx.type(query))

    multi_operation(
      "kv_cache_attention",
      [query, key, value, k_cache, v_cache],
      [offset, Plugin.f64_bits(scale), head_dim, Plugin.f64_bits(theta)],
      {output, Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  @doc false
  deftransform kv_cache_attention_tensor_offset(
                 query,
                 key,
                 value,
                 k_cache,
                 v_cache,
                 offset,
                 scale,
                 head_dim,
                 theta
               ) do
    validate_tensor_offset!(offset, query, k_cache)
    {batch, tokens, heads, width} = Nx.shape(query)
    output = Nx.template({batch, tokens, heads * width}, Nx.type(query))

    multi_operation(
      "kv_cache_attention_tensor_offset",
      [query, key, value, k_cache, v_cache, offset],
      [Plugin.f64_bits(scale), head_dim, Plugin.f64_bits(theta)],
      {output, Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  deftransform attention_residual(hidden, attention, projection) do
    single_operation(
      "attention_residual",
      [hidden, attention, projection],
      [],
      Nx.to_template(hidden)
    )
  end

  deftransform attention_block(
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
    multi_operation(
      "attention_block",
      [
        hidden,
        norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        k_cache,
        v_cache
      ],
      [
        offset,
        Plugin.f64_bits(scale),
        head_dim,
        Plugin.f64_bits(theta),
        Plugin.f64_bits(eps)
      ],
      {Nx.to_template(hidden), Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  deftransform layer_dense(
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
    multi_operation(
      "layer_dense",
      [
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
        down_proj
      ],
      [
        offset,
        Plugin.f64_bits(scale),
        head_dim,
        Plugin.f64_bits(theta),
        Plugin.f64_bits(eps)
      ],
      {Nx.to_template(hidden), Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  deftransform layer_generalized(
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
    {operands, descriptors} =
      {[hidden, norm1, q_norm, k_norm, k_cache, v_cache, norm2], []}
      |> append_linear(q_proj, false)
      |> append_linear(k_proj, false)
      |> append_linear(v_proj, false)
      |> append_linear(o_proj, false)
      |> append_linear(gate_proj, false)
      |> append_linear(up_proj, false)
      |> append_linear(down_proj, false)

    layer_generalized_flat(
      operands,
      descriptors,
      offset,
      scale,
      head_dim,
      theta,
      eps
    )
  end

  @doc false
  deftransform layer_generalized_flat(
                 operands,
                 descriptors,
                 offset,
                 scale,
                 head_dim,
                 theta,
                 eps
               ) do
    unless is_list(operands) and length(operands) >= 14 and
             Enum.all?(operands, &match?(%Nx.Tensor{}, &1)) do
      raise ArgumentError, "generalized Qwen3 layer operands must be a tensor list"
    end

    [hidden, _norm1, _q_norm, _k_norm, k_cache, v_cache | _weights] = operands
    output_type = generalized_output_type(operands, descriptors)
    output = Nx.template(Nx.shape(hidden), output_type)

    multi_operation(
      "layer_generalized",
      operands,
      [
        1,
        offset,
        Plugin.f64_bits(scale),
        head_dim,
        Plugin.f64_bits(theta),
        Plugin.f64_bits(eps),
        7
        | descriptors
      ],
      {output, Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  @doc false
  deftransform layer_dense_tensor_offset(
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
    validate_tensor_offset!(offset, hidden, k_cache)

    multi_operation(
      "layer_dense_tensor_offset",
      [
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
        offset
      ],
      [
        Plugin.f64_bits(scale),
        head_dim,
        Plugin.f64_bits(theta),
        Plugin.f64_bits(eps)
      ],
      {Nx.to_template(hidden), Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  @doc false
  def mlp_callback({hidden, norm, gate_proj, up_proj, down_proj}, opts) do
    [output] =
      Plugin.call(
        "qwen3",
        "mlp",
        [hidden, norm, gate_proj, up_proj, down_proj],
        [Plugin.f64_bits(opts[:eps])],
        Nx.to_template(hidden)
      )

    output
  end

  defp single_operation(callback, tensors, attrs, template) do
    if Plugin.traced?(tensors) do
      fused =
        Plugin.metadata(
          Nx.runtime_call(
            template,
            List.to_tuple(tensors),
            [callback: callback, attrs: attrs, templates: [template], composite: :single],
            &operation_callback/2
          ),
          "qwen3",
          callback,
          tensors,
          attrs,
          template
        )

      unsupported_gradient(fused, tensors, callback)
    else
      operation_callback(List.to_tuple(tensors),
        callback: callback,
        attrs: attrs,
        templates: [template],
        composite: :single
      )
    end
  end

  defp multi_operation(callback, tensors, attrs, templates) do
    flat_templates = Tuple.to_list(templates)

    if Plugin.traced?(tensors) do
      fused =
        Plugin.metadata(
          Nx.runtime_call(
            templates,
            List.to_tuple(tensors),
            [
              callback: callback,
              attrs: attrs,
              templates: flat_templates,
              composite: :tuple
            ],
            &operation_callback/2
          ),
          "qwen3",
          callback,
          tensors,
          attrs,
          templates
        )

      unsupported_gradient(fused, tensors, callback)
    else
      operation_callback(List.to_tuple(tensors),
        callback: callback,
        attrs: attrs,
        templates: flat_templates,
        composite: :tuple
      )
    end
  end

  @doc false
  def operation_callback(operands, opts) do
    outputs =
      Plugin.call(
        "qwen3",
        opts[:callback],
        Tuple.to_list(operands),
        opts[:attrs],
        opts[:templates]
      )

    case opts[:composite] do
      :single -> hd(outputs)
      :tuple -> List.to_tuple(outputs)
    end
  end

  defp unsupported_gradient(value, tensors, callback) do
    Nx.Defn.Kernel.custom_grad(value, tensors, fn _gradient ->
      raise ArgumentError,
            "Qwen3 native plugin operation #{callback} does not support gradients"
    end)
  end

  defp append_linear({operands, descriptors}, tensor, transpose) do
    {physical, group_size, bits, mode, quantized?} = physical_linear(tensor)
    weight_index = length(operands)
    [weight | optional] = physical
    operands = operands ++ [weight]

    {operands, scales_index, biases_index} =
      case optional do
        [] ->
          {operands, -1, -1}

        [scales] ->
          {operands ++ [scales], weight_index + 1, -1}

        [scales, biases] ->
          {operands ++ [scales, biases], weight_index + 1, weight_index + 2}
      end

    kind = if quantized?, do: 1, else: 0

    descriptor = [
      kind,
      weight_index,
      scales_index,
      biases_index,
      group_size,
      bits,
      quantization_mode(mode),
      if(quantized? or transpose, do: 1, else: 0)
    ]

    {operands, descriptors ++ descriptor}
  end

  defp physical_linear(%Nx.Tensor{
         data: %EMLX.Backend{
           ref: ref,
           quantization_config: %EMLX.Quantization.Config{} = config
         }
       }) do
    physical_weight = EMLX.Backend.to_nx(ref)
    tensors = [physical_weight, config.scales] ++ List.wrap(config.biases)
    {tensors, config.group_size, config.bits, config.mode, true}
  end

  defp physical_linear(%Nx.Tensor{
         data: %Nx.Defn.Expr{
           op: :metadata,
           args: [
             _inner,
             %{
               __EMLX_QUANT__: %{
                 weight: weight,
                 scales: scales,
                 biases: biases,
                 group_size: group_size,
                 bits: bits,
                 mode: mode
               }
             }
           ]
         }
       }) do
    {[weight, scales] ++ List.wrap(biases), group_size, bits, mode, true}
  end

  defp physical_linear(%Nx.Tensor{} = tensor), do: {[tensor], 0, 0, "affine", false}

  defp quantization_mode("affine"), do: 0
  defp quantization_mode("mxfp4"), do: 1
  defp quantization_mode("mxfp8"), do: 2
  defp quantization_mode("nvfp4"), do: 3

  defp quantization_mode(mode) do
    raise ArgumentError, "unsupported Qwen3 quantization mode: #{inspect(mode)}"
  end

  # MLX affine quantized matmul promotes the activation with its scale and bias
  # dtypes. Track those promotions across the fused layer so the static plugin
  # output template matches the lazy MLX result.
  defp generalized_output_type(operands, descriptors) do
    base_type =
      operands
      |> Enum.take(7)
      |> Enum.map(&Nx.type/1)
      |> Enum.reduce(&mlx_merge_type/2)

    descriptors
    |> Enum.chunk_every(8)
    |> Enum.reduce(base_type, fn descriptor, type ->
      case descriptor do
        [0, weight_index, -1, -1, _group_size, _bits, _mode, _transpose] ->
          mlx_merge_type(type, operands |> Enum.fetch!(weight_index) |> Nx.type())

        [1, _weight_index, scales_index, biases_index, _group_size, _bits, 0, _transpose] ->
          type
          |> mlx_merge_type(operands |> Enum.fetch!(scales_index) |> Nx.type())
          |> maybe_merge_bias_type(operands, biases_index)

        [1, _weight_index, _scales_index, _biases_index, _group_size, _bits, mode, _transpose]
        when mode in 1..3 ->
          type

        _other ->
          raise ArgumentError, "invalid generalized Qwen3 linear descriptor"
      end
    end)
  end

  defp maybe_merge_bias_type(type, _operands, -1), do: type

  defp maybe_merge_bias_type(type, operands, index) do
    mlx_merge_type(type, operands |> Enum.fetch!(index) |> Nx.type())
  end

  # MLX follows JAX promotion here, while Nx currently promotes this pair to
  # f16. The native callback must advertise the dtype MLX will actually return.
  defp mlx_merge_type({:bf, 16}, {:f, 16}), do: {:f, 32}
  defp mlx_merge_type({:f, 16}, {:bf, 16}), do: {:f, 32}
  defp mlx_merge_type(left, right), do: Nx.Type.merge(left, right)

  defp validate_tensor_offset!(offset, token_tensor, k_cache) do
    unless Nx.shape(offset) in [{}, {1}] and Nx.type(offset) in [{:s, 32}, {:s, 64}] do
      raise ArgumentError, "offset must be a scalar int32 or int64 tensor"
    end

    token_count = elem(Nx.shape(token_tensor), 1)
    capacity = elem(Nx.shape(k_cache), 2)

    unless token_count >= 1 and token_count <= capacity and capacity <= 2_147_483_647 do
      raise ArgumentError,
            "expected 1 <= token_count <= cache_capacity <= INT32_MAX"
    end
  end
end
