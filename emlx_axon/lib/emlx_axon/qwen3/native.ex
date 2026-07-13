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
    validate_attention_inputs!(query, key, value, k_cache, v_cache)
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

  deftransform final_greedy(hidden, norm, lm_head, eps) do
    {batch, _tokens, _hidden_size} = Nx.shape(hidden)

    single_operation(
      "final_greedy",
      [hidden, norm, lm_head],
      [Plugin.f64_bits(eps)],
      Nx.template({batch}, {:u, 32})
    )
  end

  @doc false
  deftransform forward_greedy_dense(
                 hidden,
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
    {operands, cache_templates, layer_count} =
      flatten_dense_layers(hidden, layers, kv_cache, norm, lm_head)

    {batch, _tokens, _hidden_size} = Nx.shape(hidden)
    token_template = Nx.template({batch}, {:u, 32})
    templates = [token_template | cache_templates]

    outputs =
      flat_multi_operation(
        "forward_greedy_dense",
        operands,
        [
          layer_count,
          offset,
          Plugin.f64_bits(scale),
          head_dim,
          Plugin.f64_bits(theta),
          Plugin.f64_bits(eps)
        ],
        templates
      )

    unpack_forward_outputs(outputs, layer_count)
  end

  @doc false
  deftransform forward_greedy_chunk_dense(
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
    validate_chunk_input!(input_ids, kv_cache, offset, count)

    {layer_operands, cache_templates, layer_count} =
      flatten_dense_layers([], layers, kv_cache, norm, lm_head)

    operands = [input_ids, embed_tokens | layer_operands]
    templates = [Nx.template({count}, {:u, 32}) | cache_templates]

    outputs =
      flat_multi_operation(
        "forward_greedy_chunk_dense",
        operands,
        [
          layer_count,
          offset,
          count,
          Plugin.f64_bits(scale),
          head_dim,
          Plugin.f64_bits(theta),
          Plugin.f64_bits(eps)
        ],
        templates
      )

    unpack_forward_outputs(outputs, layer_count)
  end

  @doc false
  def prepare_generalized_chunk(layers, lm_head) when is_list(layers) do
    {prepared_layers, descriptors, next_operand} =
      Enum.reduce(layers, {[], [], 2}, fn layer, {segments, descriptors, operand_index} ->
        {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
         down_proj} = validate_layer!(layer)

        linears =
          Enum.map(
            [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj],
            &prepare_linear(&1, false)
          )

        {linear_operands, linear_descriptors, next_operand} =
          prepare_linear_segment(linears, operand_index + 6)

        segment = {[norm1, norm2, q_norm, k_norm], linear_operands}
        {[segment | segments], descriptors ++ linear_descriptors, next_operand}
      end)

    {lm_head_operands, lm_head_descriptor, _next_operand} =
      prepare_linear_segment([prepare_linear(lm_head, true)], next_operand + 1)

    {:qwen3_generalized_chunk, Enum.reverse(prepared_layers), lm_head_operands,
     descriptors ++ lm_head_descriptor}
  end

  @doc false
  deftransform forward_greedy_chunk_generalized(
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
    validate_chunk_input!(input_ids, kv_cache, offset, count)

    {operands, descriptors, cache_templates, layer_count} =
      flatten_generalized_chunk(layers, kv_cache, input_ids, embed_tokens, norm, lm_head)

    templates = [Nx.template({count}, {:u, 32}) | cache_templates]

    outputs =
      flat_multi_operation(
        "forward_greedy_chunk_generalized",
        operands,
        [
          1,
          layer_count,
          offset,
          count,
          Plugin.f64_bits(scale),
          head_dim,
          Plugin.f64_bits(theta),
          Plugin.f64_bits(eps),
          layer_count * 7 + 1
          | descriptors
        ],
        templates
      )

    unpack_forward_outputs(outputs, layer_count)
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

  defp flat_multi_operation(callback, tensors, attrs, templates) do
    multi_operation(callback, tensors, attrs, List.to_tuple(templates))
    |> Tuple.to_list()
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
    append_prepared_linear({operands, descriptors}, prepare_linear(tensor, transpose))
  end

  defp prepare_linear(tensor, transpose) do
    {physical, group_size, bits, mode, quantized?} = physical_linear(tensor)
    {physical, group_size, bits, mode, quantized?, transpose}
  end

  defp append_prepared_linear(
         {operands, descriptors},
         {physical, group_size, bits, mode, quantized?, transpose}
       ) do
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

  defp flatten_generalized_chunk(
         {:qwen3_generalized_chunk, prepared_layers, lm_head_operands, descriptors},
         kv_cache,
         input_ids,
         embed_tokens,
         norm,
         _lm_head
       ) do
    unless length(prepared_layers) == length(kv_cache) and prepared_layers != [] do
      raise ArgumentError,
            "Qwen3 layers and KV cache must have the same nonzero length, got " <>
              "#{length(prepared_layers)} prepared layers and #{length(kv_cache)} caches"
    end

    {layer_segments, cache_templates} =
      Enum.zip_with(prepared_layers, kv_cache, fn {fixed, linear_operands}, cache ->
        {k_cache, v_cache} = cache_tensors!(cache)

        {[fixed, [k_cache, v_cache], linear_operands],
         [Nx.to_template(k_cache), Nx.to_template(v_cache)]}
      end)
      |> Enum.unzip()

    operands = List.flatten([[input_ids, embed_tokens], layer_segments, [norm], lm_head_operands])
    {operands, descriptors, List.flatten(cache_templates), length(prepared_layers)}
  end

  defp flatten_generalized_chunk(
         layers,
         kv_cache,
         input_ids,
         embed_tokens,
         norm,
         lm_head
       ) do
    {operands, descriptors, cache_templates, layer_count} =
      flatten_generalized_layers(
        layers,
        kv_cache,
        [input_ids, embed_tokens],
        [],
        []
      )

    {operands, descriptors} = append_linear({operands ++ [norm], descriptors}, lm_head, true)
    {operands, descriptors, cache_templates, layer_count}
  end

  defp prepare_linear_segment(linears, operand_index) do
    Enum.reduce(linears, {[], [], operand_index}, fn
      {physical, group_size, bits, mode, quantized?, transpose}, {operands, descriptors, index} ->
        physical_count = length(physical)
        scales_index = if physical_count >= 2, do: index + 1, else: -1
        biases_index = if physical_count == 3, do: index + 2, else: -1

        descriptor = [
          if(quantized?, do: 1, else: 0),
          index,
          scales_index,
          biases_index,
          group_size,
          bits,
          quantization_mode(mode),
          if(quantized? or transpose, do: 1, else: 0)
        ]

        {operands ++ physical, descriptors ++ descriptor, index + physical_count}
    end)
  end

  defp flatten_dense_layers(first, layers, kv_cache, norm, lm_head) do
    {reversed_operands, reversed_templates, layer_count} =
      do_flatten_dense_layers(layers, kv_cache, [], [], 0)

    if layer_count == 0 do
      raise ArgumentError, "Qwen3 layers and KV cache must not be empty"
    end

    operands =
      case first do
        [] -> :lists.reverse(reversed_operands) ++ [norm, lm_head]
        tensor -> [tensor | :lists.reverse(reversed_operands)] ++ [norm, lm_head]
      end

    {operands, :lists.reverse(reversed_templates), layer_count}
  end

  defp do_flatten_dense_layers([], [], operands, templates, layer_count),
    do: {operands, templates, layer_count}

  defp do_flatten_dense_layers(
         [layer | layers],
         [cache | caches],
         operands,
         templates,
         layer_count
       ) do
    {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} =
      validate_layer!(layer)

    {k_cache, v_cache} = cache_tensors!(cache)

    fields = [
      norm1,
      norm2,
      q_norm,
      k_norm,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      gate_proj,
      up_proj,
      down_proj,
      k_cache,
      v_cache
    ]

    do_flatten_dense_layers(
      layers,
      caches,
      Enum.reverse(fields, operands),
      [Nx.to_template(v_cache), Nx.to_template(k_cache) | templates],
      layer_count + 1
    )
  end

  defp do_flatten_dense_layers(layers, caches, _operands, _templates, _layer_count) do
    raise ArgumentError,
          "Qwen3 layers and KV cache must have the same nonzero length, got " <>
            "#{length(layers)} remaining layers and #{length(caches)} remaining caches"
  end

  defp flatten_generalized_layers([], [], operands, descriptors, cache_templates) do
    layer_count = div(length(cache_templates), 2)

    if layer_count == 0 do
      raise ArgumentError, "Qwen3 layers and KV cache must not be empty"
    end

    {operands, descriptors, cache_templates, layer_count}
  end

  defp flatten_generalized_layers(
         [layer | layers],
         [cache | caches],
         operands,
         descriptors,
         cache_templates
       ) do
    {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} =
      validate_layer!(layer)

    {k_cache, v_cache} = cache_tensors!(cache)

    {operands, descriptors} =
      {operands ++ [norm1, norm2, q_norm, k_norm, k_cache, v_cache], descriptors}
      |> append_linear(q_proj, false)
      |> append_linear(k_proj, false)
      |> append_linear(v_proj, false)
      |> append_linear(o_proj, false)
      |> append_linear(gate_proj, false)
      |> append_linear(up_proj, false)
      |> append_linear(down_proj, false)

    flatten_generalized_layers(
      layers,
      caches,
      operands,
      descriptors,
      cache_templates ++ [Nx.to_template(k_cache), Nx.to_template(v_cache)]
    )
  end

  defp flatten_generalized_layers(layers, caches, _operands, _descriptors, _templates) do
    raise ArgumentError,
          "Qwen3 layers and KV cache must have the same nonzero length, got " <>
            "#{length(layers)} remaining layers and #{length(caches)} remaining caches"
  end

  defp validate_layer!(layer) when is_tuple(layer) and tuple_size(layer) == 11, do: layer

  defp validate_layer!(layer) do
    raise ArgumentError,
          "expected a Qwen3 layer tuple with 11 tensors, got: #{inspect(layer)}"
  end

  defp cache_tensors!({%Nx.Tensor{} = k_cache, %Nx.Tensor{} = v_cache}),
    do: {k_cache, v_cache}

  defp cache_tensors!({{device, k_ref}, {device, v_ref}})
       when is_atom(device) and is_reference(k_ref) and is_reference(v_ref) do
    {EMLX.Backend.to_nx({device, k_ref}), EMLX.Backend.to_nx({device, v_ref})}
  end

  defp cache_tensors!(cache) do
    raise ArgumentError,
          "expected a Qwen3 KV cache pair on one EMLX device, got: #{inspect(cache)}"
  end

  defp unpack_forward_outputs([token | cache_outputs], layer_count)
       when length(cache_outputs) == layer_count * 2 do
    {token, cache_outputs |> unpack_cache_outputs([]) |> List.to_tuple()}
  end

  defp unpack_forward_outputs(outputs, layer_count) do
    raise ArgumentError,
          "Qwen3 plugin returned #{length(outputs)} outputs, expected #{1 + layer_count * 2}"
  end

  defp unpack_cache_outputs([k_cache, v_cache | outputs], caches),
    do: unpack_cache_outputs(outputs, [{k_cache, v_cache} | caches])

  defp unpack_cache_outputs([], caches), do: :lists.reverse(caches)

  defp validate_chunk_input!(input_ids, kv_cache, offset, count) do
    unless Nx.shape(input_ids) == {1, 1} do
      raise ArgumentError,
            "Qwen3 chunk input_ids must have shape {1, 1}, got: #{inspect(Nx.shape(input_ids))}"
    end

    unless is_integer(offset) and offset >= 0 do
      raise ArgumentError, "Qwen3 chunk offset must be a nonnegative integer"
    end

    unless is_integer(count) and count > 0 do
      raise ArgumentError, "Qwen3 chunk count must be a positive integer"
    end

    case kv_cache do
      [cache | _] ->
        {k_cache, _v_cache} = cache_tensors!(cache)
        capacity = elem(Nx.shape(k_cache), 2)

        if offset > capacity - count do
          raise ArgumentError,
                "Qwen3 chunk requires cache length #{offset + count}, capacity is #{capacity}"
        end

      [] ->
        raise ArgumentError, "Qwen3 layers and KV cache must not be empty"
    end
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

  defp validate_attention_inputs!(query, key, value, k_cache, v_cache) do
    Enum.each(
      [query: query, key: key, value: value, k_cache: k_cache, v_cache: v_cache],
      fn {name, tensor} ->
        unless match?(%Nx.Tensor{}, tensor) and tuple_size(Nx.shape(tensor)) == 4 do
          shape = if match?(%Nx.Tensor{}, tensor), do: Nx.shape(tensor), else: :not_a_tensor

          raise ArgumentError,
                "Qwen3 #{name} expects rank 4, got shape: #{inspect(shape)}"
        end
      end
    )
  end
end
