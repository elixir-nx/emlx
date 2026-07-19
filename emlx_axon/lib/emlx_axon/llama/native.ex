defmodule EMLXAxon.Llama.Native do
  @moduledoc """
  Compiler compatible dense Llama operations backed by the EMLX native plugin.

  Eager and compiled execution use the same lazy MLX callbacks. Higher level
  generation code decides when generated tokens are synchronized to the host.
  """

  import Nx.Defn

  alias EMLXAxon.Native.Plugin

  deftransform layer_dense(
                 hidden,
                 norm1,
                 q_proj,
                 k_proj,
                 v_proj,
                 o_proj,
                 k_cache,
                 v_cache,
                 norm2,
                 gate_proj,
                 up_proj,
                 down_proj,
                 offset,
                 scale,
                 head_dim,
                 rope_freqs,
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
        k_cache,
        v_cache,
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        rope_freqs
      ],
      [offset, scale, head_dim, eps],
      {Nx.to_template(hidden), Nx.to_template(k_cache), Nx.to_template(v_cache)}
    )
  end

  @doc false
  deftransform forward_greedy_dense(
                 hidden,
                 layers,
                 kv_cache,
                 norm,
                 lm_head,
                 rope_freqs,
                 offset,
                 scale,
                 head_dim,
                 eps
               ) do
    {operands, cache_templates, layer_count} =
      flatten_dense_layers(hidden, layers, kv_cache, norm, lm_head, rope_freqs)

    {batch, _tokens, _hidden_size} = Nx.shape(hidden)
    templates = [Nx.template({batch}, {:u, 32}) | cache_templates]

    outputs =
      flat_multi_operation(
        "forward_greedy_dense",
        operands,
        [
          layer_count,
          offset,
          scale,
          head_dim,
          eps
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
                 rope_freqs,
                 offset,
                 count,
                 scale,
                 head_dim,
                 eps
               ) do
    validate_chunk_input!(input_ids, kv_cache, offset, count)

    {layer_operands, cache_templates, layer_count} =
      flatten_dense_layers([], layers, kv_cache, norm, lm_head, rope_freqs)

    operands = [input_ids, embed_tokens | layer_operands]
    templates = [Nx.template({count}, {:u, 32}) | cache_templates]
    submit_each_step = if Plugin.traced?(operands), do: 0, else: 1

    outputs =
      flat_multi_operation(
        "forward_greedy_chunk_dense",
        operands,
        [
          layer_count,
          offset,
          count,
          scale,
          head_dim,
          eps,
          submit_each_step
        ],
        templates
      )

    unpack_forward_outputs(outputs, layer_count)
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
              templates: flat_templates
            ],
            &operation_callback/2
          ),
          "llama",
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
        templates: flat_templates
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
        "llama",
        opts[:callback],
        Tuple.to_list(operands),
        opts[:attrs],
        opts[:templates]
      )

    List.to_tuple(outputs)
  end

  defp unsupported_gradient(value, tensors, callback) do
    Nx.Defn.Kernel.custom_grad(value, tensors, fn _gradient ->
      raise ArgumentError,
            "Llama native plugin operation #{callback} does not support gradients"
    end)
  end

  defp flatten_dense_layers(first, layers, kv_cache, norm, lm_head, rope_freqs) do
    {reversed_operands, reversed_templates, layer_count} =
      do_flatten_dense_layers(layers, kv_cache, [], [], 0)

    if layer_count == 0 do
      raise ArgumentError, "Llama layers and KV cache must not be empty"
    end

    tail = [norm, lm_head, rope_freqs]

    operands =
      case first do
        [] -> Enum.reverse(reversed_operands) ++ tail
        tensor -> [tensor | Enum.reverse(reversed_operands)] ++ tail
      end

    {operands, Enum.reverse(reversed_templates), layer_count}
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
    {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} =
      validate_layer!(layer)

    {k_cache, v_cache} = cache_tensors!(cache)

    fields = [
      norm1,
      norm2,
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
          "Llama layers and KV cache must have the same nonzero length, got " <>
            "#{length(layers)} remaining layers and #{length(caches)} remaining caches"
  end

  defp unpack_forward_outputs([token | cache_outputs], layer_count) do
    if length(cache_outputs) != layer_count * 2 do
      raise ArgumentError,
            "Llama native plugin returned #{length(cache_outputs)} cache tensors, " <>
              "expected #{layer_count * 2}"
    end

    caches =
      cache_outputs
      |> Enum.chunk_every(2)
      |> Enum.map(&List.to_tuple/1)
      |> List.to_tuple()

    {token, caches}
  end

  defp validate_layer!(
         {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} = layer
       ) do
    if Enum.all?(Tuple.to_list(layer), &match?(%Nx.Tensor{}, &1)) do
      {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}
    else
      raise ArgumentError, "Llama layer must contain nine tensors"
    end
  end

  defp validate_layer!(_layer),
    do: raise(ArgumentError, "Llama layer must contain nine tensors")

  defp cache_tensors!({%Nx.Tensor{} = k_cache, %Nx.Tensor{} = v_cache}),
    do: {k_cache, v_cache}

  defp cache_tensors!(_cache),
    do: raise(ArgumentError, "Llama KV cache entry must contain two tensors")

  defp validate_chunk_input!(input_ids, kv_cache, offset, count) do
    unless match?(%Nx.Tensor{shape: {1, 1}}, input_ids) do
      raise ArgumentError, "Llama native chunk input must have shape {1, 1}"
    end

    unless is_list(kv_cache) and kv_cache != [] do
      raise ArgumentError, "Llama native chunk KV cache must be a nonempty list"
    end

    unless is_integer(offset) and offset >= 0 do
      raise ArgumentError, "Llama native chunk offset must be a nonnegative integer"
    end

    unless is_integer(count) and count > 0 do
      raise ArgumentError, "Llama native chunk count must be a positive integer"
    end
  end
end
