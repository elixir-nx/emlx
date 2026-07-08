defmodule EMLX.Native.Llama do
  @moduledoc """
  Fused native NIF wrappers for dense Llama transformer inference.

  The C++ host side lives in `emlx/c_src/emlx_fast/llama.cpp` and dispatches
  into the standalone Llama plugin loaded by `emlx_axon`.
  """

  import EMLX, only: [is_tensor: 2]

  def layer(
        {dev_h, ref_h},
        {_dev_norm1, ref_norm1},
        {_dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_o, ref_o},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        {_dev_norm2, ref_norm2},
        {_dev_gate, ref_gate},
        {_dev_up, ref_up},
        {_dev_down, ref_down},
        offset,
        scale,
        head_dim,
        {dev_freqs, ref_freqs},
        eps
      )
      when is_tensor(dev_h, ref_h) and is_integer(offset) and is_float(scale) and
             is_integer(head_dim) and is_tensor(dev_freqs, ref_freqs) and is_float(eps) do
    {worker, effective_device} = EMLX.resolve_worker(dev_h)

    {out_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.llama_layer(
        worker,
        ref_h,
        ref_norm1,
        ref_q,
        ref_k,
        ref_v,
        ref_o,
        ref_kc,
        ref_vc,
        ref_norm2,
        ref_gate,
        ref_up,
        ref_down,
        offset,
        scale,
        head_dim,
        ref_freqs,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, out_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  def forward_greedy_ids(
        {dev_ids, ref_ids},
        {_dev_embed, ref_embed},
        layers,
        kv_cache,
        {_dev_norm, ref_norm},
        {_dev_lm_head, ref_lm_head},
        offset,
        scale,
        head_dim,
        {dev_freqs, ref_freqs},
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_tensor(dev_freqs, ref_freqs) and is_float(eps) do
    {worker, effective_device} = EMLX.resolve_worker(dev_ids)
    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_ref, kv_updated_refs} =
      EMLX.NIF.llama_forward_greedy_ids(
        worker,
        ref_ids,
        ref_embed,
        layer_refs,
        kv_refs,
        ref_norm,
        ref_lm_head,
        offset,
        scale,
        head_dim,
        ref_freqs,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, token_ref}, wrap_kv_cache(kv_updated_refs, effective_device)}
  end

  def forward_greedy_ids_chunk(
        {dev_ids, ref_ids},
        {_dev_embed, ref_embed},
        layers,
        kv_cache,
        {_dev_norm, ref_norm},
        {_dev_lm_head, ref_lm_head},
        offset,
        count,
        scale,
        head_dim,
        {dev_freqs, ref_freqs},
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_integer(count) and count > 0 and is_float(scale) and
             is_integer(head_dim) and is_tensor(dev_freqs, ref_freqs) and is_float(eps) do
    {worker, effective_device} = EMLX.resolve_worker(dev_ids)
    assert_decode_ids!({dev_ids, ref_ids}, "llama_forward_greedy_ids_chunk")
    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_refs, kv_updated_refs} =
      EMLX.NIF.llama_forward_greedy_ids_chunk(
        worker,
        ref_ids,
        ref_embed,
        layer_refs,
        kv_refs,
        ref_norm,
        ref_lm_head,
        offset,
        count,
        scale,
        head_dim,
        ref_freqs,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {Enum.map(token_refs, &{effective_device, &1}),
     wrap_kv_cache(kv_updated_refs, effective_device)}
  end

  def forward_greedy_ids_token_id(
        {dev_ids, ref_ids},
        {_dev_embed, ref_embed},
        layers,
        kv_cache,
        {_dev_norm, ref_norm},
        {_dev_lm_head, ref_lm_head},
        offset,
        scale,
        head_dim,
        {dev_freqs, ref_freqs},
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_tensor(dev_freqs, ref_freqs) and is_float(eps) do
    {worker, effective_device} = EMLX.resolve_worker(dev_ids)
    assert_batch_size_one!({dev_ids, ref_ids}, "llama_forward_greedy_ids_token_id")
    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_id, kv_updated_refs} =
      EMLX.NIF.llama_forward_greedy_ids_token_id(
        worker,
        ref_ids,
        ref_embed,
        layer_refs,
        kv_refs,
        ref_norm,
        ref_lm_head,
        offset,
        scale,
        head_dim,
        ref_freqs,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {token_id, wrap_kv_cache(kv_updated_refs, effective_device)}
  end

  def forward_greedy_token_id(
        token_id,
        {dev_embed, ref_embed},
        layers,
        kv_cache,
        {_dev_norm, ref_norm},
        {_dev_lm_head, ref_lm_head},
        offset,
        scale,
        head_dim,
        {dev_freqs, ref_freqs},
        eps,
        device \\ nil
      )
      when is_integer(token_id) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_tensor(dev_freqs, ref_freqs) and is_float(eps) do
    device = device || dev_embed
    {worker, effective_device} = EMLX.resolve_worker(device)
    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {next_token_id, kv_updated_refs} =
      EMLX.NIF.llama_forward_greedy_token_id(
        worker,
        token_id,
        ref_embed,
        layer_refs,
        kv_refs,
        ref_norm,
        ref_lm_head,
        offset,
        scale,
        head_dim,
        ref_freqs,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {next_token_id, wrap_kv_cache(kv_updated_refs, effective_device)}
  end

  def final_greedy({dev_h, ref_h}, {_dev_norm, ref_norm}, {_dev_lm_head, ref_lm_head}, eps)
      when is_tensor(dev_h, ref_h) and is_float(eps) do
    {worker, effective_device} = EMLX.resolve_worker(dev_h)

    out_ref =
      EMLX.NIF.llama_final_greedy(worker, ref_h, ref_norm, ref_lm_head, eps, effective_device)
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {effective_device, out_ref}
  end

  defp layer_refs!({norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}) do
    {
      tensor_ref!(norm1),
      tensor_ref!(norm2),
      tensor_ref!(q_proj),
      tensor_ref!(k_proj),
      tensor_ref!(v_proj),
      tensor_ref!(o_proj),
      tensor_ref!(gate_proj),
      tensor_ref!(up_proj),
      tensor_ref!(down_proj)
    }
  end

  defp kv_refs!({{_device_k, ref_k}, {_device_v, ref_v}})
       when is_reference(ref_k) and is_reference(ref_v),
       do: {ref_k, ref_v}

  defp kv_refs!({k_cache, v_cache}), do: {tensor_ref!(k_cache), tensor_ref!(v_cache)}

  defp kv_refs([{{_device_k, ref_k}, {_device_v, ref_v}} | _rest] = kv_cache)
       when is_reference(ref_k) and is_reference(ref_v),
       do: kv_cache

  defp kv_refs(kv_cache), do: Enum.map(kv_cache, &kv_refs!/1)

  defp wrap_kv_cache(kv_refs, device) do
    Enum.map(kv_refs, fn {k_ref, v_ref} -> {{device, k_ref}, {device, v_ref}} end)
  end

  defp assert_batch_size_one!(tensor, function_name) do
    case EMLX.shape(tensor) do
      {1, _seq_len} ->
        :ok

      {batch_size, _seq_len} ->
        raise ArgumentError,
              "#{function_name} requires batch size 1, got batch size #{batch_size}"

      shape ->
        raise ArgumentError,
              "#{function_name} expects rank-2 input ids, got shape #{inspect(shape)}"
    end
  end

  defp assert_decode_ids!(tensor, function_name) do
    case EMLX.shape(tensor) do
      {1, 1} ->
        :ok

      {1, seq_len} ->
        raise ArgumentError,
              "#{function_name} requires sequence length 1, got sequence length #{seq_len}"

      {batch_size, _seq_len} ->
        raise ArgumentError,
              "#{function_name} requires batch size 1, got batch size #{batch_size}"

      shape ->
        raise ArgumentError,
              "#{function_name} expects rank-2 input ids, got shape #{inspect(shape)}"
    end
  end

  defp tensor_ref!(%Nx.Tensor{data: %EMLX.Backend{ref: {_device, ref}}}), do: ref

  defp tensor_ref!(tensor) do
    {_device, ref} = EMLX.Backend.from_nx(tensor)
    ref
  end
end
