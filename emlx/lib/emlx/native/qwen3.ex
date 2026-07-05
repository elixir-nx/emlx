defmodule EMLX.Native.Qwen3 do
  @moduledoc """
  Fused native NIF wrappers for Qwen3 transformer inference.

  These are hand-written (non-`@mlx_function`) wrappers around the
  `qwen3_*` NIFs registered directly in `EMLX.NIF` (see that module's
  `qwen3_*` stubs) — the underlying NIF/C++ names keep the `qwen3_` prefix
  (`emlx/c_src/emlx_fast/qwen3.cpp`), only this module's public API drops it
  since the module name itself already disambiguates.
  """

  import EMLX, only: [is_tensor: 2]

  @doc """
  Qwen3 fused RoPE + KV cache update + SDPA for the native decode path.

  Accepts `query`, `new_key`, and `new_value` in projection layout
  `{B, T, N, D}`. The NIF transposes Q/K/V, applies Qwen3 RoPE to Q/K, updates
  the owned KV cache, runs SDPA, and returns flattened attention output
  `{B, T, N * D}` ready for projection, plus updated cache refs.
  """
  def kv_cache_attention(
        {dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        offset,
        scale,
        head_dim,
        theta
      )
      when is_tensor(dev_q, ref_q) and is_integer(offset) and is_float(scale) and
             is_integer(head_dim) and is_number(theta) do
    device = dev_q
    {worker, effective_device} = EMLX.resolve_worker(device)

    {attn_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.qwen3_kv_cache_attention(
        worker,
        ref_q,
        ref_k,
        ref_v,
        ref_kc,
        ref_vc,
        offset,
        scale,
        head_dim,
        theta * 1.0,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, attn_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  @doc """
  Qwen3 dense MLP helper.

  Accepts hidden states `{B, T, H}`, RMSNorm weight after attention `{H}`,
  dense gate/up projections `{H, I}`, dense down projection `{I, H}`, and RMSNorm
  epsilon. Returns `hidden + mlp_output` as `{B, T, H}`.
  """
  def mlp(
        {dev_h, ref_h},
        {_dev_norm, ref_norm},
        {_dev_gate, ref_gate},
        {_dev_up, ref_up},
        {_dev_down, ref_down},
        eps
      )
      when is_tensor(dev_h, ref_h) and is_float(eps) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    out_ref =
      EMLX.NIF.qwen3_mlp(
        worker,
        ref_h,
        ref_norm,
        ref_gate,
        ref_up,
        ref_down,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {effective_device, out_ref}
  end

  @doc """
  Qwen3 dense attention output projection plus residual add.

  Accepts residual hidden state `{B, T, H}`, flattened attention output
  `{B, T, I}`, and dense output projection `{I, H}`. Returns
  `hidden + projected_attention`.
  """
  def attention_residual(
        {dev_h, ref_h},
        {_dev_attn, ref_attn},
        {_dev_o, ref_o}
      )
      when is_tensor(dev_h, ref_h) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    out_ref =
      EMLX.NIF.qwen3_attention_residual(
        worker,
        ref_h,
        ref_attn,
        ref_o,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {effective_device, out_ref}
  end

  @doc """
  Qwen3 dense transformer layer helper.

  Accepts hidden states `{B, T, H}`, input RMSNorm and RMSNorm weights after attention,
  dense attention projections, Q/K RMSNorm weights, owned KV cache refs, dense
  MLP projections, offset, scale, RoPE parameters, and RMSNorm epsilon. Returns
  `{hidden_out, k_cache, v_cache}`.
  """
  def layer(
        {dev_h, ref_h},
        {_dev_norm1, ref_norm1},
        {_dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_o, ref_o},
        {_dev_qn, ref_qn},
        {_dev_kn, ref_kn},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        {_dev_norm2, ref_norm2},
        {_dev_gate, ref_gate},
        {_dev_up, ref_up},
        {_dev_down, ref_down},
        offset,
        scale,
        head_dim,
        theta,
        eps
      )
      when is_tensor(dev_h, ref_h) and is_integer(offset) and is_float(scale) and
             is_integer(head_dim) and is_number(theta) and is_float(eps) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    {out_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.qwen3_layer(
        worker,
        ref_h,
        ref_norm1,
        ref_q,
        ref_k,
        ref_v,
        ref_o,
        ref_qn,
        ref_kn,
        ref_kc,
        ref_vc,
        ref_norm2,
        ref_gate,
        ref_up,
        ref_down,
        offset,
        scale,
        head_dim,
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, out_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  @doc """
  Qwen3 generalized transformer layer helper: same fusion as `layer/18`,
  but each of the 7 projections (q/k/v/o/gate/up/down) independently accepts
  a dense or quantized (`EMLX.quantize/2`) `Nx.Tensor`, collapsing the
  quantized native lane's per-op NIF calls down to one fused call.

  Accepts hidden states `{B, T, H}`, input RMSNorm weight, q/k/v/o projections
  (dense or quantized), Q/K RMSNorm weights, owned KV cache refs, post-attention
  RMSNorm weight, gate/up/down projections (dense or quantized), offset, scale,
  RoPE parameters, and RMSNorm epsilon. Returns `{hidden_out, k_cache, v_cache}`.
  """
  def layer_quantized(
        {dev_h, ref_h},
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        offset,
        scale,
        head_dim,
        theta,
        eps
      )
      when is_tensor(dev_h, ref_h) and is_integer(offset) and is_float(scale) and
             is_integer(head_dim) and is_number(theta) and is_float(eps) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    {out_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.qwen3_layer_quantized(
        worker,
        ref_h,
        tensor_ref!(norm1),
        linear_weight_term(q_proj),
        linear_weight_term(k_proj),
        linear_weight_term(v_proj),
        linear_weight_term(o_proj),
        tensor_ref!(q_norm),
        tensor_ref!(k_norm),
        ref_kc,
        ref_vc,
        tensor_ref!(norm2),
        linear_weight_term(gate_proj),
        linear_weight_term(up_proj),
        linear_weight_term(down_proj),
        offset,
        scale,
        head_dim,
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, out_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  @doc """
  Qwen3 embedding lookup plus dense forward through all layers and greedy token.

  This accepts token ids and embedding weights directly, so the dense greedy
  path can avoid constructing a separate embedding lookup graph before entering
  the native Qwen3 worker call.
  """
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
        theta,
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_number(theta) and is_float(eps) do
    device = dev_ids
    {worker, effective_device} = EMLX.resolve_worker(device)

    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_ref, kv_updated_refs} =
      EMLX.NIF.qwen3_forward_greedy_ids(
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
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, token_ref},
     Enum.map(kv_updated_refs, fn {k_ref, v_ref} ->
       {{effective_device, k_ref}, {effective_device, v_ref}}
     end)}
  end

  @doc """
  Qwen3 greedy decode chunk helper.

  Starting from a `{1, 1}` token id tensor, runs `count` greedy decode steps
  without returning to Elixir between steps. Returns `{token_refs, kv_cache}`,
  where `token_refs` is a list of raw EMLX token refs in generation order and
  `kv_cache` is the final updated raw cache.
  """
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
        theta,
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_integer(count) and count > 0 and is_float(scale) and
             is_integer(head_dim) and is_number(theta) and is_float(eps) do
    device = dev_ids
    {worker, effective_device} = EMLX.resolve_worker(device)

    assert_decode_ids!({dev_ids, ref_ids}, "forward_greedy_ids_chunk")

    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_refs, kv_updated_refs} =
      EMLX.NIF.qwen3_forward_greedy_ids_chunk(
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
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    tokens = Enum.map(token_refs, &{effective_device, &1})

    kv_cache =
      Enum.map(kv_updated_refs, fn {k_ref, v_ref} ->
        {{effective_device, k_ref}, {effective_device, v_ref}}
      end)

    {tokens, kv_cache}
  end

  @doc """
  Qwen3 generalized greedy decode chunk helper: same fusion as
  `forward_greedy_ids_chunk/12`, but every layer's 7 projections and the
  final `lm_head` each independently accept a dense or quantized
  (`EMLX.quantize/2`) `Nx.Tensor`. This lets the quantized native lane fuse a
  whole multi-token decode chunk into 1 NIF call, matching dense's chunk fusion.

  Starting from a `{1, 1}` token id tensor, runs `count` greedy decode steps
  without returning to Elixir between steps. Returns `{token_refs, kv_cache}`,
  where `token_refs` is a list of raw EMLX token refs in generation order and
  `kv_cache` is the final updated raw cache.
  """
  def forward_greedy_ids_chunk_quantized(
        {dev_ids, ref_ids},
        {_dev_embed, ref_embed},
        layers,
        kv_cache,
        {_dev_norm, ref_norm},
        lm_head,
        offset,
        count,
        scale,
        head_dim,
        theta,
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_integer(count) and count > 0 and is_float(scale) and
             is_integer(head_dim) and is_number(theta) and is_float(eps) do
    device = dev_ids
    {worker, effective_device} = EMLX.resolve_worker(device)

    assert_decode_ids!({dev_ids, ref_ids}, "forward_greedy_ids_chunk_quantized")

    layer_terms = Enum.map(layers, &layer_weight_terms!/1)
    kv_refs = kv_refs(kv_cache)

    {token_refs, kv_updated_refs} =
      EMLX.NIF.qwen3_forward_greedy_ids_chunk_quantized(
        worker,
        ref_ids,
        ref_embed,
        layer_terms,
        kv_refs,
        ref_norm,
        linear_weight_term(lm_head),
        offset,
        count,
        scale,
        head_dim,
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    tokens = Enum.map(token_refs, &{effective_device, &1})

    kv_cache =
      Enum.map(kv_updated_refs, fn {k_ref, v_ref} ->
        {{effective_device, k_ref}, {effective_device, v_ref}}
      end)

    {tokens, kv_cache}
  end

  @doc """
  Qwen3 embedding lookup plus dense forward through all layers and greedy token.

  This variant returns the selected token id as a BEAM integer while keeping the
  updated KV cache as raw EMLX refs. It is intended for streaming decode paths
  that need the token on the host anyway.
  """
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
        theta,
        eps
      )
      when is_tensor(dev_ids, ref_ids) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_number(theta) and is_float(eps) do
    device = dev_ids
    {worker, effective_device} = EMLX.resolve_worker(device)

    assert_batch_size_one!({dev_ids, ref_ids}, "forward_greedy_ids_token_id")

    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {token_id, kv_updated_refs} =
      EMLX.NIF.qwen3_forward_greedy_ids_token_id(
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
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {token_id,
     Enum.map(kv_updated_refs, fn {k_ref, v_ref} ->
       {{effective_device, k_ref}, {effective_device, v_ref}}
     end)}
  end

  @doc """
  Qwen3 dense forward and greedy decode for one token.

  This variant accepts the previous token id as a BEAM integer and returns the
  selected token id as a BEAM integer. It is intended for decode loops that
  already synchronize once per token for streaming or EOS checks. By default it
  runs on the embedding tensor's device; pass `device` explicitly to override.
  """
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
        theta,
        eps,
        device \\ nil
      )
      when is_integer(token_id) and is_list(layers) and is_list(kv_cache) and
             is_integer(offset) and is_float(scale) and is_integer(head_dim) and
             is_number(theta) and is_float(eps) do
    device = device || dev_embed
    {worker, effective_device} = EMLX.resolve_worker(device)

    layer_refs = Enum.map(layers, &layer_refs!/1)
    kv_refs = kv_refs(kv_cache)

    {next_token_id, kv_updated_refs} =
      EMLX.NIF.qwen3_forward_greedy_token_id(
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
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {next_token_id,
     Enum.map(kv_updated_refs, fn {k_ref, v_ref} ->
       {{effective_device, k_ref}, {effective_device, v_ref}}
     end)}
  end

  @doc """
  Qwen3 dense final RMSNorm + lm_head + greedy argmax helper.

  Accepts hidden states `{B, T, H}`, final RMSNorm weight `{H}`, dense lm_head
  `{V, H}`, and RMSNorm epsilon. Returns token ids as `{B}`.
  """
  def final_greedy(
        {dev_h, ref_h},
        {_dev_norm, ref_norm},
        {_dev_lm_head, ref_lm_head},
        eps
      )
      when is_tensor(dev_h, ref_h) and is_float(eps) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    out_ref =
      EMLX.NIF.qwen3_final_greedy(
        worker,
        ref_h,
        ref_norm,
        ref_lm_head,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {effective_device, out_ref}
  end

  @doc """
  Qwen3 dense attention block helper.

  Accepts residual hidden state before attention `{B, T, H}`, input RMSNorm weight
  `{H}`, dense Q/K/V/O projections, Q/K RMSNorm weights, owned KV cache refs,
  offset, scale, RoPE parameters, and RMSNorm epsilon. Returns
  `{hidden_out, k_cache, v_cache}`.
  """
  def attention_block(
        {dev_h, ref_h},
        {_dev_norm, ref_norm},
        {_dev_q, ref_q},
        {_dev_k, ref_k},
        {_dev_v, ref_v},
        {_dev_o, ref_o},
        {_dev_qn, ref_qn},
        {_dev_kn, ref_kn},
        {_dev_kc, ref_kc},
        {_dev_vc, ref_vc},
        offset,
        scale,
        head_dim,
        theta,
        eps
      )
      when is_tensor(dev_h, ref_h) and is_integer(offset) and is_float(scale) and
             is_integer(head_dim) and is_number(theta) and is_float(eps) do
    device = dev_h
    {worker, effective_device} = EMLX.resolve_worker(device)

    {out_ref, k_upd_ref, v_upd_ref} =
      EMLX.NIF.qwen3_attention_block(
        worker,
        ref_h,
        ref_norm,
        ref_q,
        ref_k,
        ref_v,
        ref_o,
        ref_qn,
        ref_kn,
        ref_kc,
        ref_vc,
        offset,
        scale,
        head_dim,
        theta * 1.0,
        eps,
        effective_device
      )
      |> EMLX.unwrap!()
      |> EMLX.await_worker()

    {{effective_device, out_ref}, {effective_device, k_upd_ref}, {effective_device, v_upd_ref}}
  end

  defp layer_refs!(
         {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
          down_proj}
       ) do
    {
      tensor_ref!(norm1),
      tensor_ref!(norm2),
      tensor_ref!(q_norm),
      tensor_ref!(k_norm),
      tensor_ref!(q_proj),
      tensor_ref!(k_proj),
      tensor_ref!(v_proj),
      tensor_ref!(o_proj),
      tensor_ref!(gate_proj),
      tensor_ref!(up_proj),
      tensor_ref!(down_proj)
    }
  end

  # Generalized variant of `layer_refs!/1`: q/k/v/o/gate/up/down each
  # become a `linear_weight_term/1` (dense or quantized) instead of a
  # plain ref, for `layer_quantized`/`forward_greedy_ids_chunk_quantized`.
  defp layer_weight_terms!(
         {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
          down_proj}
       ) do
    {
      tensor_ref!(norm1),
      tensor_ref!(norm2),
      tensor_ref!(q_norm),
      tensor_ref!(k_norm),
      linear_weight_term(q_proj),
      linear_weight_term(k_proj),
      linear_weight_term(v_proj),
      linear_weight_term(o_proj),
      linear_weight_term(gate_proj),
      linear_weight_term(up_proj),
      linear_weight_term(down_proj)
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

  # Builds the weight-term tuple accepted by `qwen3_get_linear_weight` in
  # emlx_fast/qwen3.cpp: `{:dense, ref}` for a plain tensor, or
  # `{:quantized, weight_ref, scales_ref, biases_ref_or_nil, group_size, bits,
  # mode, true}` for a tensor produced by `EMLX.quantize/2`. `transpose` is
  # always `true` for quantized terms here — every Qwen3 projection and
  # lm_head uses the {out,in} physical layout (same convention hardcoded by
  # `EMLX.quantized_matmul/2`); dense orientation is instead selected
  # by the C++ call site (`dense_transpose` arg of `qwen3_get_linear_weight`).
  defp linear_weight_term(%Nx.Tensor{
         data: %EMLX.Backend{ref: {_device, ref}, quantization_config: nil}
       }) do
    {:dense, ref}
  end

  defp linear_weight_term(%Nx.Tensor{
         data: %EMLX.Backend{
           ref: {_device, ref},
           quantization_config: %EMLX.Quantization.Config{} = cfg
         }
       }) do
    {_scales_device, scales_ref} = EMLX.Backend.from_nx(cfg.scales)
    biases_ref = cfg.biases && elem(EMLX.Backend.from_nx(cfg.biases), 1)

    {:quantized, ref, scales_ref, biases_ref, cfg.group_size, cfg.bits, cfg.mode, true}
  end

  defp linear_weight_term(tensor) do
    {_device, ref} = EMLX.Backend.from_nx(tensor)
    {:dense, ref}
  end

  defp tensor_ref!(%Nx.Tensor{data: %EMLX.Backend{ref: {_device, ref}}}), do: ref

  defp tensor_ref!(tensor) do
    {_device, ref} = EMLX.Backend.from_nx(tensor)
    ref
  end
end
