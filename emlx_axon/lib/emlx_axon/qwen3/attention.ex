defmodule EMLXAxon.Qwen3.Attention do
  @moduledoc """
  Grouped-query attention (GQA) for Qwen3, with a preallocated KV cache.

  KV cache layout: `{1, num_kv_heads, max_len, head_dim}` (heads before sequence).
  This layout lets us write `k` (already in `{B, N, T, D}` after transposing) into
  the cache without an extra transpose, and feeds directly into
  `EMLX.Fast.scaled_dot_product_attention` which expects `{B, N, T, D}`.

  `forward/12` delegates to `EMLXAxon.Qwen3.Native` operations backed by the
  Qwen3 plugin callbacks in `qwen3_plugin.cpp`. They transpose Q/K to
  `{B, N, T, D}` and apply RoPE internally via `mlx::core::fast::rope` before
  the fused cache update and SDPA. The `offset` is the current cache fill
  length. `EMLX.Native.Qwen3` remains the compatibility wrapper for direct
  eager callers.
  """

  alias EMLXAxon.Qwen3.Layers

  @doc """
  GQA forward.

  Inputs:
  - `hidden`      — `{1, seq_len, hidden_size}` pre-norm residual input
  - `norm`        — `{hidden_size}` input RMSNorm weight
  - `k_cache`     — `{1, num_kv_heads, max_len, head_dim}` preallocated
  - `v_cache`     — `{1, num_kv_heads, max_len, head_dim}` preallocated
  - `current_len` — number of valid positions already in the cache
  - `q_proj`, `k_proj`, `v_proj`, `o_proj` — quantized weight tensors
  - `q_norm`, `k_norm` — per-head RMSNorm weights (Qwen3 variant)
  - `cfg`         — model config map

  Returns `{attn_out, k_cache_updated, v_cache_updated}`.
  """
  def forward(
        hidden,
        norm,
        k_cache,
        v_cache,
        current_len,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        cfg
      ) do
    if Enum.any?([q_proj, k_proj, v_proj, o_proj], &EMLX.Quantization.quantized?/1) do
      forward_quantized(
        hidden,
        norm,
        k_cache,
        v_cache,
        current_len,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        cfg
      )
    else
      forward_dense(
        hidden,
        norm,
        k_cache,
        v_cache,
        current_len,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        cfg
      )
    end
  end

  defp forward_quantized(
         hidden,
         norm,
         k_cache,
         v_cache,
         current_len,
         q_proj,
         k_proj,
         v_proj,
         o_proj,
         q_norm,
         k_norm,
         cfg
       ) do
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    scale = 1.0 / :math.sqrt(head_dim)
    theta = cfg.rope_theta

    hidden_norm = Layers.rms_norm(hidden, norm, cfg.rms_norm_eps)
    hidden_shape = Nx.shape(hidden)
    batch = elem(hidden_shape, 0)
    seq_len = elem(hidden_shape, 1)

    # Quantized projections — EMLX dispatches to quantized_matmul via backend
    q = Nx.dot(hidden_norm, [2], q_proj, [1]) |> Nx.reshape({batch, seq_len, num_heads, head_dim})

    k =
      Nx.dot(hidden_norm, [2], k_proj, [1])
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})

    v =
      Nx.dot(hidden_norm, [2], v_proj, [1])
      |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})

    # Qwen3 per-head QK RMSNorm before RoPE (normalises over last axis = head_dim)
    q = Layers.rms_norm(q, q_norm, cfg.rms_norm_eps)
    k = Layers.rms_norm(k, k_norm, cfg.rms_norm_eps)

    # Fused Q/K transpose + RoPE + cache update + SDPA.
    #
    # The plugin callback updates k_cache / v_cache before
    # slice_update, enabling MLX's donation optimisation at eval time: the
    # existing 4 MB Metal buffer is reused in-place — no new allocation needed.
    # The slice and SDPA are fused in the same lazy graph, so they land in one
    # Metal command buffer submission.
    {attn_out, k_cache, v_cache} =
      EMLXAxon.Qwen3.Native.kv_cache_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        current_len,
        scale,
        head_dim,
        theta
      )

    attn_out = attention_residual(hidden, attn_out, o_proj)

    {attn_out, k_cache, v_cache}
  end

  defp forward_dense(
         hidden,
         norm,
         k_cache,
         v_cache,
         current_len,
         q_proj,
         k_proj,
         v_proj,
         o_proj,
         q_norm,
         k_norm,
         cfg
       ) do
    head_dim = cfg.head_dim
    scale = 1.0 / :math.sqrt(head_dim)
    theta = cfg.rope_theta

    EMLXAxon.Qwen3.Native.attention_block(
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
      current_len,
      scale,
      head_dim,
      theta,
      cfg.rms_norm_eps
    )
  end

  defp attention_residual(residual_hidden, attn_out, o_proj) do
    if EMLX.Quantization.quantized?(o_proj) do
      Nx.add(residual_hidden, Nx.dot(attn_out, [2], o_proj, [1]))
    else
      EMLXAxon.Qwen3.Native.attention_residual(residual_hidden, attn_out, o_proj)
    end
  end
end
