defmodule EMLX.Validation.Qwen3Quantized.Attention do
  @moduledoc """
  Grouped-query attention (GQA) for Qwen3, with a preallocated KV cache.

  KV cache layout: `{1, num_kv_heads, max_len, head_dim}` (heads before sequence).
  This layout lets us write `k` (already in `{B, N, T, D}` after transposing) into
  the cache without an extra transpose, and feeds directly into
  `EMLX.Fast.scaled_dot_product_attention` which expects `{B, N, T, D}`.

  RoPE is computed by `EMLX.Fast.rope/6` (single Metal shader, no precomputed
  cos/sin needed). The `offset` is the current cache fill length.
  """

  alias EMLX.Validation.Qwen3Quantized.Layers

  @doc """
  GQA forward.

  Inputs:
  - `hidden`      — `{1, seq_len, hidden_size}` (post-norm)
  - `k_cache`     — `{1, num_kv_heads, max_len, head_dim}` preallocated
  - `v_cache`     — `{1, num_kv_heads, max_len, head_dim}` preallocated
  - `current_len` — number of valid positions already in the cache
  - `q_proj`, `k_proj`, `v_proj`, `o_proj` — quantized weight tensors
  - `q_norm`, `k_norm` — per-head RMSNorm weights (Qwen3 variant)
  - `cfg`         — model config map

  Returns `{attn_out, k_cache_updated, v_cache_updated}`.
  """
  def forward(hidden, k_cache, v_cache, current_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, cfg) do
    num_heads    = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.head_dim
    scale        = 1.0 / :math.sqrt(head_dim)
    theta        = cfg.rope_theta

    [batch, seq_len, _hidden] = Nx.shape(hidden) |> Tuple.to_list()

    # Quantized projections — EMLX dispatches to quantized_matmul via backend
    q = Nx.dot(hidden, [2], q_proj, [1]) |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    k = Nx.dot(hidden, [2], k_proj, [1]) |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})
    v = Nx.dot(hidden, [2], v_proj, [1]) |> Nx.reshape({batch, seq_len, num_kv_heads, head_dim})

    # Qwen3 per-head QK RMSNorm before RoPE (normalises over last axis = head_dim)
    q = Layers.rms_norm(q, q_norm, cfg.rms_norm_eps)
    k = Layers.rms_norm(k, k_norm, cfg.rms_norm_eps)

    # Transpose to {B, N, T, D} — required by mlx::fast::rope and sdpa
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # Fused RoPE: computes cos/sin internally; offset = tokens already in cache
    q = EMLX.Fast.rope(q, head_dim, false, theta, 1.0, current_len)
    k = EMLX.Fast.rope(k, head_dim, false, theta, 1.0, current_len)

    # Update preallocated KV cache {B, N_kv, max_len, D} via put_slice along axis 2
    k_cache = Nx.put_slice(k_cache, [0, 0, current_len, 0], k)
    v_cache = Nx.put_slice(v_cache, [0, 0, current_len, 0], v)

    new_len = current_len + seq_len

    k_valid = k_cache[[.., .., 0..(new_len - 1)//1, ..]]
    v_valid = v_cache[[.., .., 0..(new_len - 1)//1, ..]]

    # Flash-attention SDPA — GQA-native (N_q can be a multiple of N_kv)
    attn_out =
      if seq_len > 1 do
        # Prefill: additive causal mask {1, 1, T_q, T_kv}; 0 = allowed, -1e9 = masked
        q_pos  = Nx.iota({1, 1, seq_len, 1}, type: :s32) |> Nx.add(current_len)
        kv_pos = Nx.iota({1, 1, 1, new_len}, type: :s32)
        mask   =
          Nx.less_equal(kv_pos, q_pos)
          |> Nx.select(
            Nx.broadcast(Nx.tensor(0.0, type: :f32), {1, 1, seq_len, new_len}),
            Nx.broadcast(Nx.tensor(-1.0e9, type: :f32), {1, 1, seq_len, new_len}))

        EMLX.Fast.scaled_dot_product_attention(q, k_valid, v_valid, scale, mask)
      else
        # Decode: single query token has no future positions to mask
        EMLX.Fast.scaled_dot_product_attention(q, k_valid, v_valid, scale)
      end

    # Reshape: {B, N_q, T_q, D} → {B, T_q, N_q, D} → {B, T, hidden}
    attn_out =
      attn_out
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq_len, num_heads * head_dim})

    out = Nx.dot(attn_out, [2], o_proj, [1])

    {out, k_cache, v_cache}
  end
end
