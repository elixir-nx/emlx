defmodule EMLX.Fast do
  @moduledoc """
  Single-kernel Metal shaders from `mlx::fast`, exposed as `deftransform`
  functions backed by `Nx.runtime_call`.

  Every function is defn-safe: call inside `defn`, `Nx.Defn.jit`, or from
  `Axon.rewrite_nodes/2` rewrite callbacks without restriction.

  ## Functions

  - `rms_norm/3` — fused RMS normalisation
  - `rope/6` — fused rotary position embedding
  - `scaled_dot_product_attention/4` — flash-attention SDPA (no mask)
  - `scaled_dot_product_attention/5` — flash-attention SDPA (additive/bool mask)

  ## Axon graph rewrite example

      Axon.rewrite_nodes(model, fn
        %Axon.Node{op: :rms_norm, opts: [eps: eps]} ->
          fn [x, weight], _output -> EMLX.Fast.rms_norm(x, weight, eps) end
        _ -> :skip
      end)
  """

  import Nx.Defn

  # ── RMS Norm ────────────────────────────────────────────────────────────────

  @doc """
  Fused RMS normalisation (`mlx::fast::rms_norm`).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (same size as last axis of `x`).
  - `eps`    — numerical stability constant (e.g. `1.0e-6`).

  Output shape and type match `x`.
  """
  deftransform rms_norm(x, weight, eps) do
    out = Nx.template(Nx.shape(x), Nx.type(x))
    Nx.runtime_call(out, {x, weight}, [eps: eps], &__MODULE__.rms_norm_callback/2)
  end

  @doc false
  def rms_norm_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight}, opts) do
    EMLX.fast_rms_norm(EMLX.Backend.from_nx(x), EMLX.Backend.from_nx(weight), opts[:eps])
    |> EMLX.Backend.to_nx()
  end

  # ── RoPE ────────────────────────────────────────────────────────────────────

  @doc """
  Fused rotary position embedding (`mlx::fast::rope`).

  - `a`           — input `{B, ..., T, D}`; `...` dims are passed through.
  - `dims`        — number of feature dims to rotate (≤ last-axis size, must be even).
  - `traditional` — `false` for split-half (Qwen3); `true` for interleaved.
  - `base`        — angular frequency base (e.g. `10_000` or `1_000_000`).
  - `scale`       — position scale (`1.0` unless using NTK-aware scaling).
  - `offset`      — integer position offset (tokens already in the KV cache).

  **`traditional` must match the model checkpoint's convention.**
  For Qwen3 (split-half): `traditional: false`.

  Output shape and type match `a`.
  """
  deftransform rope(a, dims, traditional, base, scale, offset) do
    out = Nx.template(Nx.shape(a), Nx.type(a))

    Nx.runtime_call(out, a,
      [dims: dims, traditional: traditional, base: base, scale: scale, offset: offset],
      &__MODULE__.rope_callback/2)
  end

  @doc false
  def rope_callback(%Nx.Tensor{} = a, opts) do
    EMLX.fast_rope(
      EMLX.Backend.from_nx(a),
      opts[:dims], opts[:traditional], opts[:base], opts[:scale], opts[:offset]
    )
    |> EMLX.Backend.to_nx()
  end

  # ── Scaled Dot-Product Attention ─────────────────────────────────────────────

  @doc """
  Flash-attention SDPA, no mask (`mlx::fast::scaled_dot_product_attention`).

  GQA-native: `k`/`v` may have fewer heads than `q` — no pre-tiling required.

  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — scalar (typically `1 / sqrt(D)`)

  Output: `{B, N_q, T_q, D}` — same dtype as `q`.
  Softmax accumulates in float32 internally regardless of input dtype.
  """
  deftransform scaled_dot_product_attention(q, k, v, scale) do
    out = Nx.template(Nx.shape(q), Nx.type(q))
    Nx.runtime_call(out, {q, k, v}, [scale: scale], &__MODULE__.sdpa_callback/2)
  end

  @doc false
  def sdpa_callback({%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v}, opts) do
    out =
      EMLX.fast_sdpa(
        EMLX.Backend.from_nx(q), EMLX.Backend.from_nx(k), EMLX.Backend.from_nx(v),
        opts[:scale]
      )
      |> EMLX.Backend.to_nx()

    # mlx::fast::sdpa may upcast to f32 internally; cast back to q's dtype
    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end

  @doc """
  Flash-attention SDPA with an additive or boolean `mask`.

  `mask` must be broadcast-compatible with `{B, N_q, T_q, T_kv}`.
  Boolean `false` entries are masked out (`-∞`); float entries are added to
  the pre-softmax scores.

  For causal masking in decode (single query token), prefer the no-mask arity
  since `T_q=1` is always trivially causal.
  """
  deftransform scaled_dot_product_attention(q, k, v, scale, mask) do
    out = Nx.template(Nx.shape(q), Nx.type(q))
    Nx.runtime_call(out, {q, k, v, mask}, [scale: scale], &__MODULE__.sdpa_masked_callback/2)
  end

  @doc false
  def sdpa_masked_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = mask},
        opts
      ) do
    out =
      EMLX.fast_sdpa_masked(
        EMLX.Backend.from_nx(q), EMLX.Backend.from_nx(k), EMLX.Backend.from_nx(v),
        EMLX.Backend.from_nx(mask), opts[:scale]
      )
      |> EMLX.Backend.to_nx()

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end
end
