defmodule EMLX.Fast do
  @moduledoc """
  Single-kernel Metal shaders from `mlx::fast`, exposed as `deftransform`
  functions backed by `Nx.runtime_call`.

  Every function is defn-safe: call inside `defn`, `Nx.Defn.jit`, or from
  `Axon.rewrite_nodes/2` rewrite callbacks without restriction.

  ## Functions

  - `rms_norm/3` — fused RMS normalisation
  - `layer_norm/4` — fused layer normalisation (with bias)
  - `layer_norm/3` — fused layer normalisation (weight-only, no bias)
  - `rope/6` — fused RoPE with scalar integer offset
  - `rope_with_positions/6` — fused RoPE accepting a `position_ids` tensor
  - `rope_with_freqs/6` — fused RoPE with precomputed inv-frequency tensor (for `:llama3` scaling)
  - `scaled_dot_product_attention/4` — flash-attention SDPA (no mask)
  - `scaled_dot_product_attention/5` — flash-attention SDPA (additive/bool mask)
  - `scaled_dot_product_attention_causal/4` — flash-attention SDPA with built-in causal mask
  - `scaled_dot_product_attention_causal_key_masked/5` — causal SDPA; checks key_mask at C++ level, fast-paths to pure causal when all-ones
  - `swiglu/2` — fused SwiGLU: `silu(gate) * up`

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

  # ── Layer Norm ───────────────────────────────────────────────────────────────

  @doc """
  Fused layer normalisation (`mlx::fast::layer_norm`).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (gamma).
  - `bias`   — `{hidden}` bias vector (beta).
  - `eps`    — numerical stability constant (e.g. `1.0e-5`).

  Output shape and type match `x`.
  """
  deftransform layer_norm(x, weight, bias, eps) do
    out = Nx.template(Nx.shape(x), Nx.type(x))
    Nx.runtime_call(out, {x, weight, bias}, [eps: eps], &__MODULE__.layer_norm_callback/2)
  end

  @doc false
  def layer_norm_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight, %Nx.Tensor{} = bias}, opts) do
    EMLX.fast_layer_norm(
      EMLX.Backend.from_nx(x), EMLX.Backend.from_nx(weight), EMLX.Backend.from_nx(bias),
      opts[:eps]
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Fused layer normalisation without bias (`mlx::fast::layer_norm`, weight-only variant).

  - `x`      — input tensor; normalised over the last axis.
  - `weight` — `{hidden}` scale vector (gamma).
  - `eps`    — numerical stability constant (e.g. `1.0e-5`).

  Output shape and type match `x`.
  """
  deftransform layer_norm(x, weight, eps) do
    out = Nx.template(Nx.shape(x), Nx.type(x))
    Nx.runtime_call(out, {x, weight}, [eps: eps], &__MODULE__.layer_norm_no_bias_callback/2)
  end

  @doc false
  def layer_norm_no_bias_callback({%Nx.Tensor{} = x, %Nx.Tensor{} = weight}, opts) do
    EMLX.fast_layer_norm_no_bias(
      EMLX.Backend.from_nx(x), EMLX.Backend.from_nx(weight),
      opts[:eps]
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Causal SDPA with the key_mask check delegated to the C++ NIF.

  At runtime the NIF evaluates `all(key_mask == 1)`:
  - **true** (no padding, e.g. single-sequence decode) → pure causal SDPA,
    no mask tensor allocated.
  - **false** (padded batch or multi-sequence) → builds a combined
    causal + key_mask additive mask and calls the masked SDPA kernel.

  This avoids the `Nx.cond` double-evaluation problem: the NIF forces eval
  of only the small `{B, T_kv}` key_mask subgraph, then branches in C++.

  Input/output layout matches `scaled_dot_product_attention_causal/4`:
  - `q`        — `{B, N_q,  T_q,  D}`
  - `k`        — `{B, N_kv, T_kv, D}`
  - `v`        — `{B, N_kv, T_kv, D}`
  - `scale`    — pre-computed scalar
  - `key_mask` — `{B, T_kv}` boolean/int tensor (1 = attend, 0 = masked)
  - Output     — `{B, N_q, T_q, D}`, same dtype as `q`
  """
  deftransform scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask) do
    out = Nx.template(Nx.shape(q), Nx.type(q))
    Nx.runtime_call(out, {q, k, v, key_mask}, [scale: scale],
      &__MODULE__.sdpa_causal_key_masked_callback/2)
  end

  @doc false
  def sdpa_causal_key_masked_callback(
        {%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v, %Nx.Tensor{} = key_mask},
        opts
      ) do
    out =
      EMLX.fast_sdpa_causal_key_masked(
        EMLX.Backend.from_nx(q), EMLX.Backend.from_nx(k), EMLX.Backend.from_nx(v),
        opts[:scale], EMLX.Backend.from_nx(key_mask)
      )
      |> EMLX.Backend.to_nx()

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
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

  @doc """
  Fused RoPE accepting a `position_ids` tensor (`mlx::fast::rope`, array-offset overload).

  Use this variant when the calling convention provides `position_ids` as a tensor
  (e.g. from Bumblebee's rotary embedding layer) rather than a scalar integer offset.

  - `a`            — input `{B, T, ..., D}` (Bumblebee convention: heads NOT yet transposed)
  - `position_ids` — `{B, T}` integer tensor; each row holds the token positions for
                     one batch example. **Positions must be sequential within each row**
                     (standard causal LM). The starting offset for batch item `b` is
                     taken as `position_ids[b, 0]`; subsequent positions are inferred
                     by MLX as `offset + 0, offset + 1, ...`.
  - `dims`         — number of feature dims to rotate.
  - `traditional`  — `false` for split-half (Bumblebee / Qwen3); `true` for interleaved.
  - `base`         — angular frequency base (e.g. `10_000`).
  - `scale`        — position scale (`1.0` unless using NTK-aware scaling).

  Output shape and type match `a`.

  > ### Sequential positions only {: .warning}
  > This function assumes positions within each batch example are contiguous
  > starting from `position_ids[b, 0]`. Non-sequential position_ids (e.g.
  > from packed sequences or custom position schemes) produce incorrect results.
  """
  deftransform rope_with_positions(a, position_ids, dims, traditional, base, scale) do
    out = Nx.template(Nx.shape(a), Nx.type(a))

    Nx.runtime_call(
      out, {a, position_ids},
      [dims: dims, traditional: traditional, base: base, scale: scale],
      &__MODULE__.rope_with_positions_callback/2
    )
  end

  @doc false
  def rope_with_positions_callback({%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids}, opts) do
    # MLX's array-offset overload takes shape {B} — one starting position per batch.
    # Extract the first position of each batch example as the per-batch offset.
    batch_size = elem(Nx.shape(position_ids), 0)
    offsets = position_ids[[.., 0]] |> Nx.reshape({batch_size})

    EMLX.fast_rope_ids(
      EMLX.Backend.from_nx(a),
      opts[:dims], opts[:traditional], opts[:base], opts[:scale],
      EMLX.Backend.from_nx(offsets)
    )
    |> EMLX.Backend.to_nx()
  end

  @doc """
  Fused RoPE with precomputed inverse-frequency vector (`mlx::fast::rope`, freqs overload).

  Use this variant when the model's RoPE scaling strategy produces a fixed
  `{dims/2}` inv-frequency tensor that can be baked at graph-rewrite time
  (e.g. `:llama3` smooth-interpolation). Strategies that are seq-len conditional
  or require cos/sin post-multiply (`:linear`, `:dynamic`, `:longrope`) should
  use `rope_with_positions/6` instead.

  - `a`            — input `{B, T, ..., D}` (Bumblebee convention: heads NOT yet transposed)
  - `position_ids` — `{B, T}` integer tensor; positions must be sequential within each row.
  - `dims`         — number of feature dims to rotate.
  - `traditional`  — `false` for split-half (Bumblebee / Qwen3); `true` for interleaved.
  - `scale`        — position scale (`1.0` for most strategies with precomputed freqs).
  - `freqs`        — `{dims/2}` tensor of precomputed inverse frequencies.

  Output shape and type match `a`.
  """
  deftransform rope_with_freqs(a, position_ids, dims, traditional, scale, freqs) do
    out = Nx.template(Nx.shape(a), Nx.type(a))

    Nx.runtime_call(
      out, {a, position_ids, freqs},
      [dims: dims, traditional: traditional, scale: scale],
      &__MODULE__.rope_with_freqs_callback/2
    )
  end

  @doc false
  def rope_with_freqs_callback(
        {%Nx.Tensor{} = a, %Nx.Tensor{} = position_ids, %Nx.Tensor{} = freqs},
        opts
      ) do
    batch_size = elem(Nx.shape(position_ids), 0)
    offsets = position_ids[[.., 0]] |> Nx.reshape({batch_size})

    EMLX.fast_rope_with_freqs(
      EMLX.Backend.from_nx(a),
      opts[:dims], opts[:traditional], opts[:scale],
      EMLX.Backend.from_nx(offsets),
      EMLX.Backend.from_nx(freqs)
    )
    |> EMLX.Backend.to_nx()
  end

  # ── SwiGLU ──────────────────────────────────────────────────────────────────

  @doc """
  Fused SwiGLU activation: `silu(gate) * up` where `silu(x) = x * sigmoid(x)`.

  Eliminates the two-op `silu(gate_proj) * up_proj` pattern that appears in
  Qwen3's FFN layers (28× per decode step).

  - `gate` — gate-projection output; silu is applied element-wise.
  - `up`   — up-projection output; same shape as `gate`.

  Output has the same shape and dtype as `gate`.
  """
  deftransform swiglu(gate, up) do
    out = Nx.template(Nx.shape(gate), Nx.type(gate))
    Nx.runtime_call(out, {gate, up}, [], &__MODULE__.swiglu_callback/2)
  end

  @doc false
  def swiglu_callback({%Nx.Tensor{} = gate, %Nx.Tensor{} = up}, _opts) do
    EMLX.fast_swiglu(EMLX.Backend.from_nx(gate), EMLX.Backend.from_nx(up))
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

  @doc """
  Flash-attention SDPA with a built-in causal mask (`mlx::fast::scaled_dot_product_attention`,
  `mask_mode="causal"`).

  MLX constructs the upper-triangular causal mask internally without materialising it,
  making this equivalent to `scaled_dot_product_attention/5` with a causal boolean mask
  but cheaper: no mask tensor allocation, and the mask is fused into the Metal kernel.

  GQA-native: `k`/`v` may have fewer heads than `q` — no pre-tiling required.

  Input/output layout matches `scaled_dot_product_attention/4`:
  - `q`     — `{B, N_q,  T_q,  D}`
  - `k`     — `{B, N_kv, T_kv, D}`
  - `v`     — `{B, N_kv, T_kv, D}`
  - `scale` — pre-computed scalar (typically `1 / sqrt(D)`)
  - Output  — `{B, N_q, T_q, D}`, same dtype as `q`
  """
  deftransform scaled_dot_product_attention_causal(q, k, v, scale) do
    out = Nx.template(Nx.shape(q), Nx.type(q))
    Nx.runtime_call(out, {q, k, v}, [scale: scale], &__MODULE__.sdpa_causal_callback/2)
  end

  @doc false
  def sdpa_causal_callback({%Nx.Tensor{} = q, %Nx.Tensor{} = k, %Nx.Tensor{} = v}, opts) do
    out =
      EMLX.fast_sdpa_causal(
        EMLX.Backend.from_nx(q), EMLX.Backend.from_nx(k), EMLX.Backend.from_nx(v),
        opts[:scale]
      )
      |> EMLX.Backend.to_nx()

    if Nx.type(out) != Nx.type(q), do: Nx.as_type(out, Nx.type(q)), else: out
  end
end
