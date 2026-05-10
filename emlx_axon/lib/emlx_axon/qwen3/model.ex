defmodule EMLXAxon.Qwen3.Model do
  @moduledoc """
  Qwen3 quantized model state struct and forward pass.

  ## Defn / JIT strategy

  Every hot-path computation that consists purely of tensor arithmetic is
  wrapped in a `defnp` kernel (declared in `Layers` and `Attention`).
  `defnp` uses `Nx.Defn.Compiler.__jit__` — the same mechanism as
  `Nx.Defn.jit/1` — to compile the function once per unique input shape and
  cache the result.  Subsequent calls skip Nx-side type/shape inference and
  dispatch directly to the compiled kernel.

  Functions that still run eagerly:
  - The quantized `Nx.dot` projections — the `Nx.Defn.Evaluator` (EMLX's
    default compiler) cannot mix jit-argument `Nx.Defn.Expr` nodes with
    captured `EMLX.Backend` tensors in closures.  Wrapping individual ops
    with `Nx.Defn.jit` would require a backend that implements
    `Nx.Defn.Compiler` (e.g. EXLA).
  - `Nx.put_slice` KV-cache update (dynamic start index) and the valid-slice
    read (dynamic end index).

  GPU sync: `EMLX.eval` is called once per token at the sampler boundary so
  the full lazy MLX graph spans all 28 layers before any CPU sync.
  """

  alias EMLXAxon.Qwen3.{Layers, Attention}

  defmodule State do
    @moduledoc "Loaded model weights and config."

    @enforce_keys [:embed_tokens, :layers, :norm, :lm_head, :config]
    defstruct [:embed_tokens, :layers, :norm, :lm_head, :config]

    @type t :: %__MODULE__{
            embed_tokens: Nx.Tensor.t(),
            layers:       [map()],
            norm:         Nx.Tensor.t(),
            lm_head:      Nx.Tensor.t(),
            config:       map()
          }
  end

  @doc """
  Initialise a preallocated KV cache for all layers.

  Returns a list of `{k_cache, v_cache}` pairs, one per transformer layer,
  where each cache is pre-allocated to `max_len` positions.
  """
  @spec init_kv_cache(State.t(), pos_integer()) :: [{Nx.Tensor.t(), Nx.Tensor.t()}]
  def init_kv_cache(%State{config: cfg, layers: layers}, max_len) do
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.head_dim
    gpu          = {EMLX.Backend, device: :gpu}

    # Cache layout: {B, N_kv, max_len, D} — heads-before-sequence so that
    # tensors transposed to {B, N, T, D} for RoPE/SDPA slot in without copies.
    for _layer <- layers do
      k = Nx.broadcast(Nx.tensor(0.0, type: :f16), {1, num_kv_heads, max_len, head_dim})
          |> Nx.backend_transfer(gpu)
      v = Nx.broadcast(Nx.tensor(0.0, type: :f16), {1, num_kv_heads, max_len, head_dim})
          |> Nx.backend_transfer(gpu)
      {k, v}
    end
  end

  @doc """
  Full forward pass for a single decode step.

  - `input_ids` — `{1, seq_len}` integer tensor (the prompt or latest token)
  - `kv_cache`  — list of `{k_cache, v_cache}` preallocated tensors
  - `current_len` — number of tokens already written into the KV cache

  Returns `{logits, kv_cache_updated}` where logits has shape `{1, vocab_size}`.
  The cache is updated in-place via `Nx.put_slice`; `kv_cache_updated` is the
  same list with updated slices.
  """
  @spec forward(Nx.Tensor.t(), [{Nx.Tensor.t(), Nx.Tensor.t()}], non_neg_integer(), State.t()) ::
          {Nx.Tensor.t(), [{Nx.Tensor.t(), Nx.Tensor.t()}]}
  def forward(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, norm: norm, lm_head: lm_head, config: cfg} = state

    # Embed input tokens: {1, seq_len} → {1, seq_len, hidden_size}
    ids_shape = Nx.shape(input_ids)

    hidden =
      if elem(ids_shape, 1) == 1 do
        # Decode hot path (single token): avoid full-row slice graph when T==1.
        Nx.take(embed_tokens, Nx.reshape(input_ids[[0, 0]], {1})) |> Nx.new_axis(0)
      else
        Nx.take(embed_tokens, input_ids[[0, ..]]) |> Nx.new_axis(0)
      end

    # Walk layers + KV pairs without Enum.zip (saves a ~28-cell list alloc per forward).
    {hidden, kv_cache_rev} =
      forward_layers(layers, kv_cache, hidden, [], current_len, cfg)

    kv_cache_updated = :lists.reverse(kv_cache_rev)

    # Final norm + lm_head on the last token position only
    last_hidden =
      if elem(ids_shape, 1) == 1 do
        Nx.squeeze(hidden, axes: [1])
      else
        hidden[[.., -1, ..]]
      end
    normed      = EMLX.Fast.rms_norm(last_hidden, norm, cfg.rms_norm_eps)
    logits      = Nx.dot(normed, [1], lm_head, [1])

    {logits, kv_cache_updated}
  end

  defp forward_layers([], [], hidden, acc, _cur_len, _cfg), do: {hidden, acc}

  defp forward_layers([layer_weights | layers_rest], [{k_cache, v_cache} | kv_rest], hidden, acc, cur_len, cfg) do
    {h_new, k_new, v_new} =
      transformer_layer(hidden, k_cache, v_cache, cur_len, layer_weights, cfg)

    forward_layers(layers_rest, kv_rest, h_new, [{k_new, v_new} | acc], cur_len, cfg)
  end

  @doc false
  def transformer_layer(hidden, k_cache, v_cache, current_len, weights, cfg) do
    %{
      input_layernorm:          norm1,
      post_attention_layernorm: norm2,
      q_norm: q_norm, k_norm: k_norm,
      q_proj: q_proj, k_proj: k_proj, v_proj: v_proj, o_proj: o_proj,
      gate_proj: gate_proj, up_proj: up_proj, down_proj: down_proj
    } = weights

    # Self-attention with pre-norm
    xn = Layers.rms_norm(hidden, norm1, cfg.rms_norm_eps)
    {attn_out, k_new, v_new} =
      Attention.forward(xn, k_cache, v_cache, current_len,
        q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, cfg)

    hidden = Nx.add(hidden, attn_out)

    # MLP with post-norm
    xn2  = Layers.rms_norm(hidden, norm2, cfg.rms_norm_eps)
    gate = Nx.dot(xn2, [2], gate_proj, [1])
    up   = Nx.dot(xn2, [2], up_proj,   [1])
    mlp  = EMLX.Fast.swiglu(gate, up)
    out  = Nx.dot(mlp, [2], down_proj, [1])

    hidden = Nx.add(hidden, out)

    {hidden, k_new, v_new}
  end
end
