defmodule EMLXAxon.Llama.Model do
  @moduledoc """
  Llama model state struct and forward pass for native dense weights.

  ## Native dense strategy

  The dense native generation path uses dedicated EMLX Llama primitives for
  layer execution, final greedy selection, and chunked decode. The generic
  tensor path remains available as a fallback for unsupported samplers.

  GPU sync: greedy paths evaluate at the native sampler boundary. Deferred host
  sync modes keep generated token tensors on the EMLX backend until the end of
  the request or the next configured chunk.
  """

  alias EMLXAxon.Llama.{Rope, Sampler}

  @type layer ::
          {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(),
           Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()}

  defmodule State do
    @moduledoc "Loaded model weights and config."

    @enforce_keys [:embed_tokens, :layers, :norm, :lm_head, :rope_freqs, :config]
    defstruct [:embed_tokens, :layers, :norm, :lm_head, :rope_freqs, :config]

    @type t :: %__MODULE__{
            embed_tokens: Nx.Tensor.t(),
            layers: [EMLXAxon.Llama.Model.layer()],
            norm: Nx.Tensor.t(),
            lm_head: Nx.Tensor.t(),
            rope_freqs: Nx.Tensor.t(),
            config: map()
          }
  end

  @doc false
  def layer(
        input_layernorm,
        post_attention_layernorm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj
      ) do
    {
      input_layernorm,
      post_attention_layernorm,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      gate_proj,
      up_proj,
      down_proj
    }
  end

  @doc """
  Initialise a preallocated KV cache for all layers.

  Returns a list of `{k_cache, v_cache}` pairs, one per transformer layer,
  where each cache is pre-allocated to `max_len` positions.
  """
  @spec init_kv_cache(State.t(), pos_integer()) :: [{Nx.Tensor.t(), Nx.Tensor.t()}]
  def init_kv_cache(%State{config: cfg, layers: layers} = state, max_len) do
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    cache_type = cache_type!(state)

    # Cache layout: {B, N_kv, max_len, D} — heads-before-sequence so that
    # tensors transposed to {B, N, T, D} for RoPE/SDPA slot in without copies.
    for _layer <- layers do
      k =
        0.0
        |> EMLX.full({1, num_kv_heads, max_len, head_dim}, cache_type, :gpu)
        |> EMLX.Backend.to_nx()

      v =
        0.0
        |> EMLX.full({1, num_kv_heads, max_len, head_dim}, cache_type, :gpu)
        |> EMLX.Backend.to_nx()

      {k, v}
    end
  end

  @doc """
  Initialise a native KV cache for dense greedy generation.

  This returns raw EMLX tensor refs instead of wrapping each cache buffer as an
  `%Nx.Tensor{}`. The native greedy Llama NIFs accept this shape directly.
  Unsupported states fall back to `init_kv_cache/2`.
  """
  @spec init_native_kv_cache(State.t(), pos_integer()) :: kv_cache()
  def init_native_kv_cache(%State{} = state, max_len) do
    if native_forward_greedy?(state) do
      %State{config: cfg, layers: layers} = state
      num_kv_heads = cfg.num_key_value_heads
      head_dim = cfg.head_dim
      cache_type = cache_type!(state)

      for _layer <- layers do
        k = EMLX.full(0.0, {1, num_kv_heads, max_len, head_dim}, cache_type, :gpu)
        v = EMLX.full(0.0, {1, num_kv_heads, max_len, head_dim}, cache_type, :gpu)
        {k, v}
      end
    else
      init_kv_cache(state, max_len)
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
    {hidden, kv_cache_updated} = forward_hidden(input_ids, kv_cache, current_len, state)

    %State{norm: norm, lm_head: lm_head, config: cfg} = state
    logits = final_logits(hidden, input_ids, norm, lm_head, cfg)

    {logits, kv_cache_updated}
  end

  @doc """
  Full forward pass for greedy generation.

  This returns the selected token id tensor directly and is only used for the
  dense native greedy path. Samplers other than greedy still use `forward/4` so
  they can inspect logits.
  """
  @type kv_cache :: [{Nx.Tensor.t() | EMLX.tensor_ref(), Nx.Tensor.t() | EMLX.tensor_ref()}]

  @spec forward_greedy(Nx.Tensor.t(), kv_cache(), non_neg_integer(), State.t()) ::
          {Nx.Tensor.t(), kv_cache()}
  def forward_greedy(input_ids, kv_cache, current_len, %State{} = state) do
    if native_forward_greedy?(state) do
      forward_native_greedy(input_ids, kv_cache, current_len, state)
    else
      {hidden, kv_cache_updated} = forward_hidden(input_ids, kv_cache, current_len, state)

      %State{norm: norm, lm_head: lm_head, config: cfg} = state
      token = final_greedy(hidden, input_ids, norm, lm_head, cfg)

      {token, kv_cache_updated}
    end
  end

  defp forward_greedy_token_id(input_ids, kv_cache, current_len, %State{} = state) do
    if native_forward_greedy?(state) do
      forward_native_greedy_token_id(input_ids, kv_cache, current_len, state)
    else
      {token, kv_cache} = forward_greedy(input_ids, kv_cache, current_len, state)
      {Nx.to_number(token), kv_cache}
    end
  end

  @doc """
  Greedy decode from one host token id.

  Native dense Llama can pass the token id directly to EMLX and avoid building
  a CPU `Nx.Tensor` plus backend transfer for every decode step. Other states
  fall back to the regular tensor based path.
  """
  @spec forward_greedy_decode_token_id(
          non_neg_integer(),
          kv_cache(),
          non_neg_integer(),
          State.t()
        ) ::
          {non_neg_integer(), kv_cache()}
  def forward_greedy_decode_token_id(token_id, kv_cache, current_len, %State{} = state) do
    if native_forward_greedy?(state) do
      forward_native_greedy_decode_token_id(token_id, kv_cache, current_len, state)
    else
      input_ids = Nx.tensor([[token_id]], type: :s64, backend: Nx.BinaryBackend)
      forward_greedy_token_id(input_ids, kv_cache, current_len, state)
    end
  end

  @doc """
  Runs multiple dense greedy decode steps in one native call.

  This is used by chunked host sync for generation without streaming. It keeps the
  generated token tensors on the EMLX backend and returns the final KV cache,
  avoiding one Elixir/NIF boundary crossing per decoded token.
  """
  @spec forward_greedy_chunk(
          Nx.Tensor.t(),
          kv_cache(),
          non_neg_integer(),
          pos_integer(),
          State.t()
        ) ::
          {[Nx.Tensor.t()], kv_cache()} | :fallback
  def forward_greedy_chunk(input_ids, kv_cache, current_len, count, %State{} = state)
      when is_integer(count) and count > 0 do
    if native_forward_greedy?(state) do
      forward_native_greedy_chunk(input_ids, kv_cache, current_len, count, state)
    else
      :fallback
    end
  end

  defp forward_hidden(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, rope_freqs: rope_freqs, config: cfg} =
      state

    hidden = embed_input(input_ids, embed_tokens)

    {hidden, kv_cache_rev} =
      if cfg[:dense_layers?] do
        # Dense state: avoid quantization checks for every layer and repeated
        # attention constant calculation in the decode hot path.
        forward_dense_layers(
          layers,
          kv_cache,
          hidden,
          [],
          current_len,
          dense_layer_ctx(cfg, rope_freqs)
        )
      else
        # Walk layers + KV pairs without Enum.zip to avoid an extra list allocation per forward.
        forward_layers(layers, kv_cache, hidden, [], current_len, cfg)
      end

    kv_cache_updated = :lists.reverse(kv_cache_rev)

    {hidden, kv_cache_updated}
  end

  defp embed_input(input_ids, embed_tokens) do
    ids_shape = Nx.shape(input_ids)

    if elem(ids_shape, 1) == 1 do
      # Decode hot path (single token): avoid a slice graph over the full row when T==1.
      Nx.take(embed_tokens, Nx.reshape(input_ids[[0, 0]], {1})) |> Nx.new_axis(0)
    else
      Nx.take(embed_tokens, input_ids[[0, ..]]) |> Nx.new_axis(0)
    end
  end

  defp native_forward_greedy?(%State{config: cfg, lm_head: lm_head}),
    do: cfg[:dense_layers?] == true and not EMLX.Quantization.quantized?(lm_head)

  defp cache_type!(%State{embed_tokens: embed_tokens}) do
    case Nx.type(embed_tokens) do
      {:f, 16} -> :float16
      {:bf, 16} -> :bfloat16
      {:f, 32} -> :float32
      type -> raise ArgumentError, "unsupported Llama KV cache tensor type: #{inspect(type)}"
    end
  end

  defp forward_native_greedy(input_ids, kv_cache, current_len, %State{} = state) do
    %State{
      embed_tokens: embed_tokens,
      layers: layers,
      norm: norm,
      lm_head: lm_head,
      rope_freqs: rope_freqs,
      config: cfg
    } =
      state

    {head_dim, scale, rope_freqs, eps} = dense_layer_ctx(cfg, rope_freqs)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)
    rope_freqs_ref = EMLX.Backend.from_nx(rope_freqs)

    {token_ref, kv_cache_refs} =
      EMLX.Native.Llama.forward_greedy_ids(
        input_ids_ref(input_ids, embed_ref),
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        rope_freqs_ref,
        eps
      )

    token =
      token_ref
      |> EMLX.Backend.to_nx()
      |> Nx.squeeze(axes: [0])

    {token, kv_cache_refs}
  end

  defp forward_native_greedy_token_id(input_ids, kv_cache, current_len, %State{} = state) do
    %State{
      embed_tokens: embed_tokens,
      layers: layers,
      norm: norm,
      lm_head: lm_head,
      rope_freqs: rope_freqs,
      config: cfg
    } =
      state

    {head_dim, scale, rope_freqs, eps} = dense_layer_ctx(cfg, rope_freqs)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)
    rope_freqs_ref = EMLX.Backend.from_nx(rope_freqs)

    {token_id, kv_cache_refs} =
      EMLX.Native.Llama.forward_greedy_ids_token_id(
        input_ids_ref(input_ids, embed_ref),
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        rope_freqs_ref,
        eps
      )

    {token_id, kv_cache_refs}
  end

  defp forward_native_greedy_decode_token_id(token_id, kv_cache, current_len, %State{} = state) do
    %State{
      embed_tokens: embed_tokens,
      layers: layers,
      norm: norm,
      lm_head: lm_head,
      rope_freqs: rope_freqs,
      config: cfg
    } =
      state

    {head_dim, scale, rope_freqs, eps} = dense_layer_ctx(cfg, rope_freqs)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)
    rope_freqs_ref = EMLX.Backend.from_nx(rope_freqs)

    {next_token_id, kv_cache_refs} =
      EMLX.Native.Llama.forward_greedy_token_id(
        token_id,
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        rope_freqs_ref,
        eps
      )

    {next_token_id, kv_cache_refs}
  end

  defp forward_native_greedy_chunk(input_ids, kv_cache, current_len, count, %State{} = state) do
    %State{
      embed_tokens: embed_tokens,
      layers: layers,
      norm: norm,
      lm_head: lm_head,
      rope_freqs: rope_freqs,
      config: cfg
    } =
      state

    {head_dim, scale, rope_freqs, eps} = dense_layer_ctx(cfg, rope_freqs)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)
    rope_freqs_ref = EMLX.Backend.from_nx(rope_freqs)

    {token_refs, kv_cache_refs} =
      EMLX.Native.Llama.forward_greedy_ids_chunk(
        input_ids_ref(input_ids, embed_ref),
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        count,
        scale,
        head_dim,
        rope_freqs_ref,
        eps
      )

    tokens =
      Enum.map(token_refs, fn token_ref ->
        token_ref
        |> EMLX.Backend.to_nx()
        |> Nx.squeeze(axes: [0])
      end)

    {tokens, kv_cache_refs}
  end

  defp input_ids_ref(%Nx.Tensor{data: %EMLX.Backend{ref: {device, ref}}}, {device, _ref}) do
    {device, ref}
  end

  defp input_ids_ref(input_ids, {device, _ref}) do
    input_ids
    |> Nx.backend_transfer({EMLX.Backend, device: device})
    |> EMLX.Backend.from_nx()
  end

  defp final_logits(hidden, input_ids, norm, lm_head, cfg) do
    ids_shape = Nx.shape(input_ids)

    # Final norm + lm_head on the last token position only
    last_hidden =
      if elem(ids_shape, 1) == 1 do
        Nx.squeeze(hidden, axes: [1])
      else
        hidden[[.., -1, ..]]
      end

    normed = EMLX.Fast.rms_norm(last_hidden, norm, cfg.rms_norm_eps)
    Nx.dot(normed, [1], lm_head, [1])
  end

  defp final_greedy(hidden, input_ids, norm, lm_head, cfg) do
    if EMLX.Quantization.quantized?(lm_head) do
      hidden
      |> final_logits(input_ids, norm, lm_head, cfg)
      |> Sampler.greedy()
    else
      hidden
      |> EMLX.Backend.from_nx()
      |> EMLX.Native.Llama.final_greedy(
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        cfg.rms_norm_eps
      )
      |> EMLX.Backend.to_nx()
      |> Nx.squeeze(axes: [0])
    end
  end

  defp forward_layers([], [], hidden, acc, _cur_len, _cfg), do: {hidden, acc}

  defp forward_layers(
         [layer_weights | layers_rest],
         [{k_cache, v_cache} | kv_rest],
         hidden,
         acc,
         cur_len,
         cfg
       ) do
    {h_new, k_new, v_new} =
      transformer_layer(hidden, k_cache, v_cache, cur_len, layer_weights, cfg)

    forward_layers(layers_rest, kv_rest, h_new, [{k_new, v_new} | acc], cur_len, cfg)
  end

  defp dense_layer_ctx(cfg, rope_freqs) do
    head_dim = cfg.head_dim
    {head_dim, 1.0 / :math.sqrt(head_dim), rope_freqs, cfg.rms_norm_eps}
  end

  defp forward_dense_layers([], [], hidden, acc, _cur_len, _ctx), do: {hidden, acc}

  defp forward_dense_layers(
         [layer_weights | layers_rest],
         [{k_cache, v_cache} | kv_rest],
         hidden,
         acc,
         cur_len,
         ctx
       ) do
    {h_new, k_new, v_new} =
      dense_transformer_layer(hidden, k_cache, v_cache, cur_len, layer_weights, ctx)

    forward_dense_layers(layers_rest, kv_rest, h_new, [{k_new, v_new} | acc], cur_len, ctx)
  end

  @doc false
  def transformer_layer(
        hidden,
        k_cache,
        v_cache,
        current_len,
        {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj},
        cfg
      ) do
    layer(
      hidden,
      norm1,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      gate_proj,
      up_proj,
      down_proj,
      norm2,
      k_cache,
      v_cache,
      current_len,
      cfg
    )
  end

  defp layer(
         hidden,
         norm1,
         q_proj,
         k_proj,
         v_proj,
         o_proj,
         gate_proj,
         up_proj,
         down_proj,
         norm2,
         k_cache,
         v_cache,
         current_len,
         cfg
       ) do
    head_dim = cfg.head_dim
    scale = 1.0 / :math.sqrt(head_dim)

    {hidden_ref, k_cache_ref, v_cache_ref} =
      EMLX.Native.Llama.layer(
        EMLX.Backend.from_nx(hidden),
        EMLX.Backend.from_nx(norm1),
        EMLX.Backend.from_nx(q_proj),
        EMLX.Backend.from_nx(k_proj),
        EMLX.Backend.from_nx(v_proj),
        EMLX.Backend.from_nx(o_proj),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        current_len,
        scale,
        head_dim,
        EMLX.Backend.from_nx(Rope.freqs_from_config!(cfg)),
        cfg.rms_norm_eps
      )

    {
      EMLX.Backend.to_nx(hidden_ref),
      EMLX.Backend.to_nx(k_cache_ref),
      EMLX.Backend.to_nx(v_cache_ref)
    }
  end

  defp dense_transformer_layer(
         hidden,
         k_cache,
         v_cache,
         current_len,
         {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj},
         {head_dim, scale, rope_freqs, eps}
       ) do
    {hidden_ref, k_cache_ref, v_cache_ref} =
      EMLX.Native.Llama.layer(
        EMLX.Backend.from_nx(hidden),
        EMLX.Backend.from_nx(norm1),
        EMLX.Backend.from_nx(q_proj),
        EMLX.Backend.from_nx(k_proj),
        EMLX.Backend.from_nx(v_proj),
        EMLX.Backend.from_nx(o_proj),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        current_len,
        scale,
        head_dim,
        EMLX.Backend.from_nx(rope_freqs),
        eps
      )

    {
      EMLX.Backend.to_nx(hidden_ref),
      EMLX.Backend.to_nx(k_cache_ref),
      EMLX.Backend.to_nx(v_cache_ref)
    }
  end
end
