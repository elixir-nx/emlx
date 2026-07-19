defmodule EMLXAxon.Llama.Model do
  @moduledoc """
  Llama model state struct and forward pass for native dense weights.

  ## Native dense strategy

  The dense native generation path uses the EMLXAxon Llama plugin for
  layer execution, greedy selection, and chunked decode. Samplers that need
  logits use `forward/4`, which composes the per-layer plugin callback with the
  final normalization and projection.

  GPU sync: greedy paths evaluate at the native sampler boundary. Deferred host
  sync modes keep generated token tensors on the EMLX backend until the end of
  the request or the next configured chunk.
  """

  alias EMLXAxon.Llama.{Native, NativeChunkRunner}

  @native_chunk_limit NativeChunkRunner.max_chunk_size()

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
  def prepare_native(%State{} = state), do: state

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
      backend = {EMLX.Backend, device: :gpu}

      k =
        0.0
        |> Nx.tensor(type: cache_type, backend: backend)
        |> Nx.broadcast({1, num_kv_heads, max_len, head_dim})

      v = Nx.backend_copy(k, backend)

      {k, v}
    end
  end

  @doc """
  Initialise a native KV cache for dense greedy generation.

  The generic plugin interface consumes regular EMLX-backed `Nx.Tensor`
  values, so this has the same representation as `init_kv_cache/2`.
  """
  @spec init_native_kv_cache(State.t(), pos_integer()) :: kv_cache()
  def init_native_kv_cache(%State{} = state, max_len), do: init_kv_cache(state, max_len)

  @doc """
  Full forward pass for a single decode step.

  - `input_ids` — `{1, seq_len}` integer tensor (the prompt or latest token)
  - `kv_cache`  — list of `{k_cache, v_cache}` preallocated tensors
  - `current_len` — number of tokens already written into the KV cache

  Returns `{logits, kv_cache_updated}` where logits has shape `{1, vocab_size}`.
  The native plugin returns updated cache tensors, so callers must use
  `kv_cache_updated` for the next generation step rather than treating the
  input cache as updated in place.
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
  @type kv_cache :: [{Nx.Tensor.t(), Nx.Tensor.t()}]

  @spec forward_greedy(Nx.Tensor.t(), kv_cache(), non_neg_integer(), State.t()) ::
          {Nx.Tensor.t(), kv_cache()}
  def forward_greedy(input_ids, kv_cache, current_len, %State{} = state),
    do: forward_native_greedy(input_ids, kv_cache, current_len, state)

  @doc """
  Greedy decode from one host token id.

  Native dense Llama builds the next input on the model device.
  """
  @spec forward_greedy_decode_token_id(
          non_neg_integer(),
          kv_cache(),
          non_neg_integer(),
          State.t()
        ) ::
          {non_neg_integer(), kv_cache()}
  def forward_greedy_decode_token_id(token_id, kv_cache, current_len, %State{} = state),
    do: forward_native_greedy_decode_token_id(token_id, kv_cache, current_len, state)

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
          {[Nx.Tensor.t()], kv_cache()}
  def forward_greedy_chunk(input_ids, kv_cache, current_len, count, %State{} = state)
      when is_integer(count) and count > 0 and count <= @native_chunk_limit,
      do: forward_native_greedy_chunk(input_ids, kv_cache, current_len, count, state)

  def forward_greedy_chunk(_input_ids, _kv_cache, _current_len, count, %State{})
      when is_integer(count) and count > @native_chunk_limit do
    raise ArgumentError,
          "a single native Llama chunk supports at most #{@native_chunk_limit} tokens, got: #{count}"
  end

  defp forward_hidden(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, rope_freqs: rope_freqs, config: cfg} =
      state

    hidden = embed_input(input_ids, embed_tokens)

    {hidden, kv_cache_rev} =
      forward_dense_layers(
        layers,
        kv_cache,
        hidden,
        [],
        current_len,
        dense_layer_ctx(cfg, rope_freqs)
      )

    kv_cache_updated = Enum.reverse(kv_cache_rev)

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

  defp cache_type!(%State{embed_tokens: embed_tokens}) do
    case Nx.type(embed_tokens) do
      t when t in [{:f, 16}, {:bf, 16}, {:f, 32}] -> t
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
    input_ids = input_ids_tensor(input_ids, embed_tokens)
    hidden = embed_input(input_ids, embed_tokens)

    {token, kv_cache} =
      Native.forward_greedy_dense(
        hidden,
        layers,
        kv_cache,
        norm,
        lm_head,
        rope_freqs,
        current_len,
        scale,
        head_dim,
        eps
      )

    {Nx.squeeze(token, axes: [0]), Tuple.to_list(kv_cache)}
  end

  defp forward_native_greedy_token_id(input_ids, kv_cache, current_len, %State{} = state) do
    {token, kv_cache} = forward_native_greedy(input_ids, kv_cache, current_len, state)
    {Nx.to_number(token), kv_cache}
  end

  defp forward_native_greedy_decode_token_id(token_id, kv_cache, current_len, %State{} = state) do
    input_ids =
      Nx.tensor([[token_id]],
        type: :s64,
        backend: backend_from_tensor(state.embed_tokens)
      )

    forward_native_greedy_token_id(input_ids, kv_cache, current_len, state)
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
    input_ids = input_ids_tensor(input_ids, embed_tokens)

    {tokens, kv_cache} =
      Native.forward_greedy_chunk_dense(
        input_ids,
        embed_tokens,
        layers,
        kv_cache,
        norm,
        lm_head,
        rope_freqs,
        current_len,
        count,
        scale,
        head_dim,
        eps
      )

    {tokens, Tuple.to_list(kv_cache)}
  end

  defp input_ids_tensor(
         %Nx.Tensor{data: %EMLX.Backend{ref: {device, _ref}}} = input_ids,
         %Nx.Tensor{data: %EMLX.Backend{ref: {device, _embed_ref}}}
       ) do
    input_ids
  end

  defp input_ids_tensor(input_ids, embed_tokens) do
    Nx.backend_transfer(input_ids, backend_from_tensor(embed_tokens))
  end

  defp backend_from_tensor(%Nx.Tensor{data: %EMLX.Backend{ref: {device, _ref}}}),
    do: {EMLX.Backend, device: device}

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

  defp dense_transformer_layer(
         hidden,
         k_cache,
         v_cache,
         current_len,
         {norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj},
         {head_dim, scale, rope_freqs, eps}
       ) do
    Native.layer_dense(
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
      current_len,
      scale,
      head_dim,
      rope_freqs,
      eps
    )
  end
end
