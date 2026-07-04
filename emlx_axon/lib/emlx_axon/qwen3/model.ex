defmodule EMLXAxon.Qwen3.Model do
  @moduledoc """
  Qwen3 model state struct and forward pass for quantized and dense weights.

  ## Defn / JIT strategy

  For the quantized (MLX-4bit) and dense native-greedy paths actually
  exercised by `bench/validate_qwen3.exs`, **every** hot-path call is a plain
  eager function call against concrete `EMLX.Backend` tensors —
  `EMLX.qwen3_*` NIFs, `EMLX.Fast.*` fused kernels, `Nx.dot` (quantized
  dispatch) — with no `Nx.Defn.Expr` tracing or `Nx.Defn.Compiler` involved
  anywhere in the loop. `Layers.swiglu/2` (the sole remaining `defn` in
  `Layers`/`Attention`) is dead code (`mlp/5` below calls
  `EMLX.Fast.swiglu/2` directly, not `Layers.swiglu/2`), and
  `Sampler.top_p_gpu/3` is only reached by the `:top_p_gpu` sampler, not
  `:greedy`. This is why a `:compiler` option threaded into
  `EMLXAxon.TextGeneration.serving/3` would be a no-op regardless of whether
  it's read out of `opts`: there is no `Nx.Defn` call site downstream for it
  to select a compiler for.

  `Nx.put_slice` KV-cache update (dynamic start index) and the valid-slice
  read (dynamic end index) also always ran eagerly, independent of the above.

  GPU sync: `EMLX.eval` is called once per token at the sampler boundary so
  the full lazy MLX graph spans all 28 layers before any CPU sync.
  """

  alias EMLXAxon.Qwen3.{Layers, Attention, Sampler}

  @type layer ::
          {Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(),
           Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(),
           Nx.Tensor.t()}

  defmodule State do
    @moduledoc "Loaded model weights and config."

    @enforce_keys [:embed_tokens, :layers, :norm, :lm_head, :config]
    defstruct [:embed_tokens, :layers, :norm, :lm_head, :config]

    @type t :: %__MODULE__{
            embed_tokens: Nx.Tensor.t(),
            layers: [EMLXAxon.Qwen3.Model.layer()],
            norm: Nx.Tensor.t(),
            lm_head: Nx.Tensor.t(),
            config: map()
          }
  end

  @doc false
  def layer(
        input_layernorm,
        post_attention_layernorm,
        q_norm,
        k_norm,
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
      q_norm,
      k_norm,
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
  def init_kv_cache(%State{config: cfg, layers: layers}, max_len) do
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    # Cache layout: {B, N_kv, max_len, D} — heads-before-sequence so that
    # tensors transposed to {B, N, T, D} for RoPE/SDPA slot in without copies.
    for _layer <- layers do
      k =
        0.0
        |> EMLX.full({1, num_kv_heads, max_len, head_dim}, :float16, :gpu)
        |> EMLX.Backend.to_nx()

      v =
        0.0
        |> EMLX.full({1, num_kv_heads, max_len, head_dim}, :float16, :gpu)
        |> EMLX.Backend.to_nx()

      {k, v}
    end
  end

  @doc """
  Initialise a native KV cache for dense greedy generation.

  This returns raw EMLX tensor refs instead of wrapping each cache buffer as an
  `%Nx.Tensor{}`. The native greedy Qwen3 NIFs accept this shape directly.
  Unsupported states fall back to `init_kv_cache/2`.
  """
  @spec init_native_kv_cache(State.t(), pos_integer()) :: kv_cache()
  def init_native_kv_cache(%State{} = state, max_len) do
    if native_forward_greedy?(state) do
      %State{config: cfg, layers: layers} = state
      num_kv_heads = cfg.num_key_value_heads
      head_dim = cfg.head_dim

      for _layer <- layers do
        k = EMLX.full(0.0, {1, num_kv_heads, max_len, head_dim}, :float16, :gpu)
        v = EMLX.full(0.0, {1, num_kv_heads, max_len, head_dim}, :float16, :gpu)
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

  Native dense Qwen3 can pass the token id directly to EMLX and avoid building
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
    if native_forward_greedy_chunk?(state) do
      forward_native_greedy_chunk(input_ids, kv_cache, current_len, count, state)
    else
      :fallback
    end
  end

  defp forward_hidden(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, config: cfg} = state

    hidden = embed_input(input_ids, embed_tokens)

    {hidden, kv_cache_rev} =
      if cfg[:dense_layers?] do
        # Dense f16 state: avoid quantization checks for every layer and repeated
        # attention constant calculation in the decode hot path.
        forward_dense_layers(layers, kv_cache, hidden, [], current_len, dense_layer_ctx(cfg))
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

  # Relaxed gate for `forward_greedy_chunk/5` only: `qwen3_layer_quantized`/
  # `qwen3_forward_greedy_ids_chunk_quantized` handle dense, quantized, and
  # mixed per-projection weights (including a quantized `lm_head`)
  # uniformly, so every `State` qualifies for the native chunked decode path
  # — unlike `native_forward_greedy?/1` above, which still gates the
  # single-step native paths (`forward_greedy`/`_token_id`/`_decode_token_id`)
  # that call the dense-only `qwen3_forward_greedy_ids*` NIFs and are
  # untouched by this change.
  defp native_forward_greedy_chunk?(%State{}), do: true

  defp forward_native_greedy(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, norm: norm, lm_head: lm_head, config: cfg} =
      state

    {head_dim, scale, theta, eps} = dense_layer_ctx(cfg)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)

    {token_ref, kv_cache_refs} =
      EMLX.qwen3_forward_greedy_ids(
        input_ids_ref(input_ids, embed_ref),
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        theta,
        eps
      )

    token =
      token_ref
      |> EMLX.Backend.to_nx()
      |> Nx.squeeze(axes: [0])

    {token, kv_cache_refs}
  end

  defp forward_native_greedy_token_id(input_ids, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, norm: norm, lm_head: lm_head, config: cfg} =
      state

    {head_dim, scale, theta, eps} = dense_layer_ctx(cfg)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)

    {token_id, kv_cache_refs} =
      EMLX.qwen3_forward_greedy_ids_token_id(
        input_ids_ref(input_ids, embed_ref),
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        theta,
        eps
      )

    {token_id, kv_cache_refs}
  end

  defp forward_native_greedy_decode_token_id(token_id, kv_cache, current_len, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, norm: norm, lm_head: lm_head, config: cfg} =
      state

    {head_dim, scale, theta, eps} = dense_layer_ctx(cfg)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)

    {next_token_id, kv_cache_refs} =
      EMLX.qwen3_forward_greedy_token_id(
        token_id,
        embed_ref,
        layers,
        kv_cache,
        EMLX.Backend.from_nx(norm),
        EMLX.Backend.from_nx(lm_head),
        current_len,
        scale,
        head_dim,
        theta,
        eps
      )

    {next_token_id, kv_cache_refs}
  end

  defp forward_native_greedy_chunk(input_ids, kv_cache, current_len, count, %State{} = state) do
    %State{embed_tokens: embed_tokens, layers: layers, norm: norm, lm_head: lm_head, config: cfg} =
      state

    {head_dim, scale, theta, eps} = dense_layer_ctx(cfg)
    embed_ref = EMLX.Backend.from_nx(embed_tokens)

    {token_refs, kv_cache_refs} =
      if native_forward_greedy?(state) do
        # All-dense layers + dense lm_head: keep using the proven, dense-only
        # fused chunk NIF — zero regression risk to the working fast path.
        EMLX.qwen3_forward_greedy_ids_chunk(
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
          theta,
          eps
        )
      else
        # Any quantized layer projection and/or quantized lm_head: the
        # generalized chunk NIF fuses the whole chunk into 1 NIF call instead
        # of falling back to per-token/per-op decoding.
        EMLX.qwen3_forward_greedy_ids_chunk_quantized(
          input_ids_ref(input_ids, embed_ref),
          embed_ref,
          layers,
          kv_cache,
          EMLX.Backend.from_nx(norm),
          lm_head,
          current_len,
          count,
          scale,
          head_dim,
          theta,
          eps
        )
      end

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
      |> EMLX.qwen3_final_greedy(
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

  defp dense_layer_ctx(cfg) do
    head_dim = cfg.head_dim
    {head_dim, 1.0 / :math.sqrt(head_dim), cfg.rope_theta, cfg.rms_norm_eps}
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
        {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
         down_proj},
        cfg
      ) do
    if Enum.any?(
         [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj],
         &EMLX.Quantization.quantized?/1
       ) do
      # `qwen3_layer_quantized` is a strict superset of `qwen3_layer`: it
      # accepts any dense-or-quantized mix of the 7 projections, fusing the
      # ~13 per-op NIF calls the old `Attention.forward` + `mlp/5` path made
      # for a quantized layer down to 1. `Attention.forward`/quantized `mlp`
      # stay defined below as an unreferenced rollback option.
      layer_generalized(
        hidden,
        norm1,
        q_norm,
        k_norm,
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
    else
      layer(
        hidden,
        norm1,
        q_norm,
        k_norm,
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
  end

  defp layer(
         hidden,
         norm1,
         q_norm,
         k_norm,
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
    theta = cfg.rope_theta

    {hidden_ref, k_cache_ref, v_cache_ref} =
      EMLX.qwen3_layer(
        EMLX.Backend.from_nx(hidden),
        EMLX.Backend.from_nx(norm1),
        EMLX.Backend.from_nx(q_proj),
        EMLX.Backend.from_nx(k_proj),
        EMLX.Backend.from_nx(v_proj),
        EMLX.Backend.from_nx(o_proj),
        EMLX.Backend.from_nx(q_norm),
        EMLX.Backend.from_nx(k_norm),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        current_len,
        scale,
        head_dim,
        theta,
        cfg.rms_norm_eps
      )

    {
      EMLX.Backend.to_nx(hidden_ref),
      EMLX.Backend.to_nx(k_cache_ref),
      EMLX.Backend.to_nx(v_cache_ref)
    }
  end

  # Generalized variant of `layer/16`: q/k/v/o/gate/up/down each independently
  # accept a dense or quantized (`EMLX.quantize/2`) `Nx.Tensor`, via
  # `EMLX.qwen3_layer_quantized/19`.
  defp layer_generalized(
         hidden,
         norm1,
         q_norm,
         k_norm,
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
    theta = cfg.rope_theta

    {hidden_ref, k_cache_ref, v_cache_ref} =
      EMLX.qwen3_layer_quantized(
        EMLX.Backend.from_nx(hidden),
        norm1,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        norm2,
        gate_proj,
        up_proj,
        down_proj,
        current_len,
        scale,
        head_dim,
        theta,
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
         {norm1, norm2, q_norm, k_norm, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
          down_proj},
         {head_dim, scale, theta, eps}
       ) do
    {hidden_ref, k_cache_ref, v_cache_ref} =
      EMLX.qwen3_layer(
        EMLX.Backend.from_nx(hidden),
        EMLX.Backend.from_nx(norm1),
        EMLX.Backend.from_nx(q_proj),
        EMLX.Backend.from_nx(k_proj),
        EMLX.Backend.from_nx(v_proj),
        EMLX.Backend.from_nx(o_proj),
        EMLX.Backend.from_nx(q_norm),
        EMLX.Backend.from_nx(k_norm),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        current_len,
        scale,
        head_dim,
        theta,
        eps
      )

    {
      EMLX.Backend.to_nx(hidden_ref),
      EMLX.Backend.to_nx(k_cache_ref),
      EMLX.Backend.to_nx(v_cache_ref)
    }
  end

  defp mlp(hidden, norm2, gate_proj, up_proj, down_proj, eps) do
    if Enum.any?([gate_proj, up_proj, down_proj], &EMLX.Quantization.quantized?/1) do
      xn2 = Layers.rms_norm(hidden, norm2, eps)
      gate = Nx.dot(xn2, [2], gate_proj, [1])
      up = Nx.dot(xn2, [2], up_proj, [1])
      mlp = EMLX.Fast.swiglu(gate, up)
      Nx.add(hidden, Nx.dot(mlp, [2], down_proj, [1]))
    else
      hidden
      |> EMLX.Backend.from_nx()
      |> EMLX.qwen3_mlp(
        EMLX.Backend.from_nx(norm2),
        EMLX.Backend.from_nx(gate_proj),
        EMLX.Backend.from_nx(up_proj),
        EMLX.Backend.from_nx(down_proj),
        eps
      )
      |> EMLX.Backend.to_nx()
    end
  end
end
