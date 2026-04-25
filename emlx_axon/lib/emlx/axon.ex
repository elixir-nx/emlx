defmodule EMLX.Axon do
  @moduledoc """
  Axon model rewrites that swap supported nodes to `EMLX.Fast` Metal shaders.

  Pass an `%Axon{}` model through `rewrite/1` before compiling it with
  `Axon.build/2` or `Bumblebee.Text.generation/4` to replace supported
  normalization and attention nodes with single-kernel MLX equivalents.

  ## Supported rewrites

  | Key                  | Matched node                           | Replaced with                                  |
  |----------------------|----------------------------------------|------------------------------------------------|
  | `:rms_norm`          | `op_name: :rms_norm`, `shift: 0.0`    | `EMLX.Fast.rms_norm/3`                        |
  | `:layer_norm`        | `op_name: :layer_norm`                 | `EMLX.Fast.layer_norm/3,4`                    |
  | `:rotary_embedding`  | Bumblebee `apply_rotary_embedding/5`   | `EMLX.Fast.rope_with_positions/6`             |
  | `:sdpa`              | Bumblebee `attention_output_impl/3`    | `EMLX.Fast.scaled_dot_product_attention_causal/4` or unmasked |

  ## Usage

      {:ok, %{model: model, params: params}} = Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})
      model = EMLX.Axon.rewrite(model)
      serving = Bumblebee.Text.generation(
        %{model: model, params: params, spec: spec},
        tokenizer, generation_config,
        compile: [batch_size: 1, sequence_length: 256]
      )

  ## Limitations

  - **`:rms_norm`** rewrite requires `shift: 0.0`. Nodes with a non-zero shift
    are skipped because `EMLX.Fast.rms_norm(x, w, eps)` computes `x/rms(x)*w`,
    not `x/rms(x)*(shift+w)`.

  - **`:rotary_embedding`** rewrite assumes sequential position IDs within each
    batch example (standard causal LM). Non-sequential schemes (packed sequences,
    custom position offsets) will produce incorrect results. Bumblebee's
    `apply_rotary_embedding/5` is matched by function identity via `function_info/1` —
    this is tied to Bumblebee's internal implementation and may break across major
    Bumblebee version changes.

    RoPE scaling strategies: `:llama3` precomputes the inv-frequency tensor at
    rewrite time and dispatches to `EMLX.Fast.rope_with_freqs/6`. Other strategies
    (`:linear`, `:dynamic`, `:longrope`) fall back to `rope_with_positions` with the
    standard base frequency — they are not frequency-precomputable because they are
    seq-len conditional or require post-multiply of cos/sin that `mlx::fast::rope`
    cannot absorb.

  - **`:sdpa`** rewrite threads `key_mask` (from input padding) through to the
    C++ NIF, which checks at runtime whether the mask is all-ones and fast-paths
    to the pure causal Metal kernel when it is. Padded batches get a combined
    causal + key_mask additive mask. Sliding-window attention falls back to the
    original `attention_output_impl`. Inference-only: dropout is elided.
  """

  @bumblebee_rope_mfa {Bumblebee.Layers, :apply_rotary_embedding, 5}
  @bumblebee_attn_mfa {Bumblebee.Layers, :attention_output_impl, 3}

  @doc """
  Rewrites all supported nodes in `model` to their `EMLX.Fast` equivalents.

  ## Options

    * `:only` — list of atoms selecting which rewrites to apply. Defaults to
      `[:rms_norm, :layer_norm, :rotary_embedding, :sdpa]`.

  ## Example

      model = EMLX.Axon.rewrite(model)
      model = EMLX.Axon.rewrite(model, only: [:rms_norm, :layer_norm])

  """
  @spec rewrite(Axon.t(), keyword()) :: Axon.t()
  def rewrite(%Axon{} = model, opts \\ []) do
    # Anonymous ETS table (no :named_table) so concurrent rewrite/2 calls don't collide.
    cache = :ets.new(:emlx_axon_rewrite_cache, [:set, :public])

    try do
      enabled = Keyword.get(opts, :only, [:rms_norm, :layer_norm, :rotary_embedding, :sdpa])

      rewriters =
        []
        |> maybe_add(:rms_norm, rms_norm_rewriter(cache), enabled)
        |> maybe_add(:layer_norm, layer_norm_rewriter(cache), enabled)
        |> maybe_add(:rotary_embedding, rotary_embedding_rewriter(cache), enabled)
        |> maybe_add(:sdpa, sdpa_rewriter(cache), enabled)

      Axon.rewrite_nodes(model, fn node ->
        Enum.find_value(rewriters, :skip, fn {_key, fun} ->
          case fun.(node) do
            :skip -> nil
            rewriter -> rewriter
          end
        end)
      end)
    after
      :ets.delete(cache)
    end
  end

  # ── rms_norm ────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `rms_norm` nodes.

  Replaces `op_name: :rms_norm` nodes with `shift: 0.0` with an Axon layer
  that calls `EMLX.Fast.rms_norm/3` — a single fused Metal shader.
  """
  @spec rms_norm_rewriter(reference() | nil) ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def rms_norm_rewriter(cache \\ nil) do
    _ = cache
    fn
      %Axon.Node{op_name: :rms_norm, opts: node_opts, name: name_fn} ->
        eps = Keyword.get(node_opts, :epsilon, 1.0e-6)
        shift = Keyword.get(node_opts, :shift, 0.0)

        if shift == 0.0 do
          fn [x], _placeholder ->
            # Recreate the weight parameter with the same name and shape as the
            # original rms_norm weight so model_state keys match after loading.
            # Bumblebee always uses channel_index: -1 (last axis) for rms_norm.
            weight = Axon.param("weight", fn input_shape ->
              {elem(input_shape, Nx.rank(input_shape) - 1)}
            end, initializer: :ones)

            Axon.layer(
              fn x, w, op_opts ->
                EMLX.Fast.rms_norm(x, w, op_opts[:epsilon])
              end,
              [x, weight],
              name: name_fn,
              op_name: :fast_rms_norm,
              epsilon: eps
            )
          end
        else
          :skip
        end

      _ ->
        :skip
    end
  end

  # ── layer_norm ───────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `layer_norm` nodes.

  Replaces `op_name: :layer_norm` nodes (Axon's built-in layer normalisation)
  with an Axon layer that calls `EMLX.Fast.layer_norm/3,4` — a single fused
  Metal shader. Skips nodes where `channel_index` is not `-1` (last axis),
  as the kernel only normalises over the last axis.
  """
  @spec layer_norm_rewriter(reference() | nil) ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def layer_norm_rewriter(cache \\ nil) do
    _ = cache
    fn
      %Axon.Node{op_name: :layer_norm, opts: node_opts, name: name_fn} ->
        channel_index = Keyword.get(node_opts, :channel_index, -1)
        eps = Keyword.get(node_opts, :epsilon, 1.0e-5)

        if channel_index == -1 do
          fn inputs, _placeholder ->
            case inputs do
              [x, gamma, beta] ->
                # With bias (gamma + beta as parents)
                Axon.layer(
                  fn x, gamma, beta, op_opts ->
                    EMLX.Fast.layer_norm(x, gamma, beta, op_opts[:epsilon])
                  end,
                  [x, gamma, beta],
                  name: name_fn,
                  op_name: :fast_layer_norm,
                  epsilon: eps
                )

              [x, gamma] ->
                # No bias (weight only)
                Axon.layer(
                  fn x, gamma, op_opts ->
                    EMLX.Fast.layer_norm(x, gamma, op_opts[:epsilon])
                  end,
                  [x, gamma],
                  name: name_fn,
                  op_name: :fast_layer_norm,
                  epsilon: eps
                )
            end
          end
        else
          :skip
        end

      _ ->
        :skip
    end
  end

  # ── rotary_embedding ─────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for Bumblebee's `rotary_embedding` nodes.

  Matches `%Axon.Node{op: &Bumblebee.Layers.apply_rotary_embedding/5}` by MFA
  identity via `function_info/1`, then replaces it with an `EMLX.Fast.rope_with_positions/6`
  call on both Q and K. The replacement node returns `{q_rotated, k_rotated}` —
  downstream `Axon.nx(_, &elem(&1, i))` unwrap nodes continue to work unchanged.

  **Assumes sequential positions** — see `EMLX.Axon` moduledoc for the limitation.
  """
  @spec rotary_embedding_rewriter(reference() | nil) ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def rotary_embedding_rewriter(cache \\ nil) do
    fn %Axon.Node{op: op, opts: node_opts, name: name_fn} ->
      if function_info(op) == @bumblebee_rope_mfa do
        size = Keyword.get(node_opts, :size, nil)
        base = Keyword.get(node_opts, :base, 10_000)
        strategy = Keyword.get(node_opts, :scaling_strategy)
        max_pos = Keyword.get(node_opts, :max_positions, 2048)

        # Precompute inv-frequency tensor once per unique hyperparameter combo.
        # Returns nil for unsupported strategies → falls back to rope_with_positions.
        freqs =
          if strategy do
            cached(cache, {:rope_freqs, strategy, size, base, max_pos}, fn ->
              precompute_rope_freqs(strategy, size, base, max_pos)
            end)
          end

        fn [q_axon, k_axon, pos_axon, _mask_axon], _placeholder ->
          Axon.layer(
            fn q, k, pos, _op_opts ->
              # Bumblebee uses rotate_half → traditional: false.
              # freqs and base/size are captured from rewrite-time scope.
              if freqs do
                q_rot = EMLX.Fast.rope_with_freqs(q, pos, size, false, 1.0, freqs)
                k_rot = EMLX.Fast.rope_with_freqs(k, pos, size, false, 1.0, freqs)
                {q_rot, k_rot}
              else
                q_rot = EMLX.Fast.rope_with_positions(q, pos, size, false, base, 1.0)
                k_rot = EMLX.Fast.rope_with_positions(k, pos, size, false, base, 1.0)
                {q_rot, k_rot}
              end
            end,
            [q_axon, k_axon, pos_axon],
            name: name_fn,
            op_name: :fast_rotary_embedding,
            dims: size,
            base: base
          )
        end
      else
        :skip
      end
    end
  end

  # ── SDPA ─────────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for Bumblebee's attention output nodes.

  Matches `%Axon.Node{op: &Bumblebee.Layers.attention_output_impl/3}` by MFA
  identity via `function_info/1`, then navigates up through the dropout and
  `attention_weights_impl` nodes to recover Q and K, and replaces the whole
  attention chain with a single `EMLX.Fast` SDPA call.

  - **`causal: true`, no window_size** — uses `scaled_dot_product_attention_causal_key_masked/5`.
    The `key_mask` is threaded through; the C++ NIF checks if it is all-ones at
    runtime and dispatches to the pure causal Metal kernel (no mask allocation) or
    builds a combined additive mask for padded batches.
  - **`causal: false`, no window_size** — uses `scaled_dot_product_attention/4`
    (unmasked, for cross-attention or prefix LM heads).
  - **`window_size` set** — re-applies the original `attention_output_impl` unchanged.

  **Inference-only**: attention dropout is elided (a no-op at inference time). Nodes
  with `dropout_rate > 0` are skipped to preserve training-time stochastic behaviour.
  """
  @spec sdpa_rewriter(reference() | nil) ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def sdpa_rewriter(cache \\ nil) do
    _ = cache
    fn %Axon.Node{op: op, opts: node_opts} ->
      dropout_rate = Keyword.get(node_opts, :dropout_rate, 0.0)

      if function_info(op) == @bumblebee_attn_mfa and dropout_rate == 0.0 do
        original_op = op

        fn [weights_dropped_axon, v_axon], _placeholder ->
          nodes = weights_dropped_axon.nodes
          weights_dropped_id = weights_dropped_axon.output

          # Navigate: weights_dropped (dropout) → attention_weights_impl
          dropout_node = nodes[weights_dropped_id]
          [attn_weights_id] = dropout_node.parent
          attn_weights_node = nodes[attn_weights_id]

          causal = Keyword.get(attn_weights_node.opts, :causal, false)
          window_size = Keyword.get(attn_weights_node.opts, :window_size)
          scale_opt = Keyword.get(attn_weights_node.opts, :scale)

          # parents: [q_id, k_id, key_mask_id, head_mask_id, bias_id, offset_id]
          [q_id, k_id, key_mask_id | _] = attn_weights_node.parent

          cond do
            not is_nil(window_size) ->
              # Sliding-window attention: fall back to original.
              Axon.layer(original_op, [weights_dropped_axon, v_axon])

            causal ->
              # Causal SDPA. The key_mask check runs at C++ level: if all-ones
              # (no padding — the common single-sequence case), dispatches to the
              # pure causal Metal kernel. Otherwise builds a combined additive
              # mask. No Nx.cond double-evaluation required.
              q_axon = %Axon{output: q_id, nodes: nodes}
              k_axon = %Axon{output: k_id, nodes: nodes}
              key_mask_axon = %Axon{output: key_mask_id, nodes: nodes}
              build_sdpa_layer(q_axon, k_axon, v_axon, key_mask_axon, scale_opt)

            true ->
              # Non-causal, no mask (e.g. cross-attention in encoder-decoder models).
              q_axon = %Axon{output: q_id, nodes: nodes}
              k_axon = %Axon{output: k_id, nodes: nodes}
              build_sdpa_layer(q_axon, k_axon, v_axon, scale_opt, :none)
          end
        end
      else
        :skip
      end
    end
  end

  # Causal SDPA with key_mask: delegates all-ones check to the C++ NIF.
  defp build_sdpa_layer(q_axon, k_axon, v_axon, key_mask_axon, scale_opt)
       when is_struct(key_mask_axon, Axon) do
    Axon.layer(
      fn q, k, v, key_mask, op_opts ->
        # Q, K, V arrive in {B, T, N, D}. SDPA expects {B, N, T, D}.
        q = Nx.transpose(q, axes: [0, 2, 1, 3])
        k = Nx.transpose(k, axes: [0, 2, 1, 3])
        v = Nx.transpose(v, axes: [0, 2, 1, 3])

        head_dim = elem(Nx.shape(q), 3)
        scale = op_opts[:scale] || 1.0 / :math.sqrt(head_dim)

        out = EMLX.Fast.scaled_dot_product_attention_causal_key_masked(q, k, v, scale, key_mask)
        Nx.transpose(out, axes: [0, 2, 1, 3])
      end,
      [q_axon, k_axon, v_axon, key_mask_axon],
      op_name: :fast_sdpa,
      scale: if(is_number(scale_opt), do: scale_opt, else: nil)
    )
  end

  # Non-causal SDPA (no mask).
  defp build_sdpa_layer(q_axon, k_axon, v_axon, scale_opt, :none) do
    Axon.layer(
      fn q, k, v, op_opts ->
        q = Nx.transpose(q, axes: [0, 2, 1, 3])
        k = Nx.transpose(k, axes: [0, 2, 1, 3])
        v = Nx.transpose(v, axes: [0, 2, 1, 3])

        head_dim = elem(Nx.shape(q), 3)
        scale = op_opts[:scale] || 1.0 / :math.sqrt(head_dim)

        out = EMLX.Fast.scaled_dot_product_attention(q, k, v, scale)
        Nx.transpose(out, axes: [0, 2, 1, 3])
      end,
      [q_axon, k_axon, v_axon],
      op_name: :fast_sdpa,
      scale: if(is_number(scale_opt), do: scale_opt, else: nil)
    )
  end

  # ── RoPE frequency precomputation ────────────────────────────────────────────

  # Returns a precomputed {dims/2} inv-frequency tensor for strategies where
  # the frequency vector can be baked at graph-rewrite time and passed to the
  # fast::rope freqs overload.  Returns nil for strategies that are seq-len
  # conditional or require post-multiply of cos/sin (:longrope, :linear,
  # :dynamic), falling back to rope_with_positions with the standard base freq.

  # :llama3 — smooth ramp interpolation between low- and high-freq components.
  # strategy_opts must provide :factor, :low_freq_factor, :high_freq_factor,
  # and :original_max_positions (or we use Meta's published defaults).
  defp precompute_rope_freqs({:llama3, strategy_opts}, size, base, _max_positions) do
    factor = Map.get(strategy_opts, :factor, 8.0)
    low_freq_factor = Map.get(strategy_opts, :low_freq_factor, 1.0)
    high_freq_factor = Map.get(strategy_opts, :high_freq_factor, 4.0)
    original_max_pos = Map.get(strategy_opts, :original_max_positions, 8_192)

    dims = div(size, 2)
    range = Nx.iota({dims}) |> Nx.multiply(2) |> Nx.divide(size)
    inv_freq = Nx.divide(1.0, Nx.pow(base, range))

    wavelen = Nx.multiply(2.0 * :math.pi(), Nx.divide(1.0, inv_freq))
    low_wavelen = original_max_pos / low_freq_factor
    high_wavelen = original_max_pos / high_freq_factor

    ramp =
      Nx.clip(
        Nx.divide(Nx.subtract(wavelen, high_wavelen), low_wavelen - high_wavelen),
        0.0,
        1.0
      )

    Nx.add(
      Nx.multiply(Nx.subtract(1.0, ramp), Nx.divide(inv_freq, factor)),
      Nx.multiply(ramp, inv_freq)
    )
  end

  # All other strategies fall back to rope_with_positions (standard base freq).
  defp precompute_rope_freqs(_strategy, _size, _base, _max_positions), do: nil

  # ── Helpers ──────────────────────────────────────────────────────────────────

  @doc """
  Extracts `{module, name, arity}` from a function reference, or returns `nil`
  for non-function values.

  Works for both named functions (`def`/`defp`/`defn`/`defnp`) and closures.
  Closures report the module where they were defined and a generated name like
  `"-foo/2-fun-0-"`, which is distinct from any hand-written function name and
  therefore safe to use in MFA comparisons.

  Note: Nx's `defnp` may compile to a closure rather than a named function,
  so this helper intentionally does not filter by `:erlang.fun_info(:type)`.
  """
  @spec function_info(term()) :: {module(), atom(), non_neg_integer()} | nil
  def function_info(fun) when is_function(fun) do
    {:module, m} = Function.info(fun, :module)
    {:name, n} = Function.info(fun, :name)
    {:arity, a} = Function.info(fun, :arity)
    {m, n, a}
  end

  def function_info(_), do: nil

  defp cached(nil, _key, compute_fn), do: compute_fn.()

  defp cached(cache, key, compute_fn) do
    case :ets.lookup(cache, key) do
      [{^key, value}] ->
        value

      [] ->
        value = compute_fn.()
        :ets.insert(cache, {key, value})
        value
    end
  end

  defp maybe_add(acc, key, fun, enabled) do
    if key in enabled, do: [{key, fun} | acc], else: acc
  end
end

defmodule EMLX.Axon.QuantizeParams do
  @moduledoc """
  Post-load param quantization for Bumblebee models.

  Traverses a Bumblebee params map and quantizes eligible 2-D weight tensors to
  4-bit so that `Nx.dot` dispatches to `EMLX.quantized_matmul` via the backend's
  transparent dispatch (A6-1 of the emlx#108 investigation).

  ## Usage

      {:ok, model_info} = Bumblebee.load_model(source, backend: {EMLX.Backend, device: :gpu})
      model_info = %{model_info | params: EMLX.Axon.QuantizeParams.quantize(model_info.params)}
      model_info = %{model_info | model: EMLX.Axon.rewrite(model_info.model)}

  ## Eligibility

  A tensor is quantized if ALL of the following hold:
  - rank is 2
  - last dimension is divisible by `group_size` (default 64)
  - first dimension < `skip_vocab_threshold` (default 100_000) — skips embed_tokens / lm_head
  - both dimensions ≥ `2 * group_size`
  """

  @doc """
  Traverse `params` and quantize all eligible weight tensors.

  ## Options

    * `:bits` — quantization bit-width, 4 (default) or 8.
    * `:group_size` — quantization group size, must evenly divide the last dim (default 64).
    * `:skip_vocab_threshold` — skip tensors whose first dim exceeds this (default 100_000).
  """
  @spec quantize(map(), keyword()) :: map()
  def quantize(params, opts \\ []) do
    bits       = Keyword.get(opts, :bits, 4)
    group_size = Keyword.get(opts, :group_size, 64)
    skip_vocab = Keyword.get(opts, :skip_vocab_threshold, 100_000)

    deep_map(params, fn tensor ->
      if eligible?(tensor, group_size, skip_vocab) do
        EMLX.quantize(tensor, type: {:s, bits}, group_size: group_size)
      else
        tensor
      end
    end)
  end

  defp eligible?(%Nx.Tensor{} = tensor, group_size, skip_vocab) do
    Nx.rank(tensor) == 2 and
      not EMLX.Quantization.quantized?(tensor) and
      (fn {rows, cols} ->
        rem(cols, group_size) == 0 and
          rows >= 2 * group_size and
          cols >= 2 * group_size and
          rows < skip_vocab
      end).(Nx.shape(tensor))
  end

  # Recursively traverse nested maps/lists, applying fun to Nx.Tensor leaves.
  defp deep_map(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)

  # Axon.ModelState: only traverse `data` (contains params); leave metadata fields alone.
  defp deep_map(%Axon.ModelState{data: data} = model_state, fun) do
    %{model_state | data: deep_map(data, fun)}
  end

  # Plain maps (not structs): recurse into values.
  defp deep_map(map, fun) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_map(v, fun)} end)
  end

  defp deep_map(list, fun) when is_list(list), do: Enum.map(list, &deep_map(&1, fun))
  defp deep_map(other, _fun), do: other
end
