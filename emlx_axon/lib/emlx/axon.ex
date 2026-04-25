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
  | `:rotary_embedding`  | Bumblebee `apply_rotary_embedding/5`   | `EMLX.Fast.rope_with_positions/6`             |
  | `:sdpa`              | Bumblebee `attention_output_impl/3`    | `EMLX.Fast.scaled_dot_product_attention_causal/4` |

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
    `apply_rotary_embedding/5` is matched by function identity — this is tied to
    Bumblebee's internal implementation and may break across major Bumblebee
    version changes.

  - **`:sdpa`** rewrite requires causal attention (`causal: true` in the upstream
    `Layers.attention` call) and no sliding-window size. The rewrite drops the
    external `key_mask` (from input padding) — suitable for single-sequence
    inference but incorrect for batches with different sequence lengths. It is also
    inference-only: the original stochastic dropout on attention weights is elided
    (a no-op at inference time anyway). Matched by Bumblebee's internal
    `attention_output_impl/3` function identity.
  """

  @doc """
  Rewrites all supported nodes in `model` to their `EMLX.Fast` equivalents.

  ## Options

    * `:only` — list of atoms selecting which rewrites to apply. Defaults to
      `[:rms_norm, :rotary_embedding, :sdpa]`.

  ## Example

      model = EMLX.Axon.rewrite(model)
      model = EMLX.Axon.rewrite(model, only: [:rms_norm])

  """
  @spec rewrite(Axon.t(), keyword()) :: Axon.t()
  def rewrite(%Axon{} = model, opts \\ []) do
    enabled = Keyword.get(opts, :only, [:rms_norm, :rotary_embedding, :sdpa])

    rewriters =
      []
      |> maybe_add(:rms_norm, rms_norm_rewriter(), enabled)
      |> maybe_add(:rotary_embedding, rotary_embedding_rewriter(), enabled)
      |> maybe_add(:sdpa, sdpa_rewriter(), enabled)

    Axon.rewrite_nodes(model, fn node ->
      Enum.find_value(rewriters, :skip, fn {_key, fun} ->
        case fun.(node) do
          :skip -> nil
          rewriter -> rewriter
        end
      end)
    end)
  end

  # ── rms_norm ────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `rms_norm` nodes.

  Replaces `op_name: :rms_norm` nodes with `shift: 0.0` with an Axon layer
  that calls `EMLX.Fast.rms_norm/3` — a single fused Metal shader.
  """
  @spec rms_norm_rewriter() :: (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def rms_norm_rewriter do
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

  # ── rotary_embedding ─────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for Bumblebee's `rotary_embedding` nodes.

  Matches `%Axon.Node{op: &Bumblebee.Layers.apply_rotary_embedding/5}` by
  inspecting the function's module and name via `Function.info/1`, then replaces
  it with an `EMLX.Fast.rope_with_positions/6` call on both Q and K.

  The replacement node returns `{q_rotated, k_rotated}` — downstream
  `Axon.nx(_, &elem(&1, i))` unwrap nodes continue to work unchanged because
  `Axon.rewrite_nodes/2` patches all parent IDs correctly.

  **Assumes sequential positions** — see `EMLX.Axon` moduledoc for the
  limitation.
  """
  @spec rotary_embedding_rewriter() ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def rotary_embedding_rewriter do
    fn %Axon.Node{op: op, opts: node_opts, name: name_fn} ->
      info = if is_function(op), do: Function.info(op), else: []

      if info[:module] == Bumblebee.Layers and
           info[:name] == :apply_rotary_embedding do
        size = Keyword.get(node_opts, :size, nil)
        base = Keyword.get(node_opts, :base, 10_000)

        fn [q_axon, k_axon, pos_axon, _mask_axon], _placeholder ->
          Axon.layer(
            fn q, k, pos, op_opts ->
              dims = op_opts[:dims]
              base_val = op_opts[:base]
              # Bumblebee's apply_rotary_embedding uses rotate_half → traditional: false
              q_rot = EMLX.Fast.rope_with_positions(q, pos, dims, false, base_val, 1.0)
              k_rot = EMLX.Fast.rope_with_positions(k, pos, dims, false, base_val, 1.0)
              {q_rot, k_rot}
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

  Matches `%Axon.Node{op: &Bumblebee.Layers.attention_output_impl/3}` by function
  identity, then navigates up through the dropout and `attention_weights_impl`
  nodes to recover Q and K, and replaces the whole attention chain with
  `EMLX.Fast.scaled_dot_product_attention_causal/4`.

  Only rewrites when all of the following are true (checked inside the rewriter):
  - `causal: true` on the upstream `attention_weights_impl` node.
  - No sliding-window size (`window_size: nil`).
  - Rewrite is inference-only (attention dropout is elided).

  If conditions are not met the original `attention_output_impl` is re-applied
  using the function reference captured from the node (no behaviour change).
  """
  @spec sdpa_rewriter() ::
          (Axon.Node.t() -> (([Axon.t()], Axon.t()) -> Axon.t()) | :skip)
  def sdpa_rewriter do
    fn %Axon.Node{op: op} ->
      info = if is_function(op), do: Function.info(op), else: []

      if info[:module] == Bumblebee.Layers and
           info[:name] == :attention_output_impl do
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

          if causal and is_nil(window_size) do
            # parents: [q_id, k_id, optional_mask_id, optional_head_mask_id, ...]
            [q_id, k_id | _] = attn_weights_node.parent
            q_axon = %Axon{output: q_id, nodes: nodes}
            k_axon = %Axon{output: k_id, nodes: nodes}

            Axon.layer(
              fn q, k, v, op_opts ->
                # Q, K, V arrive in {B, T, N, D} (Bumblebee split_heads convention).
                # SDPA expects {B, N, T, D}.
                q = Nx.transpose(q, axes: [0, 2, 1, 3])
                k = Nx.transpose(k, axes: [0, 2, 1, 3])
                v = Nx.transpose(v, axes: [0, 2, 1, 3])

                head_dim = elem(Nx.shape(q), 3)
                scale = op_opts[:scale] || :math.pow(head_dim, -0.5)

                out = EMLX.Fast.scaled_dot_product_attention_causal(q, k, v, scale)
                # Transpose back: {B, N, T, D} → {B, T, N, D}
                Nx.transpose(out, axes: [0, 2, 1, 3])
              end,
              [q_axon, k_axon, v_axon],
              op_name: :fast_sdpa,
              scale: if(is_number(scale_opt), do: scale_opt, else: nil)
            )
          else
            # Conditions not met — re-apply the original attention_output_impl.
            Axon.layer(original_op, [weights_dropped_axon, v_axon])
          end
        end
      else
        :skip
      end
    end
  end

  # ── Internal helpers ─────────────────────────────────────────────────────────

  defp maybe_add(acc, key, fun, enabled) do
    if key in enabled, do: [{key, fun} | acc], else: acc
  end
end
