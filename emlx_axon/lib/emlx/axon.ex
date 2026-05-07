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
  | `:dropout`           | `op_name: :dropout` (inference)        | identity pass-through                          |
  | `:swiglu`            | `:multiply(container(up, silu(gate)))` | `EMLX.Fast.swiglu/2`                          |
  | `:native_attention`  | Bumblebee causal self-attention        | `EMLX.kv_cache_attention_masked/8`            |

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

  - **`:dropout`** rewrite replaces `op_name: :dropout` nodes with an identity
    pass-through. Dropout at inference time is always a no-op regardless of rate;
    this eliminates 56 NIF-boundary crossings per decode step in Qwen3-0.6B without
    any functional change. Not appropriate for training graphs.

  - **`:swiglu`** rewrite matches `:multiply` nodes backed by a `:container` node whose
    two parents include one `:silu` node (the Bumblebee SwiGLU pattern:
    `multiply(container(up_proj, silu(gate_proj)))`). Replaces the multiply + container +
    silu triple with a single `EMLX.Fast.swiglu/2` call. Does not match generic
    multiplications or containers without a silu child.

  - **`:attn_weights`** rewrite replaces `:bb_attn_weights` passthrough nodes (added by
    the local Bumblebee patch) with a no-arg constant-zero layer, cutting the
    `attention_weights_impl` sub-graph and its K-side `repeat_interleave` nodes out of the
    reachable graph entirely. Inference-only: the attention weights tensor is never used
    for token generation.

  - **`:if_present`** rewrite replaces Bumblebee's KV-cache conditional nodes with their
    "cache present" branch. In compiled serving the KV cache is always initialized (never
    `%Axon.None{}`), so the else branch is dead code. Removing all 86 `:if_present` nodes
    and their ~260 `:optional` wrappers eliminates significant per-step Axon dispatch
    overhead without any functional change.

  - **`:gqa_cache_fix`** rewrite fixes a shape mismatch that arises when GQA head expansion
    (`repeat_interleave`) runs *before* `update_attention_cache`. The standard Bumblebee
    transformer block expands keys/values from `num_key_value_heads` to `num_attention_heads`
    before the cache update, but the cache is allocated with `num_key_value_heads`. This
    rewrite strips the `repeat_interleave` from the key and value inputs to every
    `update_attention_cache` Axon layer, so the cache receives the compact GQA tensors.
    The SDPA rewriter (`maybe_strip_repeat_interleave`) already handles the expanded-head
    removal on the SDPA side, and MLX fast SDPA handles GQA natively.
  """

  @bumblebee_rope_mfa {Bumblebee.Layers, :apply_rotary_embedding, 5}
  @bumblebee_attn_mfa {Bumblebee.Layers, :attention_output_impl, 3}
  @bumblebee_repeat_interleave_mfa {Bumblebee.Layers, :"-repeat_interleave/3-fun-0-", 2}
  @bumblebee_update_attn_cache_mfa {Bumblebee.Layers.Decoder, :update_attention_cache, 5}
  @bumblebee_put_block_cache_mfa {Bumblebee.Layers.Decoder, :"-put_block_cache/3-fun-0-", 3}
  @kv_cache_proc_key :"$emlx_axon_native_attention_kv_cache"
  # Memoizes Nx.to_number(offset_tensor) across all 28 layers of a single forward pass.
  # Stores {MapSet.t(layer_key), cached_integer_offset}.
  @step_offset_proc_key :"$emlx_axon_step_offset_cache"

  @doc """
  Rewrites all supported nodes in `model` to their `EMLX.Fast` equivalents.

  ## Options

    * `:only` — list of atoms selecting which rewrites to apply. Defaults to
      `[:rms_norm, :layer_norm, :rotary_embedding, :sdpa, :dropout, :swiglu,
      :attn_weights, :if_present, :native_attention, :nullify_block_cache]`.
      Pass `:gqa_cache_fix` explicitly when targeting a Bumblebee build whose
      `init_cache` allocates the KV cache with `num_key_value_heads` rather than
      `num_attention_heads` (i.e. the upstream PR branch patch).

  ## Example

      model = EMLX.Axon.rewrite(model)
      model = EMLX.Axon.rewrite(model, only: [:rms_norm, :layer_norm])

  """
  @spec rewrite(Axon.t(), keyword()) :: Axon.t()
  def rewrite(%Axon{} = model, opts \\ []) do
    # Anonymous ETS table (no :named_table) so concurrent rewrite/2 calls don't collide.
    cache = :ets.new(:emlx_axon_rewrite_cache, [:set, :public])

    try do
      enabled =
        Keyword.get(opts, :only, [
          :rms_norm,
          :layer_norm,
          :rotary_embedding,
          :sdpa,
          :dropout,
          :swiglu,
          :attn_weights,
          :if_present,
          :native_attention,
          :nullify_block_cache
        ])
        |> expand_enabled()

      rewriters =
        []
        |> maybe_add(:rms_norm, rms_norm_rewriter(cache), enabled)
        |> maybe_add(:layer_norm, layer_norm_rewriter(cache), enabled)
        |> maybe_add(:rotary_embedding, rotary_embedding_rewriter(cache), enabled)
        |> maybe_add(:sdpa, sdpa_rewriter(cache), enabled)
        |> maybe_add(:dropout, dropout_rewriter(cache), enabled)
        |> maybe_add(:swiglu, swiglu_rewriter(cache), enabled)
        |> maybe_add(:attn_weights, attn_weights_rewriter(cache), enabled)
        |> maybe_add(:if_present, if_present_rewriter(cache), enabled)
        |> maybe_add(:gqa_cache_fix, gqa_cache_fix_rewriter(cache), enabled)
        |> maybe_add(:native_attention, native_attention_rewriter(cache), enabled)
        |> maybe_add(:nullify_block_cache, nullify_block_cache_rewriter(cache), enabled)

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

  # ── load_quantized ───────────────────────────────────────────────────────────

  @doc """
  Loads a quantized Bumblebee model from an MLX-4bit checkpoint directory.

  Combines three steps into one call:

  1. Loads the Axon model structure from `config.json` via `Bumblebee.load_model/2`.
  2. Loads the MLX-4bit safetensors weights via `EMLX.Axon.MLX4BitParams.load/1`,
     dequantizing and transposing to Bumblebee `{in, out}` layout (BF16).
  3. Re-quantizes all eligible weight matrices via `EMLX.Axon.QuantizeParams.quantize/1`
     so that `Nx.dot` dispatch routes to `EMLX.quantized_matmul` at serving time.

  Returns `{:ok, model_info}` compatible with `Bumblebee.Text.generation/4`.

  ## Usage

      {:ok, model_info} = EMLX.Axon.load_quantized({:local, "~/models/Qwen3-0.6B-MLX-4bit"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:local, path})
      {:ok, gen_cfg}   = Bumblebee.load_generation_config({:local, path})
      gen_cfg = Bumblebee.configure(gen_cfg, max_new_tokens: 100)

      serving = Bumblebee.Text.generation(model_info, tokenizer, gen_cfg,
        compile: [batch_size: 1, sequence_length: 256],
        defn_options: [compiler: EMLX]
      )

      result = Nx.Serving.run(serving, "The capital of France is")

  ## Notes

  - **Do not apply `EMLX.Axon.rewrite/2` after `load_quantized`** — the rotary
    embedding rewrite is incompatible with the standard Bumblebee `native_kv_cache: false`
    path and produces incorrect outputs. BF16 fast ops (rms_norm, swiglu, dropout, sdpa)
    may be added once the rotary embedding rewrite is fixed.

  - Model architecture is inferred from `config.json` in the checkpoint directory.
    Validated with Bumblebee `~> 0.6` and Qwen3-0.6B.

  - Quantization metadata: QuantizeParams logs shape-mismatch warnings for tensors whose
    physical packed dimensions differ from the Bumblebee model's expected shapes. These
    warnings are benign — the quantized tensors are still used correctly via the EMLX
    backend's quantized_matmul dispatch.
  """
  @spec load_quantized({:local, Path.t()}, keyword()) :: {:ok, map()} | {:error, term()}
  def load_quantized(source, opts \\ [])

  def load_quantized({:local, path}, opts) do
    path = Path.expand(path)
    load_model_opts = Keyword.merge([backend: {EMLX.Backend, device: :gpu}, type: :bf16], opts)

    with {:ok, model_info} <- Bumblebee.load_model({:local, path}, load_model_opts) do
      params = EMLX.Axon.MLX4BitParams.load(path)

      bb_keys = model_info.params.data |> Map.keys() |> MapSet.new()
      mlx_keys = params.data |> Map.keys() |> MapSet.new()
      missing = MapSet.difference(bb_keys, mlx_keys)
      extra = MapSet.difference(mlx_keys, bb_keys)

      if MapSet.size(missing) > 0 or MapSet.size(extra) > 0 do
        require Logger

        Logger.warning(
          "EMLX.Axon.load_quantized: param key mismatch — " <>
            "#{MapSet.size(missing)} missing from checkpoint, " <>
            "#{MapSet.size(extra)} extra in checkpoint. " <>
            "Missing: #{inspect(Enum.take(Enum.sort(missing), 5))}. " <>
            "This may indicate an unsupported model or mismatched checkpoint."
        )
      end

      quant_params = EMLX.Axon.QuantizeParams.quantize(params)
      patched = %{model_info.params | data: quant_params.data}
      {:ok, %{model_info | params: patched}}
    end
  end

  # ── rms_norm ────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `rms_norm` nodes.

  Replaces `op_name: :rms_norm` nodes with `shift: 0.0` with an Axon layer
  that calls `EMLX.Fast.rms_norm/3` — a single fused Metal shader.
  """
  @spec rms_norm_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
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
            weight =
              Axon.param(
                "weight",
                fn input_shape ->
                  {elem(input_shape, Nx.rank(input_shape) - 1)}
                end,
                initializer: :ones
              )

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
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
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
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
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

  # ── dropout ──────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `dropout` nodes.

  At inference time, dropout is always a pass-through regardless of rate. This
  rewriter replaces every `:dropout` node with an identity layer, eliminating the
  NIF-boundary crossing without any functional change.

  **Not appropriate for training graphs** — only enable this rewriter when the
  model will be used for inference only.
  """
  @spec dropout_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def dropout_rewriter(cache \\ nil) do
    _ = cache

    fn
      %Axon.Node{op_name: :dropout, name: name_fn} ->
        fn [x], _placeholder ->
          Axon.layer(fn x, _opts -> x end, [x], name: name_fn, op_name: :dropout_identity)
        end

      _ ->
        :skip
    end
  end

  # ── Attention weights elision ─────────────────────────────────────────────

  @doc """
  Returns the rewriter function for Bumblebee aux attention-weights nodes.

  Matches `:bb_attn_weights` nodes — a passthrough layer inserted by the local
  Bumblebee patch around the `{output, weights}` return of `Layers.attention/8`.
  Replaces the node with a no-arg constant-zero layer so the entire
  `attention_weights_impl` sub-graph (and the K-side `repeat_interleave` it
  consumes) becomes unreachable. Inference-only: attention weight tensors are
  never used for token generation.
  """
  @spec attn_weights_rewriter(reference() | nil) ::
          (Axon.Node.t() -> :skip | ([Axon.t(), ...], Axon.t() -> Axon.t()))
  def attn_weights_rewriter(cache \\ nil) do
    _ = cache

    fn
      %Axon.Node{op_name: :bb_attn_weights} ->
        fn [weights_axon], _placeholder ->
          # Build a constant-zero node and merge its nodes map with the existing
          # graph so rewrite_nodes can locate original_id during the ID swap.
          # The constant has no parents, so attention_weights_impl and K's
          # repeat_interleave become orphaned and are pruned by Axon automatically.
          %Axon{output: const_id, nodes: const_nodes} = Axon.constant(Nx.tensor(0.0))
          merged = Map.merge(weights_axon.nodes, const_nodes)
          %Axon{output: const_id, nodes: merged}
        end

      _ ->
        :skip
    end
  end

  # ── if_present elision ───────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for `:if_present` nodes.

  Bumblebee wraps every KV-cache operation in `Layers.if_present(cache, ...)` to
  handle the case where no cache is provided. In compiled serving the cache is
  always initialized (never `%Axon.None{}`), so the conditional is dead code.

  The rewriter unconditionally selects the "cache present" branch (`on_true`) and
  lets the "no cache" branch (`on_false`) and all its `:optional` wrappers become
  unreachable, pruning them from the compiled graph.

  **Do not enable for training graphs** — training models typically run without a
  KV cache and rely on the `else` branch.
  """
  @spec if_present_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def if_present_rewriter(_cache \\ nil) do
    fn
      # 3-parent :if_present: [optional(condition), optional(on_true), optional(on_false)]
      %Axon.Node{op_name: :if_present, parent: [_, _, _]} ->
        fn [_cond_opt, on_true_opt, _on_false_opt], _placeholder ->
          # Skip the :optional wrapper to return the underlying on_true node.
          # The cache is always non-None in compiled serving.
          optional_node = on_true_opt.nodes[on_true_opt.output]

          on_true_id =
            case {optional_node.op_name, optional_node.parent} do
              {:optional, [id]} -> id
              # If the node isn't an :optional for some reason, use it as-is.
              _ -> on_true_opt.output
            end

          %Axon{output: on_true_id, nodes: on_true_opt.nodes}
        end

      _ ->
        :skip
    end
  end

  # ── GQA cache fix ────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for GQA key/value cache shape fix.

  In the standard Bumblebee transformer block, GQA head expansion via
  `repeat_interleave` (expanding from `num_key_value_heads` to `num_attention_heads`)
  is applied to the key and value tensors *before* `update_attention_cache`. However,
  `init_cache` allocates the preallocated buffer with `num_key_value_heads`, causing
  a shape mismatch at Axon compile time when `num_key_value_heads < num_attention_heads`.

  This rewriter fixes the graph by stripping the `repeat_interleave` node from the
  key and value inputs of every `update_attention_cache` layer. The cache update then
  operates on the compact GQA tensors. The SDPA rewriter (`maybe_strip_repeat_interleave`)
  separately handles the expanded-head removal on the SDPA path, and MLX fast SDPA
  handles GQA natively without explicit head repetition.

  Only applies when the key or value parent is a Bumblebee `repeat_interleave` node.
  Models without GQA (or where repeat_interleave is already absent) are unaffected.
  """
  @spec gqa_cache_fix_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def gqa_cache_fix_rewriter(cache \\ nil) do
    _ = cache

    fn
      %Axon.Node{op: op, parent: [_key_id, _value_id | _]} ->
        if function_info(op) == @bumblebee_update_attn_cache_mfa do
          fn [key_axon, value_axon, cache_axon, offset_axon], _placeholder ->
            # Strip GQA head expansion from K and V so the cache update receives
            # {B, 1, Hkv, D} tensors matching the preallocated cache shape.
            key_raw = maybe_strip_repeat_interleave(key_axon.nodes, key_axon.output)
            val_raw = maybe_strip_repeat_interleave(value_axon.nodes, value_axon.output)
            Axon.layer(op, [key_raw, val_raw, cache_axon, offset_axon])
          end
        else
          :skip
        end

      _ ->
        :skip
    end
  end

  # ── SwiGLU ──────────────────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for SwiGLU nodes.

  Matches `:multiply` nodes backed by a single `:container` parent whose two
  children include one `:silu` node (the Bumblebee SwiGLU pattern:
  `multiply(container(up_proj, silu(gate_proj)))`). Replaces the
  multiply + container + silu triple with a single `EMLX.Fast.swiglu/2` call,
  passing the gate's raw input (pre-silu) and the up-projection directly to the
  fused NIF.

  Generic `:multiply` nodes (no `:container` parent, or container without a
  `:silu` child) are reconstructed identically.
  """
  @spec swiglu_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def swiglu_rewriter(cache \\ nil) do
    _ = cache

    fn
      # Bumblebee's SwiGLU: multiply(container(up_proj, silu(gate))).
      # The :multiply node has ONE parent (the container), and the container
      # has TWO parents: one :silu (the gate activation) and one other node
      # (the up-projection).
      %Axon.Node{op_name: :multiply, parent: [container_id], name: name_fn} ->
        fn [container_axon], _placeholder ->
          node_map = container_axon.nodes

          case detect_swiglu_pattern(node_map, container_id) do
            {gate_id, up_id} ->
              gate_axon = %Axon{output: gate_id, nodes: node_map}
              up_axon = %Axon{output: up_id, nodes: node_map}

              Axon.layer(
                fn gate, up, _opts -> EMLX.Fast.swiglu(gate, up) end,
                [gate_axon, up_axon],
                name: name_fn,
                op_name: :fast_swiglu
              )

            :skip ->
              # Not a SwiGLU container — reconstruct the original multiply.
              Axon.layer(
                fn container, _opts -> Nx.multiply(elem(container, 0), elem(container, 1)) end,
                [container_axon],
                name: name_fn,
                op_name: :multiply
              )
          end
        end

      _ ->
        :skip
    end
  end

  # Checks if `container_id` is a :container node with exactly two parents, one
  # of which is a :silu node with a single parent (the gate input).
  # Returns {gate_id, up_id} on match, or :skip otherwise.
  defp detect_swiglu_pattern(nodes, container_id) do
    container = nodes[container_id]

    with %Axon.Node{op_name: :container, parent: [a_id, b_id]} <- container do
      cond do
        nodes[a_id].op_name == :silu and length(nodes[a_id].parent) == 1 ->
          # a is silu(gate), b is up_proj
          {hd(nodes[a_id].parent), b_id}

        nodes[b_id].op_name == :silu and length(nodes[b_id].parent) == 1 ->
          # b is silu(gate), a is up_proj
          {hd(nodes[b_id].parent), a_id}

        true ->
          :skip
      end
    else
      _ -> :skip
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
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
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
              # Strip repeat_interleave from K and V if present — after the local
              # Bumblebee GQA cache patch, expand runs after cache write, so
              # mlx::fast::sdpa receives raw 8-head K/V and handles GQA natively.
              k_axon = maybe_strip_repeat_interleave(nodes, k_id)
              v_axon_8 = maybe_strip_repeat_interleave(nodes, v_axon.output)

              if System.get_env("NATIVE_ATTN_DEBUG") in ["1", "2"] do
                km_node = nodes[key_mask_id]
                km_inner = maybe_unwrap_optional(nodes, key_mask_id)
                if km_node do
                  IO.puts("[sdpa_build] key_mask_id=#{inspect(key_mask_id)} op_name=#{km_node.op_name} inner_id=#{inspect(km_inner)}")
                end
              end

              key_mask_axon = %Axon{output: key_mask_id, nodes: nodes}
              build_sdpa_layer(q_axon, k_axon, v_axon_8, key_mask_axon, scale_opt)

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

  # If `node_id` is a `repeat_interleave` custom node (GQA head expand), return
  # an Axon wrapping its single parent (the 8-head tensor). Otherwise return the
  # node unchanged. Safe to call when the patch is absent — the else branch is a no-op.
  defp maybe_strip_repeat_interleave(nodes, node_id) do
    node = nodes[node_id]

    if node.op_name == :custom and
         function_info(node.op) == @bumblebee_repeat_interleave_mfa and
         length(node.parent) == 1 do
      %Axon{output: hd(node.parent), nodes: nodes}
    else
      %Axon{output: node_id, nodes: nodes}
    end
  end

  # Causal SDPA with key_mask: delegates all-ones check to the C++ NIF.
  defp build_sdpa_layer(q_axon, k_axon, v_axon, key_mask_axon, scale_opt)
       when is_struct(key_mask_axon, Axon) do
    Axon.layer(
      fn q, k, v, key_mask, op_opts ->
        # Q, K, V arrive in {B, T, N, D}. SDPA expects {B, N, T, D}.
        q_t = Nx.transpose(q, axes: [0, 2, 1, 3])
        k_t = Nx.transpose(k, axes: [0, 2, 1, 3])
        v_t = Nx.transpose(v, axes: [0, 2, 1, 3])

        head_dim = elem(Nx.shape(q_t), 3)
        scale = op_opts[:scale] || 1.0 / :math.sqrt(head_dim)

        out = EMLX.Fast.scaled_dot_product_attention_causal_key_masked(q_t, k_t, v_t, scale, key_mask)
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

  # ── Native KV attention ─────────────────────────────────────────────────────

  @doc """
  Returns the rewriter function for Bumblebee causal self-attention nodes.

  This rewrite replaces the attention output with a single `Nx.runtime_call`
  callback that updates a process-local ETS K/V cache and calls
  `EMLX.kv_cache_attention_masked/8`. It intentionally only matches causal
  attention without sliding-window masking; cross-attention and local attention
  fall back to the original graph.
  """
  @spec native_attention_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def native_attention_rewriter(cache \\ nil) do
    _ = cache

    fn %Axon.Node{op: op, opts: node_opts} ->
      dropout_rate = Keyword.get(node_opts, :dropout_rate, 0.0)

      if function_info(op) == @bumblebee_attn_mfa and dropout_rate == 0.0 do
        original_op = op

        fn [weights_dropped_axon, v_axon], _placeholder ->
          nodes = weights_dropped_axon.nodes
          weights_dropped_id = weights_dropped_axon.output

          dropout_node = nodes[weights_dropped_id]
          [attn_weights_id] = dropout_node.parent
          attn_weights_node = nodes[attn_weights_id]

          causal = Keyword.get(attn_weights_node.opts, :causal, false)
          window_size = Keyword.get(attn_weights_node.opts, :window_size)
          scale_opt = Keyword.get(attn_weights_node.opts, :scale)

          # parents: [q_id, k_id, key_mask_id, head_mask_id, bias_id, offset_id]
          [q_id, k_id, key_mask_id, _head_mask_id, _bias_id, _offset_id] =
            attn_weights_node.parent

          with true <- causal,
               true <- is_nil(window_size),
               {:ok, k_key_id, _k_value_id, _k_cache_id, k_offset_id} <-
                 find_update_attention_cache(nodes, k_id),
               {:ok, _v_key_id, v_value_id, _v_cache_id, v_offset_id} <-
                 find_update_attention_cache(nodes, v_axon.output),
               true <- k_offset_id == v_offset_id do
            q_axon = %Axon{output: q_id, nodes: nodes}
            new_k_axon = maybe_strip_repeat_interleave(nodes, k_key_id)
            new_v_axon = maybe_strip_repeat_interleave(nodes, v_value_id)
            offset_axon = %Axon{output: k_offset_id, nodes: nodes}

            key_mask_axon = %Axon{output: key_mask_id, nodes: nodes}

            build_native_attention_layer(
              q_axon,
              new_k_axon,
              new_v_axon,
              offset_axon,
              key_mask_axon,
              scale_opt
            )
          else
            reason ->
              IO.puts("[native_attention_rewriter] FALLTHROUGH causal=#{causal} window=#{inspect(window_size)} reason=#{inspect(reason)}")
              Axon.layer(original_op, [weights_dropped_axon, v_axon])
          end
        end
      else
        :skip
      end
    end
  end

  defp build_native_attention_layer(
         q_axon,
         new_k_axon,
         new_v_axon,
         offset_axon,
         key_mask_axon,
         scale_opt
       ) do
    layer_key = make_ref()

    Axon.layer(
      fn q, new_k, new_v, offset, key_mask, op_opts ->
        out = Nx.template(Nx.shape(q), Nx.type(q))
        head_dim = elem(Nx.shape(q), 3)
        scale = op_opts[:scale] || 1.0 / :math.sqrt(head_dim)

        # Both prefill and decode are handled entirely inside the callback:
        # - Prefill (t_new > 1): callback computes SDPA eagerly, stores K/V in ETS.
        # - Decode  (t_new == 1): callback reads ETS cache, calls kv_cache_attention_masked.
        Nx.runtime_call(
          out,
          {q, new_k, new_v, offset, key_mask},
          [layer_key: op_opts[:layer_key], scale: scale],
          &__MODULE__.native_kv_attn_callback/2
        )
      end,
      [q_axon, new_k_axon, new_v_axon, offset_axon, key_mask_axon],
      op_name: :native_kv_attention,
      layer_key: layer_key,
      scale: if(is_number(scale_opt), do: scale_opt, else: nil)
    )
  end

  @doc false
  def native_kv_attn_callback(
        {query, new_k, new_v, offset_tensor, key_mask},
        opts
      ) do
    table = get_or_create_kv_table()
    layer_key = Keyword.fetch!(opts, :layer_key)
    t_new = elem(Nx.shape(new_k), 1)

    if t_new > 1 do
      # Prefill path: always read the actual offset tensor.
      #
      # The step-offset cache may hold a stale value from a previous serving's
      # last decode step when this layer_key is brand new (never seen before).
      # For prefill the optimization is irrelevant (only 1 GPU→CPU sync per
      # layer anyway since it's a single step), so bypass get_step_offset.
      #
      # Also accumulate this layer_key into the seen set so decode step 1 finds
      # it already "seen" and issues a fresh read instead of using the (now
      # correct) prefill cached_offset at the wrong decode position.
      offset = Nx.to_number(offset_tensor)
      register_prefill_layer(layer_key, offset)

      if offset == 0 do
        native_kv_prefill(query, new_k, new_v, key_mask, table, layer_key, opts)
      else
        native_kv_decode(query, new_k, new_v, offset, key_mask, table, layer_key, opts)
      end
    else
      offset = get_step_offset(offset_tensor, layer_key)
      native_kv_decode(query, new_k, new_v, offset, key_mask, table, layer_key, opts)
    end
  end

  # Registers a layer_key seen during the prefill step.
  # Accumulates all 28 prefill keys into the seen set so that decode step 1
  # will find each layer_key already "seen" and issue a fresh step-boundary read
  # at the correct decode offset rather than reusing the prefill offset (0).
  defp register_prefill_layer(layer_key, offset) do
    new_state =
      case Process.get(@step_offset_proc_key) do
        nil -> {MapSet.new([layer_key]), offset}
        {seen, _prev} -> {MapSet.put(seen, layer_key), offset}
      end

    Process.put(@step_offset_proc_key, new_state)
  end

  defp native_kv_prefill(query, new_k, new_v, key_mask, table, layer_key, opts) do
    t_new = elem(Nx.shape(new_k), 1)
    scale = Keyword.fetch!(opts, :scale)

    # Bumblebee compiles with max_length = seq_length + max_new_tokens, so
    # key_mask is {B, max_length} while new_k is {B, seq_length, N_kv, D}.
    # Pad new_k/new_v with zeros to max_length before storing in ETS so the
    # decode path can retrieve a buffer of the expected size.
    max_len =
      case Nx.shape(key_mask) do
        {_, max} -> max
        {_, _, _, max} -> max
      end

    pad_len = max_len - t_new
    {b, _, nkv, d} = Nx.shape(new_k)
    type = Nx.type(new_k)

    {k_full, v_full} =
      if pad_len > 0 do
        zeros_k = Nx.broadcast(Nx.tensor(0, type: type), {b, pad_len, nkv, d})
        zeros_v = Nx.broadcast(Nx.tensor(0, type: type), {b, pad_len, nkv, d})
        {Nx.concatenate([new_k, zeros_k], axis: 1),
         Nx.concatenate([new_v, zeros_v], axis: 1)}
      else
        {new_k, new_v}
      end

    :ets.insert(table, {{layer_key, :key}, k_full})
    :ets.insert(table, {{layer_key, :value}, v_full})

    # Compute prefill SDPA eagerly here in the callback (avoids splitting the
    # computation across the Axon layer boundary and the Nx.add(sdpa, zeros) trick).
    # Use k_full/v_full (padded to max_length) and the FULL key_mask rather than
    # slicing both to t_new. This exactly matches the default sdpa rewriter path
    # (T_kv = max_length), ensuring identical NaN-propagation behavior on Metal
    # for left-padded input sequences.

    # Q/K/V: {B, T, N, D} → transpose to {B, N, T, D} for the NIF.
    q_t = Nx.transpose(query, axes: [0, 2, 1, 3])
    k_t = Nx.transpose(k_full, axes: [0, 2, 1, 3])
    v_t = Nx.transpose(v_full, axes: [0, 2, 1, 3])

    # kv_offset = 0 for prefill (lower-triangular causal mask from position 0).
    sdpa_t =
      EMLX.fast_sdpa_causal_key_masked(
        EMLX.Backend.from_nx(q_t),
        EMLX.Backend.from_nx(k_t),
        EMLX.Backend.from_nx(v_t),
        scale,
        EMLX.Backend.from_nx(key_mask),
        0
      )
      |> EMLX.Backend.to_nx()

    # Transpose back to {B, T, N, D} and match query dtype.
    out = Nx.transpose(sdpa_t, axes: [0, 2, 1, 3])
    if Nx.type(out) == Nx.type(query), do: out, else: Nx.as_type(out, Nx.type(query))
  end

  defp native_kv_decode(query, new_k, new_v, offset, key_mask, table, layer_key, opts) do
    t_new = elem(Nx.shape(new_k), 1)
    valid_len = offset + t_new

    if offset == 0 do
      # No prefill ran for this layer — initialize ETS slots so ets_get_or_init works.
      :ets.insert(table, {{layer_key, :key}, :not_initialized})
      :ets.insert(table, {{layer_key, :value}, :not_initialized})
    end

    {batch_size, max_length, mask_axis} =
      case Nx.shape(key_mask) do
        {batch_size, max_length} -> {batch_size, max_length, 1}
        {batch_size, _heads, _query_len, max_length} -> {batch_size, max_length, 3}
      end

    {_, _, kv_heads, head_dim} = Nx.shape(new_k)
    full_shape = {batch_size, max_length, kv_heads, head_dim}
    type = Nx.type(new_k)
    scale = Keyword.fetch!(opts, :scale)

    k_cache = ets_get_or_init(table, {layer_key, :key}, full_shape, type)
    v_cache = ets_get_or_init(table, {layer_key, :value}, full_shape, type)

    key_mask_sliced = Nx.slice_along_axis(key_mask, 0, valid_len, axis: mask_axis)

    {attn_ref, k_upd_ref, v_upd_ref} =
      EMLX.kv_cache_attention_masked(
        EMLX.Backend.from_nx(query),
        EMLX.Backend.from_nx(new_k),
        EMLX.Backend.from_nx(new_v),
        EMLX.Backend.from_nx(k_cache),
        EMLX.Backend.from_nx(v_cache),
        offset,
        scale,
        EMLX.Backend.from_nx(key_mask_sliced)
      )

    attn_out = EMLX.Backend.to_nx(attn_ref)
    k_upd = EMLX.Backend.to_nx(k_upd_ref)
    v_upd = EMLX.Backend.to_nx(v_upd_ref)

    :ets.insert(table, {{layer_key, :key}, k_upd})
    :ets.insert(table, {{layer_key, :value}, v_upd})

    if Nx.type(attn_out) == Nx.type(query) do
      attn_out
    else
      Nx.as_type(attn_out, Nx.type(query))
    end
  end

  defp find_update_attention_cache(nodes, node_id, seen \\ MapSet.new()) do
    cond do
      MapSet.member?(seen, node_id) ->
        :error

      node = nodes[node_id] ->
        if function_info(node.op) == @bumblebee_update_attn_cache_mfa do
          [key_id, value_id, cache_id, offset_id] = node.parent
          {:ok, key_id, value_id, cache_id, offset_id}
        else
          seen = MapSet.put(seen, node_id)

          # For :if_present nodes, only follow parent[1] (the on_true / cache-present branch).
          # parent[0] is optional(condition) which chains through put_block_cache to OTHER
          # layers' UAC nodes. parent[2] is the fallback (no cache). Only parent[1] leads
          # to the current layer's own UAC.
          parents_to_search =
            if node.op_name == :if_present do
              case node.parent do
                [_cond, on_true, _on_false] -> [on_true]
                _ -> node.parent
              end
            else
              node.parent
            end

          Enum.find_value(parents_to_search, :error, fn parent_id ->
            case find_update_attention_cache(nodes, parent_id, seen) do
              {:ok, _key_id, _value_id, _cache_id, _offset_id} = found -> found
              :error -> nil
            end
          end)
        end

      true ->
        :error
    end
  end

  defp maybe_unwrap_optional(nodes, node_id) do
    case nodes[node_id] do
      %Axon.Node{op_name: :optional, parent: [inner_id]} -> inner_id
      _ -> node_id
    end
  end

  @doc """
  Returns the rewriter function for Bumblebee block-cache update nodes.

  When native attention owns K/V state in an ETS table, the Axon block-cache
  update chain is dead. Replacing `put_block_cache` with an identity lets DCE
  prune `get_block_cache`, `update_attention_cache`, and container plumbing.
  """
  @spec nullify_block_cache_rewriter(reference() | nil) ::
          (Axon.Node.t() -> ([Axon.t()], Axon.t() -> Axon.t()) | :skip)
  def nullify_block_cache_rewriter(cache \\ nil) do
    _ = cache

    fn %Axon.Node{op: op} ->
      if function_info(op) == @bumblebee_put_block_cache_mfa do
        fn [cache_axon, _block_cache_axon], _placeholder -> cache_axon end
      else
        :skip
      end
    end
  end

  defp get_or_create_kv_table do
    case Process.get(@kv_cache_proc_key) do
      nil ->
        table = :ets.new(:emlx_axon_native_attention_kv_cache, [:set, :public])
        Process.put(@kv_cache_proc_key, table)
        table

      table ->
        table
    end
  end

  # Memoizes Nx.to_number(offset_tensor) within a single forward pass (decode step).
  #
  # All 28 layer callbacks within one forward pass share the same offset value. Calling
  # Nx.to_number 28× forces 28 GPU→CPU syncs. Instead, call it once per step: detect
  # the step boundary by tracking which layer_keys have been called in the current step.
  # When a layer_key appears AGAIN (it was already seen in the previous step), we know
  # a new step has begun and issue one fresh Nx.to_number; all other layers reuse the cache.
  defp get_step_offset(offset_tensor, layer_key) do
    case Process.get(@step_offset_proc_key) do
      nil ->
        offset = Nx.to_number(offset_tensor)
        Process.put(@step_offset_proc_key, {MapSet.new([layer_key]), offset})
        offset

      {seen, cached_offset} ->
        if MapSet.member?(seen, layer_key) do
          # layer_key already seen in the prior step cycle → this is the start of a new step.
          offset = Nx.to_number(offset_tensor)
          Process.put(@step_offset_proc_key, {MapSet.new([layer_key]), offset})
          offset
        else
          Process.put(@step_offset_proc_key, {MapSet.put(seen, layer_key), cached_offset})
          cached_offset
        end
    end
  end

  defp ets_get_or_init(table, key, shape, type) do
    case :ets.lookup(table, key) do
      [{_, :not_initialized}] ->
        zeros = Nx.broadcast(Nx.tensor(0, type: type), shape)
        :ets.insert(table, {key, zeros})
        zeros

      [{_, cached}] ->
        cached

      [] ->
        zeros = Nx.broadcast(Nx.tensor(0, type: type), shape)
        :ets.insert(table, {key, zeros})
        zeros
    end
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

  defp expand_enabled(enabled) do
    if :native_attention in enabled do
      Enum.uniq([:if_present | enabled])
    else
      enabled
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
  - first dimension (in_features) is divisible by `group_size` (default 64)
  - first dimension < `skip_vocab_threshold` (default 100_000) — skips embed_tokens / lm_head
  - both dimensions ≥ `2 * group_size`
  """

  @doc """
  Traverse `params` and quantize all eligible weight tensors.

  ## Options

    * `:bits` — quantization bit-width, 4 (default) or 8.
    * `:group_size` — quantization group size, must evenly divide in_features (default 64).
    * `:skip_vocab_threshold` — skip tensors whose first dim exceeds this (default 100_000).
  """
  @spec quantize(map(), keyword()) :: map()
  def quantize(params, opts \\ []) do
    bits = Keyword.get(opts, :bits, 4)
    group_size = Keyword.get(opts, :group_size, 64)
    skip_vocab = Keyword.get(opts, :skip_vocab_threshold, 100_000)

    deep_map(params, fn tensor ->
      if eligible?(tensor, group_size, skip_vocab) do
        original_type = Nx.type(tensor)
        original_shape = Nx.shape(tensor)

        # Bumblebee uses {in_features, out_features} but MLX quantize expects
        # {out_features, in_features} and packs along the last (in_features) dim.
        # Transpose first, then quantize, so the physical storage is {out, in/8}.
        # Set cfg.transpose=true so quantized_dot calls mlx::quantized_matmul with
        # transpose=true (act @ dequant(w).T = act_{...,in} @ {in,out} = {..out}) ✓
        qw = EMLX.quantize(Nx.transpose(tensor), type: {:s, bits}, group_size: group_size)

        # Patch the config and restore the Bumblebee {in, out} logical shape + type
        # so Axon's shape-checking and Nx.dot's right_axes=[0] remain unaware of
        # the internal {out, in/8} physical layout.
        new_cfg = %{qw.data.quantization_config | transpose: true}
        new_data = %{qw.data | quantization_config: new_cfg}
        %Nx.Tensor{} = qw
        %{qw | data: new_data, shape: original_shape, type: original_type}
      else
        tensor
      end
    end)
  end

  # in_features is the first dim (rows) in Bumblebee {in, out} convention.
  # After transposing to {out, in}, quantize packs along in_features (last dim),
  # so we check rem(rows, group_size) == 0.
  defp eligible?(%Nx.Tensor{} = tensor, group_size, skip_vocab) do
    Nx.rank(tensor) == 2 and
      not EMLX.Quantization.quantized?(tensor) and
      (fn {rows, cols} ->
         rem(rows, group_size) == 0 and
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
