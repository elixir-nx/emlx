defmodule EMLX.Axon do
  @moduledoc """
  Axon model rewrites that swap supported nodes to `EMLX.Fast` Metal shaders.

  Pass an `%Axon{}` model through `rewrite/1` before compiling it with
  `Axon.build/2` or `Bumblebee.build_serving/2` to replace supported
  normalization and attention nodes with single-kernel MLX equivalents.

  ## Supported rewrites

  | Axon `op_name`   | Original Bumblebee impl                    | Replaced with                |
  |------------------|--------------------------------------------|------------------------------|
  | `:rms_norm`      | `rms_norm_impl_upcast_normalization` (defnp) | `EMLX.Fast.rms_norm/3`      |

  ## Planned rewrites (not yet supported)

  - `:rotary_embedding` — blocked pending a `position_ids`-aware variant of
    `EMLX.Fast.rope` (current `rope/6` takes a scalar integer offset, but
    Bumblebee's rotary embedding takes a `position_ids` tensor).
  - Attention SDPA — Bumblebee decomposes attention into multiple Axon nodes
    (`attention_weights_impl` + `attention_output_impl`); no single `op_name`
    to match.

  ## Usage

      {:ok, %{model: model, params: params}} = Bumblebee.load_model({:hf, "..."})
      model = EMLX.Axon.rewrite(model)
      {run, _} = Axon.build(model, compiler: EXLA)

  ## Limitations

  - `rms_norm` rewrite assumes `shift: 0.0` (standard Qwen3 / Llama convention).
    Nodes with a non-zero shift are left untouched because
    `EMLX.Fast.rms_norm(x, w, eps)` computes `x / rms(x) * w`, not
    `x / rms(x) * (shift + w)`.

  - The rewrite assumes the `:normalization` upcast mode (cast to f32 for the
    norm, cast result back), which is `EMLX.Fast.rms_norm`'s internal behaviour.
    Models built with `Bumblebee.Layers.rms_norm(x, upcast: :all)` are not
    detected from the graph and will be rewritten regardless — their numerical
    output will be equivalent in practice but may differ in rounding.
  """

  @doc """
  Rewrites all supported nodes in `model` to their `EMLX.Fast` equivalents.

  ## Options

    * `:only` — list of atoms selecting which rewrites to apply. Defaults to
      all supported rewrites. Currently the only supported key is `:rms_norm`.

  ## Example

      model = EMLX.Axon.rewrite(model)
      model = EMLX.Axon.rewrite(model, only: [:rms_norm])

  """
  @spec rewrite(Axon.t(), keyword()) :: Axon.t()
  def rewrite(%Axon{} = model, opts \\ []) do
    enabled = Keyword.get(opts, :only, [:rms_norm])

    rewriters =
      []
      |> maybe_add(:rms_norm, rms_norm_rewriter(), enabled)

    Axon.rewrite_nodes(model, fn node ->
      Enum.find_value(rewriters, :skip, fn {_key, fun} -> fun.(node) end)
    end)
  end

  # ── rms_norm ────────────────────────────────────────────────────────────────

  @doc """
  Returns the `Axon.rewrite_nodes/2` rewriter function for `rms_norm` nodes.

  Replaces nodes with `op_name: :rms_norm` and `shift: 0.0` with an Axon layer
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
          # Non-zero shift: x / rms(x) * (shift + w) — not equivalent to EMLX.Fast.rms_norm
          :skip
        end

      _ ->
        :skip
    end
  end

  # ── Internal helpers ─────────────────────────────────────────────────────────

  defp maybe_add(acc, key, fun, enabled) do
    if key in enabled, do: [{key, fun} | acc], else: acc
  end
end
