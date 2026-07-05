defmodule EMLXAxon.MLX4BitParams do
  @moduledoc """
  Loads Qwen3 weights from an MLX-4bit safetensors checkpoint into Bumblebee
  Axon params format (BF16, Bumblebee `{in, out}` key convention).

  ## How it works

  MLX-4bit checkpoints store linear weights in `{out, in/8}` packed u32 format
  (MLX's `{out, in}` convention, packing the in-dimension). Loading proceeds in
  two steps:

  1. Dequantize each weight to BF16 — yielding the logical shape `{out, in}`.
  2. Transpose to `{in, out}` — Bumblebee's Axon convention where `Nx.dot/6`
     contracts the weight's first axis (in-features).

  After calling `load/1`, pass the result through
  `EMLXAxon.QuantizeParams.quantize/1` to re-apply 4-bit quantization in
  Bumblebee's convention so `quantized_matmul` dispatch is active at serving
  time.

  ## Usage

      {:ok, model_info} = Bumblebee.load_model({:local, mlx_path})
      params = EMLXAxon.MLX4BitParams.load(mlx_path)
      params = EMLXAxon.QuantizeParams.quantize(params)
      serving = Bumblebee.Text.generation(
        %{model_info | params: params}, tokenizer, gen_cfg, ...
      )
  """

  alias EMLX.Quantization

  @group_size 64

  @doc """
  Load Qwen3 params from an MLX-4bit checkpoint directory.

  Returns `%Axon.ModelState{}` with BF16 tensors in Bumblebee key layout.
  All linear kernels are transposed from MLX `{out, in}` to Bumblebee `{in, out}`.
  """
  def load(mlx_path) do
    mlx_path = Path.expand(mlx_path)
    config = read_config(mlx_path)
    num_layers = config["num_hidden_layers"]
    tensors = read_safetensors(mlx_path)

    params =
      %{}
      |> put_t(
        ["embedder", "token_embedding", "kernel"],
        load_embed(tensors, "model.embed_tokens")
      )
      |> put_t(
        ["output_norm", "weight"],
        to_bf16_gpu(tensors["model.norm.weight"])
      )
      |> load_lm_head(tensors, config)
      |> load_layers(tensors, num_layers)

    %Axon.ModelState{data: params}
  end

  # ── Per-section loaders ────────────────────────────────────────────────────

  defp load_lm_head(params, tensors, _config) do
    # Bumblebee uses dense_transposed for lm_head: contracts weight's last axis with
    # input. So the kernel shape must be {vocab, hidden} — MLX convention, no transpose.
    kernel =
      if Map.has_key?(tensors, "lm_head.weight") do
        # lm_head is a separate quantized weight: dequantize but don't transpose.
        load_weight_dequant(tensors, "lm_head")
      else
        # tie_word_embeddings: embed_tokens is already {vocab, hidden}, reuse as-is.
        get_in(params, ["embedder.token_embedding", "kernel"])
      end

    put_t(params, ["language_modeling_head", "output", "kernel"], kernel)
  end

  defp load_layers(params, tensors, num_layers) do
    Enum.reduce(0..(num_layers - 1), params, fn n, acc ->
      mlx = "model.layers.#{n}"
      bb = ["decoder", "blocks", "#{n}"]

      acc
      # Dense norms — shape {hidden}, same in both conventions.
      # Bumblebee's RMSNorm Axon layers use "weight" as the parameter name.
      |> put_t(
        bb ++ ["self_attention_norm", "weight"],
        to_bf16_gpu(tensors["#{mlx}.input_layernorm.weight"])
      )
      |> put_t(
        bb ++ ["output_norm", "weight"],
        to_bf16_gpu(tensors["#{mlx}.post_attention_layernorm.weight"])
      )
      |> put_t(
        bb ++ ["self_attention", "query_norm", "weight"],
        to_bf16_gpu(tensors["#{mlx}.self_attn.q_norm.weight"])
      )
      |> put_t(
        bb ++ ["self_attention", "key_norm", "weight"],
        to_bf16_gpu(tensors["#{mlx}.self_attn.k_norm.weight"])
      )
      # Quantized linear projections — dequantize → transpose {out, in} → {in, out}.
      |> put_t(
        bb ++ ["self_attention", "query", "kernel"],
        load_linear(tensors, "#{mlx}.self_attn.q_proj")
      )
      |> put_t(
        bb ++ ["self_attention", "key", "kernel"],
        load_linear(tensors, "#{mlx}.self_attn.k_proj")
      )
      |> put_t(
        bb ++ ["self_attention", "value", "kernel"],
        load_linear(tensors, "#{mlx}.self_attn.v_proj")
      )
      |> put_t(
        bb ++ ["self_attention", "output", "kernel"],
        load_linear(tensors, "#{mlx}.self_attn.o_proj")
      )
      |> put_t(
        bb ++ ["ffn", "gate", "kernel"],
        load_linear(tensors, "#{mlx}.mlp.gate_proj")
      )
      |> put_t(
        bb ++ ["ffn", "intermediate", "kernel"],
        load_linear(tensors, "#{mlx}.mlp.up_proj")
      )
      |> put_t(
        bb ++ ["ffn", "output", "kernel"],
        load_linear(tensors, "#{mlx}.mlp.down_proj")
      )
    end)
  end

  # ── Weight loaders ─────────────────────────────────────────────────────────

  # Dequantize without transpose — for dense_transposed layers (lm_head) where
  # the weight convention matches MLX's {out, in} exactly.
  defp load_weight_dequant(tensors, name) do
    weight = tensors["#{name}.weight"]
    scales = tensors["#{name}.scales"]
    biases = tensors["#{name}.biases"]
    {out_f, num_groups} = Nx.shape(scales)
    original_shape = {out_f, num_groups * @group_size}

    qt =
      Quantization.quantized_tensor(
        tensors_to_ref(weight),
        tensors_to_ref(scales),
        tensors_to_ref(biases),
        original_shape,
        type: {:s, 4},
        group_size: @group_size
      )

    Quantization.dequantize(qt)
  end

  # Quantized linear: dequantize {out, in} → transpose → {in, out} (Bumblebee).
  defp load_linear(tensors, name) do
    weight = tensors["#{name}.weight"]
    scales = tensors["#{name}.scales"]
    biases = tensors["#{name}.biases"]

    # Scales shape {out, num_groups} → original logical shape {out, in}.
    {out_f, num_groups} = Nx.shape(scales)
    original_shape = {out_f, num_groups * @group_size}

    weight_ref = tensors_to_ref(weight)
    scales_ref = tensors_to_ref(scales)
    biases_ref = tensors_to_ref(biases)

    qt =
      Quantization.quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape,
        type: {:s, 4},
        group_size: @group_size
      )

    # Dequantize to BF16, result shape = original_shape = {out, in} (MLX convention).
    # Transpose to {in, out} (Bumblebee convention) so Nx.dot contracts axis 0.
    qt
    |> Quantization.dequantize()
    |> Nx.transpose()
  end

  # Embedding matrix: may be quantized but stays {vocab, hidden} — no transpose.
  defp load_embed(tensors, name) do
    if Map.has_key?(tensors, "#{name}.scales") do
      # Quantized embed_tokens: dequantize to BF16, keep shape {vocab, hidden}.
      weight = tensors["#{name}.weight"]
      scales = tensors["#{name}.scales"]
      biases = tensors["#{name}.biases"]
      {out_f, num_groups} = Nx.shape(scales)
      original_shape = {out_f, num_groups * @group_size}

      weight_ref = tensors_to_ref(weight)
      scales_ref = tensors_to_ref(scales)
      biases_ref = tensors_to_ref(biases)

      qt =
        Quantization.quantized_tensor(weight_ref, scales_ref, biases_ref, original_shape,
          type: {:s, 4},
          group_size: @group_size
        )

      Quantization.dequantize(qt)
    else
      to_bf16_gpu(tensors["#{name}.weight"])
    end
  end

  defp tensors_to_ref(tensor) do
    EMLX.Backend.from_nx(Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu}))
  end

  defp to_bf16_gpu(tensor) do
    tensor
    |> Nx.as_type(:bf16)
    |> Nx.backend_transfer({EMLX.Backend, device: :gpu})
  end

  # ── Params map helpers ─────────────────────────────────────────────────────
  #
  # Axon's ModelState.data is a flat map:
  #   %{"layer.name" => %{"kernel" => tensor, "bias" => tensor}, ...}
  # The full dot-joined path (minus the last component) is the layer name;
  # the last component is the parameter name.

  defp put_t(params, path, tensor) do
    {layer_parts, [param_name]} = Enum.split(path, length(path) - 1)
    layer_name = Enum.join(layer_parts, ".")
    layer_entry = Map.get(params, layer_name, %{})
    Map.put(params, layer_name, Map.put(layer_entry, param_name, tensor))
  end

  # ── I/O helpers ────────────────────────────────────────────────────────────

  defp read_config(path) do
    path |> Path.join("config.json") |> File.read!() |> Jason.decode!()
  end

  defp read_safetensors(path) do
    path
    |> File.ls!()
    |> Enum.filter(&String.ends_with?(&1, ".safetensors"))
    |> Enum.sort()
    |> Enum.reduce(%{}, fn shard, acc ->
      Map.merge(acc, Safetensors.read!(Path.join(path, shard)))
    end)
  end
end
