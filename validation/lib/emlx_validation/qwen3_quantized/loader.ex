defmodule EMLX.Validation.Qwen3Quantized.Loader do
  @moduledoc """
  Loads a `lmstudio-community/Qwen3-*-MLX-4bit` checkpoint from disk into
  an `%EMLX.Validation.Qwen3Quantized.Model.State{}` struct.

  Each linear weight is stored as a triplet in the safetensors file:
    - `<name>.weight`  — `:u32`, packed int4 data
    - `<name>.scales`  — `:bf16` or `:f16`
    - `<name>.biases`  — same dtype as scales

  The loader constructs an annotated `Nx.Tensor` via
  `EMLX.Quantization.quantized_tensor/5` so plain `Nx.dot(act, weight, …)`
  dispatches transparently to `EMLX.quantized_matmul` via the backend.

  ## Usage

      {:ok, state} = Loader.load("~/models/Qwen3-0.6B-MLX-4bit")
  """

  alias EMLX.Validation.Qwen3Quantized.Model.State

  @type config :: %{
          hidden_size: pos_integer(),
          intermediate_size: pos_integer(),
          num_attention_heads: pos_integer(),
          num_key_value_heads: pos_integer(),
          head_dim: pos_integer(),
          num_hidden_layers: pos_integer(),
          vocab_size: pos_integer(),
          rms_norm_eps: float(),
          rope_theta: float(),
          tie_word_embeddings: boolean()
        }

  @doc """
  Loads a Qwen3 MLX-4bit checkpoint directory.

  Reads `config.json` to extract model dimensions, then loads all
  `.safetensors` shards found in the directory.

  Returns `{:ok, %State{}}` or `{:error, reason}`.
  """
  @spec load(Path.t()) :: {:ok, State.t()} | {:error, term()}
  def load(path) do
    path = Path.expand(path)

    with {:ok, config}  <- read_config(path),
         {:ok, tensors} <- read_safetensors(path) do
      build_state(tensors, config)
    end
  end

  # ── Config ──────────────────────────────────────────────────────────────────

  defp read_config(path) do
    config_path = Path.join(path, "config.json")

    with {:ok, json} <- File.read(config_path),
         {:ok, raw}  <- Jason.decode(json) do
      config = %{
        hidden_size:          raw["hidden_size"],
        intermediate_size:    raw["intermediate_size"],
        num_attention_heads:  raw["num_attention_heads"],
        num_key_value_heads:  raw["num_key_value_heads"],
        head_dim:             raw["head_dim"] || div(raw["hidden_size"], raw["num_attention_heads"]),
        num_hidden_layers:    raw["num_hidden_layers"],
        vocab_size:           raw["vocab_size"],
        rms_norm_eps:         raw["rms_norm_eps"] || 1.0e-6,
        rope_theta:           raw["rope_theta"] || 10_000.0,
        tie_word_embeddings:  raw["tie_word_embeddings"] || false,
        eos_token_id:         raw["eos_token_id"] || 2,
        bos_token_id:         raw["bos_token_id"] || 1
      }
      {:ok, config}
    end
  end

  # ── Safetensors ──────────────────────────────────────────────────────────────

  defp read_safetensors(path) do
    shards =
      path
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".safetensors"))
      |> Enum.sort()

    if shards == [] do
      {:error, "No .safetensors files found in #{path}"}
    else
      tensors =
        Enum.reduce(shards, %{}, fn shard, acc ->
          full = Path.join(path, shard)
          Map.merge(acc, Safetensors.read!(full))
        end)
      {:ok, tensors}
    end
  end

  # ── State builder ────────────────────────────────────────────────────────────

  defp build_state(tensors, config) do
    gpu = {EMLX.Backend, device: :gpu}

    to_gpu = fn t -> Nx.backend_transfer(t, gpu) end

    # Embedding: quantized in MLX 4-bit format; dequantize to dense f16 once
    # at load time so Nx.take works for embedding lookup and lm_head projection.
    embed_tokens =
      if Map.has_key?(tensors, "model.embed_tokens.scales") do
        qw = quantized_linear(tensors, "model.embed_tokens", config)
        EMLX.Quantization.dequantize(qw)
      else
        tensors["model.embed_tokens.weight"] |> to_gpu.()
      end

    # Norm weights are dense f16/bf16
    final_norm = tensors["model.norm.weight"] |> to_gpu.()

    # lm_head: quantize for 4× bandwidth reduction on the {vocab_size, hidden} matmul.
    # embed_tokens stays dense f16 because token lookups use Nx.take (gather), not dot.
    lm_head =
      cond do
        config.tie_word_embeddings ->
          # embed_tokens weight was already moved to GPU (and the checkpoint tensors consumed
          # by backend_transfer). Re-quantize from the dequantized f16 tensor — precision
          # loss is acceptable for a timing benchmark; the dispatched op is identical.
          EMLX.quantize(embed_tokens, type: {:s, 4}, group_size: 64)

        Map.has_key?(tensors, "lm_head.scales") ->
          quantized_linear(tensors, "lm_head", config)

        Map.has_key?(tensors, "lm_head.weight") ->
          # Dense lm_head (bf16 in checkpoint): quantize for 4× bandwidth reduction
          tensors["lm_head.weight"]
          |> to_gpu.()
          |> EMLX.quantize(type: {:s, 4}, group_size: 64)

        true ->
          embed_tokens
      end

    layers =
      for i <- 0..(config.num_hidden_layers - 1) do
        prefix = "model.layers.#{i}"

        %{
          input_layernorm:
            tensors["#{prefix}.input_layernorm.weight"] |> to_gpu.(),
          post_attention_layernorm:
            tensors["#{prefix}.post_attention_layernorm.weight"] |> to_gpu.(),
          # Qwen3 has per-head RMSNorm on q and k after projection
          q_norm: tensors["#{prefix}.self_attn.q_norm.weight"] |> to_gpu.(),
          k_norm: tensors["#{prefix}.self_attn.k_norm.weight"] |> to_gpu.(),
          q_proj: quantized_linear(tensors, "#{prefix}.self_attn.q_proj", config),
          k_proj: quantized_linear(tensors, "#{prefix}.self_attn.k_proj", config),
          v_proj: quantized_linear(tensors, "#{prefix}.self_attn.v_proj", config),
          o_proj: quantized_linear(tensors, "#{prefix}.self_attn.o_proj", config),
          gate_proj: quantized_linear(tensors, "#{prefix}.mlp.gate_proj", config),
          up_proj:   quantized_linear(tensors, "#{prefix}.mlp.up_proj", config),
          down_proj: quantized_linear(tensors, "#{prefix}.mlp.down_proj", config)
        }
      end

    state = %State{
      embed_tokens: embed_tokens,
      layers:       layers,
      norm:         final_norm,
      lm_head:      lm_head,
      config:       config
    }

    {:ok, state}
  end

  # Build an annotated quantized tensor from the weight/scales/biases triplet.
  defp quantized_linear(tensors, name, _config) do
    weight = tensors["#{name}.weight"]
    scales = tensors["#{name}.scales"]
    biases = tensors["#{name}.biases"]

    # MLX stores quantized weight in packed u32; shape is {out, in / elements_per_u32}.
    # scales/biases shape is {out, in / group_size}.
    # The original (logical) shape is inferred from scales: {out, num_groups * group_size}.
    group_size = 64
    {out_f, num_groups} = Nx.shape(scales)
    original_shape      = {out_f, num_groups * group_size}

    weight_ref = EMLX.Backend.from_nx(Nx.backend_transfer(weight, {EMLX.Backend, device: :gpu}))
    scales_ref = EMLX.Backend.from_nx(Nx.backend_transfer(scales, {EMLX.Backend, device: :gpu}))
    biases_ref = EMLX.Backend.from_nx(Nx.backend_transfer(biases, {EMLX.Backend, device: :gpu}))

    EMLX.Quantization.quantized_tensor(
      weight_ref, scales_ref, biases_ref, original_shape,
      type: {:s, 4}, group_size: group_size
    )
  end
end
