defmodule EMLXAxon.Qwen3.DenseLoader do
  @moduledoc """
  Builds native Qwen3 `%EMLXAxon.Qwen3.Model.State{}` structs from standard
  Bumblebee Qwen3 dense parameters or Hugging Face safetensors files.

  Bumblebee stores dense linear kernels in `{in, out}` convention. The native
  dense Qwen3 bridge keeps that layout so MLX can use direct
  `matmul(input, weight)` calls without adding a transpose node for every
  projection in the decode graph.
  """

  alias EMLXAxon.Qwen3.Model
  alias EMLXAxon.Qwen3.Model.State

  @gpu {EMLX.Backend, device: :gpu}

  @doc """
  Converts a Bumblebee Qwen3 `model_info` map into a native dense state.
  """
  @spec from_model_info(map()) :: {:ok, State.t()} | {:error, term()}
  def from_model_info(%{params: %{data: params}, spec: %Bumblebee.Text.Qwen3{} = spec}) do
    config = config_from_spec(spec)

    layers =
      for i <- 0..(config.num_hidden_layers - 1) do
        block = "decoder.blocks.#{i}"

        Model.layer(
          tensor!(params, "#{block}.self_attention_norm", "weight"),
          tensor!(params, "#{block}.output_norm", "weight"),
          tensor!(params, "#{block}.self_attention.query_norm", "weight"),
          tensor!(params, "#{block}.self_attention.key_norm", "weight"),
          tensor!(params, "#{block}.self_attention.query", "kernel"),
          tensor!(params, "#{block}.self_attention.key", "kernel"),
          tensor!(params, "#{block}.self_attention.value", "kernel"),
          tensor!(params, "#{block}.self_attention.output", "kernel"),
          tensor!(params, "#{block}.ffn.gate", "kernel"),
          tensor!(params, "#{block}.ffn.intermediate", "kernel"),
          tensor!(params, "#{block}.ffn.output", "kernel")
        )
      end

    embed_tokens = tensor!(params, "embedder.token_embedding", "kernel")
    norm = tensor!(params, "output_norm", "weight")
    lm_head = tensor!(params, "language_modeling_head.output", "kernel")

    {:ok,
     %State{
       embed_tokens: embed_tokens,
       layers: layers,
       norm: norm,
       lm_head: lm_head,
       config: config
     }}
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  def from_model_info(%{spec: spec}) do
    {:error, "expected Bumblebee.Text.Qwen3 spec, got: #{inspect(spec)}"}
  end

  def from_model_info(_model_info),
    do: {:error, "expected Bumblebee model_info with params and spec"}

  @doc """
  Loads a native dense Qwen3 state directly from a local safetensors directory.

  This fast path is intended for standard dense Qwen3 checkpoints using
  Hugging Face tensor names, such as local f16/bf16 conversions. Projection
  weights are transposed once at load time from safetensors `{out, in}` layout
  into the native dense `{in, out}` layout.
  """
  @spec from_safetensors_dir(Path.t(), keyword()) :: {:ok, State.t()} | {:error, term()}
  def from_safetensors_dir(path, opts \\ []) do
    path = Path.expand(path)

    with {:ok, shards} <- safetensors_paths(path) do
      from_safetensors_files(Path.join(path, "config.json"), shards, opts)
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc """
  Loads a native dense Qwen3 state from explicit config and safetensors files.

  This is useful when a repository cache stores files by content addressed names
  rather than as a model directory.
  """
  @spec from_safetensors_files(Path.t(), [Path.t()], keyword()) ::
          {:ok, State.t()} | {:error, term()}
  def from_safetensors_files(config_path, safetensors_paths, opts \\ []) do
    device = Keyword.get(opts, :device, :gpu)
    type = Keyword.get(opts, :type)

    with :ok <- validate_type(type),
         :ok <- validate_device(device),
         {:ok, config} <- read_config_file(config_path),
         {:ok, tensors} <- read_safetensors_files(safetensors_paths) do
      backend = {EMLX.Backend, device: device}

      state =
        Nx.with_default_backend(backend, fn ->
          build_safetensors_state(tensors, config, type)
        end)

      {:ok, state}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp config_from_spec(spec) do
    %{
      hidden_size: spec.hidden_size,
      intermediate_size: spec.intermediate_size,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      head_dim: spec.attention_head_size,
      num_hidden_layers: spec.num_blocks,
      vocab_size: spec.vocab_size,
      rms_norm_eps: spec.layer_norm_epsilon,
      rope_theta: spec.rotary_embedding_base,
      tie_word_embeddings: spec.tie_word_embeddings,
      eos_token_id: nil,
      bos_token_id: nil,
      dense_layers?: true
    }
  end

  defp read_config_file(config_path) do
    with {:ok, json} <- File.read(config_path),
         {:ok, raw} <- Jason.decode(json) do
      {:ok,
       %{
         hidden_size: raw["hidden_size"],
         intermediate_size: raw["intermediate_size"],
         num_attention_heads: raw["num_attention_heads"],
         num_key_value_heads: raw["num_key_value_heads"],
         head_dim: raw["head_dim"] || div(raw["hidden_size"], raw["num_attention_heads"]),
         num_hidden_layers: raw["num_hidden_layers"],
         vocab_size: raw["vocab_size"],
         rms_norm_eps: raw["rms_norm_eps"] || 1.0e-6,
         rope_theta: raw["rope_theta"] || 10_000.0,
         tie_word_embeddings: raw["tie_word_embeddings"] || false,
         eos_token_id: raw["eos_token_id"],
         bos_token_id: raw["bos_token_id"],
         dense_layers?: true
       }}
    end
  end

  defp safetensors_paths(path) do
    shards =
      path
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".safetensors"))
      |> Enum.sort()
      |> Enum.map(&Path.join(path, &1))

    if shards == [] do
      {:error, "No .safetensors files found in #{path}"}
    else
      {:ok, shards}
    end
  end

  defp read_safetensors_files([]), do: {:error, "No .safetensors files given"}

  defp read_safetensors_files([path]), do: {:ok, Safetensors.read!(path, lazy: true)}

  defp read_safetensors_files(paths) do
    tensors =
      Enum.reduce(paths, %{}, fn path, acc ->
        Map.merge(acc, Safetensors.read!(path, lazy: true))
      end)

    {:ok, tensors}
  end

  defp build_safetensors_state(tensors, config, type) do
    tensor = fn key -> safetensors_tensor!(tensors, key, type) end
    linear = fn key -> tensor.(key) |> Nx.transpose() end

    layers =
      for i <- 0..(config.num_hidden_layers - 1) do
        prefix = "model.layers.#{i}"

        Model.layer(
          tensor.("#{prefix}.input_layernorm.weight"),
          tensor.("#{prefix}.post_attention_layernorm.weight"),
          tensor.("#{prefix}.self_attn.q_norm.weight"),
          tensor.("#{prefix}.self_attn.k_norm.weight"),
          linear.("#{prefix}.self_attn.q_proj.weight"),
          linear.("#{prefix}.self_attn.k_proj.weight"),
          linear.("#{prefix}.self_attn.v_proj.weight"),
          linear.("#{prefix}.self_attn.o_proj.weight"),
          linear.("#{prefix}.mlp.gate_proj.weight"),
          linear.("#{prefix}.mlp.up_proj.weight"),
          linear.("#{prefix}.mlp.down_proj.weight")
        )
      end

    embed_tokens = tensor.("model.embed_tokens.weight")

    lm_head =
      cond do
        config.tie_word_embeddings ->
          embed_tokens

        Map.has_key?(tensors, "lm_head.weight") ->
          tensor.("lm_head.weight")

        true ->
          tensor.("lm_head.weight")
      end

    %State{
      embed_tokens: embed_tokens,
      layers: layers,
      norm: tensor.("model.norm.weight"),
      lm_head: lm_head,
      config: config
    }
  end

  defp validate_type(type) when type in [nil, :f16, :bf16], do: :ok

  defp validate_type(type) do
    {:error, "expected :type to be nil, :f16, or :bf16, got: #{inspect(type)}"}
  end

  defp validate_device(:gpu), do: :ok

  defp validate_device(device) do
    {:error,
     "native dense Qwen3 generation currently supports device: :gpu only, got: #{inspect(device)}"}
  end

  defp safetensors_tensor!(tensors, key, type) do
    tensors
    |> Map.fetch!(key)
    |> Nx.to_tensor()
    |> maybe_cast(type)
  end

  defp maybe_cast(tensor, nil), do: tensor
  defp maybe_cast(tensor, :f16), do: Nx.as_type(tensor, :f16)
  defp maybe_cast(tensor, :bf16), do: Nx.as_type(tensor, :bf16)

  defp tensor!(params, scope, name) do
    case params do
      %{^scope => %{^name => tensor}} ->
        Nx.backend_transfer(tensor, @gpu)

      _missing ->
        raise ArgumentError, "missing Bumblebee parameter #{scope}.#{name}"
    end
  end
end
