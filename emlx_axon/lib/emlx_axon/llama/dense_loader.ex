defmodule EMLXAxon.Llama.DenseLoader do
  @moduledoc """
  Builds native Llama `%EMLXAxon.Llama.Model.State{}` structs from standard
  Bumblebee Llama dense parameters or Hugging Face safetensors files.

  Bumblebee stores dense linear kernels in `{in, out}` convention. The native
  dense Llama bridge keeps that layout so MLX can use direct
  `matmul(input, weight)` calls without adding a transpose node for every
  projection in the decode graph.
  """

  alias EMLXAxon.Llama.{Model, Rope}
  alias EMLXAxon.Llama.Model.State

  @gpu {EMLX.Backend, device: :gpu}

  @doc """
  Converts a Bumblebee Llama `model_info` map into a native dense state.

  ## Options

    * `:generation_config` - a `%Bumblebee.Text.GenerationConfig{}` returned by
      `Bumblebee.load_generation_config/2`. Its `:eos_token_id` and
      `:bos_token_id` fields are copied into the native state.

    * `:eos_token_id` - overrides the EOS token id. Accepts a non-negative
      integer or a non-empty list of non-negative integers.

    * `:bos_token_id` - overrides the BOS token id. Accepts a non-negative
      integer or a non-empty list of non-negative integers.
  """
  @spec from_model_info(map()) :: {:ok, State.t()} | {:error, term()}
  @spec from_model_info(map(), keyword()) :: {:ok, State.t()} | {:error, term()}
  def from_model_info(model_info, opts \\ [])

  def from_model_info(%{params: %{data: params}, spec: %Bumblebee.Text.Llama{} = spec}, opts)
      when is_list(opts) do
    opts = Keyword.validate!(opts, [:generation_config, :eos_token_id, :bos_token_id])
    config = config_from_spec(spec, opts)

    layers =
      for i <- 0..(config.num_hidden_layers - 1) do
        block = "decoder.blocks.#{i}"

        Model.layer(
          tensor!(params, "#{block}.self_attention_norm", "weight"),
          tensor!(params, "#{block}.output_norm", "weight"),
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
       rope_freqs: Rope.freqs_from_config!(config),
       config: config
     }}
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  def from_model_info(%{params: %{data: _params}, spec: %Bumblebee.Text.Llama{}}, opts) do
    {:error, "expected opts to be a keyword list, got: #{inspect(opts)}"}
  end

  def from_model_info(%{spec: spec}, _opts) do
    {:error, "expected Bumblebee.Text.Llama spec, got: #{inspect(spec)}"}
  end

  def from_model_info(_model_info, _opts),
    do: {:error, "expected Bumblebee model_info with params and spec"}

  @doc """
  Loads a native dense Llama state directly from a local safetensors directory.

  This fast path is intended for standard dense Llama checkpoints using
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
  Loads a native dense Llama state from explicit config and safetensors files.

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

  defp config_from_spec(spec, opts) do
    validate_supported_spec!(spec)

    %{
      hidden_size: spec.hidden_size,
      intermediate_size: spec.intermediate_size,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads || spec.num_attention_heads,
      head_dim: spec.attention_head_size || div(spec.hidden_size, spec.num_attention_heads),
      num_hidden_layers: spec.num_blocks,
      vocab_size: spec.vocab_size,
      rms_norm_eps: spec.layer_norm_epsilon,
      rope_theta: spec.rotary_embedding_base,
      rope_scaling: normalize_rope_scaling(spec.rotary_embedding_scaling_strategy),
      tie_word_embeddings: spec.tie_word_embeddings,
      eos_token_id: token_id_from_opts(opts, :eos_token_id),
      bos_token_id: token_id_from_opts(opts, :bos_token_id)
    }
  end

  defp token_id_from_opts(opts, key) do
    token_id =
      case Keyword.fetch(opts, key) do
        {:ok, value} -> value
        :error -> generation_config_token_id(opts[:generation_config], key)
      end

    validate_token_id!(token_id, key)
  end

  defp generation_config_token_id(%Bumblebee.Text.GenerationConfig{} = config, key),
    do: Map.get(config, key)

  defp generation_config_token_id(nil, _key), do: nil

  defp generation_config_token_id(other, _key) do
    raise ArgumentError,
          "expected :generation_config to be a Bumblebee.Text.GenerationConfig struct, got: #{inspect(other)}"
  end

  defp validate_token_id!(nil, _key), do: nil

  defp validate_token_id!(token_id, _key) when is_integer(token_id) and token_id >= 0,
    do: token_id

  defp validate_token_id!(token_ids, key)
       when is_list(token_ids) and token_ids != [] do
    if Enum.all?(token_ids, &(is_integer(&1) and &1 >= 0)) do
      token_ids
    else
      raise ArgumentError,
            "expected #{inspect(key)} to be a non-negative integer or a list of non-negative integers, got: #{inspect(token_ids)}"
    end
  end

  defp validate_token_id!(token_id, key) do
    raise ArgumentError,
          "expected #{inspect(key)} to be a non-negative integer or a list of non-negative integers, got: #{inspect(token_id)}"
  end

  defp read_config_file(config_path) do
    with {:ok, json} <- File.read(config_path),
         {:ok, raw} <- Jason.decode(json),
         :ok <- validate_supported_config(raw) do
      {:ok,
       %{
         hidden_size: raw["hidden_size"],
         intermediate_size: raw["intermediate_size"],
         num_attention_heads: raw["num_attention_heads"],
         num_key_value_heads: raw["num_key_value_heads"] || raw["num_attention_heads"],
         head_dim: raw["head_dim"] || div(raw["hidden_size"], raw["num_attention_heads"]),
         num_hidden_layers: raw["num_hidden_layers"],
         vocab_size: raw["vocab_size"],
         rms_norm_eps: raw["rms_norm_eps"] || 1.0e-6,
         rope_theta: raw["rope_theta"] || 10_000.0,
         rope_scaling: normalize_rope_scaling(raw["rope_scaling"]),
         max_position_embeddings: raw["max_position_embeddings"],
         tie_word_embeddings: raw["tie_word_embeddings"] || false,
         eos_token_id: raw["eos_token_id"],
         bos_token_id: raw["bos_token_id"]
       }}
    end
  end

  defp validate_supported_spec!(%{activation: activation}) when activation in [nil, :silu],
    do: :ok

  defp validate_supported_spec!(%{activation: activation}) do
    raise ArgumentError,
          "native dense Llama generation supports activation :silu only, got: #{inspect(activation)}"
  end

  defp validate_supported_config(raw) do
    with :ok <- validate_model_type(raw),
         :ok <- validate_sliding_window(raw),
         :ok <- validate_false_config(raw, "attention_bias"),
         :ok <- validate_false_config(raw, "mlp_bias"),
         :ok <- validate_hidden_act(raw) do
      :ok
    end
  end

  defp validate_model_type(%{"model_type" => "llama"}), do: :ok

  defp validate_model_type(raw) do
    {:error,
     "native dense Llama generation expected model_type \"llama\", got: #{inspect(raw["model_type"])}"}
  end

  defp validate_sliding_window(raw) do
    case raw["sliding_window"] do
      nil ->
        :ok

      sliding_window ->
        {:error,
         "native dense Llama generation does not support sliding_window, got: #{inspect(sliding_window)}"}
    end
  end

  defp validate_false_config(raw, key) do
    case Map.get(raw, key, false) do
      false ->
        :ok

      nil ->
        :ok

      true ->
        {:error, "native dense Llama generation does not support #{key}=true"}

      other ->
        {:error,
         "native dense Llama generation expected #{key} to be false or absent, got: #{inspect(other)}"}
    end
  end

  defp validate_hidden_act(raw) do
    case raw["hidden_act"] || raw["hidden_activation"] do
      nil ->
        :ok

      "silu" ->
        :ok

      other ->
        {:error,
         "native dense Llama generation supports hidden_act \"silu\" only, got: #{inspect(other)}"}
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
      rope_freqs: Rope.freqs_from_config!(config),
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
     "native dense Llama generation currently supports device: :gpu only, got: #{inspect(device)}"}
  end

  defp normalize_rope_scaling(nil), do: nil

  defp normalize_rope_scaling(%{type: :llama3} = scaling), do: scaling

  defp normalize_rope_scaling(%{"rope_type" => "llama3"} = scaling) do
    %{
      type: :llama3,
      factor: scaling["factor"],
      low_frequency_factor: scaling["low_freq_factor"],
      high_frequency_factor: scaling["high_freq_factor"],
      original_max_positions: scaling["original_max_position_embeddings"]
    }
  end

  defp normalize_rope_scaling(%{"type" => "llama3"} = scaling) do
    normalize_rope_scaling(Map.put(scaling, "rope_type", "llama3"))
  end

  defp normalize_rope_scaling(%{"rope_type" => type} = scaling) do
    raise ArgumentError,
          "native dense Llama generation supports nil or llama3 rope scaling, got: #{inspect(%{scaling | "rope_type" => type})}"
  end

  defp normalize_rope_scaling(%{type: type} = scaling) do
    raise ArgumentError,
          "native dense Llama generation supports nil or :llama3 rope scaling, got: #{inspect(%{scaling | type: type})}"
  end

  defp normalize_rope_scaling(scaling) do
    raise ArgumentError,
          "native dense Llama generation supports nil or :llama3 rope scaling, got: #{inspect(scaling)}"
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
