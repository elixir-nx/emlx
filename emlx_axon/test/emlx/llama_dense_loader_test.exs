defmodule EMLXAxon.LlamaDenseLoaderTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Llama.{DenseLoader, Generate, Model, Rope}

  test "builds native state from Bumblebee dense Llama params" do
    params = %Axon.ModelState{data: params()}

    assert {:ok, state} = DenseLoader.from_model_info(%{params: params, spec: spec()})

    assert Nx.shape(state.embed_tokens) == {16, 4}
    assert Nx.shape(state.lm_head) == {16, 4}
    assert Nx.shape(state.norm) == {4}
    assert state.config.hidden_size == 4
    assert state.config.num_hidden_layers == 1

    [layer] = state.layers

    {_norm1, _norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} = layer

    assert Nx.shape(q_proj) == {4, 4}
    assert Nx.shape(k_proj) == {4, 2}
    assert Nx.shape(v_proj) == {4, 2}
    assert Nx.shape(o_proj) == {4, 4}
    assert Nx.shape(gate_proj) == {4, 8}
    assert Nx.shape(up_proj) == {4, 8}
    assert Nx.shape(down_proj) == {8, 4}
  end

  test "rejects unsupported Bumblebee Llama activation" do
    params = %Axon.ModelState{data: params()}
    spec = spec(activation: :gelu)

    assert {:error, message} = DenseLoader.from_model_info(%{params: params, spec: spec})
    assert message =~ "supports activation :silu only"
    assert message =~ ":gelu"
  end

  test "rejects non-keyword model_info options with a clear error" do
    model_info = %{params: %Axon.ModelState{data: params()}, spec: spec()}

    assert {:error, message} = DenseLoader.from_model_info(model_info, :bad_opts)
    assert message =~ "expected opts to be a keyword list"
    assert message =~ ":bad_opts"
  end

  test "uses token ids from Bumblebee generation config" do
    model_info = %{params: %Axon.ModelState{data: params()}, spec: spec()}

    assert {:ok, probe_state} = DenseLoader.from_model_info(model_info)

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {[first_token], _metadata} =
      Generate.generate(input_ids, probe_state,
        max_new_tokens: 1,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    generation_config = %Bumblebee.Text.GenerationConfig{
      eos_token_id: [first_token, 128_009],
      bos_token_id: 128_000
    }

    assert {:ok, state} =
             DenseLoader.from_model_info(model_info, generation_config: generation_config)

    assert state.config.eos_token_id == [first_token, 128_009]
    assert state.config.bos_token_id == 128_000

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 5,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    assert tokens == [first_token]
    assert metadata.finish_reason == :stop
  end

  test "keeps params already loaded on EMLX GPU" do
    params = params() |> gpu_params()
    model_state = %Axon.ModelState{data: params}

    assert {:ok, state} = DenseLoader.from_model_info(%{params: model_state, spec: spec()})

    assert emlx_ref(state.embed_tokens) == emlx_ref(params["embedder.token_embedding"]["kernel"])
    assert emlx_ref(state.norm) == emlx_ref(params["output_norm"]["weight"])
  end

  test "builds native state directly from dense safetensors directory" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:ok, state} = DenseLoader.from_safetensors_dir(dir, type: :f16)

    assert {:ok, file_state} =
             DenseLoader.from_safetensors_files(config_path, [safetensors_path], type: :f16)

    assert Nx.shape(state.embed_tokens) == {16, 4}
    assert Nx.shape(state.lm_head) == {16, 4}
    assert emlx_ref(state.lm_head) == emlx_ref(state.embed_tokens)
    assert Nx.shape(file_state.embed_tokens) == {16, 4}
    assert state.config.eos_token_id == 2

    [layer] = state.layers

    {_norm1, _norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj} = layer

    assert Nx.shape(q_proj) == {4, 4}
    assert Nx.shape(k_proj) == {4, 2}
    assert Nx.shape(v_proj) == {4, 2}
    assert Nx.shape(o_proj) == {4, 4}
    assert Nx.shape(gate_proj) == {4, 8}
    assert Nx.shape(up_proj) == {4, 8}
    assert Nx.shape(down_proj) == {8, 4}
    assert match?(%EMLX.Backend{}, q_proj.data)

    assert tensor_values(q_proj) == [
             0.0,
             4.0,
             8.0,
             12.0,
             1.0,
             5.0,
             9.0,
             13.0,
             2.0,
             6.0,
             10.0,
             14.0,
             3.0,
             7.0,
             11.0,
             15.0
           ]

    assert tensor_values(k_proj) == [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
  end

  test "dense state loaded from safetensors can run native generation" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:ok, state} = DenseLoader.from_safetensors_files(config_path, [safetensors_path])

    input_ids =
      Nx.tensor([[1, 2]], type: :s64)
      |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

    {tokens, metadata} =
      Generate.generate(input_ids, state,
        max_new_tokens: 2,
        max_len: 8,
        sampler: :greedy,
        host_sync: :per_token,
        profile_timing: false
      )

    assert length(tokens) == 2
    assert Enum.all?(tokens, &is_integer/1)
    assert metadata.finish_reason in [:length, :stop]
  end

  test "KV cache dtype follows bf16 safetensors state dtype" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:ok, state} =
             DenseLoader.from_safetensors_files(config_path, [safetensors_path], type: :bf16)

    [{k_cache, v_cache}] = Model.init_kv_cache(state, 8)
    [{k_ref, v_ref}] = Model.init_native_kv_cache(state, 8)

    assert Nx.type(state.embed_tokens) == {:bf, 16}
    assert Nx.type(k_cache) == {:bf, 16}
    assert Nx.type(v_cache) == {:bf, 16}
    assert EMLX.scalar_type(k_ref) == :bfloat16
    assert EMLX.scalar_type(v_ref) == :bfloat16
  end

  test "rejects unsupported safetensors cast type" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:error, message} =
             DenseLoader.from_safetensors_files(config_path, [safetensors_path], type: :fp16)

    assert message =~ "expected :type"
  end

  test "rejects CPU safetensors device for native dense generation" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:error, message} =
             DenseLoader.from_safetensors_files(config_path, [safetensors_path], device: :cpu)

    assert message =~ "supports device: :gpu only"
  end

  test "rejects unsupported safetensors RoPE scaling config" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    config =
      Map.put(config_json(), "rope_scaling", %{
        "rope_type" => "linear",
        "factor" => 2.0
      })

    File.write!(config_path, Jason.encode!(config))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:error, message} = DenseLoader.from_safetensors_files(config_path, [safetensors_path])
    assert message =~ "supports nil or llama3 rope scaling"
  end

  test "rejects unsupported safetensors config features" do
    for {field, value, expected} <- [
          {"attention_bias", true, "attention_bias=true"},
          {"mlp_bias", true, "mlp_bias=true"},
          {"hidden_act", "gelu", ~s(hidden_act "silu")}
        ] do
      dir =
        Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")

      File.mkdir_p!(dir)
      on_exit(fn -> File.rm_rf!(dir) end)

      config_path = Path.join(dir, "config.json")
      safetensors_path = Path.join(dir, "model.safetensors")

      config = Map.put(config_json(), field, value)

      File.write!(config_path, Jason.encode!(config))
      Safetensors.write!(safetensors_path, safetensors_params())

      assert {:error, message} =
               DenseLoader.from_safetensors_files(config_path, [safetensors_path])

      assert message =~ expected
    end
  end

  test "rejects invalid safetensors device for native dense generation" do
    assert {:error, message} =
             DenseLoader.from_safetensors_files("missing-config.json", [], device: :tpu)

    assert message =~ "supports device: :gpu only"
    assert message =~ ":tpu"
  end

  test "returns an error when a safetensors checkpoint is missing a required key" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    tensors = Map.delete(safetensors_params(), "model.layers.0.self_attn.k_proj.weight")

    File.write!(config_path, Jason.encode!(config_json()))
    Safetensors.write!(safetensors_path, tensors)

    assert {:error, message} = DenseLoader.from_safetensors_files(config_path, [safetensors_path])
    assert message =~ "model.layers.0.self_attn.k_proj.weight"
  end

  test "requires lm_head weight when safetensors embeddings are not tied" do
    dir = Path.join(System.tmp_dir!(), "llama_dense_loader_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    config_path = Path.join(dir, "config.json")
    safetensors_path = Path.join(dir, "model.safetensors")

    config = Map.put(config_json(), "tie_word_embeddings", false)

    File.write!(config_path, Jason.encode!(config))
    Safetensors.write!(safetensors_path, safetensors_params())

    assert {:error, message} = DenseLoader.from_safetensors_files(config_path, [safetensors_path])
    assert message =~ "lm_head.weight"
  end

  test "precomputes RoPE frequencies in MLX fast rope convention" do
    freqs =
      %{
        head_dim: 8,
        rope_theta: 10_000.0,
        rope_scaling: nil
      }
      |> Rope.freqs_from_config!()
      |> tensor_values()

    assert_all_close(freqs, [1.0, 10.0, 100.0, 1000.0], atol: 1.0e-4)

    inverse_freqs = Enum.map(freqs, &Kernel./(1.0, &1))
    assert_all_close(inverse_freqs, [1.0, 0.1, 0.01, 0.001], atol: 1.0e-6)
  end

  test "precomputes Llama 3 RoPE scaling frequencies in MLX fast rope convention" do
    freqs =
      %{
        head_dim: 8,
        rope_theta: 10_000.0,
        rope_scaling: %{
          type: :llama3,
          factor: 8.0,
          low_frequency_factor: 1.0,
          high_frequency_factor: 4.0,
          original_max_positions: 8192
        }
      }
      |> Rope.freqs_from_config!()
      |> tensor_values()

    inverse_freqs = Enum.map(freqs, &Kernel./(1.0, &1))
    assert_all_close(inverse_freqs, [1.0, 0.1, 0.01, 0.000213761], atol: 1.0e-6)
  end

  defp spec(opts \\ []) do
    defaults = [
      architecture: :for_causal_language_modeling,
      vocab_size: 16,
      hidden_size: 4,
      intermediate_size: 8,
      attention_head_size: 2,
      num_blocks: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      rotary_embedding_base: 10_000,
      layer_norm_epsilon: 1.0e-6,
      tie_word_embeddings: true
    ]

    struct!(Bumblebee.Text.Llama, Keyword.merge(defaults, opts))
  end

  defp config_json do
    %{
      "hidden_size" => 4,
      "intermediate_size" => 8,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 1,
      "head_dim" => 2,
      "num_hidden_layers" => 1,
      "vocab_size" => 16,
      "rms_norm_eps" => 1.0e-6,
      "rope_theta" => 10_000,
      "tie_word_embeddings" => true,
      "eos_token_id" => 2,
      "bos_token_id" => 1
    }
  end

  defp safetensors_params do
    %{
      "model.embed_tokens.weight" => Nx.iota({16, 4}, type: :f16),
      "model.norm.weight" => f16_ones({4}),
      "model.layers.0.input_layernorm.weight" => f16_ones({4}),
      "model.layers.0.post_attention_layernorm.weight" => f16_ones({4}),
      "model.layers.0.self_attn.q_proj.weight" => Nx.iota({4, 4}, type: :f16),
      "model.layers.0.self_attn.k_proj.weight" => Nx.iota({2, 4}, type: :f16),
      "model.layers.0.self_attn.v_proj.weight" => Nx.iota({2, 4}, type: :f16),
      "model.layers.0.self_attn.o_proj.weight" => Nx.iota({4, 4}, type: :f16),
      "model.layers.0.mlp.gate_proj.weight" => Nx.iota({8, 4}, type: :f16),
      "model.layers.0.mlp.up_proj.weight" => Nx.iota({8, 4}, type: :f16),
      "model.layers.0.mlp.down_proj.weight" => Nx.iota({4, 8}, type: :f16)
    }
  end

  defp f16_ones(shape), do: Nx.broadcast(Nx.tensor(1.0, type: :f16), shape)

  defp params do
    %{}
    |> put("embedder.token_embedding", "kernel", Nx.iota({16, 4}, type: :f32))
    |> put("language_modeling_head.output", "kernel", Nx.iota({16, 4}, type: :f32))
    |> put("output_norm", "weight", Nx.broadcast(1.0, {4}))
    |> put("decoder.blocks.0.self_attention_norm", "weight", Nx.broadcast(1.0, {4}))
    |> put("decoder.blocks.0.output_norm", "weight", Nx.broadcast(1.0, {4}))
    |> put("decoder.blocks.0.self_attention.query", "kernel", Nx.iota({4, 4}, type: :f32))
    |> put("decoder.blocks.0.self_attention.key", "kernel", Nx.iota({4, 2}, type: :f32))
    |> put("decoder.blocks.0.self_attention.value", "kernel", Nx.iota({4, 2}, type: :f32))
    |> put("decoder.blocks.0.self_attention.output", "kernel", Nx.iota({4, 4}, type: :f32))
    |> put("decoder.blocks.0.ffn.gate", "kernel", Nx.iota({4, 8}, type: :f32))
    |> put("decoder.blocks.0.ffn.intermediate", "kernel", Nx.iota({4, 8}, type: :f32))
    |> put("decoder.blocks.0.ffn.output", "kernel", Nx.iota({8, 4}, type: :f32))
  end

  defp put(params, scope, name, tensor) do
    Map.update(params, scope, %{name => tensor}, &Map.put(&1, name, tensor))
  end

  defp gpu_params(params) do
    Map.new(params, fn {scope, values} ->
      values =
        Map.new(values, fn {name, tensor} ->
          {name, Nx.backend_transfer(tensor, {EMLX.Backend, device: :gpu})}
        end)

      {scope, values}
    end)
  end

  defp emlx_ref(%Nx.Tensor{data: %EMLX.Backend{ref: ref}}), do: ref

  defp tensor_values(tensor) do
    tensor
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_flat_list()
  end

  defp assert_all_close(left, right, opts) do
    atol = Keyword.fetch!(opts, :atol)

    assert Enum.zip(left, right)
           |> Enum.all?(fn {left, right} -> abs(left - right) <= atol end)
  end
end
