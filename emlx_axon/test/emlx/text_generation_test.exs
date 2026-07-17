defmodule EMLXAxon.TextGenerationTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Qwen3.DenseLoader

  describe "run/4" do
    test "rejects list inputs because native generation accepts one sequence" do
      assert {:ok, state} = dense_state()

      assert_raise ArgumentError, ~r/one input at a time/, fn ->
        apply(EMLXAxon.TextGeneration, :run, [:tokenizer, state, ["one", "two"]])
      end
    end

    test "rejects map inputs without text before tokenization" do
      assert {:ok, state} = dense_state()

      assert_raise ArgumentError, ~r/expected input/, fn ->
        apply(EMLXAxon.TextGeneration, :run, [:tokenizer, state, %{prompt: "hello"}])
      end
    end

    test "rejects invalid max_new_tokens before tokenization" do
      assert {:ok, state} = dense_state()

      for max_new_tokens <- [0, -1, 1.5, "1"] do
        assert_raise ArgumentError, ~r/expected :max_new_tokens to be a positive integer/, fn ->
          EMLXAxon.TextGeneration.run(
            __MODULE__.FakeTokenizer.new(),
            state,
            "hello",
            max_new_tokens: max_new_tokens
          )
        end
      end
    end
  end

  describe "stream/5" do
    test "emits chunks for top_p_cpu with default stream host sync" do
      assert {:ok, state} = stable_dense_state()
      parent = self()

      result =
        EMLXAxon.TextGeneration.stream(
          __MODULE__.FakeTokenizer.new(),
          state,
          "hello",
          fn chunk -> send(parent, {:chunk, chunk}) end,
          max_new_tokens: 2,
          max_len: 4,
          sampler: :top_p_cpu,
          stream_chunking: :token
        )

      assert result.token_summary.output == 2
      assert_receive {:chunk, _chunk}
    end

    test "rejects invalid max_new_tokens before tokenization" do
      assert {:ok, state} = dense_state()

      for max_new_tokens <- [0, -1, 1.5, "1"] do
        assert_raise ArgumentError, ~r/expected :max_new_tokens to be a positive integer/, fn ->
          EMLXAxon.TextGeneration.stream(
            __MODULE__.FakeTokenizer.new(),
            state,
            "hello",
            fn _chunk -> :ok end,
            max_new_tokens: max_new_tokens
          )
        end
      end
    end
  end

  describe "serving/3" do
    test "prepares generalized state when the serving is built" do
      assert {:ok, state} = dense_state()

      state = %{
        state
        | layers: [:invalid_layer],
          config: %{state.config | dense_layers?: false}
      }

      assert_raise ArgumentError, ~r/expected a Qwen3 layer tuple/, fn ->
        EMLXAxon.TextGeneration.serving(__MODULE__.FakeTokenizer.new(), state)
      end
    end

    test "rejects list inputs during client preprocessing" do
      assert {:ok, state} = dense_state()
      serving = EMLXAxon.TextGeneration.serving(:tokenizer, state)

      assert_raise ArgumentError, ~r/one input at a time/, fn ->
        Nx.Serving.run(serving, ["one", "two"])
      end
    end

    test "forwards temperature to generation" do
      assert {:ok, state} = dense_state()

      serving =
        EMLXAxon.TextGeneration.serving(
          __MODULE__.FakeTokenizer.new(),
          state,
          max_new_tokens: 1,
          max_len: 4,
          sampler: :top_p_cpu,
          temperature: :invalid,
          top_p: 0.9
        )

      assert catch_error(Nx.Serving.run(serving, "hello"))
    end

    test "forwards top_p to generation" do
      assert {:ok, state} = dense_state()

      serving =
        EMLXAxon.TextGeneration.serving(
          __MODULE__.FakeTokenizer.new(),
          state,
          max_new_tokens: 1,
          max_len: 4,
          sampler: :top_p_cpu,
          temperature: 0.95,
          top_p: :invalid
        )

      assert catch_error(Nx.Serving.run(serving, "hello"))
    end

    test "rejects invalid max_new_tokens before generating" do
      assert {:ok, state} = dense_state()

      for max_new_tokens <- [0, -1, 1.5, "1"] do
        assert_raise ArgumentError, ~r/expected :max_new_tokens to be a positive integer/, fn ->
          serving =
            EMLXAxon.TextGeneration.serving(
              __MODULE__.FakeTokenizer.new(),
              state,
              max_new_tokens: max_new_tokens
            )

          Nx.Serving.run(serving, "hello")
        end
      end
    end
  end

  defmodule FakeTokenizer do
    defstruct []

    def new, do: %__MODULE__{}

    def apply(%__MODULE__{}, [_text]) do
      %{"input_ids" => Nx.tensor([[1]], type: :s64)}
    end

    def decode(%__MODULE__{}, ids), do: inspect(ids)
  end

  defp spec do
    %Bumblebee.Text.Qwen3{
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
    }
  end

  defp dense_state do
    DenseLoader.from_model_info(%{
      params: %Axon.ModelState{data: params()},
      spec: spec()
    })
  end

  defp stable_dense_state do
    DenseLoader.from_model_info(%{
      params: %Axon.ModelState{data: stable_params()},
      spec: spec()
    })
  end

  defp params do
    %{}
    |> put("embedder.token_embedding", "kernel", Nx.iota({16, 4}, type: :f16))
    |> put("language_modeling_head.output", "kernel", Nx.iota({16, 4}, type: :f16))
    |> put("output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention.query_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.key_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.query", "kernel", Nx.iota({4, 4}, type: :f16))
    |> put("decoder.blocks.0.self_attention.key", "kernel", Nx.iota({4, 2}, type: :f16))
    |> put("decoder.blocks.0.self_attention.value", "kernel", Nx.iota({4, 2}, type: :f16))
    |> put("decoder.blocks.0.self_attention.output", "kernel", Nx.iota({4, 4}, type: :f16))
    |> put("decoder.blocks.0.ffn.gate", "kernel", Nx.iota({4, 8}, type: :f16))
    |> put("decoder.blocks.0.ffn.intermediate", "kernel", Nx.iota({4, 8}, type: :f16))
    |> put("decoder.blocks.0.ffn.output", "kernel", Nx.iota({8, 4}, type: :f16))
  end

  defp stable_params do
    %{}
    |> put("embedder.token_embedding", "kernel", zeros({16, 4}))
    |> put("language_modeling_head.output", "kernel", zeros({16, 4}))
    |> put("output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.output_norm", "weight", ones({4}))
    |> put("decoder.blocks.0.self_attention.query_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.key_norm", "weight", ones({2}))
    |> put("decoder.blocks.0.self_attention.query", "kernel", zeros({4, 4}))
    |> put("decoder.blocks.0.self_attention.key", "kernel", zeros({4, 2}))
    |> put("decoder.blocks.0.self_attention.value", "kernel", zeros({4, 2}))
    |> put("decoder.blocks.0.self_attention.output", "kernel", zeros({4, 4}))
    |> put("decoder.blocks.0.ffn.gate", "kernel", zeros({4, 8}))
    |> put("decoder.blocks.0.ffn.intermediate", "kernel", zeros({4, 8}))
    |> put("decoder.blocks.0.ffn.output", "kernel", zeros({8, 4}))
  end

  defp ones(shape), do: Nx.broadcast(Nx.tensor(1.0, type: :f16), shape)
  defp zeros(shape), do: Nx.broadcast(Nx.tensor(0.0, type: :f16), shape)

  defp put(params, scope, name, tensor) do
    Map.update(params, scope, %{name => tensor}, &Map.put(&1, name, tensor))
  end
end
