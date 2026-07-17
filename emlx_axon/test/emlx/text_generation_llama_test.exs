defmodule EMLXAxon.TextGenerationLlamaTest do
  use ExUnit.Case, async: true

  alias EMLXAxon.Llama.DenseLoader

  test "run returns generated text for Llama state" do
    assert {:ok, state} = dense_state()

    result =
      EMLXAxon.TextGeneration.run(__MODULE__.FakeTokenizer.new(), state, "hello",
        max_new_tokens: 2,
        max_len: 4
      )

    assert [%{generated_text: text, num_tokens: 2}] = result.results
    assert is_binary(text)
  end

  test "stream emits chunks for Llama state" do
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
        stream_chunking: :token
      )

    assert result.token_summary.output == 2
    assert_receive {:chunk, _chunk}
  end

  test "serving runs Llama state" do
    assert {:ok, state} = dense_state()

    serving =
      EMLXAxon.TextGeneration.serving(__MODULE__.FakeTokenizer.new(), state,
        max_new_tokens: 2,
        max_len: 4
      )

    assert %{results: [%{generated_text: text, num_tokens: 2}]} =
             Nx.Serving.run(serving, "hello")

    assert is_binary(text)
  end

  test "serving reports Llama finish reason from generation metadata" do
    assert {:ok, state} = dense_state()

    serving =
      EMLXAxon.TextGeneration.serving(__MODULE__.FakeTokenizer.new(), state,
        max_new_tokens: 2,
        max_len: 4
      )

    assert %{results: [%{generated_text: text, num_tokens: 2}], finish_reason: finish_reason} =
             Nx.Serving.run(serving, "hello")

    assert is_binary(text)
    assert finish_reason in [:length, :stop]
  end

  test "serving forwards unsupported sampler options to generation" do
    assert {:ok, state} = dense_state()

    serving =
      EMLXAxon.TextGeneration.serving(__MODULE__.FakeTokenizer.new(), state,
        max_new_tokens: 1,
        max_len: 4,
        sampler: :top_p_cpu,
        temperature: :invalid,
        top_p: 0.9
      )

    assert catch_error(Nx.Serving.run(serving, "hello"))
  end

  test "run rejects invalid max_len before cache allocation" do
    assert {:ok, state} = dense_state()

    for max_len <- [0, -1, 1.5, "4"] do
      assert_raise ArgumentError, ~r/expected :max_len to be a positive integer/, fn ->
        EMLXAxon.TextGeneration.run(__MODULE__.FakeTokenizer.new(), state, "hello",
          max_new_tokens: 1,
          max_len: max_len
        )
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
    %Bumblebee.Text.Llama{
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
