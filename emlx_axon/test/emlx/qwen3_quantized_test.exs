defmodule EMLXAxon.Qwen3QuantizedTest do
  use ExUnit.Case, async: false

  # Excluded from default CI runs — requires local checkpoint.
  # Run with: mix test --only quantized_inference
  @moduletag :quantized_inference

  alias EMLXAxon.Qwen3.{Loader, Generate}

  @model_path System.get_env("EMLX_QWEN3_MODEL_PATH") ||
                "~/models/Qwen3-0.6B-MLX-4bit"

  setup_all do
    Nx.default_backend({EMLX.Backend, device: :gpu})

    path = Path.expand(@model_path)

    unless File.dir?(path) do
      flunk("""
      Model not found at #{path}.
      Set the EMLX_QWEN3_MODEL_PATH env var to a local Qwen3-0.6B-MLX-4bit checkout, e.g.:

          huggingface-cli download lmstudio-community/Qwen3-0.6B-MLX-4bit \\
            --local-dir ~/models/Qwen3-0.6B-MLX-4bit
          export EMLX_QWEN3_MODEL_PATH=~/models/Qwen3-0.6B-MLX-4bit
      """)
    end

    {:ok, state} = Loader.load(path)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:local, path})

    {:ok, %{state: state, tokenizer: tokenizer}}
  end

  defp tokenize(tokenizer, text) do
    %{"input_ids" => input_ids} = Bumblebee.apply_tokenizer(tokenizer, text)
    Nx.backend_transfer(input_ids, {EMLX.Backend, device: :gpu})
  end

  test "greedy decode produces deterministic token sequence", %{state: state, tokenizer: tok} do
    input_ids = tokenize(tok, "Write one sentence about Elixir.")

    {tokens1, _} = Generate.generate(input_ids, state, max_new_tokens: 20, sampler: :greedy)
    {tokens2, _} = Generate.generate(input_ids, state, max_new_tokens: 20, sampler: :greedy)

    # Greedy must be deterministic across runs
    assert tokens1 == tokens2

    # Reference from mlx_lm greedy decode on the same local MLX-4bit checkpoint:
    #   prompt = "Write one sentence about Elixir."  (raw encode, no chat template)
    #   max_tokens = 20, temp = 0
    # Decodes to:
    #   " - How does Elixir handle the case when there is no data in the database? - What is"
    #
    # Regenerate with:
    #   python -c 'from mlx_lm import load; from mlx_lm.generate import generate_step; ...'
    expected = [
      481,
      2585,
      1558,
      468,
      97772,
      3705,
      279,
      1142,
      979,
      1052,
      374,
      902,
      821,
      304,
      279,
      4625,
      30,
      481,
      3555,
      374
    ]

    assert tokens1 == expected
  end

  test "greedy decode is faster than top_p_cpu", %{state: state, tokenizer: tok} do
    input_ids = tokenize(tok, "Hello")

    {_tokens_g, %{timing: t_greedy}} =
      Generate.generate(input_ids, state,
        max_new_tokens: 10,
        sampler: :greedy,
        profile_timing: true
      )

    {_tokens_c, %{timing: t_cpu}} =
      Generate.generate(input_ids, state,
        max_new_tokens: 10,
        sampler: :top_p_cpu,
        profile_timing: true
      )

    greedy_median = median(t_greedy.per_token_ms)
    cpu_median = median(t_cpu.per_token_ms)

    # CPU sampler must add measurable overhead vs greedy (A0 showed ~42 ms)
    assert cpu_median > greedy_median,
           "Expected top_p_cpu (#{cpu_median} ms) > greedy (#{greedy_median} ms)"
  end

  defp median([]), do: 0

  defp median(list) do
    sorted = Enum.sort(list)
    n = length(sorted)
    Enum.at(sorted, div(n, 2))
  end
end
