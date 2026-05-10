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

    # Reference tokens from this emlx implementation's forward pass.
    # Captured on M4 Max, 64 GB, macOS 26.3, EMLX 0.2.0 / MLX 0.31.2.
    #
    # Note: tokens differ from `mlx_lm` because embed_tokens is dequantised to
    # f16 at load time (vs bf16 in mlx_lm), and fast-ops (rms_norm, rope, sdpa)
    # may accumulate in f32 then cast back. Tokens are internally consistent
    # and deterministic on this hardware + software stack.
    #
    # Captured on M4 Max, 64 GB, macOS 26.3, EMLX 0.2.0 / MLX 0.31.2, A4 fast-ops.
    # To regenerate: run `mix test --only quantized_inference` and copy `left:`.
    assert tokens1 == [
             481,
             2585,
             1558,
             468,
             97772,
             3705,
             279,
             990,
             315,
             5746,
             304,
             264,
             15473,
             4128,
             30,
             481,
             3555,
             374,
             279,
             7428
           ]
  end

  test "greedy decode is faster than top_p_cpu", %{state: state, tokenizer: tok} do
    input_ids = tokenize(tok, "Hello")

    {_tokens_g, %{timing: t_greedy}} =
      Generate.generate(input_ids, state, max_new_tokens: 10, sampler: :greedy)

    {_tokens_c, %{timing: t_cpu}} =
      Generate.generate(input_ids, state, max_new_tokens: 10, sampler: :top_p_cpu)

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
