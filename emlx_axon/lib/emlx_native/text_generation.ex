defmodule EMLX.Native.TextGeneration do
  @moduledoc """
  A `Nx.Serving`-compatible wrapper around the native Qwen3 quantized model.

  Bypasses the Axon graph entirely — the 28-layer forward pass runs as a single
  `mlx::eval` per token (via `EMLX.Validation.Qwen3Quantized.Generate`), avoiding
  the 28 separate Metal command buffer submissions that the Bumblebee + Axon path
  incurs.

  Only Bumblebee tokenization is used from upstream Bumblebee. No Bumblebee model
  function or Axon graph is involved in the decode forward pass.

  ## Usage

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:local, "~/models/Qwen3-0.6B-MLX-4bit"})
      serving = EMLX.Native.TextGeneration.from_mlx4bit(
        "~/models/Qwen3-0.6B-MLX-4bit",
        tokenizer,
        max_new_tokens: 100,
        sampler: :greedy
      )

      result = Nx.Serving.run(serving, "Write a short story about a robot who learns to love.")
      IO.puts(result.results |> hd() |> Map.fetch!(:generated_text))
  """

  alias EMLX.Validation.Qwen3Quantized.{Model, Generate, Loader}

  @cpu_backend Nx.BinaryBackend

  @doc """
  Builds an `Nx.Serving` wrapping the native Qwen3 quantized model.

  Accepts the same text-string input format as `Bumblebee.Text.generation/4`:
  a plain binary or `%{text: binary()}`. Returns `%{results: [%{generated_text: binary(),
  num_tokens: pos_integer()}]}` for a single input and a list of those maps for a batch input.

  ## Options

  - `:max_new_tokens` — max tokens to generate per request (default 100)
  - `:max_len`        — KV cache preallocated token budget (default 2048)
  - `:sampler`        — `:greedy | :top_p_cpu | :top_p_gpu` (default `:greedy`)
  - `:profile_timing` — forwarded to `Generate.generate/3`; when `false`, skips per-token
                        `System.monotonic_time` in the decode loop (default `true`)
  """
  @spec serving(Bumblebee.Tokenizer.t(), Model.State.t(), keyword()) :: Nx.Serving.t()
  def serving(tokenizer, state, opts \\ []) do
    max_new = Keyword.get(opts, :max_new_tokens, 100)
    max_len = Keyword.get(opts, :max_len, 2048)
    sampler = Keyword.get(opts, :sampler, :greedy)
    profile_timing = Keyword.get(opts, :profile_timing, true)

    Nx.Serving.new(fn _init_opts ->
      fn batch ->
        # `Nx.Batch` stays lazy until a defn/jit entry; `jit_apply(identity)` is the
        # supported way to concatenate stacked entries into concrete tensors.
        %{"input_ids" => input_ids} = Nx.Defn.jit_apply(&Function.identity/1, [batch])

        # Pre-alloc KV cache per call. The cache is safe to reuse because
        # Model.forward/4 always slices to 0..current_len-1, so stale entries
        # beyond current_len are never read (advisor confirmed: no zero-fill needed).
        kv_cache = Model.init_kv_cache(state, max_len)

        {tokens, _timing} =
          Generate.generate(input_ids, state,
            kv_cache: kv_cache,
            max_new_tokens: max_new,
            sampler: sampler,
            profile_timing: profile_timing
          )

        # Return as 2-D tensor {1, num_tokens} matching the shape that
        # Bumblebee.Tokenizer.decode/2 expects for batch input.
        tokens_to_batch_tensor(tokens)
      end
    end)
    |> Nx.Serving.client_preprocessing(fn input ->
      {inputs, multi?} = normalize_input(input)
      texts = Enum.map(inputs, & &1.text)

      # Tokenize on CPU backend (variable length — no compile-time padding needed
      # since the forward pass is a plain Elixir function, not a defn).
      encoded =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      input_ids =
        encoded["input_ids"]
        |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

      {Nx.Batch.concatenate([%{"input_ids" => input_ids}]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {token_ids, _metadata}, multi? ->
      # token_ids: {1, num_generated_tokens} on Nx.BinaryBackend
      num_tokens = elem(Nx.shape(token_ids), 1)
      decoded = Bumblebee.Tokenizer.decode(tokenizer, token_ids)

      case {List.wrap(decoded), multi?} do
        {[text], false} ->
          %{results: [%{generated_text: text, num_tokens: num_tokens}]}

        {texts, _} ->
          texts
          |> Enum.map(&%{results: [%{generated_text: &1, num_tokens: num_tokens}]})
          |> normalize_output(multi?)
      end
    end)
  end

  @doc """
  Convenience: load `%State{}` from an MLX-4bit checkpoint directory and build a serving.

  The tokenizer is expected to come from the same directory (same `tokenizer.json`).
  Loading both from the same directory avoids chat-template / BOS-token divergence.
  """
  @spec from_mlx4bit(Path.t(), Bumblebee.Tokenizer.t(), keyword()) :: Nx.Serving.t()
  def from_mlx4bit(checkpoint_path, tokenizer, opts \\ []) do
    {:ok, state} = Loader.load(checkpoint_path)
    serving(tokenizer, state, opts)
  end

  # ── Input normalisation ──────────────────────────────────────────────────────

  defp normalize_input(text) when is_binary(text), do: {[%{text: text}], false}
  defp normalize_input(%{text: _} = item), do: {[item], false}

  defp normalize_input(inputs) when is_list(inputs) do
    {Enum.map(inputs, &normalize_single/1), true}
  end

  defp normalize_single(text) when is_binary(text), do: %{text: text}
  defp normalize_single(%{text: _} = item), do: item

  defp normalize_output([result], false), do: result
  defp normalize_output(results, _multi?), do: results

  defp tokens_to_batch_tensor(tokens) when is_list(tokens) and tokens != [] do
    tokens
    |> Nx.tensor(type: :s64, backend: @cpu_backend)
    |> Nx.reshape({1, :auto})
  end
end
