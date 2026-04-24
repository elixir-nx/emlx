defmodule EMLX.Validation.Qwen3Quantized.Generate do
  @moduledoc """
  Autoregressive token generation loop.

  `EMLX.eval` is called once per token, at the sampler boundary, so the
  full 28-layer forward runs as a single lazy MLX graph before any CPU
  synchronisation.

  ## Usage

      {:ok, state}     = Loader.load(model_path)
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-0.6B"})

      encoded   = Bumblebee.apply_tokenizer(tokenizer, "Hello")
      input_ids = encoded["input_ids"]

      {tokens, timing} = Generate.generate(input_ids, state,
        max_new_tokens: 100,
        sampler: :greedy
      )
  """

  alias EMLX.Validation.Qwen3Quantized.{Model, Sampler}

  @default_max_new_tokens 100
  @default_max_len 2048

  @doc """
  Run the autoregressive generation loop.

  ## Options

  - `:max_new_tokens` — max number of tokens to generate (default 100)
  - `:max_len`        — KV cache allocation size (default 2048)
  - `:sampler`        — `:greedy | :top_p_cpu | :top_p_gpu` (default `:greedy`)
  - `:temperature`    — float, passed to samplers that use it (default 0.95)
  - `:top_p`          — float, passed to nucleus samplers (default 0.9)
  - `:rng_key`        — `Nx.Random.key/1`, used by `:top_p_gpu`

  ## Returns

  `{generated_token_ids, %{timing: timing_map}}` where `timing_map` has:
  - `:prefill_ms`     — first-token time in milliseconds
  - `:per_token_ms`   — list of per-token decode times (median ≈ steady-state)
  - `:total_ms`       — wall time for the whole call
  """
  @spec generate(Nx.Tensor.t(), Model.State.t(), keyword()) ::
          {[non_neg_integer()], map()}
  def generate(input_ids, %Model.State{} = state, opts \\ []) do
    max_new  = Keyword.get(opts, :max_new_tokens, @default_max_new_tokens)
    max_len  = Keyword.get(opts, :max_len, @default_max_len)
    sampler  = Keyword.get(opts, :sampler, :greedy)
    temp     = Keyword.get(opts, :temperature, 0.95)
    top_p    = Keyword.get(opts, :top_p, 0.9)
    rng_key  = Keyword.get(opts, :rng_key, Nx.Random.key(42))

    kv_cache = Model.init_kv_cache(state, max_len)

    # Prefill: pass the full prompt in one forward
    t0 = System.monotonic_time(:millisecond)

    {logits, kv_cache} = Model.forward(input_ids, kv_cache, 0, state)
    first_token        = sample(logits, sampler, temp, top_p, rng_key)
    # Sync to GPU here (once per token)
    EMLX.eval(EMLX.Backend.from_nx(first_token))

    t1 = System.monotonic_time(:millisecond)
    prefill_ms = t1 - t0

    [seq_len] = Nx.shape(input_ids) |> Tuple.to_list() |> tl()
    current_len = seq_len

    # Decode loop: max_new - 1 additional steps (prefill already produced one token)
    {tokens, per_token_ms, _kv, _cur} =
      Enum.reduce_while(
        1..max(max_new - 1, 0),
        {[Nx.to_number(first_token)], [], kv_cache, current_len}, fn
        _step, {acc_tokens, acc_times, kv, cur} ->
          last_id = hd(acc_tokens)

          if last_id == eos_token_id(state) do
            {:halt, {acc_tokens, acc_times, kv, cur}}
          else
            ts = System.monotonic_time(:millisecond)

            next_input = Nx.tensor([[last_id]], type: :s64)
                         |> Nx.backend_transfer({EMLX.Backend, device: :gpu})

            {logits, kv_new} = Model.forward(next_input, kv, cur, state)
            next_token       = sample(logits, sampler, temp, top_p, rng_key)
            EMLX.eval(EMLX.Backend.from_nx(next_token))

            te = System.monotonic_time(:millisecond)

            {:cont,
             {[Nx.to_number(next_token) | acc_tokens], [te - ts | acc_times], kv_new, cur + 1}}
          end
      end)

    t_end = System.monotonic_time(:millisecond)

    timing = %{
      prefill_ms:    prefill_ms,
      per_token_ms:  Enum.reverse(per_token_ms),
      total_ms:      t_end - t0
    }

    {Enum.reverse(tokens), %{timing: timing}}
  end

  # ── Helpers ───────────────────────────────────────────────────────────────

  defp sample(logits, :greedy, _temp, _top_p, _key), do: Sampler.greedy(logits)

  defp sample(logits, :top_p_cpu, temp, top_p, _key),
    do: Sampler.top_p_cpu(logits, temp, top_p)

  defp sample(logits, :top_p_gpu, temp, _top_p, _key) do
    # Generate a fresh key per step so the sample is independent across tokens.
    key = Nx.Random.key(:os.system_time(:nanosecond))
          |> Nx.backend_transfer({EMLX.Backend, device: :gpu})
    Sampler.top_p_gpu(logits, key, temperature: temp)
  end

  defp eos_token_id(%Model.State{config: cfg}) do
    cfg[:eos_token_id] || 151_645
  end
end
